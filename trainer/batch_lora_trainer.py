"""
BatchUserLoRATrainer — 基础模型只加载一次，顺序为多个用户训练独立 LoRA。

训练流程（每个用户）：
  1. 将轨迹写入临时 Parquet 文件
  2. 确定 LoRA 起点（历史 LoRA 增量训练，或随机初始化）
  3. 调用 verl SFT Trainer（torchrun 子进程）训练 LoRA
  4. 将 LoRA 权重发布到仓库，通知推理服务热加载
  5. 标记轨迹状态为 TRAINED

容错：单用户失败不影响后续用户，失败轨迹回滚为 FAILED。
"""

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from inference.notifier import InferenceNotifier
from storage.lora_repo import LoRARepository
from storage.models import UserTrainingJob
from storage.trajectory_store import TrajectoryStore
from trainer.trajectory_dataset import trajectories_to_parquet

logger = logging.getLogger(__name__)


def run_verl_lora_sft(
    base_model: str,
    train_parquet: str,
    output_dir: str,
    config_path: str,
    lora_adapter_path: Optional[str] = None,
    nproc_per_node: int = 1,
    extra_overrides: Optional[list[str]] = None,
) -> None:
    """以子进程方式运行 verl SFT Trainer 训练 LoRA。

    Args:
        base_model:        基础模型路径（HuggingFace 格式）
        train_parquet:     训练数据 Parquet 文件路径
        output_dir:        LoRA 权重输出目录
        config_path:       verl LoRA 训练配置文件路径（YAML）
        lora_adapter_path: 增量训练时的历史 LoRA 路径，None 表示随机初始化
        nproc_per_node:    每节点 GPU 数量
        extra_overrides:   额外 Hydra 配置覆盖项
    """
    overrides = [
        f"model.path={base_model}",
        f"data.train_files={train_parquet}",
        f"trainer.default_local_dir={output_dir}",
    ]
    if lora_adapter_path:
        overrides.append(f"model.lora_adapter_path={lora_adapter_path}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "-m", "verl.trainer.sft_trainer_engine",
        f"--config-path={Path(config_path).parent}",
        f"--config-name={Path(config_path).stem}",
    ] + overrides

    logger.info("Running verl SFT: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=False)
    logger.info("verl SFT completed with return code %d", result.returncode)


class BatchUserLoRATrainer:
    """批量顺序 LoRA 训练器。

    Args:
        base_model_path:  基础模型路径
        lora_repo:        LoRA 版本化仓库
        notifier:         推理服务热加载通知器
        store:            轨迹存储
        verl_config_path: verl LoRA 训练配置文件路径
        nproc_per_node:   每节点 GPU 数量（默认自动检测）
        tmp_root:         临时文件根目录（默认 /tmp/agent_rl）
    """

    def __init__(
        self,
        base_model_path: str,
        lora_repo: LoRARepository,
        notifier: InferenceNotifier,
        store: TrajectoryStore,
        verl_config_path: str,
        nproc_per_node: int = 0,
        tmp_root: str = "/tmp/agent_rl",
    ):
        self.base_model_path = base_model_path
        self.lora_repo = lora_repo
        self.notifier = notifier
        self.store = store
        self.verl_config_path = verl_config_path
        self.nproc_per_node = nproc_per_node or _detect_gpu_count()
        self.tmp_root = tmp_root

    def run(self, user_batch: list[UserTrainingJob]) -> None:
        """顺序训练 user_batch 中的每个用户，单用户失败不影响其他用户。"""
        logger.info("Starting batch training for %d users", len(user_batch))
        for job in user_batch:
            try:
                self._train_one_user(job)
            except Exception:
                logger.exception("Training failed for user %s", job.user_id)
                self.store.mark_failed(job.trajectory_ids)

    def _train_one_user(self, job: UserTrainingJob) -> None:
        """完整训练单个用户的 LoRA 并发布。"""
        logger.info("Training LoRA for user %s (%d trajectories)", job.user_id, len(job.trajectory_ids))
        trajectories = self.store.load(job.user_id, job.trajectory_ids)
        if not trajectories:
            logger.warning("No trajectories found for user %s, skipping", job.user_id)
            return

        # 过滤掉没有 reward 的轨迹（Judge 计算失败的）
        scored = [t for t in trajectories if t.reward is not None]
        if not scored:
            logger.warning("All trajectories for user %s lack reward scores, skipping", job.user_id)
            return

        run_dir = Path(self.tmp_root) / job.user_id / str(uuid.uuid4())
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._run_training(job, scored, run_dir)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def _run_training(self, job: UserTrainingJob, trajectories, run_dir: Path) -> None:
        # 1. 写入训练数据 Parquet
        parquet_path = str(run_dir / "train.parquet")
        trajectories_to_parquet(trajectories, parquet_path)

        # 2. 确定 LoRA 起点
        existing = self.lora_repo.get_latest(job.user_id)
        lora_adapter_path = existing.path if existing else None
        if lora_adapter_path:
            logger.info("Incremental training from %s for user %s", lora_adapter_path, job.user_id)
        else:
            logger.info("Fresh LoRA init for user %s", job.user_id)

        # 3. 训练
        output_dir = str(run_dir / "output")
        run_verl_lora_sft(
            base_model=self.base_model_path,
            train_parquet=parquet_path,
            output_dir=output_dir,
            config_path=self.verl_config_path,
            lora_adapter_path=lora_adapter_path,
            nproc_per_node=self.nproc_per_node,
        )

        # 4. 发布 LoRA 到仓库
        reward_avg = sum(t.reward for t in trajectories) / len(trajectories)
        lora_version = self.lora_repo.publish(
            user_id=job.user_id,
            lora_path=output_dir,
            metadata={
                "trajectory_count": len(trajectories),
                "reward_avg": reward_avg,
            },
            base_model=self.base_model_path,
        )
        logger.info("Published LoRA %s for user %s (reward_avg=%.3f)", lora_version.version, job.user_id, reward_avg)

        # 5. 通知推理服务热加载
        try:
            self.notifier.notify_update(job.user_id, lora_version.path)
        except Exception:
            logger.warning("Failed to notify inference service for user %s (non-fatal)", job.user_id, exc_info=True)

        # 6. 标记训练完成
        self.store.mark_trained(job.trajectory_ids)


def _detect_gpu_count() -> int:
    """自动检测可用 GPU 数量，无 GPU 时返回 1（CPU 模式）。"""
    try:
        import torch
        n = torch.cuda.device_count()
        return n if n > 0 else 1
    except Exception:
        return 1
