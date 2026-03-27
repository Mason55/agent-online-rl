"""
ResourceScheduler — 可插拔资源调度抽象接口。

内置实现：
  LocalProcessScheduler  本地子进程（开发调试）
  RayJobScheduler        Ray Job（与 verl 天然集成，生产推荐）
  K8sJobScheduler        K8s Job（大规模集群）
"""

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from storage.models import JobStatus, UserTrainingJob

logger = logging.getLogger(__name__)


class ResourceScheduler(ABC):
    @abstractmethod
    def submit_batch_training_job(self, user_jobs: list[UserTrainingJob]) -> str:
        """提交批量训练任务，返回 job_id。"""

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """查询任务状态。"""

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """取消任务。"""


class LocalProcessScheduler(ResourceScheduler):
    """本地子进程调度器，用于开发调试。

    Args:
        script_path: train_batch_lora.py 脚本路径
        base_model: 基础模型路径
        config_path: verl LoRA 训练配置路径
        lora_repo_root: LoRA 仓库根目录
        vllm_url: vLLM 服务地址
        db_path: 轨迹数据库路径
        extra_env: 额外环境变量
    """

    def __init__(
        self,
        script_path: str,
        base_model: str,
        config_path: str,
        lora_repo_root: str,
        vllm_url: str,
        db_path: str = "trajectories.db",
        extra_env: Optional[dict] = None,
    ):
        self.script_path = script_path
        self.base_model = base_model
        self.config_path = config_path
        self.lora_repo_root = lora_repo_root
        self.vllm_url = vllm_url
        self.db_path = db_path
        self.extra_env = extra_env or {}
        self._processes: dict[str, subprocess.Popen] = {}

    def submit_batch_training_job(self, user_jobs: list[UserTrainingJob]) -> str:
        jobs_json = json.dumps([
            {"user_id": j.user_id, "trajectory_ids": j.trajectory_ids}
            for j in user_jobs
        ])
        import os
        env = os.environ.copy()
        env.update(self.extra_env)

        cmd = [
            "python", self.script_path,
            "--jobs", jobs_json,
            "--base-model", self.base_model,
            "--config", self.config_path,
            "--lora-repo-root", self.lora_repo_root,
            "--vllm-url", self.vllm_url,
            "--db-path", self.db_path,
        ]
        proc = subprocess.Popen(cmd, env=env)
        job_id = str(proc.pid)
        self._processes[job_id] = proc
        logger.info("Launched local training process pid=%s for %d users", job_id, len(user_jobs))
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        proc = self._processes.get(job_id)
        if proc is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        retcode = proc.poll()
        if retcode is None:
            return JobStatus.RUNNING
        return JobStatus.COMPLETED if retcode == 0 else JobStatus.FAILED

    def cancel_job(self, job_id: str) -> None:
        proc = self._processes.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            logger.info("Terminated local training process pid=%s", job_id)


class RayJobScheduler(ResourceScheduler):
    """Ray Job 调度器，与 verl 天然集成（生产推荐）。

    Args:
        ray_address: Ray cluster address（如 "ray://head:10001"）
        working_dir: 提交任务的工作目录（包含代码）
        script_path: train_batch_lora.py 路径（相对 working_dir）
        base_model / config_path / lora_repo_root / vllm_url / db_path: 同 Local
        num_cpus / num_gpus: 任务资源需求
    """

    def __init__(
        self,
        ray_address: str,
        working_dir: str,
        script_path: str,
        base_model: str,
        config_path: str,
        lora_repo_root: str,
        vllm_url: str,
        db_path: str = "trajectories.db",
        num_cpus: int = 4,
        num_gpus: int = 1,
    ):
        self.ray_address = ray_address
        self.working_dir = working_dir
        self.script_path = script_path
        self.base_model = base_model
        self.config_path = config_path
        self.lora_repo_root = lora_repo_root
        self.vllm_url = vllm_url
        self.db_path = db_path
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

    def _get_client(self):
        try:
            from ray.job_submission import JobSubmissionClient
            return JobSubmissionClient(self.ray_address)
        except ImportError as e:
            raise ImportError("ray[default] is required for RayJobScheduler") from e

    def submit_batch_training_job(self, user_jobs: list[UserTrainingJob]) -> str:
        jobs_json = json.dumps([
            {"user_id": j.user_id, "trajectory_ids": j.trajectory_ids}
            for j in user_jobs
        ])
        entrypoint = (
            f"python {self.script_path}"
            f" --jobs '{jobs_json}'"
            f" --base-model {self.base_model}"
            f" --config {self.config_path}"
            f" --lora-repo-root {self.lora_repo_root}"
            f" --vllm-url {self.vllm_url}"
            f" --db-path {self.db_path}"
        )
        client = self._get_client()
        job_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env={"working_dir": self.working_dir},
            entrypoint_resources={"CPU": self.num_cpus, "GPU": self.num_gpus},
        )
        logger.info("Submitted Ray job %s for %d users", job_id, len(user_jobs))
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        from ray.job_submission import JobStatus as RayStatus
        client = self._get_client()
        status = client.get_job_status(job_id)
        mapping = {
            RayStatus.RUNNING: JobStatus.RUNNING,
            RayStatus.SUCCEEDED: JobStatus.COMPLETED,
            RayStatus.FAILED: JobStatus.FAILED,
            RayStatus.STOPPED: JobStatus.FAILED,
        }
        return mapping.get(status, JobStatus.RUNNING)

    def cancel_job(self, job_id: str) -> None:
        client = self._get_client()
        client.stop_job(job_id)
        logger.info("Cancelled Ray job %s", job_id)


class K8sJobScheduler(ResourceScheduler):
    """K8s Job 调度器（存根，需配置 kubeconfig）。"""

    def __init__(self, namespace: str = "default", image: str = "", **kwargs):
        self.namespace = namespace
        self.image = image

    def submit_batch_training_job(self, user_jobs: list[UserTrainingJob]) -> str:
        raise NotImplementedError("K8sJobScheduler is not yet implemented")

    def get_job_status(self, job_id: str) -> JobStatus:
        raise NotImplementedError("K8sJobScheduler is not yet implemented")

    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError("K8sJobScheduler is not yet implemented")
