"""
TrajectoryDataset — 将轨迹数据转换为 verl SFT Trainer 可消费的格式。

verl MultiTurnSFTDataset 期望 Parquet 文件包含 `messages` 列，
每行是 list[dict{"role": str, "content": str}]，即标准多轮对话格式。

本模块提供两个公开接口：
  1. trajectories_to_parquet()  — 将 Trajectory 列表序列化为 Parquet 文件
  2. TrajectoryDataset          — 直接封装为 torch Dataset（可选用，用于调试或非 verl 路径）
"""

import logging
from pathlib import Path

from storage.models import Trajectory

logger = logging.getLogger(__name__)


def trajectories_to_parquet(
    trajectories: list[Trajectory],
    output_path: str,
    reward_key: str = "reward",
) -> str:
    """将 Trajectory 列表序列化为 Parquet 文件，供 verl MultiTurnSFTDataset 读取。

    输出 schema：
      messages: list[dict]   — 多轮对话（role + content）
      reward:   float        — trajectory 级别 reward，范围 [-1, 1]
      trajectory_id: str     — 来源 ID（调试用）
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for trajectories_to_parquet") from e

    records = []
    for traj in trajectories:
        messages = [{"role": t.role, "content": t.content} for t in traj.turns]
        records.append({
            "messages": messages,
            reward_key: traj.reward if traj.reward is not None else 0.0,
            "trajectory_id": traj.trajectory_id,
        })

    df = pd.DataFrame(records)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d trajectories to %s", len(records), output_path)
    return output_path


class TrajectoryDataset:
    """轻量 PyTorch Dataset 封装，用于调试和非 verl 训练路径。

    在使用 verl SFT Trainer 的正式训练路径中，请使用 trajectories_to_parquet()
    将数据写成 Parquet 后交由 verl 的 MultiTurnSFTDataset 处理。
    """

    def __init__(self, trajectories: list[Trajectory], tokenizer, max_length: int = 2048):
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is required for TrajectoryDataset") from e

        self._torch = torch
        self.samples: list[dict] = []

        for traj in trajectories:
            messages = [{"role": t.role, "content": t.content} for t in traj.turns]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = "\n".join(f"{t.role}: {t.content}" for t in traj.turns)

            tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors=None)
            self.samples.append({
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "reward": traj.reward if traj.reward is not None else 0.0,
                "trajectory_id": traj.trajectory_id,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        seq_len = len(sample["input_ids"])
        token_rewards = self._torch.zeros(seq_len)
        token_rewards[-1] = sample["reward"]  # reward 分配到最后一个有效 token
        return {
            "input_ids": self._torch.tensor(sample["input_ids"], dtype=self._torch.long),
            "attention_mask": self._torch.tensor(sample["attention_mask"], dtype=self._torch.long),
            "token_level_rewards": token_rewards,
            "trajectory_id": sample["trajectory_id"],
        }
