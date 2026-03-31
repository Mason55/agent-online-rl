#!/usr/bin/env python
"""
批量 LoRA 训练入口脚本，由 ResourceScheduler 调用。

用法：
    python trainer/train_batch_lora.py \
        --jobs '[{"user_id": "u1", "trajectory_ids": ["t1", "t2"]}]' \
        --base-model /models/qwen3-7b \
        --config config/ppo_lora_trainer.yaml \
        --lora-repo-root /data/lora_repo \
        --vllm-url http://localhost:8000 \
        --db-path trajectories.db
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Batch LoRA training for multiple users")
    p.add_argument("--jobs", required=True, help="JSON list of UserTrainingJob dicts")
    p.add_argument("--base-model", required=True, help="Base model path (HuggingFace format)")
    p.add_argument("--config", default="config/ppo_lora_trainer.yaml", help="verl LoRA training config")
    p.add_argument("--lora-repo-root", required=True, help="LoRA repository root directory")
    p.add_argument("--vllm-url", required=True, help="vLLM inference service URL")
    p.add_argument("--db-path", default="trajectories.db", help="Trajectory SQLite database path")
    p.add_argument("--nproc-per-node", type=int, default=0, help="GPUs per node (0 = auto-detect)")
    p.add_argument("--tmp-root", default="/tmp/agent_rl", help="Temp directory for training artifacts")
    return p.parse_args()


def main():
    args = parse_args()

    # 延迟导入，避免在没有 torch 的环境中解析 --help 时报错
    from inference.notifier import InferenceNotifier
    from storage.lora_repo import LoRARepository
    from storage.models import UserTrainingJob
    from storage.trajectory_store import TrajectoryStore
    from trainer.batch_lora_trainer import BatchUserLoRATrainer

    try:
        jobs_data = json.loads(args.jobs)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse --jobs: %s", e)
        sys.exit(1)

    user_jobs = [
        UserTrainingJob(
            user_id=j["user_id"],
            trajectory_ids=j["trajectory_ids"],
        )
        for j in jobs_data
    ]

    logger.info("Starting batch LoRA training for %d users", len(user_jobs))

    store = TrajectoryStore(args.db_path)
    lora_repo = LoRARepository(args.lora_repo_root)
    notifier = InferenceNotifier(args.vllm_url)

    trainer = BatchUserLoRATrainer(
        base_model_path=args.base_model,
        lora_repo=lora_repo,
        notifier=notifier,
        store=store,
        verl_config_path=args.config,
        nproc_per_node=args.nproc_per_node,
        tmp_root=args.tmp_root,
    )
    failures = trainer.run(user_jobs)
    if failures:
        logger.error("Batch LoRA training finished with %d/%d user failures", failures, len(user_jobs))
        sys.exit(1)
    logger.info("Batch LoRA training completed successfully")


if __name__ == "__main__":
    main()
