"""
TrainingScheduler — 定时扫描 + 批量触发 LoRA 训练。

每 scan_interval_seconds（默认 10 分钟）扫描一次：
  1. 找出所有 pending_count >= threshold 的用户
  2. 原子获取这批轨迹并标记为 TRAINING
  3. 提交单次批量训练任务（多用户共享同一集群启动成本）
"""

import logging
import threading
import time
from typing import Optional

from scheduler.resource_scheduler import ResourceScheduler
from storage.models import UserTrainingJob
from storage.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


class TrainingScheduler:
    def __init__(
        self,
        store: TrajectoryStore,
        resource_scheduler: ResourceScheduler,
        threshold: int = 200,
        scan_interval_seconds: int = 600,  # 10 分钟
    ):
        self.store = store
        self.resource_scheduler = resource_scheduler
        self.threshold = threshold
        self.scan_interval_seconds = scan_interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """启动后台扫描线程（daemon，进程退出时自动终止）。"""
        if self._thread and self._thread.is_alive():
            logger.warning("TrainingScheduler is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scan_loop, daemon=True, name="TrainingScheduler")
        self._thread.start()
        logger.info(
            "TrainingScheduler started: threshold=%d scan_interval=%ds",
            self.threshold, self.scan_interval_seconds,
        )

    def stop(self) -> None:
        """停止后台扫描线程。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("TrainingScheduler stopped")

    def _scan_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception:
                logger.exception("Error during training scheduler scan")
            self._stop_event.wait(self.scan_interval_seconds)

    def _scan_once(self) -> Optional[str]:
        """执行一次扫描，若有满足条件的用户则提交批量训练任务，返回 job_id 或 None。"""
        eligible_users = self.store.get_users_above_threshold(self.threshold)
        if not eligible_users:
            logger.debug("No users above training threshold (%d)", self.threshold)
            return None

        logger.info("Found %d users above training threshold", len(eligible_users))

        user_jobs: list[UserTrainingJob] = []
        for user_id in eligible_users:
            trajectories = self.store.fetch_and_mark_training(user_id, self.threshold)
            if trajectories:
                user_jobs.append(UserTrainingJob(
                    user_id=user_id,
                    trajectory_ids=[t.trajectory_id for t in trajectories],
                ))
                logger.debug("Queued %d trajectories for user %s", len(trajectories), user_id)

        if not user_jobs:
            # 极少发生：race condition 下其他进程已抢先处理
            return None

        try:
            job_id = self.resource_scheduler.submit_batch_training_job(user_jobs)
            logger.info(
                "Submitted batch training job %s for %d users (%d total trajectories)",
                job_id,
                len(user_jobs),
                sum(len(j.trajectory_ids) for j in user_jobs),
            )
            return job_id
        except Exception:
            # 提交失败：将已标记为 TRAINING 的轨迹回滚为 FAILED，等待下次重试
            logger.exception("Failed to submit batch training job, marking trajectories as failed")
            for job in user_jobs:
                self.store.mark_failed(job.trajectory_ids)
            return None
