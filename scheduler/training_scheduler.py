"""
TrainingScheduler — 定时扫描 + 批量触发 LoRA 训练。

每 scan_interval_seconds（默认 10 分钟）扫描一次：
  1. 检查之前提交的训练任务是否已完成/失败，回退失败的轨迹为 pending
  2. 找出所有 pending_count >= threshold 的用户
  3. 原子获取这批轨迹并标记为 TRAINING
  4. 提交单次批量训练任务（多用户共享同一集群启动成本）
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from scheduler.resource_scheduler import ResourceScheduler
from storage.models import JobStatus, UserTrainingJob
from storage.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class _ActiveJob:
    """Tracks a submitted training job so we can monitor it."""
    job_id: str
    user_jobs: list[UserTrainingJob]


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
        self._active_jobs: list[_ActiveJob] = []

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
                self._check_active_jobs()
                self._scan_once()
            except Exception:
                logger.exception("Error during training scheduler scan")
            self._stop_event.wait(self.scan_interval_seconds)

    def _check_active_jobs(self) -> None:
        """Check previously submitted jobs; roll back trajectories on failure, clean up on success."""
        still_active: list[_ActiveJob] = []
        for active in self._active_jobs:
            try:
                status = self.resource_scheduler.get_job_status(active.job_id)
            except KeyError:
                logger.warning("Job %s no longer tracked by scheduler, assuming failed", active.job_id)
                status = JobStatus.FAILED

            if status == JobStatus.RUNNING:
                still_active.append(active)
                continue

            all_ids = [tid for j in active.user_jobs for tid in j.trajectory_ids]
            if status == JobStatus.COMPLETED:
                logger.info("Training job %s completed (%d trajectories)", active.job_id, len(all_ids))
                self.store.mark_trained(all_ids)
            else:
                logger.warning(
                    "Training job %s failed — resetting %d trajectories to pending for retry",
                    active.job_id, len(all_ids),
                )
                self.store.reset_to_pending(all_ids)

        self._active_jobs = still_active

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
            return None

        try:
            job_id = self.resource_scheduler.submit_batch_training_job(user_jobs)
            logger.info(
                "Submitted batch training job %s for %d users (%d total trajectories)",
                job_id,
                len(user_jobs),
                sum(len(j.trajectory_ids) for j in user_jobs),
            )
            self._active_jobs.append(_ActiveJob(job_id=job_id, user_jobs=user_jobs))
            return job_id
        except Exception:
            logger.exception("Failed to submit batch training job, resetting trajectories to pending")
            for job in user_jobs:
                self.store.reset_to_pending(job.trajectory_ids)
            return None
