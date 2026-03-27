"""单元测试：TrainingScheduler 扫描逻辑"""

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from scheduler.resource_scheduler import ResourceScheduler
from scheduler.training_scheduler import TrainingScheduler
from storage.models import Trajectory, Turn, UserTrainingJob
from storage.trajectory_store import TrajectoryStore


def _make_traj(traj_id: str, user_id: str) -> Trajectory:
    return Trajectory(
        trajectory_id=traj_id,
        user_id=user_id,
        session_id="s",
        turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
        created_at=datetime.now(),
        reward=0.5,
    )


class MockScheduler(ResourceScheduler):
    def __init__(self):
        self.submitted_jobs: list[list[UserTrainingJob]] = []

    def submit_batch_training_job(self, user_jobs):
        self.submitted_jobs.append(user_jobs)
        return "mock-job-id"

    def get_job_status(self, job_id):
        from storage.models import JobStatus
        return JobStatus.COMPLETED

    def cancel_job(self, job_id):
        pass


class TestTrainingScheduler:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = TrajectoryStore(self.tmp.name)
        self.mock_scheduler = MockScheduler()
        self.scheduler = TrainingScheduler(
            store=self.store,
            resource_scheduler=self.mock_scheduler,
            threshold=3,
        )

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_scan_submits_eligible_users(self):
        """3 个达到阈值的用户 → submit_batch_training_job 被调用一次，参数正确。"""
        for uid in ["u1", "u2", "u3"]:
            for i in range(3):
                self.store.save(_make_traj(f"{uid}-t{i}", uid))

        self.scheduler._scan_once()

        assert len(self.mock_scheduler.submitted_jobs) == 1
        submitted_user_ids = {j.user_id for j in self.mock_scheduler.submitted_jobs[0]}
        assert submitted_user_ids == {"u1", "u2", "u3"}

    def test_scan_skips_users_below_threshold(self):
        """只有 2 条轨迹（< 3）的用户不触发训练。"""
        for i in range(2):
            self.store.save(_make_traj(f"t{i}", "user1"))

        self.scheduler._scan_once()
        assert len(self.mock_scheduler.submitted_jobs) == 0

    def test_scan_marks_trajectories_training(self):
        """扫描后，已提交的轨迹状态变为 TRAINING 或 TRAINED（取决于 mock）。"""
        from storage.models import TrajectoryStatus
        for i in range(3):
            self.store.save(_make_traj(f"t{i}", "user1"))

        self.scheduler._scan_once()

        # 轨迹应被标记为 TRAINING（mock scheduler 不实际训练）
        pending = self.store.get_pending_count("user1")
        assert pending == 0

    def test_scan_rollback_on_submit_failure(self):
        """submit_batch_training_job 抛异常时，轨迹被标记为 FAILED。"""
        from storage.models import TrajectoryStatus
        for i in range(3):
            self.store.save(_make_traj(f"t{i}", "user1"))

        self.mock_scheduler.submit_batch_training_job = MagicMock(side_effect=RuntimeError("cluster down"))
        self.scheduler._scan_once()

        loaded = self.store.load("user1", [f"t{i}" for i in range(3)])
        assert all(t.status == TrajectoryStatus.FAILED for t in loaded)
