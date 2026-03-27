"""单元测试：storage 层（TrajectoryStore + LoRARepository）"""

import os
import tempfile
from datetime import datetime

import pytest

from storage.models import Trajectory, TrajectoryStatus, Turn
from storage.trajectory_store import TrajectoryStore


def _make_traj(traj_id: str, user_id: str, reward: float = 0.5) -> Trajectory:
    return Trajectory(
        trajectory_id=traj_id,
        user_id=user_id,
        session_id="sess-1",
        turns=[Turn(role="user", content="hello", timestamp=datetime.now())],
        created_at=datetime.now(),
        reward=reward,
    )


class TestTrajectoryStore:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = TrajectoryStore(self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_save_and_load(self):
        traj = _make_traj("t1", "user1")
        self.store.save(traj)
        loaded = self.store.load("user1", ["t1"])
        assert len(loaded) == 1
        assert loaded[0].trajectory_id == "t1"
        assert loaded[0].reward == pytest.approx(0.5)

    def test_get_pending_count(self):
        self.store.save(_make_traj("t1", "user1"))
        self.store.save(_make_traj("t2", "user1"))
        assert self.store.get_pending_count("user1") == 2
        assert self.store.get_pending_count("user2") == 0

    def test_fetch_and_mark_training_atomicity(self):
        """两次调用 fetch_and_mark_training，第二次应返回空（已被第一次标记）。"""
        for i in range(5):
            self.store.save(_make_traj(f"t{i}", "user1"))

        first = self.store.fetch_and_mark_training("user1", 10)
        assert len(first) == 5
        assert all(t.status == TrajectoryStatus.TRAINING for t in first)

        second = self.store.fetch_and_mark_training("user1", 10)
        assert len(second) == 0

    def test_mark_trained_and_failed(self):
        self.store.save(_make_traj("t1", "user1"))
        self.store.save(_make_traj("t2", "user1"))
        self.store.mark_trained(["t1"])
        self.store.mark_failed(["t2"])
        loaded = self.store.load("user1", ["t1", "t2"])
        status_map = {t.trajectory_id: t.status for t in loaded}
        assert status_map["t1"] == TrajectoryStatus.TRAINED
        assert status_map["t2"] == TrajectoryStatus.FAILED

    def test_get_users_above_threshold(self):
        for i in range(3):
            self.store.save(_make_traj(f"t{i}", "user1"))
        for i in range(5):
            self.store.save(_make_traj(f"u{i}", "user2"))

        assert set(self.store.get_users_above_threshold(3)) == {"user1", "user2"}
        assert set(self.store.get_users_above_threshold(4)) == {"user2"}
        assert self.store.get_users_above_threshold(6) == []


class TestLoRARepository:
    def setup_method(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        from storage.lora_repo import LoRARepository
        self.repo = LoRARepository(self.tmpdir)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _make_lora_dir(self, name: str = "adapter") -> str:
        """创建一个包含 dummy 文件的临时 LoRA 目录。"""
        import tempfile, os
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_model.safetensors"), "w") as f:
            f.write("dummy")
        return d

    def test_publish_and_get_latest(self):
        import shutil
        lora_dir = self._make_lora_dir()
        v = self.repo.publish("user1", lora_dir, metadata={"trajectory_count": 10, "reward_avg": 0.6})
        shutil.rmtree(lora_dir)

        assert v.version == "v1"
        assert v.trajectory_count == 10

        latest = self.repo.get_latest("user1")
        assert latest is not None
        assert latest.version == "v1"

    def test_latest_points_to_newest(self):
        import shutil
        for i in range(3):
            d = self._make_lora_dir()
            self.repo.publish("user1", d, metadata={"trajectory_count": i, "reward_avg": 0.0})
            shutil.rmtree(d)

        latest = self.repo.get_latest("user1")
        assert latest.version == "v3"

    def test_get_latest_returns_none_for_new_user(self):
        assert self.repo.get_latest("no_such_user") is None
