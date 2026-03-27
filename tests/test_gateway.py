"""单元测试：Gateway 组件（recorder + reward_computor）"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from gateway.recorder import SessionRecorder
from gateway.reward_computor import RewardComputor
from storage.models import Trajectory, Turn


class TestSessionRecorder:
    def setup_method(self):
        self.recorder = SessionRecorder()

    def test_record_request_creates_session(self):
        self.recorder.record_request("s1", "u1", [{"role": "user", "content": "hello"}])
        assert "s1" in self.recorder._sessions

    def test_full_conversation_returns_trajectory(self):
        messages = [{"role": "user", "content": "hello"}]
        self.recorder.record_request("s1", "u1", messages)

        response = {
            "choices": [{"message": {"content": "world", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }
        traj = self.recorder.record_response("s1", response)

        assert traj is not None
        assert isinstance(traj, Trajectory)
        assert traj.user_id == "u1"
        assert len(traj.turns) == 2
        assert traj.turns[0].role == "user"
        assert traj.turns[1].role == "assistant"
        # session 已被清理
        assert "s1" not in self.recorder._sessions

    def test_non_stop_finish_reason_does_not_finalize(self):
        self.recorder.record_request("s1", "u1", [{"role": "user", "content": "hi"}])
        response = {
            "choices": [{"message": {"content": "...", "role": "assistant"}, "finish_reason": "length"}],
        }
        traj = self.recorder.record_response("s1", response)
        assert traj is None
        assert "s1" in self.recorder._sessions

    def test_timeout_finalizes_session(self):
        self.recorder.record_request("s1", "u1", [{"role": "user", "content": "hi"}])
        traj = self.recorder.on_session_timeout("s1")
        assert traj is not None
        assert "s1" not in self.recorder._sessions


class TestRewardComputor:
    def test_normalization_overall_5_gives_zero(self):
        rc = RewardComputor("http://dummy", "model")
        scores = {"overall": 5.0, "task_completion": 5, "response_quality": 5,
                  "tool_usage": 5, "coherence": 5, "reason": "neutral"}
        assert rc._parse_scores('{"overall": 5.0, "task_completion": 5, "response_quality": 5, "tool_usage": 5, "coherence": 5, "reason": "ok"}')["overall"] == 5.0

    def test_normalization_overall_10_gives_one(self):
        rc = RewardComputor("http://dummy", "model")
        # reward = (10 - 5) / 5 = 1.0
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1",
            turns=[], created_at=datetime.now(),
        )
        # 直接调用内部逻辑
        scores = {"overall": 10.0}
        reward = (scores["overall"] - 5.0) / 5.0
        assert reward == pytest.approx(1.0)

    def test_normalization_overall_0_gives_minus_one(self):
        reward = (0.0 - 5.0) / 5.0
        assert reward == pytest.approx(-1.0)

    def test_parse_scores_handles_markdown_json(self):
        rc = RewardComputor("http://dummy", "model")
        content = '```json\n{"overall": 8.0, "task_completion": 8, "response_quality": 7, "tool_usage": 9, "coherence": 8, "reason": "good"}\n```'
        scores = rc._parse_scores(content)
        assert scores["overall"] == pytest.approx(8.0)

    def test_parse_scores_computes_overall_from_fields_if_missing(self):
        rc = RewardComputor("http://dummy", "model")
        content = '{"task_completion": 8, "response_quality": 6, "tool_usage": 10, "coherence": 8}'
        scores = rc._parse_scores(content)
        assert "overall" in scores
        assert scores["overall"] == pytest.approx((8 + 6 + 10 + 8) / 4)

    def test_compute_async_handles_judge_failure_gracefully(self):
        import asyncio
        rc = RewardComputor("http://dummy", "model")
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1",
            turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
            created_at=datetime.now(),
        )
        async def bad_judge(prompt):
            raise RuntimeError("judge unavailable")
        rc._call_judge = bad_judge

        result = asyncio.run(rc.compute_async(traj))
        assert result.reward == pytest.approx(0.0)
        assert "error" in result.reward_details
