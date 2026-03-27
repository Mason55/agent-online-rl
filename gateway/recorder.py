import logging
import threading
import uuid
from datetime import datetime
from typing import Optional

from storage.models import Trajectory, Turn

logger = logging.getLogger(__name__)

# session 超时（秒）
SESSION_TTL = 3600


class SessionRecorder:
    """管理所有 session 的轨迹录制生命周期（内存存储，生产可替换为 Redis）。"""

    def __init__(self):
        self._sessions: dict[str, dict] = {}  # session_id → session state
        self._lock = threading.Lock()

    def record_request(self, session_id: str, user_id: str, messages: list) -> None:
        """录制用户请求，更新 session 上下文。"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "turns": [],
                    "created_at": datetime.now(),
                    "last_active": datetime.now(),
                }
            session = self._sessions[session_id]
            session["last_active"] = datetime.now()

            # 提取最后一条 user 消息（避免重复录制历史消息）
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                last_user = user_msgs[-1]
                session["turns"].append(Turn(
                    role="user",
                    content=last_user.get("content", ""),
                    timestamp=datetime.now(),
                ))

    def record_response(self, session_id: str, response: dict) -> Optional[Trajectory]:
        """
        录制 Assistant 响应。
        若 finish_reason == 'stop'，返回完整 Trajectory 并清理 session。
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found in recorder")
                return None

            # 提取响应内容
            choices = response.get("choices", [])
            if not choices:
                return None

            choice = choices[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")

            session["turns"].append(Turn(
                role="assistant",
                content=message.get("content") or "",
                timestamp=datetime.now(),
                token_count=response.get("usage", {}).get("completion_tokens", 0),
            ))
            session["last_active"] = datetime.now()

            if finish_reason == "stop":
                return self._finalize_session(session_id, session)
            return None

    def on_session_timeout(self, session_id: str) -> Optional[Trajectory]:
        """session 超时，强制结束录制。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return self._finalize_session(session_id, session)

    def _finalize_session(self, session_id: str, session: dict) -> Optional[Trajectory]:
        """构建 Trajectory 并从内存中移除 session（调用方持有 lock）。"""
        turns = session.get("turns", [])
        if not turns:
            self._sessions.pop(session_id, None)
            return None

        trajectory = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            user_id=session["user_id"],
            session_id=session_id,
            turns=turns,
            created_at=session["created_at"],
        )
        self._sessions.pop(session_id, None)
        return trajectory
