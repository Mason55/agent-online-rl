import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from .models import Trajectory, TrajectoryStatus, Turn

logger = logging.getLogger(__name__)


class TrajectoryStore:
    def __init__(self, db_path: str = "trajectories.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turns_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reward REAL,
                    reward_details_json TEXT DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    metadata_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_status ON trajectories (user_id, status)")

    def save(self, trajectory: Trajectory) -> None:
        turns_json = json.dumps([
            {"role": t.role, "content": t.content,
             "timestamp": t.timestamp.isoformat(), "token_count": t.token_count}
            for t in trajectory.turns
        ])
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trajectories
                (trajectory_id, user_id, session_id, turns_json, created_at,
                 reward, reward_details_json, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.trajectory_id,
                trajectory.user_id,
                trajectory.session_id,
                turns_json,
                trajectory.created_at.isoformat(),
                trajectory.reward,
                json.dumps(trajectory.reward_details),
                trajectory.status.value,
                json.dumps(trajectory.metadata),
            ))

    def get_pending_count(self, user_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE user_id=? AND status=?",
                (user_id, TrajectoryStatus.PENDING.value)
            ).fetchone()
            return row[0]

    def get_users_above_threshold(self, threshold: int) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT user_id, COUNT(*) as cnt
                FROM trajectories
                WHERE status=?
                GROUP BY user_id
                HAVING cnt >= ?
            """, (TrajectoryStatus.PENDING.value, threshold)).fetchall()
            return [row["user_id"] for row in rows]

    def fetch_and_mark_training(self, user_id: str, limit: int) -> list[Trajectory]:
        """原子操作：获取 PENDING 轨迹并标记为 TRAINING"""
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT * FROM trajectories
                    WHERE user_id=? AND status=?
                    LIMIT ?
                """, (user_id, TrajectoryStatus.PENDING.value, limit)).fetchall()

                if not rows:
                    return []

                ids = [r["trajectory_id"] for r in rows]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE trajectories SET status=? WHERE trajectory_id IN ({placeholders})",
                    [TrajectoryStatus.TRAINING.value] + ids,
                )
                trajectories = [self._row_to_trajectory(r) for r in rows]
                for traj in trajectories:
                    traj.status = TrajectoryStatus.TRAINING
                return trajectories

    def mark_trained(self, trajectory_ids: list[str]) -> None:
        self._update_status(trajectory_ids, TrajectoryStatus.TRAINED)

    def mark_failed(self, trajectory_ids: list[str]) -> None:
        self._update_status(trajectory_ids, TrajectoryStatus.FAILED)

    def load(self, user_id: str, trajectory_ids: list[str]) -> list[Trajectory]:
        if not trajectory_ids:
            return []
        placeholders = ",".join("?" * len(trajectory_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trajectories WHERE user_id=? AND trajectory_id IN ({placeholders})",
                [user_id] + trajectory_ids,
            ).fetchall()
        return [self._row_to_trajectory(r) for r in rows]

    def _update_status(self, trajectory_ids: list[str], status: TrajectoryStatus) -> None:
        if not trajectory_ids:
            return
        placeholders = ",".join("?" * len(trajectory_ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE trajectories SET status=? WHERE trajectory_id IN ({placeholders})",
                [status.value] + trajectory_ids,
            )

    def _row_to_trajectory(self, row: sqlite3.Row) -> Trajectory:
        turns_data = json.loads(row["turns_json"])
        turns = [
            Turn(
                role=t["role"],
                content=t["content"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                token_count=t.get("token_count", 0),
            )
            for t in turns_data
        ]
        return Trajectory(
            trajectory_id=row["trajectory_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            turns=turns,
            created_at=datetime.fromisoformat(row["created_at"]),
            reward=row["reward"],
            reward_details=json.loads(row["reward_details_json"]),
            status=TrajectoryStatus(row["status"]),
            metadata=json.loads(row["metadata_json"]),
        )
