from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TrajectoryStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class JobStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Turn:
    role: str  # "user" | "assistant" | "tool"
    content: str
    timestamp: datetime
    token_count: int = 0


@dataclass
class Trajectory:
    trajectory_id: str
    user_id: str
    session_id: str
    turns: list[Turn]
    created_at: datetime
    reward: Optional[float] = None  # [-1, 1]
    reward_details: dict = field(default_factory=dict)
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    metadata: dict = field(default_factory=dict)


@dataclass
class UserTrainingJob:
    user_id: str
    trajectory_ids: list[str]


@dataclass
class LoRAVersion:
    user_id: str
    version: str  # "v1", "v2", ...
    path: str
    created_at: datetime
    trajectory_count: int
    reward_avg: float
    base_model: str
