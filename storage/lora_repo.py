import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import LoRAVersion

logger = logging.getLogger(__name__)


class LoRARepository:
    def __init__(self, root: str = "lora_repo"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        user_id: str,
        lora_path: str,
        metadata: Optional[dict] = None,
        base_model: str = "",
    ) -> LoRAVersion:
        user_dir = self.root / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # 计算下一个版本号
        existing = self._list_version_dirs(user_dir)
        next_num = len(existing) + 1
        version = f"v{next_num}"
        version_dir = user_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # 复制 LoRA 权重文件
        src = Path(lora_path)
        for f in src.iterdir() if src.is_dir() else [src]:
            shutil.copy2(f, version_dir / f.name)

        # 计算 reward 平均值
        reward_avg = metadata.get("reward_avg", 0.0) if metadata else 0.0
        trajectory_count = metadata.get("trajectory_count", 0) if metadata else 0

        # 写 metadata.json
        meta = {
            "user_id": user_id,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "trajectory_count": trajectory_count,
            "reward_avg": reward_avg,
            "base_model": base_model,
        }
        (version_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # 原子更新 latest 软链
        latest_link = user_dir / "latest"
        tmp_link = user_dir / ".latest_tmp"
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        tmp_link.symlink_to(version)
        tmp_link.rename(latest_link)

        lora_version = LoRAVersion(
            user_id=user_id,
            version=version,
            path=str(version_dir),
            created_at=datetime.fromisoformat(meta["created_at"]),
            trajectory_count=trajectory_count,
            reward_avg=reward_avg,
            base_model=base_model,
        )
        logger.info(f"Published LoRA {version} for user {user_id} at {version_dir}")
        return lora_version

    def get_latest(self, user_id: str) -> Optional[LoRAVersion]:
        latest_link = self.root / user_id / "latest"
        if not latest_link.exists():
            return None
        version_dir = latest_link.resolve()
        meta_file = version_dir / "metadata.json"
        if not meta_file.exists():
            return None
        meta = json.loads(meta_file.read_text())
        return LoRAVersion(
            user_id=meta["user_id"],
            version=meta["version"],
            path=str(version_dir),
            created_at=datetime.fromisoformat(meta["created_at"]),
            trajectory_count=meta["trajectory_count"],
            reward_avg=meta["reward_avg"],
            base_model=meta["base_model"],
        )

    def list_versions(self, user_id: str) -> list[LoRAVersion]:
        user_dir = self.root / user_id
        if not user_dir.exists():
            return []
        versions = []
        for version_dir in sorted(self._list_version_dirs(user_dir)):
            meta_file = version_dir / "metadata.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text())
            versions.append(LoRAVersion(
                user_id=meta["user_id"],
                version=meta["version"],
                path=str(version_dir),
                created_at=datetime.fromisoformat(meta["created_at"]),
                trajectory_count=meta["trajectory_count"],
                reward_avg=meta["reward_avg"],
                base_model=meta["base_model"],
            ))
        return versions

    def _list_version_dirs(self, user_dir: Path) -> list[Path]:
        return [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
