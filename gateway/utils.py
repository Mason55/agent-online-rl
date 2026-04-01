"""Small helpers used across gateway modules."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fit_list(values: list[float], expected_len: int) -> list[float]:
    """Truncate or pad *values* so it has exactly *expected_len* entries."""
    if expected_len <= 0:
        return []
    if len(values) > expected_len:
        return values[:expected_len]
    if len(values) < expected_len:
        return values + [0.0] * (expected_len - len(values))
    return values
