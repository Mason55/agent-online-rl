"""Trajectory output routing: print / file / queue / http.

Ported from agent-gateway/gateway_core/trajectory_output_router.py
with mode names adapted for LLM-as-Judge (judge_log, judge_output).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Optional

import httpx

logger = logging.getLogger("online_rl.gateway")


def trajectory_for_log(trajectory: dict[str, Any], dump_token_ids: bool) -> dict[str, Any]:
    if not isinstance(trajectory, dict):
        return {}
    if dump_token_ids:
        return trajectory
    out = dict(trajectory)
    for key in ("input_ids", "prompt_ids", "response_ids", "response_logprobs"):
        value = out.pop(key, None)
        if isinstance(value, list):
            out[f"{key}_len"] = len(value)
    return out


class TrajectoryOutputRouter:
    """Route emitted trajectory batches to configured outputs."""

    _SUPPORTED = {"print", "queue", "file", "http"}
    _DEFAULTS = {
        "log": {"print", "file"},
        "judge_log": {"print", "file"},
        "judge_output": {"print", "file", "queue"},
    }

    def __init__(
        self,
        *,
        mode: str,
        outputs: str = "",
        http_url: str = "",
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
    ) -> None:
        self.mode = mode
        self.outputs = self._parse_outputs(mode=mode, outputs=outputs)
        self.http_url = http_url.rstrip("/")
        self._owned_http_client = http_client is None and "http" in self.outputs and bool(self.http_url)
        self._http_client = http_client or (
            httpx.AsyncClient(timeout=max(1.0, float(timeout)))
            if self._owned_http_client else None
        )

    @classmethod
    def _parse_outputs(cls, *, mode: str, outputs: str) -> set[str]:
        if not outputs.strip():
            return set(cls._DEFAULTS.get(mode, {"print", "file"}))
        parsed = {item.strip().lower() for item in outputs.split(",") if item.strip()}
        valid = parsed & cls._SUPPORTED
        if not valid:
            return set(cls._DEFAULTS.get(mode, {"print", "file"}))
        return valid

    @property
    def should_write_file(self) -> bool:
        return "file" in self.outputs

    @property
    def should_enqueue(self) -> bool:
        return "queue" in self.outputs

    @property
    def should_print(self) -> bool:
        return "print" in self.outputs

    @property
    def should_post_http(self) -> bool:
        return "http" in self.outputs and bool(self.http_url) and self._http_client is not None

    async def close(self) -> None:
        if self._owned_http_client and self._http_client is not None:
            await self._http_client.aclose()

    async def publish_batches(
        self,
        *,
        batches: Sequence[dict[str, Any]],
        dump_token_ids: bool,
        get_training_queue_size: Optional[Callable[[], Awaitable[int]]] = None,
    ) -> None:
        for batch in batches:
            traj_dump = [
                {
                    "sample_id": s.get("sample_id"),
                    "session_id": s.get("session_id"),
                    "turn_num": s.get("turn_num"),
                    "trajectory": trajectory_for_log(s.get("trajectory", {}), dump_token_ids),
                    "judge": s.get("judge"),
                }
                for s in batch.get("samples", [])
            ]
            if self.should_print:
                await self._log_batch(batch, traj_dump, get_training_queue_size)
            if self.should_post_http:
                await self._post_batch(batch, traj_dump)

    async def _log_batch(
        self,
        batch: dict[str, Any],
        traj_dump: list[dict[str, Any]],
        get_training_queue_size: Optional[Callable[[], Awaitable[int]]],
    ) -> None:
        bid = batch.get("batch_id")
        size = batch.get("size")
        if self.mode in ("log", "judge_log"):
            scores = [s.get("judge", {}).get("score", 0.0) for s in batch.get("samples", [])]
            logger.info(
                "[Gateway] batch=%s size=%s judge_scores=%s",
                bid, size, scores,
            )
            return
        queue_size = -1
        if get_training_queue_size is not None:
            queue_size = await get_training_queue_size()
        logger.info("[Gateway] batch=%s size=%s queued_for_training (queue=%s)", bid, size, queue_size)

    async def _post_batch(self, batch: dict, traj_dump: list[dict]) -> None:
        if self._http_client is None or not self.http_url:
            return
        try:
            payload = {
                "mode": self.mode,
                "batch_id": batch.get("batch_id"),
                "size": batch.get("size"),
                "batch": batch,
                "trajectory_preview": traj_dump,
            }
            resp = await self._http_client.post(self.http_url, json=payload)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("[Gateway] http output failed batch=%s err=%s", batch.get("batch_id"), exc)
