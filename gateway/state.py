"""Gateway mutable runtime state.

Manages tokenizer, HTTP clients, per-session tracking, sample batching,
training queue, and verl DataProto integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Any, Optional

import httpx
from fastapi import HTTPException, Request
from transformers import AutoTokenizer

from gateway.config import GatewayConfig
from gateway.forwarder import StreamCollector, StringForwarder, TokenForwarder
from gateway.message_utils import (
    flatten_message_content,
    normalize_messages_for_template,
)
from gateway.output_router import TrajectoryOutputRouter
from gateway.utils import utc_now_iso
from gateway.verl_converter import VerlDataProtoConverter

logger = logging.getLogger("online_rl.gateway")


class GatewayState:
    """Central gateway state: HTTP clients, tokenizer, batching, verl."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config

        tokenizer_name = config.model_path or config.model_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True,
        )
        self._http_client = httpx.AsyncClient(timeout=config.request_timeout)

        os.makedirs(config.record_dir, exist_ok=True)
        self.trajectory_file = os.path.join(config.record_dir, "trajectories.jsonl")
        self.batch_file = os.path.join(config.record_dir, "batches.jsonl")

        self._session_turn_count: dict[str, int] = {}
        self._pending_samples: list[dict[str, Any]] = []
        self._pending_judge_samples: dict[str, dict[str, Any]] = {}
        self._training_queue: deque[dict[str, Any]] = deque()

        self._output_to_verl = bool(config.output_to_verl)
        self._max_pending_verl_batches = max(0, config.max_pending_verl_batches)
        self._verl_converter: Optional[VerlDataProtoConverter] = (
            VerlDataProtoConverter(pad_token_id=self._tokenizer.pad_token_id or 0)
            if self._output_to_verl else None
        )
        self._pending_verl_dataproto: deque[Any] = deque()
        self._pending_verl_errors: deque[Exception] = deque()
        self._verl_waiters: deque[asyncio.Future[Any]] = deque()
        self._dropped_verl_batches = 0

        self._total_requests = 0
        self._total_samples = 0
        self._emitted_batches = 0
        self._batch_seq = 0
        self._lock = asyncio.Lock()

        self._string_forwarder = StringForwarder(
            http_client=self._http_client,
            llm_url=config.llm_url,
            model_id=config.model_id,
        )
        self._token_forwarder = TokenForwarder(
            http_client=self._http_client,
            llm_url=config.llm_url,
            model_id=config.model_id,
        )
        self._output_router = TrajectoryOutputRouter(
            mode=config.mode,
            outputs=config.trajectory_outputs,
            http_url=config.trajectory_http_url,
            http_client=self._http_client,
            timeout=config.request_timeout,
        )
        logger.info(
            "[GatewayState] mode=%s io_mode=%s batch_size=%d verl=%s",
            config.mode, config.io_mode, config.rollout_batch_size, self._output_to_verl,
        )

    async def close(self) -> None:
        await self._output_router.close()
        await self._http_client.aclose()
        if self._pending_judge_samples:
            logger.warning("[Gateway] dropping %d pending judge samples", len(self._pending_judge_samples))
        for fut in list(self._verl_waiters):
            if not fut.done():
                fut.cancel()
        self._verl_waiters.clear()
        self._pending_verl_dataproto.clear()

    async def ensure_auth(self, authorization: Optional[str]) -> None:
        if not self.config.gateway_api_key:
            return
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.config.gateway_api_key:
            raise HTTPException(status_code=403, detail="invalid bearer token")

    def make_upstream_headers(self, request: Request) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in request.headers.items():
            lk = key.lower()
            if lk in {"host", "content-length", "connection"}:
                continue
            if lk.startswith("x-forwarded-"):
                continue
            headers[key] = value
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        return headers

    async def next_turn_num(self, session_id: str) -> int:
        async with self._lock:
            self._session_turn_count[session_id] = self._session_turn_count.get(session_id, 0) + 1
            return self._session_turn_count[session_id]

    async def inc_request_counter(self) -> None:
        async with self._lock:
            self._total_requests += 1

    def tokenize_text(self, text: str) -> list[int]:
        return self._tokenizer.encode(text or "", add_special_tokens=False)

    def build_prompt_text_and_ids(self, messages: list[dict], tools: Any) -> tuple[str, list[int]]:
        norm_msgs = normalize_messages_for_template(messages)
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
            )
        except TypeError:
            prompt_text = self._tokenizer.apply_chat_template(
                norm_msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt_text = "\n".join(
                f"{m.get('role', 'unknown')}: {flatten_message_content(m.get('content', ''))}"
                for m in norm_msgs
            )
        return prompt_text, self.tokenize_text(prompt_text)

    async def forward_string(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        self._string_forwarder.http_client = self._http_client
        return await self._string_forwarder.forward(body=body, headers=headers)

    async def forward_string_stream(self, body: dict[str, Any], headers: dict[str, str]):
        self._string_forwarder.http_client = self._http_client
        return await self._string_forwarder.forward_stream(body=body, headers=headers)

    def parse_collected_stream(self, collector: StreamCollector) -> dict[str, Any]:
        return self._string_forwarder.parse_collected_stream(collector)

    async def forward_token(self, body: dict[str, Any], prompt_ids: list[int], headers: dict[str, str]) -> dict[str, Any]:
        self._token_forwarder.http_client = self._http_client
        return await self._token_forwarder.forward(body=body, prompt_ids=prompt_ids, headers=headers)

    async def proxy_request(
        self, *, method: str, url: str, params: dict[str, Any], headers: dict[str, str], content: bytes,
    ) -> httpx.Response:
        return await self._http_client.request(method=method, url=url, params=params, headers=headers, content=content)

    # ---- Delayed Judge: stage / pop pending samples ----

    async def stage_pending_judge_sample(self, session_id: str, sample: dict[str, Any]) -> None:
        async with self._lock:
            self._pending_judge_samples[session_id] = sample

    async def pop_pending_judge_sample(self, session_id: str) -> Optional[dict[str, Any]]:
        async with self._lock:
            return self._pending_judge_samples.pop(session_id, None)

    # ---- Sample recording and batching ----

    def _append_jsonl(self, path: str, payload: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def record_sample(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        async with self._lock:
            self._total_samples += 1
            if self._output_router.should_write_file:
                self._append_jsonl(self.trajectory_file, sample)
            self._pending_samples.append(sample)

            while len(self._pending_samples) >= self.config.rollout_batch_size:
                self._batch_seq += 1
                self._emitted_batches += 1
                batch_samples = self._pending_samples[:self.config.rollout_batch_size]
                self._pending_samples = self._pending_samples[self.config.rollout_batch_size:]
                batch = {
                    "batch_id": self._batch_seq,
                    "created_at": utc_now_iso(),
                    "mode": self.config.mode,
                    "size": len(batch_samples),
                    "samples": batch_samples,
                }
                if self._output_router.should_write_file:
                    self._append_jsonl(self.batch_file, batch)
                emitted.append(batch)
                if self._output_router.should_enqueue:
                    self._training_queue.append(batch)
        return emitted

    async def publish_batches(self, batches: list[dict[str, Any]]) -> None:
        if self._output_to_verl:
            for batch in batches:
                await self._emit_verl_dataproto_for_batch(batch)
        await self._output_router.publish_batches(
            batches=batches,
            dump_token_ids=self.config.dump_token_ids,
            get_training_queue_size=self.get_training_queue_size,
        )

    async def get_stats(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "mode": self.config.mode,
                "io_mode": self.config.io_mode,
                "rollout_batch_size": self.config.rollout_batch_size,
                "total_requests": self._total_requests,
                "total_samples": self._total_samples,
                "pending_samples": len(self._pending_samples),
                "pending_judge_samples": len(self._pending_judge_samples),
                "emitted_batches": self._emitted_batches,
                "training_queue_batches": len(self._training_queue),
                "known_sessions": len(self._session_turn_count),
                "llm_url": self.config.llm_url,
                "judge_url": self.config.judge_url,
                "model_id": self.config.model_id,
                "output_to_verl": self._output_to_verl,
                "pending_verl_dataproto": len(self._pending_verl_dataproto),
            }

    async def get_training_queue_size(self) -> int:
        async with self._lock:
            return len(self._training_queue)

    async def pop_training_queue(self, max_batches: int) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        async with self._lock:
            while self._training_queue and len(batches) < max_batches:
                batches.append(self._training_queue.popleft())
        return batches

    # ---- verl DataProto ----

    def get_batch_trajectories(self) -> asyncio.Future[Any]:
        if not self._output_to_verl:
            raise RuntimeError("output_to_verl is disabled")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        loop.create_task(self._register_verl_waiter(fut))
        return fut

    async def _register_verl_waiter(self, fut: asyncio.Future[Any]) -> None:
        async with self._lock:
            while self._pending_verl_errors:
                err = self._pending_verl_errors.popleft()
                if not fut.done():
                    fut.set_exception(err)
                    return
            while self._pending_verl_dataproto:
                data = self._pending_verl_dataproto.popleft()
                if not fut.done():
                    fut.set_result(data)
                    return
            self._verl_waiters.append(fut)

    async def _emit_verl_dataproto_for_batch(self, batch: dict[str, Any]) -> None:
        if not self._output_to_verl or self._verl_converter is None:
            return
        try:
            data = self._verl_converter.convert_batch(batch)
        except Exception as exc:
            logger.warning("[Gateway] verl convert failed batch=%s err=%s", batch.get("batch_id"), exc)
            await self._deliver_verl_exception(RuntimeError(str(exc)))
            return
        await self._deliver_verl_dataproto(data)

    async def _deliver_verl_dataproto(self, data: Any) -> None:
        async with self._lock:
            while self._verl_waiters:
                fut = self._verl_waiters.popleft()
                if fut.cancelled() or fut.done():
                    continue
                fut.set_result(data)
                return
            if self._max_pending_verl_batches > 0 and len(self._pending_verl_dataproto) >= self._max_pending_verl_batches:
                self._pending_verl_dataproto.popleft()
                self._dropped_verl_batches += 1
            self._pending_verl_dataproto.append(data)

    async def _deliver_verl_exception(self, err: Exception) -> None:
        async with self._lock:
            while self._verl_waiters:
                fut = self._verl_waiters.popleft()
                if fut.cancelled() or fut.done():
                    continue
                fut.set_exception(err)
                return
            self._pending_verl_errors.append(err)

    # ---- Synthetic SSE streaming (for non-streaming upstream, stream to client) ----

    async def stream_chat_response(self, response_json: dict[str, Any]):
        created = int(response_json.get("created", int(time.time())))
        resp_id = response_json.get("id", f"chatcmpl-gw-{created}")
        model = response_json.get("model", self.config.model_id)
        choices = response_json.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        finish_reason = choice.get("finish_reason", "stop") if isinstance(choice, dict) else "stop"

        delta: dict[str, Any] = {}
        role = message.get("role")
        if role:
            delta["role"] = role
        content = message.get("content")
        if isinstance(content, str) and content:
            delta["content"] = content
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        if message.get("reasoning_content"):
            delta["reasoning_content"] = message["reasoning_content"]

        first = {
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        last = {
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
