"""LLM request forwarders with true-streaming support.

Two modes:
  - StringForwarder: chat/completions (string I/O), supports true SSE streaming
  - TokenForwarder: completions (token I/O), non-streaming with synthetic chat response
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any, AsyncIterator, Callable

import httpx
from fastapi import HTTPException

from gateway.constants import NON_STANDARD_BODY_KEYS
from gateway.message_utils import (
    extract_logprobs_from_chat_response,
    extract_logprobs_from_completion_response,
    extract_tool_calls_from_text,
    flatten_message_content,
)

logger = logging.getLogger("online_rl.gateway")


def _extract_runtime_tokens_from_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-token runtime data from vLLM's logprobs.content[].

    Returns a list of dicts with ``token`` (str), ``logprob`` (float), and
    ``bytes`` (list[int] or None).  This is the "runtime token truth" —
    the exact token sequence the model generated, before any re-tokenisation.
    """
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list) or not content:
        return []
    result: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        result.append({
            "token": item.get("token", ""),
            "logprob": float(item.get("logprob", 0.0)),
            "bytes": item.get("bytes"),
        })
    return result


#TODO 删除这种实现
class StringForwarder:
    """Forward chat requests to /v1/chat/completions.

    Supports two paths:
      - forward(): non-streaming, returns full response + parsed fields
      - forward_stream(): true SSE streaming, yields lines to client while
        collecting the full response for trajectory recording
    """

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        llm_url: str,
        model_id: str,
    ) -> None:
        self.http_client = http_client
        self.llm_url = llm_url.rstrip("/")
        self.model_id = model_id

    def _clean_body(self, body: dict[str, Any]) -> dict[str, Any]:
        send_body = {k: v for k, v in copy.deepcopy(body).items() if k not in NON_STANDARD_BODY_KEYS}
        send_body["stream"] = False
        send_body.pop("stream_options", None)
        if "model" not in send_body:
            send_body["model"] = self.model_id
        send_body["messages"] = body.get("messages", [])
        send_body["logprobs"] = True
        send_body["top_logprobs"] = 1
        return send_body

    async def forward(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        send_body = self._clean_body(body)
        t0 = time.perf_counter()
        resp = await self.http_client.post(
            f"{self.llm_url}/v1/chat/completions",
            json=send_body,
            headers=headers,
        )
        logger.debug("forward_string status=%s cost_ms=%.1f", resp.status_code, (time.perf_counter() - t0) * 1000)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise HTTPException(status_code=502, detail=f"upstream error: {detail}") from exc

        data = resp.json()
        return self._parse_response(data)

    async def forward_stream(
        self, body: dict[str, Any], headers: dict[str, str],
    ) -> tuple[AsyncIterator[str], "StreamCollector"]:
        """Start a true-streaming request to upstream vLLM.

        Returns (line_iterator, collector). The caller yields lines from
        line_iterator to the HTTP client; after the stream ends, collector
        holds the assembled full response for trajectory recording.
        """
        send_body = {k: v for k, v in copy.deepcopy(body).items() if k not in NON_STANDARD_BODY_KEYS}
        send_body["stream"] = True
        send_body.pop("stream_options", None)
        if "model" not in send_body:
            send_body["model"] = self.model_id
        send_body["messages"] = body.get("messages", [])
        send_body["logprobs"] = True
        send_body["top_logprobs"] = 1
        send_body["stream_options"] = {"include_usage": False}

        collector = StreamCollector()

        async def _generate() -> AsyncIterator[str]:
            async with self.http_client.stream(
                "POST",
                f"{self.llm_url}/v1/chat/completions",
                json=send_body,
                headers=headers,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            choice0 = chunk.get("choices", [{}])[0]
                            delta = choice0.get("delta", {})
                            if delta.get("content"):
                                collector.content_parts.append(delta["content"])
                            chunk_logprobs = choice0.get("logprobs")
                            if isinstance(chunk_logprobs, dict):
                                lp_content = chunk_logprobs.get("content")
                                if isinstance(lp_content, list):
                                    for lp_item in lp_content:
                                        if isinstance(lp_item, dict):
                                            collector.runtime_tokens.append({
                                                "token": lp_item.get("token", ""),
                                                "logprob": float(lp_item.get("logprob", 0.0)),
                                                "bytes": lp_item.get("bytes"),
                                            })
                            fr = choice0.get("finish_reason")
                            if fr:
                                collector.finish_reason = fr
                            if not collector.model:
                                collector.model = chunk.get("model", "")
                            if not collector.response_id:
                                collector.response_id = chunk.get("id", "")
                        except Exception:
                            pass
                    yield line + "\n"

        return _generate(), collector

    def _parse_response(self, data: dict) -> dict[str, Any]:
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        msg = choice.get("message", {})
        if not isinstance(msg, dict):
            msg = {"role": "assistant", "content": ""}
            choice["message"] = msg

        content = flatten_message_content(msg.get("content"))
        has_tool_calls = bool(msg.get("tool_calls"))
        has_reasoning = bool(msg.get("reasoning_content"))
        if content and (not has_tool_calls or not has_reasoning):
            cleaned, parsed_tools, parsed_reasoning = extract_tool_calls_from_text(content)
            if parsed_tools or parsed_reasoning:
                msg["content"] = cleaned
                if parsed_reasoning and not has_reasoning:
                    msg["reasoning_content"] = parsed_reasoning
                if parsed_tools and not has_tool_calls:
                    msg["tool_calls"] = parsed_tools
                    choice["finish_reason"] = "tool_calls"

        tool_calls = msg.get("tool_calls") or []
        response_text = flatten_message_content(msg.get("content"))
        if not response_text and tool_calls:
            response_text = json.dumps(tool_calls, ensure_ascii=False)

        runtime_tokens = _extract_runtime_tokens_from_logprobs(choice)

        return {
            "response_json": data,
            "assistant_message": msg,
            "response_text": response_text,
            "tool_calls": tool_calls,
            "response_logprobs": [t["logprob"] for t in runtime_tokens],
            "runtime_tokens": runtime_tokens,
        }

    def parse_collected_stream(self, collector: "StreamCollector") -> dict[str, Any]:
        """Build a full response dict from a finished StreamCollector."""
        content = "".join(collector.content_parts)
        msg: dict[str, Any] = {"role": "assistant", "content": content}

        cleaned, parsed_tools, reasoning = extract_tool_calls_from_text(content)
        if parsed_tools or reasoning:
            msg["content"] = cleaned
            if reasoning:
                msg["reasoning_content"] = reasoning
            if parsed_tools:
                msg["tool_calls"] = parsed_tools

        tool_calls = msg.get("tool_calls") or []
        response_text = flatten_message_content(msg.get("content"))
        if not response_text and tool_calls:
            response_text = json.dumps(tool_calls, ensure_ascii=False)

        finish_reason = "tool_calls" if parsed_tools else (collector.finish_reason or "stop")

        response_json = {
            "id": collector.response_id or f"chatcmpl-gw-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": collector.model or self.model_id,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        }

        runtime_tokens = collector.runtime_tokens

        return {
            "response_json": response_json,
            "assistant_message": msg,
            "response_text": response_text,
            "tool_calls": tool_calls,
            "response_logprobs": [t["logprob"] for t in runtime_tokens],
            "runtime_tokens": runtime_tokens,
        }


class StreamCollector:
    """Accumulates content and logprobs from SSE chunks during streaming."""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.runtime_tokens: list[dict[str, Any]] = []
        self.finish_reason: str | None = None
        self.model: str = ""
        self.response_id: str = ""


class TokenForwarder:
    """Forward tokenized prompts to /v1/completions (non-streaming)."""

    _ALLOWED_KEYS = {
        "model", "max_tokens", "temperature", "top_p", "top_k", "min_p",
        "n", "stop", "presence_penalty", "frequency_penalty", "repetition_penalty",
        "seed", "logit_bias", "ignore_eos", "skip_special_tokens",
        "spaces_between_special_tokens", "guided_json", "guided_regex",
        "guided_choice", "guided_grammar", "guided_decoding_backend", "response_format",
    }

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        llm_url: str,
        model_id: str,
    ) -> None:
        self.http_client = http_client
        self.llm_url = llm_url.rstrip("/")
        self.model_id = model_id

    async def forward(
        self,
        body: dict[str, Any],
        prompt_ids: list[int],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        #TODO 这里要配置 vllm token in token out的参数
        send_body: dict[str, Any] = {k: v for k, v in body.items() if k in self._ALLOWED_KEYS}
        send_body["model"] = send_body.get("model") or self.model_id
        send_body["prompt"] = prompt_ids
        send_body["stream"] = False
        send_body["logprobs"] = max(1, int(body.get("logprobs", 1) or 1))
        send_body["max_tokens"] = int(body.get("max_tokens") or 2048)

        t0 = time.perf_counter()
        resp = await self.http_client.post(
            f"{self.llm_url}/v1/completions",
            json=send_body,
            headers=headers,
        )
        logger.debug("forward_token status=%s cost_ms=%.1f", resp.status_code, (time.perf_counter() - t0) * 1000)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise HTTPException(status_code=502, detail=f"upstream completions error: {detail}") from exc

        completion_data = resp.json()
        choices = completion_data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        raw_text = choice.get("text") or ""
        #TODO check this
        cleaned, parsed_tools, reasoning = extract_tool_calls_from_text(raw_text)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": cleaned}
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        if parsed_tools:
            assistant_msg["tool_calls"] = parsed_tools

        completion_lps = extract_logprobs_from_completion_response(choice)
        lp_content = [{"token": "", "logprob": float(lp), "top_logprobs": []} for lp in completion_lps]

        finish_reason = "tool_calls" if parsed_tools else (choice.get("finish_reason") or "stop")
        chat_response = {
            "id": completion_data.get("id", f"chatcmpl-gw-{int(time.time())}"),
            "object": "chat.completion",
            "created": completion_data.get("created", int(time.time())),
            "model": completion_data.get("model", send_body["model"]),
            "choices": [{
                "index": 0,
                "message": assistant_msg,
                "finish_reason": finish_reason,
                "logprobs": {"content": lp_content},
            }],
            "usage": completion_data.get("usage", {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_lps),
                "total_tokens": len(prompt_ids) + len(completion_lps),
            }),
        }

        response_text = cleaned
        if not response_text and parsed_tools:
            response_text = json.dumps(parsed_tools, ensure_ascii=False)

        return {
            "response_json": chat_response,
            "assistant_message": assistant_msg,
            "response_text": response_text,
            "tool_calls": parsed_tools,
            "response_logprobs": completion_lps,
        }
