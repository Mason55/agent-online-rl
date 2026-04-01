"""Per-turn message processing pipeline.

Handles: tokenize → forward to LLM → record trajectory (per-turn) →
delayed judge scoring → batch emission.

Ported from agent-gateway/gateway_core/llm_message_processor.py, adapted
for LLM-as-Judge (delayed reward) and true streaming.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from fastapi import HTTPException, Request

from gateway.judge_scorer import JudgeScorer
from gateway.message_utils import extract_last_user_instruction
from gateway.utils import fit_list, utc_now_iso

logger = logging.getLogger("online_rl.gateway")


class LLMMessageProcessor:
    """Per-turn processing pipeline for the gateway."""

    def __init__(self, *, state: Any, config: Any, judge_scorer: Optional[JudgeScorer] = None) -> None:
        self._state = state
        self._config = config
        self._judge_scorer = judge_scorer

    @staticmethod
    def _resolve_trace_id(request: Request) -> str:
        req_headers = getattr(request, "headers", {}) or {}
        return (
            req_headers.get("x-request-id") if hasattr(req_headers, "get") else None
        ) or uuid.uuid4().hex[:8]

    @staticmethod
    def _extract_finish_reason(response_json: dict[str, Any]) -> Optional[str]:
        choices = response_json.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            return choices[0].get("finish_reason")
        return None

    @staticmethod
    def _require_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        return messages

    def _build_current_sample(
        self,
        *,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        session_id: str,
        turn_num: int,
        response_json: dict[str, Any],
        assistant_msg: dict[str, Any],
        finish_reason: Optional[str],
        prompt_text: str,
        prompt_ids: list[int],
        response_text: str,
        response_ids: list[int],
        response_logprobs: list[float],
        tool_calls: list[dict[str, Any]],
        prompt_mask: Optional[list[int]] = None,
        render_fingerprint: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        input_ids = prompt_ids + response_ids

        # Per-turn training (OpenClaw-RL style): each sample is one turn.
        # prompt = full context up to this turn → mask 0 (not in loss).
        # response = current turn's model output → mask 1 (in loss).
        # Historical assistant content lives in the prompt but belongs to
        # earlier turns that have their own training samples, so prompt is
        # always 0 here.
        #
        # prompt_mask is available for episode-level training where the
        # entire multi-turn trajectory is one sequence and historical
        # assistant spans need mask=1 too.
        response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        attention_mask = [1] * len(input_ids)

        return {
            "sample_id": str(uuid.uuid4()),
            "created_at": utc_now_iso(),
            "session_id": session_id,
            "turn_num": turn_num,
            "mode": self._config.mode,
            "io_mode": self._config.io_mode,
            "model": response_json.get("model", body.get("model", self._config.model_id)),
            "render_fingerprint": render_fingerprint,
            "request": {
                "messages": messages,
                "tools": body.get("tools"),
                "tool_choice": body.get("tool_choice"),
                "temperature": body.get("temperature"),
                "max_tokens": body.get("max_tokens"),
            },
            "response": {
                "message": assistant_msg,
                "usage": response_json.get("usage", {}),
                "finish_reason": finish_reason,
            },
            "trajectory": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "response_text": response_text,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "tool_calls": tool_calls,
            },
        }

    # ---- Token resolution ----

    def _resolve_response_tokens(
        self,
        *,
        runtime_tokens: list[dict[str, Any]],
        response_text: str,
        fallback_logprobs: list[float],
    ) -> tuple[list[int], list[float]]:
        """Resolve response_ids and logprobs from runtime token truth.

        Prefers vLLM's runtime token data (from logprobs) to avoid
        re-tokenisation drift.  Falls back to tokenize_text when runtime
        tokens are unavailable.
        """
        if runtime_tokens:
            response_ids = self._state.convert_runtime_tokens_to_ids(runtime_tokens)
            response_logprobs = [t["logprob"] for t in runtime_tokens]
            if len(response_ids) != len(response_logprobs):
                response_logprobs = fit_list(response_logprobs, len(response_ids))
            return response_ids, response_logprobs

        response_ids = self._state.tokenize_text(response_text)
        response_logprobs = fit_list(fallback_logprobs, len(response_ids))
        return response_ids, response_logprobs

    # ---- Delayed Judge scoring ----

    async def _collect_scored_previous_sample(
        self,
        *,
        messages: list[dict[str, Any]],
        session_id: str,
        turn_num: int,
        trace_id: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Score the previous turn's sample using current user feedback."""
        if self._config.mode not in {"judge_log", "judge_output"}:
            return [], ""

        samples_to_record: list[dict[str, Any]] = []
        feedback_instruction = extract_last_user_instruction(messages)
        previous_sample = await self._state.pop_pending_judge_sample(session_id)

        if previous_sample is None:
            return samples_to_record, feedback_instruction

        if not feedback_instruction.strip():
            await self._state.stage_pending_judge_sample(session_id, previous_sample)
            return samples_to_record, feedback_instruction

        prev_trajectory = previous_sample.get("trajectory", {})
        previous_response_text = str(prev_trajectory.get("response_text") or "")
        prev_request = previous_sample.get("request", {})
        prev_messages = prev_request.get("messages", [])
        previous_instruction_text = extract_last_user_instruction(prev_messages) if isinstance(prev_messages, list) else ""
        if not previous_instruction_text.strip():
            previous_instruction_text = feedback_instruction

        judge_result = await self._score_with_judge(
            response_text=previous_response_text,
            instruction_text=previous_instruction_text,
            followup_user_feedback=feedback_instruction,
            session_id=session_id,
            turn_num=int(previous_sample.get("turn_num") or 0),
        )
        previous_sample["judge"] = judge_result
        previous_sample["judge_feedback"] = {
            "instruction_text": feedback_instruction,
            "feedback_turn_num": turn_num,
        }
        samples_to_record.append(previous_sample)
        logger.info(
            "[Processor %s] scored prev turn=%d score=%.3f",
            trace_id, previous_sample.get("turn_num", 0), judge_result.get("score", 0.0),
        )
        return samples_to_record, feedback_instruction

    async def _score_with_judge(
        self,
        response_text: str,
        instruction_text: str,
        followup_user_feedback: str,
        session_id: str,
        turn_num: int,
    ) -> dict[str, Any]:
        if self._judge_scorer is None:
            return {"score": 0.0, "votes": ["noop"], "details": {}, "error": "judge scorer disabled"}
        try:
            return await self._judge_scorer.score(
                response_text=response_text,
                instruction_text=instruction_text,
                followup_user_feedback=followup_user_feedback,
                session_id=session_id,
                turn_num=turn_num,
            )
        except Exception as exc:
            logger.warning("[Processor] judge failed session=%s turn=%d: %s", session_id, turn_num, exc)
            return {"score": 0.0, "votes": ["fail"], "details": {}, "error": str(exc)}

    async def _queue_current_sample(
        self,
        *,
        sample: dict[str, Any],
        body: dict[str, Any],
        feedback_instruction: str,
        session_id: str,
        turn_num: int,
        trace_id: str,
        samples_to_record: list[dict[str, Any]],
    ) -> None:
        if self._config.mode not in {"judge_log", "judge_output"}:
            samples_to_record.append(sample)
            return

        if bool(body.get("session_done", False)):
            sample["judge"] = {"score": 0.0, "votes": ["skip"], "details": {}, "error": "session_done"}
            samples_to_record.append(sample)
            return

        if not feedback_instruction.strip():
            sample["judge"] = {"score": 0.0, "votes": ["skip"], "details": {}, "error": "no user instruction"}
            samples_to_record.append(sample)
            return

        sample["judge_feedback"] = {"awaiting_followup_user_feedback": True}
        await self._state.stage_pending_judge_sample(session_id, sample)
        logger.debug("[Processor %s] staged sample=%s for delayed judge", trace_id, sample["sample_id"])

    # ---- Main entry point (non-streaming) ----

    async def process_chat_completions(
        self,
        *,
        request: Request,
        body: dict[str, Any],
        x_session_id: Optional[str],
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        messages = self._require_messages(body)
        stream = bool(body.get("stream", False))
        trace_id = self._resolve_trace_id(request)
        session_id = str(x_session_id or "").strip() or "default"
        turn_num = await self._state.next_turn_num(session_id)
        tools = body.get("tools")

        prompt_text, prompt_ids = self._state.build_prompt_text_and_ids(messages=messages, tools=tools)
        upstream_headers = self._state.make_upstream_headers(request)

        if self._config.io_mode == "token":
            forward = await self._state.forward_token(body=body, prompt_ids=prompt_ids, headers=upstream_headers)
        else:
            forward = await self._state.forward_string(body=body, headers=upstream_headers)

        response_json = forward["response_json"]
        assistant_msg = forward["assistant_message"]
        response_text = forward["response_text"]
        tool_calls = forward["tool_calls"]
        runtime_tokens = forward.get("runtime_tokens", [])

        response_ids, response_logprobs = self._resolve_response_tokens(
            runtime_tokens=runtime_tokens,
            response_text=response_text,
            fallback_logprobs=forward.get("response_logprobs", []),
        )
        finish_reason = self._extract_finish_reason(response_json)
        render_fp = self._state.get_render_fingerprint()

        samples_to_record, feedback_instruction = await self._collect_scored_previous_sample(
            messages=messages, session_id=session_id, turn_num=turn_num, trace_id=trace_id,
        )

        sample = self._build_current_sample(
            body=body, messages=messages, session_id=session_id, turn_num=turn_num,
            response_json=response_json, assistant_msg=assistant_msg,
            finish_reason=finish_reason, prompt_text=prompt_text,
            prompt_ids=prompt_ids, response_text=response_text,
            response_ids=response_ids, response_logprobs=response_logprobs,
            tool_calls=tool_calls, render_fingerprint=render_fp,
        )
        await self._queue_current_sample(
            sample=sample, body=body, feedback_instruction=feedback_instruction,
            session_id=session_id, turn_num=turn_num, trace_id=trace_id,
            samples_to_record=samples_to_record,
        )

        emitted_batches: list[dict[str, Any]] = []
        for ready_sample in samples_to_record:
            emitted_batches.extend(await self._state.record_sample(ready_sample))
        await self._state.publish_batches(emitted_batches)

        logger.debug(
            "[Processor %s] done session=%s turn=%d samples=%d batches=%d cost_ms=%.1f",
            trace_id, session_id, turn_num, len(samples_to_record), len(emitted_batches),
            (time.perf_counter() - t0) * 1000,
        )

        return {"response_json": response_json, "stream": stream}

    # ---- Streaming entry point ----

    async def process_chat_completions_stream(
        self,
        *,
        request: Request,
        body: dict[str, Any],
        x_session_id: Optional[str],
    ):
        """True streaming: yield SSE lines to client, record trajectory after stream ends."""
        messages = self._require_messages(body)
        trace_id = self._resolve_trace_id(request)
        session_id = str(x_session_id or "").strip() or "default"
        turn_num = await self._state.next_turn_num(session_id)
        tools = body.get("tools")

        prompt_text, prompt_ids = self._state.build_prompt_text_and_ids(messages=messages, tools=tools)
        upstream_headers = self._state.make_upstream_headers(request)

        line_iter, collector = await self._state.forward_string_stream(body=body, headers=upstream_headers)

        async def _generate():
            async for line in line_iter:
                yield line

            forward_result = self._state.parse_collected_stream(collector)
            response_text = forward_result["response_text"]
            tool_calls = forward_result["tool_calls"]
            runtime_tokens = forward_result.get("runtime_tokens", [])

            response_ids, response_logprobs = self._resolve_response_tokens(
                runtime_tokens=runtime_tokens,
                response_text=response_text,
                fallback_logprobs=forward_result.get("response_logprobs", []),
            )
            finish_reason = self._extract_finish_reason(forward_result["response_json"])
            render_fp = self._state.get_render_fingerprint()

            samples_to_record, feedback_instruction = await self._collect_scored_previous_sample(
                messages=messages, session_id=session_id, turn_num=turn_num, trace_id=trace_id,
            )

            sample = self._build_current_sample(
                body=body, messages=messages, session_id=session_id, turn_num=turn_num,
                response_json=forward_result["response_json"], assistant_msg=forward_result["assistant_message"],
                finish_reason=finish_reason, prompt_text=prompt_text,
                prompt_ids=prompt_ids, response_text=response_text,
                response_ids=response_ids, response_logprobs=response_logprobs,
                tool_calls=tool_calls, render_fingerprint=render_fp,
            )
            await self._queue_current_sample(
                sample=sample, body=body, feedback_instruction=feedback_instruction,
                session_id=session_id, turn_num=turn_num, trace_id=trace_id,
                samples_to_record=samples_to_record,
            )
            emitted_batches: list[dict[str, Any]] = []
            for ready_sample in samples_to_record:
                emitted_batches.extend(await self._state.record_sample(ready_sample))
            await self._state.publish_batches(emitted_batches)

        return _generate()
