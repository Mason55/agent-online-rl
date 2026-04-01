"""Convert gateway trajectory samples into verl DataProto.

Ported from agent-gateway/gateway_core/trajectory_to_verl_dataproto_converter.py.
Field name 'prm' is generalized to 'judge' but the tensor layout is identical.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional, Sequence

import torch


class VerlDataProtoConverter:
    """Build a verl DataProto from gateway trajectory samples."""

    def __init__(
        self,
        *,
        dataproto_cls: Optional[type] = None,
        pad_token_id: int = 0,
        default_score: float = 0.0,
    ) -> None:
        self._dataproto_cls = dataproto_cls
        self.pad_token_id = int(pad_token_id)
        self.default_score = float(default_score)

    def convert_batch(self, batch: dict[str, Any]) -> Any:
        samples = batch.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("batch.samples must be a non-empty list")
        return self.convert_samples(samples)

    def convert_samples(self, samples: Sequence[dict[str, Any]]) -> Any:
        if not samples:
            raise ValueError("samples must be non-empty")

        rows = [self._normalize_sample(sample=s, idx=i) for i, s in enumerate(samples)]
        input_max = max(max((len(r["input_ids"]) for r in rows), default=0), 1)
        prompt_max = max(max((len(r["prompt_ids"]) for r in rows), default=0), 1)
        response_max = max(max((len(r["response_ids"]) for r in rows), default=0), 1)

        bs = len(rows)
        input_ids = torch.full((bs, input_max), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, input_max), dtype=torch.long)
        position_ids = torch.zeros((bs, input_max), dtype=torch.long)
        # Per-turn training (OpenClaw-RL style): response_mask covers only
        # the response portion.  prompt=0 (context, not in loss),
        # response=1 (current turn model output, in loss).
        response_mask = torch.zeros((bs, response_max), dtype=torch.long)
        old_log_probs = torch.zeros((bs, response_max), dtype=torch.float32)
        token_level_scores = torch.zeros((bs, response_max), dtype=torch.float32)
        prompts = torch.full((bs, prompt_max), self.pad_token_id, dtype=torch.long)
        responses = torch.full((bs, response_max), self.pad_token_id, dtype=torch.long)

        for row_idx, row in enumerate(rows):
            il = len(row["input_ids"])
            pl = len(row["prompt_ids"])
            rl = len(row["response_ids"])

            input_ids[row_idx, :il] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask[row_idx, :il] = 1
            position_ids[row_idx, :il] = torch.arange(il, dtype=torch.long)
            prompts[row_idx, :pl] = torch.tensor(row["prompt_ids"], dtype=torch.long)
            responses[row_idx, :rl] = torch.tensor(row["response_ids"], dtype=torch.long)

            if rl > 0:
                response_mask[row_idx, :rl] = 1
                old_log_probs[row_idx, :rl] = torch.tensor(row["response_logprobs"], dtype=torch.float32)
                token_level_scores[row_idx, :rl] = row["judge_score"]

        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "token_level_scores": token_level_scores,
        }
        non_tensors = {
            "sample_id": [r["sample_id"] for r in rows],
            "session_id": [r["session_id"] for r in rows],
            "turn_num": [r["turn_num"] for r in rows],
            "created_at": [r["created_at"] for r in rows],
            "mode": [r["mode"] for r in rows],
            "io_mode": [r["io_mode"] for r in rows],
            "model": [r["model"] for r in rows],
            "prompt_text": [r["prompt_text"] for r in rows],
            "response_text": [r["response_text"] for r in rows],
            "judge_score": [r["judge_score"] for r in rows],
        }
        meta_info = {
            "source": "agent-online-rl",
            "converter": "verl_dataproto",
            "num_samples": bs,
            "pad_token_id": self.pad_token_id,
        }

        dataproto_cls = self._resolve_dataproto_cls()
        return dataproto_cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def _resolve_dataproto_cls(self) -> type:
        if self._dataproto_cls is not None:
            return self._dataproto_cls
        try:
            module = importlib.import_module("verl.protocol")
            cls = getattr(module, "DataProto")
        except Exception as exc:
            raise RuntimeError("Failed to import verl DataProto. Is verl installed?") from exc
        self._dataproto_cls = cls
        return cls

    def _normalize_sample(self, *, sample: dict[str, Any], idx: int) -> dict[str, Any]:
        trajectory = sample.get("trajectory", {})
        if not isinstance(trajectory, dict):
            trajectory = {}

        prompt_ids = self._coerce_int_list(trajectory.get("prompt_ids"))
        response_ids = self._coerce_int_list(trajectory.get("response_ids"))
        input_ids = self._coerce_int_list(trajectory.get("input_ids"))

        if not input_ids:
            input_ids = prompt_ids + response_ids
        if not response_ids and input_ids and prompt_ids and len(input_ids) >= len(prompt_ids):
            response_ids = input_ids[len(prompt_ids):]
        if not prompt_ids and input_ids and response_ids and len(input_ids) >= len(response_ids):
            prompt_ids = input_ids[:len(input_ids) - len(response_ids)]
        if not input_ids:
            raise ValueError(f"sample[{idx}] has no valid token ids")

        response_len = len(response_ids)
        response_logprobs = self._coerce_float_list(trajectory.get("response_logprobs"))
        response_logprobs = response_logprobs[:response_len]
        if response_len > len(response_logprobs):
            response_logprobs.extend([0.0] * (response_len - len(response_logprobs)))

        judge_obj = sample.get("judge", {})
        judge_score = self.default_score
        if isinstance(judge_obj, dict):
            sv = judge_obj.get("score")
            if isinstance(sv, (int, float)):
                judge_score = float(sv)

        return {
            "sample_id": str(sample.get("sample_id") or ""),
            "session_id": str(sample.get("session_id") or "default"),
            "turn_num": int(sample.get("turn_num") or 0),
            "created_at": str(sample.get("created_at") or ""),
            "mode": str(sample.get("mode") or ""),
            "io_mode": str(sample.get("io_mode") or ""),
            "model": str(sample.get("model") or ""),
            "prompt_text": str(trajectory.get("prompt_text") or ""),
            "response_text": str(trajectory.get("response_text") or ""),
            "input_ids": input_ids,
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_logprobs": response_logprobs,
            "judge_score": judge_score,
        }

    @staticmethod
    def _coerce_int_list(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        return [int(x) for x in value if isinstance(x, (int, float))]

    @staticmethod
    def _coerce_float_list(value: Any) -> list[float]:
        if not isinstance(value, list):
            return []
        return [float(x) for x in value if isinstance(x, (int, float))]
