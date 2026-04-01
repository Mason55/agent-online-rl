"""LLM-as-Judge scorer adapter.

Calls the Judge service (which may be a dedicated judge_server with voting,
or a raw vLLM endpoint) to score a single turn.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("online_rl.gateway")

JUDGE_PROMPT_TEMPLATE = """你是一个专业的 AI Agent 质量评估器。请对以下 Agent 对话轮次打分。

## 用户指令
{instruction_text}

## Agent 回复
{response_text}

## 用户反馈（下一轮输入）
{followup_user_feedback}

## 评分维度（各 0-10 分）
1. 任务完成度：Agent 是否完成了用户意图？
2. 响应质量：回答是否准确、有帮助、简洁？
3. 工具使用合理性：工具调用是否必要且正确？
4. 对话连贯性：多轮对话是否自然流畅？

请严格以 JSON 格式返回，不要添加任何其他文字：
{{"task_completion": 8, "response_quality": 7, "tool_usage": 9, "coherence": 8, "overall": 8.0, "reason": "..."}}"""


class JudgeScorer:
    """Call LLM-as-Judge to score a single (instruction, response, feedback) triple."""

    def __init__(
        self,
        *,
        judge_url: str,
        judge_model: str,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        num_votes: int = 1,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.judge_url = judge_url.rstrip("/")
        self.judge_model = judge_model
        self.api_key = api_key
        self.num_votes = max(1, num_votes)
        self._owned_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        if self._owned_client:
            await self._http_client.aclose()

    async def score(
        self,
        *,
        response_text: str,
        instruction_text: str,
        followup_user_feedback: str,
        session_id: str = "",
        turn_num: int = 0,
    ) -> dict[str, Any]:
        """Score a turn and return {score, details, votes}."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction_text=instruction_text or "(无)",
            response_text=response_text or "(无回复)",
            followup_user_feedback=followup_user_feedback or "(无反馈)",
        )

        scores_list = []
        for _ in range(self.num_votes):
            try:
                scores = await self._call_judge(prompt)
                scores_list.append(scores)
            except Exception as exc:
                logger.warning("[JudgeScorer] vote failed session=%s turn=%d: %s", session_id, turn_num, exc)
                scores_list.append({"overall": 5.0, "error": str(exc)})

        overall_values = [s.get("overall", 5.0) for s in scores_list]
        avg_overall = sum(overall_values) / len(overall_values) if overall_values else 5.0
        normalized_score = (avg_overall - 5.0) / 5.0  # [-1, 1]

        return {
            "score": normalized_score,
            "overall_raw": avg_overall,
            "details": scores_list[0] if len(scores_list) == 1 else scores_list,
            "votes": overall_values,
        }

    async def _call_judge(self, prompt: str) -> dict[str, Any]:
        resp = await self._http_client.post(
            f"{self.judge_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._parse_scores(content)

    @staticmethod
    def _parse_scores(content: str) -> dict[str, Any]:
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            scores = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                scores = json.loads(m.group())
            else:
                raise ValueError(f"Cannot parse judge response: {content[:200]}")

        if "overall" not in scores:
            fields = ["task_completion", "response_quality", "tool_usage", "coherence"]
            values = [scores.get(f, 5.0) for f in fields]
            scores["overall"] = sum(values) / len(values)
        return scores
