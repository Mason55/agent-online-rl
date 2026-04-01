"""LLM-as-Judge scoring service with voting and retry logic.

Inspired by agent-gateway/prm_server.py but uses a multi-dimensional
LLM-as-Judge prompt (0-10 per dimension) instead of discrete PRM votes
({-1, 0, +1}).  Retains the voting mechanism (majority over num_votes)
and the retry-on-max-tokens fallback logic.

Usage:
    python -m judge.judge_server \
        --llm-url http://127.0.0.1:18001 \
        --model-id Qwen3-32B \
        --num-votes 3
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("online_rl.judge_server")

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


def _sanitize_text(text: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", "[tool_call block]", text, flags=re.DOTALL)
    text = re.sub(r"<[a-zA-Z_][^>]{0,80}>", "[tag]", text)
    text = re.sub(r"</[a-zA-Z_][^>]{0,80}>", "[/tag]", text)
    return text


def _parse_json_scores(content: str) -> Optional[dict[str, Any]]:
    """Try to extract a JSON object with an 'overall' field."""
    content = content.strip()
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.startswith("json"):
                block = block[4:]
            content = block.strip()
    try:
        scores = json.loads(content)
        if isinstance(scores, dict):
            return scores
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            scores = json.loads(m.group())
            if isinstance(scores, dict):
                return scores
        except json.JSONDecodeError:
            pass
    return None


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    if content is None:
        return ""
    return str(content)


@dataclass
class JudgeConfig:
    llm_url: str
    model_id: str
    api_key: str = ""
    num_votes: int = 1
    temperature: float = 0.1
    max_completion_tokens: int = 4096
    timeout: float = 120.0
    expected_api_key: str = ""


class ScoreRequest(BaseModel):
    response_text: str = Field(default="")
    instruction_text: str = Field(default="")
    followup_user_feedback: str = Field(default="")
    session_id: str = Field(default="")
    turn_num: int = Field(default=0)


class ScoreResponse(BaseModel):
    score: float
    overall_raw: float
    votes: list[float]
    details: Any
    model: str
    session_id: str
    turn_num: int


class JudgeEngine:
    """Multi-vote LLM-as-Judge scoring engine."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.api_base = config.llm_url.rstrip("/")

    async def close(self) -> None:
        await self.client.aclose()

    async def evaluate(
        self,
        response_text: str,
        instruction_text: str,
        followup_user_feedback: str = "",
        session_id: str = "",
        turn_num: int = 0,
    ) -> dict[str, Any]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction_text=_sanitize_text(instruction_text) or "(无)",
            response_text=_sanitize_text(response_text) or "(无回复)",
            followup_user_feedback=_sanitize_text(followup_user_feedback) or "(无反馈)",
        )
        messages = [{"role": "user", "content": prompt}]

        tasks = [self._query_once(messages, i) for i in range(self.config.num_votes)]
        results = await asyncio.gather(*tasks)

        all_scores = [r[0] for r in results]
        all_details = [r[1] for r in results]

        overalls = [s.get("overall", 5.0) if s else 5.0 for s in all_scores]
        avg_overall = sum(overalls) / len(overalls) if overalls else 5.0
        normalized_score = (avg_overall - 5.0) / 5.0

        best_detail = all_details[0] if len(all_details) == 1 else all_details

        logger.info(
            "[Judge] session=%s turn=%d votes=%s -> overall=%.2f score=%.3f",
            session_id, turn_num, overalls, avg_overall, normalized_score,
        )

        return {
            "score": normalized_score,
            "overall_raw": avg_overall,
            "votes": overalls,
            "details": best_detail,
            "model": self.config.model_id,
            "session_id": session_id,
            "turn_num": turn_num,
        }

    async def _query_once(self, messages: list[dict], vote_id: int) -> tuple[Optional[dict], str]:
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_completion_tokens,
            "stream": False,
        }
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            resp = await self.client.post(f"{self.api_base}/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            content = _flatten_content(choice.get("message", {}).get("content", ""))
            finish_reason = str(choice.get("finish_reason") or "")
            scores = _parse_json_scores(content)

            if scores is None and finish_reason == "length":
                retry_payload = dict(payload)
                retry_payload["temperature"] = 0.0
                retry_payload["max_tokens"] = max(payload.get("max_tokens", 1024), 1024)
                retry_resp = await self.client.post(
                    f"{self.api_base}/v1/chat/completions", json=retry_payload, headers=headers,
                )
                retry_resp.raise_for_status()
                retry_data = retry_resp.json()
                retry_choice = retry_data.get("choices", [{}])[0]
                retry_content = _flatten_content(retry_choice.get("message", {}).get("content", ""))
                retry_scores = _parse_json_scores(retry_content)
                if retry_scores is not None:
                    return retry_scores, retry_content
                content = retry_content

            if scores is None:
                logger.warning("[Judge] vote %d unparseable: %s", vote_id, content[:200])
                return None, content

            if "overall" not in scores:
                fields = ["task_completion", "response_quality", "tool_usage", "coherence"]
                vals = [scores.get(f, 5.0) for f in fields]
                scores["overall"] = sum(vals) / len(vals)

            return scores, content
        except Exception as exc:
            logger.warning("[Judge] vote %d failed: %s", vote_id, exc)
            return None, ""


def create_app(config: JudgeConfig) -> FastAPI:
    engine = JudgeEngine(config)

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(title="Judge Server", lifespan=_lifespan)
    app.state.engine = engine

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "model": config.model_id, "num_votes": config.num_votes}

    @app.post("/score", response_model=ScoreResponse)
    async def score(
        req: ScoreRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        if config.expected_api_key:
            if not authorization or not authorization.lower().startswith("bearer "):
                raise HTTPException(status_code=401, detail="missing bearer token")
            token = authorization.split(" ", 1)[1].strip()
            if token != config.expected_api_key:
                raise HTTPException(status_code=403, detail="invalid bearer token")

        return await engine.evaluate(
            response_text=req.response_text,
            instruction_text=req.instruction_text,
            followup_user_feedback=req.followup_user_feedback,
            session_id=req.session_id,
            turn_num=req.turn_num,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-Judge scoring server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--llm-url", required=True, help="Judge vLLM base URL")
    parser.add_argument("--model-id", required=True, help="Judge model name")
    parser.add_argument("--api-key", default="", help="vLLM bearer token")
    parser.add_argument("--judge-api-key", default="", help="API key for /score endpoint")
    parser.add_argument("--num-votes", type=int, default=1, help="Number of judge votes")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = JudgeConfig(
        llm_url=args.llm_url,
        model_id=args.model_id,
        api_key=args.api_key,
        num_votes=max(1, args.num_votes),
        temperature=args.temperature,
        max_completion_tokens=max(1, args.max_completion_tokens),
        timeout=max(1.0, args.timeout),
        expected_api_key=args.judge_api_key,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
