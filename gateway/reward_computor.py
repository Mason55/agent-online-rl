import json
import logging
from typing import Optional

import httpx

from storage.models import Trajectory

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """你是一个专业的 AI Agent 质量评估器。请对以下 Agent 对话轨迹打分。

## 对话轨迹
{trajectory_text}

## 评分维度（各 0-10 分）
1. 任务完成度：Agent 是否完成了用户意图？
2. 响应质量：回答是否准确、有帮助、简洁？
3. 工具使用合理性：工具调用是否必要且正确？
4. 对话连贯性：多轮对话是否自然流畅？

请严格以 JSON 格式返回，不要添加任何其他文字：
{{"task_completion": 8, "response_quality": 7, "tool_usage": 9, "coherence": 8, "overall": 8.0, "reason": "..."}}"""


class RewardComputor:
    def __init__(self, judge_base_url: str, judge_model: str, api_key: str = "EMPTY"):
        self.judge_base_url = judge_base_url.rstrip("/")
        self.judge_model = judge_model
        self.api_key = api_key

    async def compute_async(self, trajectory: Trajectory) -> Trajectory:
        """异步调用 LLM-as-Judge，填入 reward 后返回。"""
        trajectory_text = self._format_trajectory(trajectory)
        prompt = JUDGE_PROMPT_TEMPLATE.format(trajectory_text=trajectory_text)
        try:
            scores = await self._call_judge(prompt)
            trajectory.reward = (scores["overall"] - 5.0) / 5.0
            trajectory.reward_details = scores
        except Exception as e:
            logger.warning(f"Reward computation failed for {trajectory.trajectory_id}: {e}")
            trajectory.reward = 0.0  # 中性 reward，不影响训练
            trajectory.reward_details = {"error": str(e)}
        return trajectory

    def compute(self, trajectory: Trajectory) -> Trajectory:
        """同步版本，用于调试。"""
        import asyncio
        return asyncio.run(self.compute_async(trajectory))

    def _format_trajectory(self, traj: Trajectory) -> str:
        lines = []
        for i, turn in enumerate(traj.turns):
            lines.append(f"[Turn {i+1}] {turn.role.upper()}: {turn.content}")
        return "\n".join(lines)

    async def _call_judge(self, prompt: str) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.judge_base_url}/v1/chat/completions",
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

    def _parse_scores(self, content: str) -> dict:
        """解析 JSON 打分，容错处理。"""
        content = content.strip()
        # 尝试提取 JSON 块
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            scores = json.loads(content)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                scores = json.loads(m.group())
            else:
                raise ValueError(f"Cannot parse judge response: {content[:200]}")

        # 确保 overall 字段存在
        if "overall" not in scores:
            fields = ["task_completion", "response_quality", "tool_usage", "coherence"]
            values = [scores.get(f, 5.0) for f in fields]
            scores["overall"] = sum(values) / len(values)
        return scores
