"""Client for talking to the Judge scoring service."""

from __future__ import annotations

from typing import Any, Optional

import httpx


class JudgeClient:
    """Async HTTP client for the Judge scoring server (/score endpoint)."""

    def __init__(
        self,
        judge_url: str,
        timeout: float = 120.0,
        api_key: str = "",
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = judge_url.rstrip("/")
        self.api_key = api_key
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def score(
        self,
        response_text: str,
        instruction_text: str,
        followup_user_feedback: str = "",
        session_id: str = "",
        turn_num: int = 0,
    ) -> dict[str, Any]:
        payload = {
            "response_text": response_text,
            "instruction_text": instruction_text,
            "followup_user_feedback": followup_user_feedback,
            "session_id": session_id,
            "turn_num": turn_num,
        }
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = await self._client.post(f"{self.base_url}/score", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("invalid judge response")
        return data


if __name__ == "__main__":
    import argparse
    import asyncio
    import json

    parser = argparse.ArgumentParser(description="Judge Client CLI")
    parser.add_argument("--judge-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--feedback", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--turn-num", type=int, default=0)
    args = parser.parse_args()

    async def _main() -> None:
        client = JudgeClient(judge_url=args.judge_url)
        try:
            result = await client.score(
                response_text=args.response,
                instruction_text=args.instruction,
                followup_user_feedback=args.feedback,
                session_id=args.session_id,
                turn_num=args.turn_num,
            )
        finally:
            await client.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(_main())
