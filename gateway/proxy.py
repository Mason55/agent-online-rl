"""
Agent RL Gateway — OpenAI-compatible 透明代理。

启动方式:
    uvicorn gateway.proxy:create_app --factory --host 0.0.0.0 --port 8080

环境变量:
    INFERENCE_URL      推理服务地址（如 http://localhost:8000）
    JUDGE_URL          Judge LLM 地址（如 http://localhost:8001）
    JUDGE_MODEL        Judge 模型名
    DB_PATH            轨迹数据库路径（默认 trajectories.db）
    LORA_REPO_ROOT     LoRA 仓库根目录（默认 lora_repo）
    VLLM_URL           vLLM 推理服务地址（用于热加载通知，默认同 INFERENCE_URL）
"""

import asyncio
import copy
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.recorder import SessionRecorder
from gateway.reward_computor import RewardComputor
from inference.notifier import InferenceNotifier
from storage.lora_repo import LoRARepository
from storage.models import Trajectory
from storage.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)

# ---- 全局单例（由 create_app 初始化）----
_store: Optional[TrajectoryStore] = None
_lora_repo: Optional[LoRARepository] = None
_recorder: Optional[SessionRecorder] = None
_reward_computor: Optional[RewardComputor] = None
_notifier: Optional[InferenceNotifier] = None
_inference_url: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _lora_repo, _recorder, _reward_computor, _notifier, _inference_url

    _inference_url = os.environ["INFERENCE_URL"].rstrip("/")
    judge_url = os.environ.get("JUDGE_URL", _inference_url)
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4")
    db_path = os.environ.get("DB_PATH", "trajectories.db")
    lora_root = os.environ.get("LORA_REPO_ROOT", "lora_repo")
    vllm_url = os.environ.get("VLLM_URL", _inference_url)

    _store = TrajectoryStore(db_path)
    _lora_repo = LoRARepository(lora_root)
    _recorder = SessionRecorder()
    _reward_computor = RewardComputor(judge_url, judge_model)
    _notifier = InferenceNotifier(vllm_url)

    logger.info("Gateway started: inference=%s judge=%s", _inference_url, judge_url)
    yield
    logger.info("Gateway stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Agent RL Gateway", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        user_id = request.headers.get("X-User-ID", "anonymous")
        session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))

        body = await request.json()

        # 注入用户私有 LoRA（若存在）
        latest_lora = _lora_repo.get_latest(user_id)
        if latest_lora:
            body.setdefault("extra_body", {})["lora_name"] = user_id

        # 录制请求
        _recorder.record_request(session_id, user_id, body.get("messages", []))

        # 转发到推理服务
        is_stream = body.get("stream", False)
        if is_stream:
            return await _stream_forward(body, session_id)
        else:
            return await _forward(body, session_id)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


async def _forward(body: dict, session_id: str) -> JSONResponse:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{_inference_url}/v1/chat/completions",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    # 录制响应，检测 session 是否结束
    trajectory = _recorder.record_response(session_id, data)
    if trajectory:
        asyncio.create_task(_handle_trajectory(trajectory))

    return JSONResponse(content=data, status_code=resp.status_code)


async def _stream_forward(body: dict, session_id: str) -> StreamingResponse:
    """流式转发：透传 SSE 给 Agent，同时收集完整响应用于录制。"""
    collected_content = []
    finish_reason = None

    async def generate():
        nonlocal finish_reason
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{_inference_url}/v1/chat/completions",
                json=body,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                collected_content.append(delta["content"])
                            fr = chunk.get("choices", [{}])[0].get("finish_reason")
                            if fr:
                                finish_reason = fr
                        except Exception:
                            pass
                    yield line + "\n"

        # 流结束后录制
        if finish_reason == "stop":
            fake_response = {
                "choices": [{
                    "message": {"content": "".join(collected_content), "role": "assistant"},
                    "finish_reason": "stop",
                }]
            }
            trajectory = _recorder.record_response(session_id, fake_response)
            if trajectory:
                asyncio.create_task(_handle_trajectory(trajectory))

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _handle_trajectory(trajectory: Trajectory) -> None:
    """异步：计算 reward → 存储。调度由 TrainingScheduler 负责。"""
    try:
        scored = await _reward_computor.compute_async(trajectory)
        _store.save(scored)
        logger.info(
            "Trajectory saved: user=%s reward=%.3f",
            trajectory.user_id, scored.reward or 0,
        )
    except Exception as e:
        logger.error("Failed to handle trajectory %s: %s", trajectory.trajectory_id, e)
