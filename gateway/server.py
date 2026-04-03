"""FastAPI application and routes for the online-RL gateway.

Ported from agent-gateway/gateway_core/gateway_frontend_server.py, adapted
for LLM-as-Judge delayed reward, true streaming, and LoRA injection.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("online_rl.gateway")


class QueuePopRequest(BaseModel):
    max_batches: int = Field(default=1, ge=1, le=100)


class GatewayServer:
    """Build the FastAPI app for the online-RL gateway."""

    def __init__(self, *, config: Any, state: Any, processor: Any, lora_repo: Any = None) -> None:
        self.config = config
        self.state = state
        self.processor = processor
        self.lora_repo = lora_repo

    def build_app(self) -> FastAPI:
        @asynccontextmanager
        async def _lifespan(_: FastAPI):
            try:
                yield
            finally:
                await self.state.close()

        app = FastAPI(title="Online-RL Gateway", lifespan=_lifespan)
        app.state.owner = self.state

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {"ok": True, "mode": self.config.mode, "io_mode": self.config.io_mode}

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok"}

        @app.get("/v1/gateway/stats")
        async def gateway_stats(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
            await self.state.ensure_auth(authorization)
            return await self.state.get_stats()

        @app.post("/v1/gateway/training_queue/pop")
        async def training_queue_pop(
            req: QueuePopRequest,
            authorization: Optional[str] = Header(default=None),
        ) -> dict[str, Any]:
            await self.state.ensure_auth(authorization)
            batches = await self.state.pop_training_queue(req.max_batches)
            stats = await self.state.get_stats()
            return {
                "ok": True,
                "num_batches": len(batches),
                "batches": batches,
                "remaining_batches": stats["training_queue_batches"],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            session_id: Optional[str] = Header(default=None, alias="session-id"),
            x_session_id: Optional[str] = Header(default=None),
        ):
            t0 = time.perf_counter()
            await self.state.ensure_auth(authorization)
            await self.state.inc_request_counter()
            resolved_session = str(session_id or x_session_id or "").strip() or None

            try:
                body = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

            user_id = request.headers.get("x-user-id", "anonymous")

            # LoRA injection
            if self.lora_repo is not None:
                latest_lora = self.lora_repo.get_latest(user_id)
                if latest_lora:
                    body.setdefault("extra_body", {})["lora_name"] = user_id

            stream = bool(body.get("stream", False))

            # Streaming workaround: the original streaming path puts trajectory
            # recording inside the async generator's post-yield code, which
            # Starlette never executes.  Instead, force non-streaming to vLLM,
            # record the trajectory normally, then re-wrap as SSE if the client
            # originally asked for streaming.
            client_wants_stream = stream
            if stream:
                body["stream"] = False

            result = await self.processor.process_chat_completions(
                request=request, body=body, x_session_id=resolved_session,
            )
            response_json = result["response_json"]

            if client_wants_stream:
                return StreamingResponse(
                    self.state.stream_chat_response(response_json),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=response_json)

        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        )
        async def proxy_other(
            path: str,
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            await self.state.ensure_auth(authorization)
            await self.state.inc_request_counter()
            target_url = f"{self.config.llm_url.rstrip('/')}/{path}"
            upstream_headers = self.state.make_upstream_headers(request)
            body_bytes = await request.body()
            try:
                resp = await self.state.proxy_request(
                    method=request.method, url=target_url,
                    params=dict(request.query_params),
                    headers=upstream_headers, content=body_bytes,
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"proxy failed: {exc}") from exc
            response_headers = {}
            for key, value in resp.headers.items():
                if key.lower() in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
                    continue
                response_headers[key] = value
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )

        return app
