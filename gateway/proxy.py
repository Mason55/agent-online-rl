"""
Agent Online-RL Gateway — OpenAI-compatible proxy with per-turn trajectory
recording, delayed LLM-as-Judge reward, streaming, and LoRA injection.

Usage:
    # Factory mode (uvicorn)
    uvicorn gateway.proxy:create_app --factory --host 0.0.0.0 --port 18080

    # CLI mode
    python -m gateway.proxy --llm-url http://localhost:18000 --model-id Qwen/Qwen3-4B

Environment variables (override CLI flags):
    LLM_URL, JUDGE_URL, JUDGE_MODEL, MODEL_ID, GATEWAY_PORT, LLM_API_KEY,
    JUDGE_API_KEY, GATEWAY_API_KEY, RECORD_DIR, LORA_REPO_ROOT, MODE, IO_MODE,
    ROLLOUT_BATCH_SIZE, REQUEST_TIMEOUT, TRAJECTORY_OUTPUTS, OUTPUT_TO_VERL
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI

from gateway.config import GatewayConfig
from gateway.judge_scorer import JudgeScorer
from gateway.processor import LLMMessageProcessor
from gateway.server import GatewayServer
from gateway.state import GatewayState

logger = logging.getLogger("online_rl.gateway")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def build_config_from_env() -> GatewayConfig:
    """Build config purely from environment variables (for uvicorn factory mode)."""
    inference_url = _env("INFERENCE_URL", _env("LLM_URL", "http://127.0.0.1:18000"))
    return GatewayConfig(
        host=_env("GATEWAY_HOST", "0.0.0.0"),
        port=int(_env("GATEWAY_PORT", "18080")),
        rollout_batch_size=int(_env("ROLLOUT_BATCH_SIZE", "8")),
        llm_url=inference_url,
        judge_url=_env("JUDGE_URL", inference_url),
        model_id=_env("MODEL_ID", _env("SERVED_MODEL_NAME", "")),
        model_path=_env("MODEL_PATH", ""),
        judge_model=_env("JUDGE_MODEL", ""),
        mode=_env("MODE", "judge_log"),
        io_mode=_env("IO_MODE", "string"),
        request_timeout=float(_env("REQUEST_TIMEOUT", "120")),
        llm_api_key=_env("LLM_API_KEY", ""),
        judge_api_key=_env("JUDGE_API_KEY", "EMPTY"),
        gateway_api_key=_env("GATEWAY_API_KEY", ""),
        record_dir=_env("RECORD_DIR", "records"),
        log_level=_env("LOG_LEVEL", "INFO"),
        dump_token_ids=_env("DUMP_TOKEN_IDS", "").lower() in ("1", "true"),
        trajectory_outputs=_env("TRAJECTORY_OUTPUTS", ""),
        trajectory_http_url=_env("TRAJECTORY_HTTP_URL", ""),
        trace_stages=_env("TRACE_STAGES", "").lower() in ("1", "true"),
        output_to_verl=_env("OUTPUT_TO_VERL", "").lower() in ("1", "true"),
        max_pending_verl_batches=int(_env("MAX_PENDING_VERL_BATCHES", "0")),
        lora_repo_root=_env("LORA_REPO_ROOT", ""),
    )


def build_app_from_config(config: GatewayConfig) -> FastAPI:
    """Assemble the full gateway app from a config object."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    state = GatewayState(config)

    judge_scorer: Optional[JudgeScorer] = None
    if config.mode in ("judge_log", "judge_output") and config.judge_url:
        judge_scorer = JudgeScorer(
            judge_url=config.judge_url,
            judge_model=config.judge_model or config.model_id,
            api_key=config.judge_api_key or "EMPTY",
        )

    processor = LLMMessageProcessor(state=state, config=config, judge_scorer=judge_scorer)

    lora_repo = None
    if config.lora_repo_root:
        try:
            from storage.lora_repo import LoRARepository
            lora_repo = LoRARepository(config.lora_repo_root)
        except Exception:
            logger.warning("LoRA repo not available at %s", config.lora_repo_root)

    server = GatewayServer(config=config, state=state, processor=processor, lora_repo=lora_repo)
    return server.build_app()


def create_app() -> FastAPI:
    """Factory for ``uvicorn gateway.proxy:create_app --factory``."""
    config = build_config_from_env()
    return build_app_from_config(config)


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description="Online-RL Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--llm-url", default="http://127.0.0.1:18000")
    parser.add_argument("--judge-url", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--mode", default="judge_log", choices=["judge_log", "judge_output", "log"])
    parser.add_argument("--io-mode", default="string", choices=["string", "token"])
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--record-dir", default="records")
    parser.add_argument("--lora-repo-root", default="")
    parser.add_argument("--output-to-verl", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    config = GatewayConfig(
        host=args.host,
        port=args.port,
        rollout_batch_size=args.rollout_batch_size,
        llm_url=args.llm_url,
        judge_url=args.judge_url or args.llm_url,
        model_id=args.model_id,
        judge_model=args.judge_model,
        mode=args.mode,
        io_mode=args.io_mode,
        record_dir=args.record_dir,
        lora_repo_root=args.lora_repo_root,
        output_to_verl=args.output_to_verl,
        log_level=args.log_level,
    )
    app = build_app_from_config(config)

    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())


if __name__ == "__main__":
    main()
