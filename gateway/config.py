"""Gateway runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 18080
    rollout_batch_size: int = 8

    llm_url: str = "http://127.0.0.1:18000"
    judge_url: str = "http://127.0.0.1:18001"
    model_id: str = ""
    model_path: str = ""               # local path for tokenizer; falls back to model_id
    judge_model: str = ""
    mode: str = "judge_log"          # judge_log | judge_output
    io_mode: str = "string"          # string | token

    request_timeout: float = 120.0
    llm_api_key: str = ""
    judge_api_key: str = ""
    gateway_api_key: str = ""

    record_dir: str = "records"
    log_level: str = "INFO"
    dump_token_ids: bool = False
    trajectory_outputs: str = ""
    trajectory_http_url: str = ""
    trace_stages: bool = False

    output_to_verl: bool = False
    max_pending_verl_batches: int = 0

    lora_repo_root: str = ""
