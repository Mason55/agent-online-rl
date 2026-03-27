"""
InferenceNotifier — 通知 vLLM 运行时热加载用户 LoRA。

vLLM 原生支持 /v1/load_lora_adapter 接口，无需重启服务。
加载后，对应 lora_name 的请求将自动应用新权重。
"""

import logging

import httpx

logger = logging.getLogger(__name__)


class InferenceNotifier:
    def __init__(self, vllm_base_url: str, timeout: float = 30.0):
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.timeout = timeout

    def notify_update(self, user_id: str, lora_path: str) -> None:
        """通知 vLLM 热加载指定用户的 LoRA。

        Args:
            user_id: 用户 ID，作为 vLLM 中的 lora_name。
            lora_path: LoRA 权重目录的绝对路径。
        """
        resp = httpx.post(
            f"{self.vllm_base_url}/v1/load_lora_adapter",
            json={"lora_name": user_id, "lora_path": lora_path},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        logger.info("LoRA hot-loaded for user %s: %s", user_id, lora_path)

    def unload(self, user_id: str) -> None:
        """卸载用户 LoRA（可选，用于清理不活跃用户）。"""
        resp = httpx.post(
            f"{self.vllm_base_url}/v1/unload_lora_adapter",
            json={"lora_name": user_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        logger.info("LoRA unloaded for user %s", user_id)
