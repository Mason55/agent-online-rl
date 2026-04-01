"""OnlineTrainingScheduler — consumes batches from the gateway's in-memory
queue (or verl DataProto buffer) and triggers training.

Unlike TrainingScheduler (which polls SQLite), this scheduler pulls from
the gateway's HTTP queue endpoint or from the verl DataProto future API.

Two operating modes:
  - "queue":  Polls ``GET /v1/gateway/training_queue/pop`` for raw batches,
              converts them to parquet, and launches verl SFT.
  - "verl":   Awaits ``state.get_batch_trajectories()`` futures for pre-built
              DataProto objects, and feeds them directly to verl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

from inference.notifier import InferenceNotifier
from storage.lora_repo import LoRARepository
from storage.models import JobStatus

logger = logging.getLogger("online_rl.scheduler")


class OnlineTrainingScheduler:
    """Pull batches from gateway queue and trigger LoRA training.

    This is designed for the online RL loop where the gateway accumulates
    per-turn samples with token_ids/logprobs and emits fixed-size batches.
    """

    def __init__(
        self,
        *,
        gateway_url: str = "http://127.0.0.1:18080",
        gateway_api_key: str = "",
        poll_interval: float = 30.0,
        min_samples_for_training: int = 32,
        base_model_path: str = "",
        verl_config_path: str = "",
        lora_repo: Optional[LoRARepository] = None,
        notifier: Optional[InferenceNotifier] = None,
        nproc_per_node: int = 1,
        training_gpu_ids: str = "",
        tmp_root: str = "/tmp/agent_rl_online",
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.gateway_api_key = gateway_api_key
        self.poll_interval = poll_interval
        self.min_samples_for_training = min_samples_for_training
        self.base_model_path = base_model_path
        self.verl_config_path = verl_config_path
        self.lora_repo = lora_repo
        self.notifier = notifier
        self.nproc_per_node = nproc_per_node
        self.training_gpu_ids = training_gpu_ids
        self.tmp_root = tmp_root

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._accumulated_samples: list[dict[str, Any]] = []
        self._training_count = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("OnlineTrainingScheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="OnlineTrainScheduler")
        self._thread.start()
        logger.info(
            "OnlineTrainingScheduler started: gateway=%s min_samples=%d poll=%.0fs",
            self.gateway_url, self.min_samples_for_training, self.poll_interval,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("OnlineTrainingScheduler stopped (accumulated=%d)", len(self._accumulated_samples))

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("Error in online training scheduler poll")
            self._stop_event.wait(self.poll_interval)

    def _poll_once(self) -> None:
        """Pull batches from gateway and accumulate samples."""
        headers: dict[str, str] = {}
        if self.gateway_api_key:
            headers["Authorization"] = f"Bearer {self.gateway_api_key}"

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.gateway_url}/v1/gateway/training_queue/pop",
                    json={"max_batches": 10},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug("Failed to poll gateway queue: %s", exc)
            return

        batches = data.get("batches", [])
        if not batches:
            logger.debug("No batches in queue (accumulated=%d)", len(self._accumulated_samples))
            return

        for batch in batches:
            samples = batch.get("samples", [])
            self._accumulated_samples.extend(samples)
            logger.info(
                "Pulled batch=%s size=%d (total accumulated=%d)",
                batch.get("batch_id"), len(samples), len(self._accumulated_samples),
            )

        if len(self._accumulated_samples) >= self.min_samples_for_training:
            self._trigger_training()

    def _trigger_training(self) -> None:
        """Launch a training run with accumulated samples."""
        samples = self._accumulated_samples
        self._accumulated_samples = []
        self._training_count += 1

        logger.info(
            "Triggering training #%d with %d samples",
            self._training_count, len(samples),
        )

        run_dir = Path(self.tmp_root) / f"run_{self._training_count}_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._run_sft_training(samples, run_dir)
        except Exception:
            logger.exception("Training #%d failed", self._training_count)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def _run_sft_training(self, samples: list[dict], run_dir: Path) -> None:
        """Write samples to parquet, run verl SFT, publish LoRA."""
        parquet_path = str(run_dir / "train.parquet")
        self._samples_to_parquet(samples, parquet_path)

        verl_output_dir = str(run_dir / "verl_output")

        from trainer.batch_lora_trainer import run_verl_lora_sft, _find_latest_checkpoint, _convert_fsdp_to_peft

        env_overrides = {}
        if self.training_gpu_ids:
            env_overrides["CUDA_VISIBLE_DEVICES"] = self.training_gpu_ids

        existing_lora = None
        if self.lora_repo:
            existing_lora = self.lora_repo.get_latest("online")

        old_env = os.environ.copy()
        try:
            os.environ.update(env_overrides)
            run_verl_lora_sft(
                base_model=self.base_model_path,
                train_parquet=parquet_path,
                output_dir=verl_output_dir,
                config_path=self.verl_config_path,
                lora_adapter_path=existing_lora.path if existing_lora else None,
                nproc_per_node=self.nproc_per_node,
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        ckpt_dir = _find_latest_checkpoint(verl_output_dir)
        peft_dir = str(run_dir / "peft_adapter")
        _convert_fsdp_to_peft(ckpt_dir, self.base_model_path, peft_dir)

        if self.lora_repo:
            scores = [s.get("judge", {}).get("score", 0.0) for s in samples]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            version = self.lora_repo.publish(
                user_id="online",
                lora_path=peft_dir,
                metadata={"sample_count": len(samples), "avg_score": avg_score},
                base_model=self.base_model_path,
            )
            logger.info("Published LoRA %s (avg_score=%.3f)", version.version, avg_score)

            if self.notifier:
                try:
                    self.notifier.notify_update("online", version.path)
                except Exception:
                    logger.warning("Failed to notify vLLM for LoRA hot-load (non-fatal)")

    def _samples_to_parquet(self, samples: list[dict], output_path: str) -> None:
        """Convert gateway samples to a parquet file for verl SFT."""
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas is required for parquet conversion")

        rows = []
        for s in samples:
            traj = s.get("trajectory", {})
            prompt_text = traj.get("prompt_text", "")
            response_text = traj.get("response_text", "")
            if not prompt_text and not response_text:
                continue
            rows.append({
                "prompt": prompt_text,
                "response": response_text,
                "reward": s.get("judge", {}).get("score", 0.0),
            })

        if not rows:
            raise ValueError("No valid samples for training")

        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
        logger.info("Wrote %d training samples to %s", len(rows), output_path)
