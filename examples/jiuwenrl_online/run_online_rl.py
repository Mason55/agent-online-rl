# -*- coding: UTF-8 -*-
"""JiuwenClaw 在线 RL 闭环启动脚本。

拉起完整的在线强化学习闭环：

    1. vLLM 推理服务  (TP=2, LoRA 热加载, GPU 0,1, port 18000)
    2. agent-online-rl Gateway  (port 18080, 记录轨迹 + LLM-as-Judge)
       Judge 默认复用推理 vLLM；可通过 --judge-url 指定独立 Judge 服务
    3. TrainingScheduler  (后台线程, 轨迹累积达阈值自动触发 SFT LoRA 训练)
    4. JiuwenClaw 全栈  (jiuwenclaw-start, 含 Web 前端 + AgentServer)

用户通过 Web 前端 (http://localhost:5173) 与 JiuwenClaw 交互，
交互产生的 LLM 请求经 Gateway 透明代理到 vLLM，同时记录轨迹。
Gateway 调用 Judge (默认复用推理模型) 对轨迹进行多维度评分。
轨迹累积到阈值后，TrainingScheduler 自动启动 LoRA 微调 (GPU 2-7)，
训练完成后热加载到 vLLM，后续请求立即使用新 LoRA。

Usage:
    python run_online_rl.py
    python run_online_rl.py --threshold 10 --scan-interval 120
    python run_online_rl.py --inference-url http://localhost:18000  # 跳过 vLLM 启动
    python run_online_rl.py --judge-url http://localhost:18001      # 使用独立 Judge 服务
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
# agent-online-rl repo root: examples/jiuwenrl_online/../../
AGENT_ONLINE_RL = (_SCRIPT_DIR / '..' / '..').resolve()
# Workspace root (parent of agent-online-rl) — contains peer repos
REPO_ROOT = AGENT_ONLINE_RL.parent
JIUWENCLAW_REPO = REPO_ROOT / 'jiuwenclaw'
WORKSPACE = Path.home() / '.jiuwenclaw'
CONFIG_ENV = WORKSPACE / 'config' / '.env'

DEFAULT_MODEL_PATH = '/data1/models/Qwen/Qwen3-4B-Instruct-2507'
DEFAULT_MODEL_NAME = 'Qwen3-4B-Instruct-2507'

DEFAULT_JUDGE_MODEL_PATH = '/data1/models/Qwen/Qwen3-32B'
DEFAULT_JUDGE_MODEL_NAME = 'Qwen3-32B'

for p in [str(JIUWENCLAW_REPO), str(AGENT_ONLINE_RL)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('online_rl')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_port_free(host: str, port: int) -> None:
    """Abort early if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f'Port {host}:{port} is already in use. '
                f'Kill the occupying process first: lsof -i :{port}'
            )


def _wait_for_port(host: str, port: int, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f'Port {host}:{port} did not open within {timeout}s')


def _wait_for_health(url: str, timeout: float = 300.0) -> None:
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f'Health check {url} did not pass within {timeout}s')


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _flush_log(proc: subprocess.Popen | None, name: str, log_dir: Path) -> None:
    if proc is None or proc.stdout is None:
        return
    try:
        output = proc.stdout.read()
    except Exception:
        return
    if not output:
        return
    log_path = log_dir / f'{name}.log'
    if isinstance(output, bytes):
        output = output.decode('utf-8', errors='replace')
    log_path.write_text(output + '\n', encoding='utf-8')
    log.info('Saved %s log -> %s', name, log_path)


# ---------------------------------------------------------------------------
# Service launchers
# ---------------------------------------------------------------------------

def _start_vllm(
    model_path: str,
    port: int,
    gpu_ids: str = '0,1',
    served_model_name: str | None = None,
    tp: int = 2,
) -> subprocess.Popen:
    """Start vLLM with TP and LoRA hot-loading enabled."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_ids

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model_path,
        '--served-model-name', served_model_name or model_path,
        '--port', str(port),
        '--host', '0.0.0.0',
        '--tensor-parallel-size', str(tp),
        '--enable-lora',
        '--max-loras', '4',
        '--max-lora-rank', '32',
        '--enable-auto-tool-choice',
        '--tool-call-parser', 'hermes',
        '--max-model-len', '32768',
        '--gpu-memory-utilization', '0.85',
    ]
    log.info('[1a/5] Starting vLLM (TP=%d, LoRA=on) on GPU [%s], port %d ...', tp, gpu_ids, port)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _start_judge_vllm(
    model_path: str,
    port: int,
    gpu_ids: str = '4,5,6,7',
    served_model_name: str | None = None,
    tp: int = 4,
) -> subprocess.Popen:
    """Start a dedicated vLLM instance for LLM-as-Judge (Qwen3-32B, TP=4)."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_ids

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model_path,
        '--served-model-name', served_model_name or model_path,
        '--port', str(port),
        '--host', '0.0.0.0',
        '--tensor-parallel-size', str(tp),
        '--max-model-len', '8192',
        '--gpu-memory-utilization', '0.85',
        '--max-num-seqs', '16',
    ]
    log.info('[1b/5] Starting Judge vLLM (TP=%d) on GPU [%s], port %d ...', tp, gpu_ids, port)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _start_gateway(
    inference_url: str,
    judge_url: str,
    judge_model: str,
    model_id: str,
    model_path: str,
    lora_repo_root: str,
    port: int,
    mode: str = 'judge_log',
    rollout_batch_size: int = 8,
) -> subprocess.Popen:
    """Start agent-online-rl Gateway (new architecture).

    The new gateway uses per-turn trajectory recording, delayed LLM-as-Judge
    scoring, true streaming, and optional verl DataProto output.
    """
    env = os.environ.copy()
    env['LLM_URL'] = inference_url
    env['JUDGE_URL'] = judge_url
    env['JUDGE_MODEL'] = judge_model
    env['MODEL_ID'] = model_id
    env['MODEL_PATH'] = model_path
    env['MODE'] = mode
    env['ROLLOUT_BATCH_SIZE'] = str(rollout_batch_size)
    env['GATEWAY_PORT'] = str(port)
    env['RECORD_DIR'] = 'records'
    if lora_repo_root:
        env['LORA_REPO_ROOT'] = lora_repo_root

    cmd = [
        sys.executable, '-m', 'uvicorn',
        'gateway.proxy:create_app',
        '--factory',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--log-level', 'info',
    ]
    log.info('[2/5] Starting Gateway on port %d (mode=%s) ...', port, mode)
    return subprocess.Popen(
        cmd,
        cwd=str(AGENT_ONLINE_RL),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _ensure_workspace(gateway_url: str, model_name: str) -> None:
    """Ensure JiuwenClaw .env points to the Gateway."""
    from jiuwenclaw.utils import prepare_workspace

    if not CONFIG_ENV.exists():
        prepare_workspace(overwrite=False, preferred_language='zh')

    updates = {
        'API_BASE': gateway_url,
        'API_KEY': 'EMPTY',
        'MODEL_NAME': model_name,
        'MODEL_PROVIDER': 'OpenAI',
        'EMBED_API_BASE': gateway_url,
        'EMBED_API_KEY': 'EMPTY',
        'EMBED_MODEL': model_name,
        'BROWSER_RUNTIME_MCP_ENABLED': '0',
        'EVOLUTION_AUTO_SCAN': 'false',
    }

    existing: dict[str, str] = {}
    if CONFIG_ENV.exists():
        for line in CONFIG_ENV.read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                existing[key.strip()] = value.strip()

    quoted_keys = {
        'API_BASE', 'API_KEY', 'MODEL_NAME', 'MODEL_PROVIDER',
        'EMBED_API_BASE', 'EMBED_API_KEY', 'EMBED_MODEL',
    }
    for key, value in updates.items():
        existing[key] = f'"{value}"' if key in quoted_keys else value

    lines = [f'{key}={value}' for key, value in existing.items()]
    CONFIG_ENV.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _start_jiuwenclaw() -> tuple[subprocess.Popen, subprocess.Popen | None]:
    """Start JiuwenClaw app + web frontend (if dist exists)."""
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{JIUWENCLAW_REPO}:{env.get('PYTHONPATH', '')}".rstrip(':')

    cmd = [sys.executable, '-m', 'jiuwenclaw.app']
    log.info('[4/5] Starting JiuwenClaw app ...')
    app_proc = subprocess.Popen(
        cmd,
        cwd=str(JIUWENCLAW_REPO),
        env=env,
        stdout=None,
        stderr=None,
    )

    web_proc = None
    dist_dir = JIUWENCLAW_REPO / 'jiuwenclaw' / 'web' / 'dist'
    if not dist_dir.exists():
        dist_dir = WORKSPACE / 'web' / 'dist'
    if dist_dir.exists():
        web_cmd = [sys.executable, '-m', 'jiuwenclaw.app_web', '--host', '0.0.0.0', '--dist', str(dist_dir)]
        log.info('[5/5] Starting JiuwenClaw web frontend (dist=%s) ...', dist_dir)
        web_proc = subprocess.Popen(
            web_cmd,
            cwd=str(JIUWENCLAW_REPO),
            env=env,
            stdout=None,
            stderr=None,
        )
    else:
        log.warning('[5/5] Web dist not found, skipping frontend. '
                     'Build it: cd jiuwenclaw/jiuwenclaw/web && npm install && npm run build')

    return app_proc, web_proc


# ---------------------------------------------------------------------------
# Training scheduler setup
# ---------------------------------------------------------------------------

def _start_online_training_scheduler(
    gateway_url: str,
    model_path: str,
    lora_repo: str,
    vllm_url: str,
    train_gpu: str,
    threshold: int,
    scan_interval: int,
):
    """Start the OnlineTrainingScheduler that polls the gateway queue."""
    from scheduler.online_training_scheduler import OnlineTrainingScheduler
    from storage.lora_repo import LoRARepository
    from inference.notifier import InferenceNotifier

    lora_repo_obj = LoRARepository(lora_repo)
    notifier = InferenceNotifier(vllm_url)

    train_gpu_count = len(train_gpu.split(','))
    scheduler = OnlineTrainingScheduler(
        gateway_url=gateway_url,
        poll_interval=float(scan_interval),
        min_samples_for_training=threshold,
        base_model_path=model_path,
        verl_config_path=str(AGENT_ONLINE_RL / 'config' / 'ppo_lora_trainer.yaml'),
        lora_repo=lora_repo_obj,
        notifier=notifier,
        nproc_per_node=train_gpu_count,
        training_gpu_ids=train_gpu,
    )
    scheduler.start()
    return scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='JiuwenClaw 在线 RL 闭环：交互 → 轨迹采集 → 自动训练 → LoRA 热加载',
    )
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='基座模型路径')
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='vLLM 注册的模型名')
    parser.add_argument('--vllm-gpu', default='0,1', help='vLLM 推理使用的 GPU (逗号分隔)')
    parser.add_argument('--vllm-tp', type=int, default=2, help='推理 Tensor Parallel 大小')
    parser.add_argument('--vllm-port', type=int, default=18000, help='vLLM 推理端口')

    parser.add_argument('--judge-model-path', default=DEFAULT_JUDGE_MODEL_PATH, help='Judge 模型路径')
    parser.add_argument('--judge-model-name', default=DEFAULT_JUDGE_MODEL_NAME, help='Judge vLLM 注册的模型名')
    parser.add_argument('--judge-gpu', default='4,5,6,7', help='Judge vLLM 使用的 GPU (逗号分隔)')
    parser.add_argument('--judge-tp', type=int, default=4, help='Judge Tensor Parallel 大小')
    parser.add_argument('--judge-port', type=int, default=18001, help='Judge vLLM 端口')

    parser.add_argument('--gateway-port', type=int, default=18080, help='Gateway 端口')
    parser.add_argument('--gateway-mode', default='judge_output',
                        choices=['judge_log', 'judge_output', 'log'],
                        help='Gateway 模式: judge_output(打分+记录+训练队列, 默认), judge_log(打分+记录,不入队), log(仅记录)')
    parser.add_argument('--rollout-batch-size', type=int, default=8, help='Gateway batch 大小')
    parser.add_argument('--threshold', type=int, default=4, help='触发训练的样本数阈值')
    parser.add_argument('--scan-interval', type=int, default=30, help='TrainingScheduler 扫描间隔 (秒)')
    parser.add_argument('--train-gpu', default='2,3', help='训练使用的 GPU (逗号分隔)')
    parser.add_argument('--lora-repo', default=None, help='LoRA 存储目录 (默认: ./lora_repo)')
    parser.add_argument(
        '--inference-url', default=None,
        help='跳过推理 vLLM 启动，直接连接已有推理服务',
    )
    parser.add_argument(
        '--judge-url', default=None,
        help='跳过 Judge vLLM 启动，直接连接已有 Judge 服务',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    lora_repo = args.lora_repo or str(script_dir / 'lora_repo')
    skip_vllm = args.inference_url is not None
    inference_url = args.inference_url or f'http://127.0.0.1:{args.vllm_port}'
    vllm_url = inference_url
    gateway_base_url = f'http://127.0.0.1:{args.gateway_port}'
    gateway_url = f'{gateway_base_url}/v1'

    # Judge: if --judge-url given, use it; otherwise reuse inference vLLM
    if args.judge_url:
        judge_url = args.judge_url
        skip_judge = True
    elif args.judge_model_name != args.model_name:
        judge_url = f'http://127.0.0.1:{args.judge_port}'
        skip_judge = False
    else:
        judge_url = inference_url
        skip_judge = True
        log.info('Judge will reuse inference vLLM (%s)', args.model_name)

    # Pre-flight: ensure ports are free
    ports_to_check = [('Gateway', args.gateway_port)]
    if not skip_vllm:
        ports_to_check.append(('vLLM-Inference', args.vllm_port))
    if not skip_judge:
        ports_to_check.append(('vLLM-Judge', args.judge_port))
    ports_to_check.extend([
        ('JiuwenClaw-AgentServer', 18092),
        ('JiuwenClaw-WS', 19000),
    ])
    for name, port in ports_to_check:
        _check_port_free('127.0.0.1', port)
        log.info('  Port %d (%s) is free', port, name)

    vllm_proc = None
    judge_proc = None
    gateway_proc = None
    claw_proc = None
    web_proc = None
    training_scheduler = None

    def _shutdown(signum=None, frame=None):
        print()
        log.info('Shutting down all services ...')
        if training_scheduler:
            training_scheduler.stop()
        _terminate(web_proc)
        _terminate(claw_proc)
        _terminate(gateway_proc)
        _terminate(judge_proc)
        _terminate(vllm_proc)
        _flush_log(gateway_proc, 'gateway', log_dir)
        if judge_proc:
            _flush_log(judge_proc, 'judge_vllm', log_dir)
        if vllm_proc:
            _flush_log(vllm_proc, 'vllm', log_dir)
        log.info('All services stopped.')

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        # ---- 1a. vLLM Inference ----
        if not skip_vllm:
            vllm_proc = _start_vllm(
                args.model_path, args.vllm_port,
                gpu_ids=args.vllm_gpu, served_model_name=args.model_name, tp=args.vllm_tp,
            )
        else:
            log.info('[1a/5] Using existing inference at %s', inference_url)

        # ---- 1b. vLLM Judge ----
        if not skip_judge:
            judge_proc = _start_judge_vllm(
                args.judge_model_path, args.judge_port,
                gpu_ids=args.judge_gpu, served_model_name=args.judge_model_name, tp=args.judge_tp,
            )
        else:
            log.info('[1b/5] Using existing Judge at %s', judge_url)

        # Wait for both vLLM services to be healthy
        if not skip_vllm:
            log.info('  Waiting for Inference vLLM health check (may take 1-3 min) ...')
            _wait_for_health(f'{inference_url}/health', timeout=300.0)
            log.info('  Inference vLLM ready at %s', inference_url)
        else:
            _wait_for_health(f'{inference_url}/health', timeout=30.0)

        if not skip_judge:
            log.info('  Waiting for Judge vLLM health check (may take 2-5 min) ...')
            _wait_for_health(f'{judge_url}/health', timeout=600.0)
            log.info('  Judge vLLM ready at %s', judge_url)
        else:
            _wait_for_health(f'{judge_url}/health', timeout=30.0)

        # ---- 2. Gateway ----
        gateway_proc = _start_gateway(
            inference_url=inference_url,
            judge_url=judge_url,
            judge_model=args.judge_model_name,
            model_id=args.model_name,
            model_path=args.model_path,
            lora_repo_root=lora_repo,
            port=args.gateway_port,
            mode=args.gateway_mode,
            rollout_batch_size=args.rollout_batch_size,
        )
        _wait_for_health(f'{gateway_base_url}/health', timeout=30.0)
        log.info('  Gateway ready at port %d', args.gateway_port)

        # ---- 3. OnlineTrainingScheduler ----
        log.info('[3/5] Starting OnlineTrainingScheduler (threshold=%d, interval=%ds) ...',
                 args.threshold, args.scan_interval)
        training_scheduler = _start_online_training_scheduler(
            gateway_url=gateway_base_url,
            model_path=args.model_path,
            lora_repo=lora_repo,
            vllm_url=vllm_url,
            train_gpu=args.train_gpu,
            threshold=args.threshold,
            scan_interval=args.scan_interval,
        )
        log.info('  OnlineTrainingScheduler running (train GPU: [%s])', args.train_gpu)

        # ---- 4/5. JiuwenClaw (app + web) ----
        _ensure_workspace(gateway_url, args.model_name)
        claw_proc, web_proc = _start_jiuwenclaw()
        time.sleep(5)
        log.info('  JiuwenClaw app started (pid=%d)', claw_proc.pid)
        if web_proc:
            log.info('  JiuwenClaw web started (pid=%d)', web_proc.pid)

        # ---- Ready ----
        has_web = web_proc is not None
        print()
        print('=' * 60)
        print('  JiuwenClaw 在线 RL 闭环已启动 (v2: per-turn + Judge)')
        print()
        if has_web:
            print(f'  Web 前端:        http://localhost:5173')
        print(f'  JiuwenClaw WS:   ws://localhost:19000/ws')
        print(f'  vLLM 推理:       {inference_url}')
        judge_label = '复用推理' if judge_url == inference_url else args.judge_model_name
        print(f'  vLLM Judge:      {judge_url} ({judge_label})')
        print(f'  Gateway 代理:    {gateway_base_url}')
        print(f'  Gateway 模式:    {args.gateway_mode}')
        print(f'  轨迹记录:        records/ (JSONL, per-turn)')
        print(f'  LoRA 仓库:       {lora_repo}')
        print(f'  训练阈值:        {args.threshold} 条样本')
        print(f'  扫描间隔:        {args.scan_interval} 秒')
        print(f'  训练 GPU:        [{args.train_gpu}]')
        print()
        if has_web:
            print('  打开 http://localhost:5173 开始对话，')
        else:
            print('  通过 WebSocket (ws://localhost:19000/ws) 对话，')
        print('  每轮对话自动记录 token_ids + logprobs，')
        print('  下一轮到来时触发延迟 Judge 打分，')
        print('  样本累积达阈值后自动触发 LoRA 训练。')
        print('  按 Ctrl+C 停止所有服务。')
        print('=' * 60)
        print()

        # Block until a critical child exits or interrupt
        while True:
            for name, proc in [('vllm', vllm_proc), ('judge_vllm', judge_proc), ('gateway', gateway_proc), ('jiuwenclaw', claw_proc)]:
                if proc is not None and proc.poll() is not None:
                    log.error('%s exited unexpectedly with code %d — stopping', name, proc.returncode)
                    return

            try:
                import urllib.request
                import json as _json
                req = urllib.request.Request(f'{gateway_base_url}/v1/gateway/stats')
                with urllib.request.urlopen(req, timeout=5) as resp:
                    stats = _json.loads(resp.read())
                    log.info(
                        'Gateway stats: requests=%d samples=%d pending=%d batches=%d',
                        stats.get('total_requests', 0),
                        stats.get('total_samples', 0),
                        stats.get('pending_samples', 0),
                        stats.get('emitted_batches', 0),
                    )
            except Exception:
                pass

            time.sleep(30)

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        log.exception('Fatal error')
    finally:
        _shutdown()


if __name__ == '__main__':
    main()
