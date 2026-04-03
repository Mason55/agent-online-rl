#!/usr/bin/env bash
#
# check_env.sh — 在线 RL 环境检测脚本
#
# 在新环境部署前运行，检测所有依赖是否就绪：
#   - Python 版本 & 关键包（openjiuwen / jiuwenclaw / vllm / verl / torch …）
#   - NVIDIA 驱动 & CUDA & GPU 显存
#   - 模型文件是否存在
#   - 端口是否被占用
#   - vLLM 能否正常拉起（dry-run 模式）
#   - Web 前端 dist 是否已编译
#
# 用法：
#   bash check_env.sh                             # 完整检测（含 vLLM dry-run）
#   bash check_env.sh --quick                     # 快速检测（跳过 vLLM dry-run）
#   bash check_env.sh --model-path /path/to/model # 自定义推理模型路径
#   bash check_env.sh --judge-model-path /path    # 自定义 Judge 模型路径
#   bash check_env.sh --vllm-gpu 0,1              # 指定推理 GPU
#   bash check_env.sh --judge-gpu 4,5,6,7         # 指定 Judge GPU

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ONLINE_RL="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_ONLINE_RL/.." && pwd)"

# ---- 默认参数 ----
MODEL_PATH="/data1/models/Qwen/Qwen3-4B-Instruct-2507"
JUDGE_MODEL_PATH="/data1/models/Qwen/Qwen3-32B"
VLLM_GPU="0,1"
JUDGE_GPU="4,5,6,7"
TRAIN_GPU="2,3"
QUICK=false
VLLM_DRY_RUN_TIMEOUT=60

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---- 计数器 ----
PASS=0
FAIL=0
WARN=0

pass()  { PASS=$((PASS + 1)); echo -e "  ${GREEN}✓${NC} $*"; }
fail()  { FAIL=$((FAIL + 1)); echo -e "  ${RED}✗${NC} $*"; }
warn()  { WARN=$((WARN + 1)); echo -e "  ${YELLOW}!${NC} $*"; }
section() { echo -e "\n${CYAN}${BOLD}[$1]${NC} $2"; }

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)            QUICK=true; shift ;;
        --model-path)       MODEL_PATH="$2"; shift 2 ;;
        --judge-model-path) JUDGE_MODEL_PATH="$2"; shift 2 ;;
        --vllm-gpu)         VLLM_GPU="$2"; shift 2 ;;
        --judge-gpu)        JUDGE_GPU="$2"; shift 2 ;;
        --train-gpu)        TRAIN_GPU="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash check_env.sh [--quick] [--model-path PATH] [--judge-model-path PATH]"
            echo "                         [--vllm-gpu IDS] [--judge-gpu IDS] [--train-gpu IDS]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  JiuwenClaw 在线 RL — 环境检测${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  主机: $(hostname)"
echo -e "  模式: $([ "$QUICK" = true ] && echo '快速' || echo '完整')"

# =====================================================================
# 1. Python
# =====================================================================
section "1/7" "Python 环境"

if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 11 && "$PY_MINOR" -le 13 ]]; then
        pass "Python $PY_VER (要求 3.11-3.13)"
    else
        fail "Python $PY_VER — 需要 3.11-3.13"
    fi
else
    fail "python3 未找到"
fi

# ---- 关键 Python 包 ----
section "2/7" "Python 依赖包"

check_package() {
    local pkg="$1"
    local display="${2:-$1}"
    local required="${3:-true}"
    local ver
    ver=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$ver" ]; then
        pass "$display==$ver"
        return 0
    else
        if [ "$required" = "true" ]; then
            fail "$display 未安装"
        else
            warn "$display 未安装（可选）"
        fi
        return 1
    fi
}

check_import() {
    local module="$1"
    local display="${2:-$1}"
    local required="${3:-true}"
    if python3 -c "import $module" 2>/dev/null; then
        pass "$display 可导入"
        return 0
    else
        if [ "$required" = "true" ]; then
            fail "$display 无法导入"
        else
            warn "$display 无法导入（可选）"
        fi
        return 1
    fi
}

# 核心依赖
check_import "openjiuwen" "openjiuwen (agent-core)"
check_import "jiuwenclaw" "jiuwenclaw"

# agent-online-rl 自身依赖
check_package "fastapi"       "fastapi"
check_package "uvicorn"       "uvicorn"
check_package "httpx"         "httpx"
check_package "pydantic"      "pydantic"
check_package "pandas"        "pandas"
check_package "pyarrow"       "pyarrow"

# 训练 & 推理
check_package "torch"         "torch"
check_package "transformers"  "transformers"
check_package "peft"          "peft"
check_package "vllm"          "vllm"
check_package "verl"          "verl"

# CUDA via torch
TORCH_CUDA=$(python3 -c "import torch; print(f'CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}')" 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$TORCH_CUDA" ]; then
    pass "torch: $TORCH_CUDA"
else
    fail "torch CUDA 不可用（torch.cuda 未就绪）"
fi

TORCH_GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$TORCH_GPU_COUNT" -gt 0 ] 2>/dev/null; then
    pass "torch.cuda.device_count() = $TORCH_GPU_COUNT"
else
    fail "torch 未检测到 GPU"
fi

# =====================================================================
# 3. NVIDIA 驱动 & GPU
# =====================================================================
section "3/7" "GPU & NVIDIA 驱动"

if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    pass "NVIDIA 驱动: $DRIVER_VER"

    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    pass "GPU 数量: $GPU_COUNT"

    # 逐卡显示
    echo ""
    echo -e "  ${BOLD}GPU 状态:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader | while IFS= read -r line; do
        echo "    $line"
    done
    echo ""

    # 检查目标 GPU 是否空闲
    check_gpu_free() {
        local label="$1"
        local gpu_ids="$2"
        IFS=',' read -ra IDS <<< "$gpu_ids"
        local all_free=true
        for gid in "${IDS[@]}"; do
            local mem_used
            mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
            if [ -z "$mem_used" ]; then
                fail "$label: GPU $gid 不存在或无法查询"
                all_free=false
            elif [ "$mem_used" -gt 100 ] 2>/dev/null; then
                warn "$label: GPU $gid 显存已占用 ${mem_used} MiB"
                all_free=false
            fi
        done
        if [ "$all_free" = true ]; then
            pass "$label: GPU [$gpu_ids] 空闲"
        fi
    }

    check_gpu_free "推理 vLLM"  "$VLLM_GPU"
    check_gpu_free "Judge vLLM" "$JUDGE_GPU"
    check_gpu_free "训练"       "$TRAIN_GPU"
else
    fail "nvidia-smi 未找到 — 未安装 NVIDIA 驱动？"
fi

# =====================================================================
# 4. 模型文件
# =====================================================================
section "4/7" "模型文件"

check_model() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        # 检查关键文件
        local has_config=false has_weights=false
        [ -f "$path/config.json" ] && has_config=true
        if ls "$path"/*.safetensors &>/dev/null || ls "$path"/*.bin &>/dev/null; then
            has_weights=true
        fi
        if [ "$has_config" = true ] && [ "$has_weights" = true ]; then
            local size
            size=$(du -sh "$path" 2>/dev/null | cut -f1)
            pass "$label: $path ($size)"
        elif [ "$has_config" = true ]; then
            warn "$label: $path 有 config.json 但未找到权重文件 (.safetensors / .bin)"
        else
            fail "$label: $path 目录存在但缺少 config.json"
        fi
    else
        fail "$label: $path 不存在"
    fi
}

check_model "推理模型" "$MODEL_PATH"
check_model "Judge 模型" "$JUDGE_MODEL_PATH"

# =====================================================================
# 5. 端口占用
# =====================================================================
section "5/7" "端口占用检测"

PORTS=(18000 18001 18080 18092 19000 5173)
PORT_LABELS=("vLLM推理" "vLLM-Judge" "Gateway" "JiuwenClaw-AgentServer" "JiuwenClaw-WS" "Web前端")

for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    label="${PORT_LABELS[$i]}"
    if lsof -ti :"$port" >/dev/null 2>&1; then
        pid=$(lsof -ti :"$port" | head -1)
        proc_name=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
        warn "端口 $port ($label) 被 PID=$pid ($proc_name) 占用"
    else
        pass "端口 $port ($label) 空闲"
    fi
done

# =====================================================================
# 6. 仓库 & 文件结构
# =====================================================================
section "6/7" "仓库 & 文件结构"

# agent-online-rl
if [ -d "$AGENT_ONLINE_RL/gateway" ] && [ -d "$AGENT_ONLINE_RL/scheduler" ] && [ -d "$AGENT_ONLINE_RL/trainer" ]; then
    pass "agent-online-rl 目录结构完整"
else
    fail "agent-online-rl 目录结构不完整（缺少 gateway/scheduler/trainer）"
fi

# jiuwenclaw
JIUWENCLAW_REPO="$REPO_ROOT/jiuwenclaw"
if [ -d "$JIUWENCLAW_REPO" ]; then
    pass "jiuwenclaw 仓库: $JIUWENCLAW_REPO"
else
    fail "jiuwenclaw 仓库不存在: $JIUWENCLAW_REPO"
fi

# agent-core
AGENT_CORE_REPO="$REPO_ROOT/agent-core"
if [ -d "$AGENT_CORE_REPO" ]; then
    pass "agent-core 仓库: $AGENT_CORE_REPO"
else
    fail "agent-core 仓库不存在: $AGENT_CORE_REPO"
fi

# Web dist
WEB_DIST_1="$JIUWENCLAW_REPO/jiuwenclaw/web/dist"
WEB_DIST_2="$HOME/.jiuwenclaw/web/dist"
if [ -d "$WEB_DIST_1" ]; then
    pass "Web 前端已编译: $WEB_DIST_1"
elif [ -d "$WEB_DIST_2" ]; then
    pass "Web 前端已编译: $WEB_DIST_2"
else
    warn "Web 前端未编译（可选，编译命令: cd jiuwenclaw/jiuwenclaw/web && npm install && npm run build）"
fi

# verl config
VERL_CONFIG="$AGENT_ONLINE_RL/config/ppo_lora_trainer.yaml"
if [ -f "$VERL_CONFIG" ]; then
    pass "verl 训练配置: $VERL_CONFIG"
else
    fail "verl 训练配置不存在: $VERL_CONFIG"
fi

# =====================================================================
# 7. vLLM Dry-Run（可选）
# =====================================================================
if [ "$QUICK" = false ]; then
    section "7/7" "vLLM Dry-Run（启动测试）"

    vllm_dry_run() {
        local label="$1"
        local model="$2"
        local gpu_ids="$3"
        local tp="$4"
        local port="$5"
        local max_model_len="${6:-4096}"

        if lsof -ti :"$port" >/dev/null 2>&1; then
            warn "$label: 端口 $port 已占用，跳过 dry-run"
            return
        fi

        if [ ! -d "$model" ]; then
            fail "$label: 模型不存在 ($model)，跳过 dry-run"
            return
        fi

        echo -e "  ${CYAN}→${NC} 尝试拉起 $label (最多 ${VLLM_DRY_RUN_TIMEOUT}s) ..."

        local env_str="CUDA_VISIBLE_DEVICES=$gpu_ids"
        local log_file="/tmp/check_env_vllm_${port}.log"

        env CUDA_VISIBLE_DEVICES="$gpu_ids" python3 -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --port "$port" \
            --host 127.0.0.1 \
            --tensor-parallel-size "$tp" \
            --max-model-len "$max_model_len" \
            --gpu-memory-utilization 0.5 \
            > "$log_file" 2>&1 &
        local vllm_pid=$!

        local waited=0
        local started=false
        while [ $waited -lt $VLLM_DRY_RUN_TIMEOUT ]; do
            if ! kill -0 "$vllm_pid" 2>/dev/null; then
                local exit_code
                wait "$vllm_pid" 2>/dev/null
                exit_code=$?
                fail "$label: 进程提前退出 (exit=$exit_code)"
                echo -e "    日志尾部:"
                tail -5 "$log_file" 2>/dev/null | while IFS= read -r line; do
                    echo "      $line"
                done
                return
            fi

            if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
                started=true
                break
            fi
            sleep 3
            waited=$((waited + 3))
        done

        # 清理
        kill "$vllm_pid" 2>/dev/null
        wait "$vllm_pid" 2>/dev/null || true

        if [ "$started" = true ]; then
            pass "$label: 启动成功 (${waited}s)，已自动关闭"
        else
            fail "$label: ${VLLM_DRY_RUN_TIMEOUT}s 内未就绪"
            echo -e "    日志尾部:"
            tail -10 "$log_file" 2>/dev/null | while IFS= read -r line; do
                echo "      $line"
            done
        fi
        rm -f "$log_file"
    }

    IFS=',' read -ra VLLM_IDS <<< "$VLLM_GPU"
    VLLM_TP=${#VLLM_IDS[@]}
    vllm_dry_run "推理 vLLM (Qwen3-4B)" "$MODEL_PATH" "$VLLM_GPU" "$VLLM_TP" 18000 4096

    IFS=',' read -ra JUDGE_IDS <<< "$JUDGE_GPU"
    JUDGE_TP=${#JUDGE_IDS[@]}
    vllm_dry_run "Judge vLLM (Qwen3-32B)" "$JUDGE_MODEL_PATH" "$JUDGE_GPU" "$JUDGE_TP" 18001 4096
else
    section "7/7" "vLLM Dry-Run（已跳过 —— 使用 --quick）"
    warn "跳过 vLLM 启动测试（去掉 --quick 启用）"
fi

# =====================================================================
# 汇总
# =====================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  检测结果汇总${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}通过: $PASS${NC}    ${YELLOW}警告: $WARN${NC}    ${RED}失败: $FAIL${NC}"
echo ""

if [ "$FAIL" -eq 0 ] && [ "$WARN" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}所有检测项通过! 可以启动在线 RL 闭环。${NC}"
    echo -e "  ${CYAN}cd agent-online-rl/examples/jiuwenrl_online && python run_online_rl.py${NC}"
elif [ "$FAIL" -eq 0 ]; then
    echo -e "  ${YELLOW}${BOLD}有 $WARN 项警告，但无阻塞性问题。建议处理后启动。${NC}"
    echo -e "  ${CYAN}cd agent-online-rl/examples/jiuwenrl_online && python run_online_rl.py${NC}"
else
    echo -e "  ${RED}${BOLD}有 $FAIL 项失败，请先修复后再启动。${NC}"
fi
echo ""

exit "$FAIL"
