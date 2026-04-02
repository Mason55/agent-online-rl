#!/usr/bin/env bash
#
# restart_online_rl.sh — 一键清理环境 & 重新拉起 JiuwenClaw 在线 RL 闭环
#
# 使用方式:
#   bash restart_online_rl.sh                     # 默认配置
#   bash restart_online_rl.sh --train-gpu 2,3,4,5 # 自定义训练 GPU
#   bash restart_online_rl.sh --skip-judge         # 跳过 Judge vLLM（复用推理）
#   bash restart_online_rl.sh --dry-run            # 只清理不拉起
#
# 会按以下顺序执行:
#   Phase 1: 杀掉所有相关进程 (vLLM / Gateway / JiuwenClaw / torchrun)
#   Phase 2: 释放 GPU 显存 (等待 CUDA context 回收)
#   Phase 3: 确认端口空闲
#   Phase 4: 拉起 run_online_rl.py
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# agent-online-rl repo root: examples/jiuwenrl_online/../../
AGENT_ONLINE_RL="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AGENT_ONLINE_RL/.." && pwd)"

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

# ---- 参数解析 ----
DRY_RUN=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        *)          EXTRA_ARGS+=("$arg") ;;
    esac
done

# ---- 需要关注的端口 ----
PORTS=(18000 18001 18080 18092 19000 5173)

# ---- Phase 1: 杀进程 ----
step "Phase 1/4: 杀掉所有相关进程"

kill_by_pattern() {
    local pattern="$1"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        info "  杀掉匹配 '$pattern' 的进程: $pids"
        echo "$pids" | xargs kill -15 2>/dev/null || true
        sleep 1
        # 如果还活着就强杀
        local still_alive
        still_alive=$(echo "$pids" | xargs -I{} sh -c 'kill -0 {} 2>/dev/null && echo {}' || true)
        if [ -n "$still_alive" ]; then
            warn "  强杀残留进程: $still_alive"
            echo "$still_alive" | xargs kill -9 2>/dev/null || true
        fi
    fi
}

kill_by_pattern "vllm.entrypoints"
kill_by_pattern "uvicorn.*gateway"
kill_by_pattern "jiuwenclaw.app"
kill_by_pattern "jiuwenclaw.app_web"
kill_by_pattern "run_online_rl"
kill_by_pattern "online_training_scheduler"
kill_by_pattern "torchrun.*verl"
kill_by_pattern "verl.trainer.sft_trainer"

# 按端口杀 (兜底)
for port in "${PORTS[@]}"; do
    pid=$(lsof -ti :"$port" 2>/dev/null | head -1 || true)
    if [ -n "$pid" ]; then
        warn "  端口 $port 仍被 PID=$pid 占用，强杀"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# 杀掉所有占用 GPU 的残留进程
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)
if [ -n "$gpu_pids" ]; then
    info "  清理 GPU 上的残留进程: $(echo $gpu_pids | tr '\n' ' ')"
    echo "$gpu_pids" | xargs kill -9 2>/dev/null || true
fi

info "  进程清理完成"

# ---- Phase 2: 等待 GPU 释放 ----
step "Phase 2/4: 等待 GPU 显存释放"

wait_gpu_free() {
    local max_wait=15
    local waited=0
    while [ $waited -lt $max_wait ]; do
        local gpu_pids_alive=0
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
            if kill -0 "$pid" 2>/dev/null; then
                gpu_pids_alive=$((gpu_pids_alive + 1))
            fi
        done
        if [ "$gpu_pids_alive" -eq 0 ]; then
            info "  所有 GPU 进程已终止"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        info "  等待 GPU 进程终止... (${waited}s/${max_wait}s, 活跃进程 ${gpu_pids_alive} 个)"
    done
    warn "  超时: 仍有活跃 GPU 进程，尝试强杀"
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null || true
    done
    return 0
}

wait_gpu_free

# 检查是否有僵尸 GPU context（进程已死但显存未释放）
zombie_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | wc -l)
if [ "$zombie_count" -gt 0 ]; then
    warn "  检测到 ${zombie_count} 个僵尸 GPU context（进程已死，显存未回收）"
    warn "  这不影响新服务启动（会使用空闲 GPU），但显存会被占用"
    warn "  如需彻底清理，可执行: sudo nvidia-smi --gpu-reset -i <gpu_id>"
fi

# 显示 GPU 状态
echo ""
info "  当前 GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | while IFS= read -r line; do
    echo "    GPU $line"
done
echo ""

# ---- Phase 3: 确认端口空闲 ----
step "Phase 3/4: 确认端口空闲"

all_free=true
for port in "${PORTS[@]}"; do
    if lsof -ti :"$port" >/dev/null 2>&1; then
        error "  端口 $port 仍被占用！"
        lsof -i :"$port" 2>/dev/null | head -3
        all_free=false
    else
        info "  端口 $port ✓"
    fi
done

if [ "$all_free" = false ]; then
    error "部分端口未释放，请手动处理后重试"
    exit 1
fi

# ---- Phase 4: 拉起 ----
if [ "$DRY_RUN" = true ]; then
    info "DRY-RUN 模式: 清理完成，跳过拉起"
    exit 0
fi

step "Phase 4/4: 拉起 JiuwenClaw 在线 RL 闭环"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/online_rl_${TIMESTAMP}.log"

cd "$SCRIPT_DIR"

info "  日志文件: ${LOG_FILE}"
info "  启动命令: python run_online_rl.py ${EXTRA_ARGS[*]:-}"
echo ""
echo "============================================================"
echo "  拉起中 (Ctrl+C 停止所有服务)"
echo "  实时日志: tail -f ${LOG_FILE}"
echo "============================================================"
echo ""

# 前台运行 Python，tee 到日志文件，确保 Ctrl+C 能正确传递
python3 "${SCRIPT_DIR}/run_online_rl.py" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -ne 0 ]; then
    error "run_online_rl.py 退出码: ${EXIT_CODE}"
    error "查看日志: ${LOG_FILE}"
fi
exit "$EXIT_CODE"
