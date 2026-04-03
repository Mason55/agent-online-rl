# JiuwenClaw 在线强化学习闭环

基于真实用户交互的在线 RL 闭环系统。用户与 JiuwenClaw Agent 正常对话，Gateway **逐轮**采集轨迹（含 `token_ids` + `logprobs`），并使用**延迟奖励**机制——当下一轮用户消息到来时，将上一轮的 `user(N) + assistant(N) + user(N+1)` 三元组发送给独立的 Qwen3-32B Judge 模型进行多维度评分。样本累积达阈值后自动触发 SFT LoRA 训练，训练完成后热加载到 vLLM 立即生效。

## 架构总览

```
                          用户 (浏览器)
                               │
                               │ http://HOST:5173 (Web 前端)
                               ▼
                   ┌──────────────────────┐
                   │  JiuwenClaw 全栈服务  │
                   │  app  (ws://HOST:19000)│
                   │  web  (http://HOST:5173)│
                   └──────────┬───────────┘
                              │ POST /v1/chat/completions
                              ▼
                   ┌──────────────────────┐
                   │   Gateway :18080     │─────────────────┐
                   │   逐轮记录轨迹       │                  │ 延迟触发
                   │   token_ids+logprobs │                  │ (下一轮到来时)
                   │   注入 LoRA (若有)   │                  ▼
                   │   真流式透传         │       ┌──────────────────┐
                   └──────┬───────────────┘       │ vLLM Judge :18001│
                          │                       │ Qwen3-32B (TP=4) │
                          ▼                       │ GPU 4,5,6,7      │
                   ┌──────────────┐               └────────┬─────────┘
                   │ vLLM :18000  │                        │ 四维度评分
                   │ Qwen3-4B    │                        ▼
                   │ TP=2, LoRA  │             ┌────────────────────┐
                   │ GPU 0,1     │             │  JSONL + 内存队列   │
                   └──────┬──────┘             │  records/*.jsonl   │
                          ↑                    └────────┬───────────┘
                          │  热加载 LoRA               │ poll (每 N 秒)
                          │                   ┌────────▼───────────┐
                          └───────────────────│ OnlineTraining     │
                                              │ Scheduler          │
                                              │ → verl SFT LoRA   │
                                              │   GPU 2,3          │
                                              └────────────────────┘
```

**数据流：** 用户对话 → Gateway 逐轮透传推理 + 记录 token_ids/logprobs → 下一轮到来时触发 Judge (32B) 延迟打分 → 写入 JSONL + 推入内存队列 → 累积达 batch_size → 发射 batch → OnlineTrainingScheduler 拉取 → SFT LoRA 训练 → 热加载 → 后续请求自动使用新 LoRA

## 核心机制

### 延迟奖励（Delayed Reward）

传统方式在 assistant 回复后立即打分，缺乏用户反馈信号。本系统使用延迟奖励：

```
Turn N:   user(N) → assistant(N)     →  暂存为 pending_judge
Turn N+1: user(N+1) 到来时            →  取出 pending，组装评分输入：
                                          user(N) + assistant(N) + user(N+1)
                                          ↓
                                        Judge 打分（user(N+1) 作为隐式反馈）
                                          ↓
                                        记录带分数的样本
```

这样 Judge 能看到用户的后续反应（如"回答得很好"或"答案错了"），给出更准确的奖励信号。

### 逐轮数据粒度

每一轮交互作为独立样本记录，包含：

| 字段 | 说明 |
|------|------|
| `prompt_ids` | 该轮 prompt 的 token ID 序列 |
| `response_ids` | 模型回复的 token ID 序列 |
| `response_logprobs` | 每个 response token 的 log probability |
| `judge.score` | 归一化奖励 ∈ [0, 1] |
| `judge.details` | 四维度原始分（task_completion, response_quality, tool_usage, coherence） |

`logprobs` 数据支持未来升级到 PPO/GRPO 训练。

### Gateway 模式

| 模式 | 输出 | 用途 |
|------|------|------|
| `judge_log` | JSONL 文件 + 日志打印 | 开发观察，不触发训练 |
| `judge_output` | JSONL 文件 + 日志打印 + **训练队列** | 生产闭环，连接 OnlineTrainingScheduler |
| `log` | JSONL 文件 + 日志打印（不调 Judge） | 纯采集模式 |

## GPU 分配（默认 8 卡，RTX 3090 24GB）

| GPU | 用途 | 备注 |
|-----|------|------|
| 0, 1 | vLLM 推理 (Qwen3-4B, TP=2) | 含 LoRA 热加载 |
| 2, 3 | SFT LoRA 训练 | 空闲待 OnlineTrainingScheduler 调度 |
| 4, 5, 6, 7 | vLLM Judge (Qwen3-32B, TP=4) | 仅做打分，不需要 LoRA |

> GPU 不够？可以让 Judge 和训练共享卡（不同时运行），或用外部 API 替代 Judge。

## 前置准备

### 1. 安装依赖

```bash
# 1) agent-core (openjiuwen SDK) — jiuwenclaw 的底层依赖
cd agent-core
make install

# 2) jiuwenclaw 应用（自动拉入 openjiuwen）
cd jiuwenclaw
pip install -e ".[dev]"

# 3) agent-online-rl 自身依赖
cd agent-online-rl
pip install -r requirements.txt

# 4) vLLM（推理服务）和 verl（训练框架）
pip install vllm verl
```

> **依赖关系**：`agent-online-rl` → `jiuwenclaw` → `openjiuwen (agent-core)`。三者都需要安装。

### 2. 确认模型路径

脚本默认使用以下路径，可通过参数覆盖：

```
/data1/models/Qwen/Qwen3-4B-Instruct-2507    # 推理模型
/data1/models/Qwen/Qwen3-32B                  # Judge 模型
```

### 3. 编译 Web 前端（可选但推荐）

```bash
cd jiuwenclaw/jiuwenclaw/web
npm install
npm run build
```

编译产出 `dist/` 目录后，启动脚本会自动在 `http://HOST:5173` 提供 Web UI。
不编译也可以用，但只能通过 WebSocket 对话。

### 4. 环境检测

提供了一键检测脚本，在新环境部署前先跑一遍：

```bash
cd agent-online-rl/examples/jiuwenrl_online
bash check_env.sh
```

脚本会检测：Python 依赖包、GPU 驱动 / 显存、模型路径、端口占用、vLLM 能否正常拉起等。
详见 [check_env.sh](check_env.sh)。

### 5. 确认 GPU 空闲

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 确保目标 GPU 的 memory.used 接近 0 MiB
```

## 一键启动

```bash
cd agent-online-rl/examples/jiuwenrl_online

# 默认配置（8 卡完整部署）
python run_online_rl.py
```

脚本会依次拉起 5 个服务：

1. **vLLM 推理** — Qwen3-4B, TP=2, GPU 0,1, port 18000（约 1-2 分钟）
2. **vLLM Judge** — Qwen3-32B, TP=4, GPU 4,5,6,7, port 18001（约 1-2 分钟）
3. **Gateway** — port 18080，逐轮轨迹记录 + 延迟 Judge 打分 + 真流式透传
4. **OnlineTrainingScheduler** — 后台线程，轮询 Gateway 训练队列触发训练
5. **JiuwenClaw** — app (ws://HOST:19000) + web (http://HOST:5173)

启动完成后会打印以下信息：

```
============================================================
  JiuwenClaw 在线 RL 闭环已启动 (v2: per-turn + Judge)

  Web 前端:        http://localhost:5173
  JiuwenClaw WS:   ws://localhost:19000/ws
  vLLM 推理:       http://127.0.0.1:18000
  vLLM Judge:      http://127.0.0.1:18001 (Qwen3-32B)
  Gateway 代理:    http://127.0.0.1:18080
  Gateway 模式:    judge_log
  轨迹记录:        records/ (JSONL, per-turn)
  LoRA 仓库:       .../lora_repo
  训练阈值:        20 条样本
  扫描间隔:        300 秒
  训练 GPU:        [2,3]

  打开 http://localhost:5173 开始对话，
  每轮对话自动记录 token_ids + logprobs，
  下一轮到来时触发延迟 Judge 打分，
  样本累积达阈值后自动触发 LoRA 训练。
  按 Ctrl+C 停止所有服务。
============================================================
```

### 远程访问

所有服务默认绑定 `0.0.0.0`，可以直接从远程浏览器访问：

```
http://<服务器IP>:5173      # Web 前端
ws://<服务器IP>:19000/ws    # WebSocket
```

> **注意**：确保服务器防火墙放行了 5173 和 19000 端口。如果有安全要求，也可以用 SSH 隧道：
>
> ```bash
> ssh -L 5173:127.0.0.1:5173 -L 19000:127.0.0.1:19000 user@服务器IP
> ```
>
> 然后本地浏览器打开 `http://localhost:5173`。

### 常用启动模式

```bash
# 开发调试：降低阈值 + 加快扫描 + judge_log 模式（不触发训练）
python run_online_rl.py --threshold 5 --scan-interval 60 --gateway-mode judge_log

# 生产闭环：启用训练队列
python run_online_rl.py --gateway-mode judge_output --rollout-batch-size 8

# 推理 vLLM 已在运行（跳过启动）
python run_online_rl.py --inference-url http://127.0.0.1:18000

# Judge vLLM 也已在运行（跳过两个 vLLM 的启动，秒级启动）
python run_online_rl.py \
  --inference-url http://127.0.0.1:18000 \
  --judge-url http://127.0.0.1:18001

# 自定义 GPU 分配
python run_online_rl.py \
  --vllm-gpu 0,1 \
  --judge-gpu 4,5,6,7 \
  --train-gpu 2,3

# 更换推理模型
python run_online_rl.py \
  --model-path /path/to/your/model \
  --model-name YourModelName

# 纯采集模式（不调 Judge，仅记录轨迹）
python run_online_rl.py --gateway-mode log
```

### 停止服务

按 `Ctrl+C` 会优雅关闭所有子进程。如果异常退出，手动清理：

```bash
# 查看占用端口的进程
lsof -i :18000 -i :18001 -i :18080

# 按需 kill
kill <pid>
```

## 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **推理 vLLM** | | |
| `--model-path` | `/data1/models/Qwen/Qwen3-4B-Instruct-2507` | 推理基座模型路径 |
| `--model-name` | `Qwen3-4B-Instruct-2507` | vLLM 注册的模型名 |
| `--vllm-gpu` | `0,1` | 推理 GPU（逗号分隔） |
| `--vllm-tp` | `2` | 推理 Tensor Parallel 大小 |
| `--vllm-port` | `18000` | 推理服务端口 |
| `--inference-url` | — | 设置后跳过推理 vLLM 启动 |
| **Judge vLLM** | | |
| `--judge-model-path` | `/data1/models/Qwen/Qwen3-32B` | Judge 模型路径 |
| `--judge-model-name` | `Qwen3-32B` | Judge 模型名 |
| `--judge-gpu` | `4,5,6,7` | Judge GPU（逗号分隔） |
| `--judge-tp` | `4` | Judge Tensor Parallel 大小 |
| `--judge-port` | `18001` | Judge 服务端口 |
| `--judge-url` | — | 设置后跳过 Judge vLLM 启动 |
| **Gateway** | | |
| `--gateway-port` | `18080` | Gateway 代理端口 |
| `--gateway-mode` | `judge_log` | 模式：`judge_log` / `judge_output` / `log` |
| `--rollout-batch-size` | `8` | 凑够多少样本发射一个 batch |
| **训练 & 调度** | | |
| `--threshold` | `20` | 触发训练的累积样本数 |
| `--scan-interval` | `300` | OnlineTrainingScheduler 轮询间隔（秒） |
| `--train-gpu` | `2,3` | 训练 GPU（逗号分隔） |
| `--lora-repo` | `./lora_repo` | LoRA adapter 版本化存储目录 |

## 运行时产物

| 路径 | 说明 |
|------|------|
| `logs/vllm.log` | 推理 vLLM 日志 |
| `logs/judge_vllm.log` | Judge vLLM 日志 |
| `logs/gateway.log` | Gateway 日志 |
| `records/trajectories.jsonl` | 逐轮轨迹记录（含 token_ids、logprobs、Judge 评分） |
| `records/batches.jsonl` | 已发射的 batch 汇总 |
| `lora_repo/<user>/<version>/` | 训练产出的 LoRA adapter（PEFT 格式） |

### 查看轨迹

```bash
# 查看最近 5 条轨迹摘要
python3 -c "
import json
with open('records/trajectories.jsonl') as f:
    recs = [json.loads(l) for l in f if l.strip()]
for r in recs[-5:]:
    j = r.get('judge', {})
    t = r.get('trajectory', {})
    print(f\"session={r['session_id']} turn={r['turn_num']} \"
          f\"score={j.get('score','?')} overall={j.get('overall_raw','?')} \"
          f\"prompt_ids={len(t.get('prompt_ids',[]))} response_ids={len(t.get('response_ids',[]))}\")
    print(f\"  reason: {j.get('details',{}).get('reason','')[:80]}\")
"
```

### 查看 Gateway 实时状态

```bash
curl -s http://127.0.0.1:18080/v1/gateway/stats | python3 -m json.tool
```

返回字段说明：

| 字段 | 说明 |
|------|------|
| `total_requests` | 累计处理的请求数 |
| `total_samples` | 已完成 Judge 打分的样本数 |
| `pending_samples` | 待发射到 batch 的样本数 |
| `pending_judge_samples` | 等待下一轮触发打分的暂存样本数 |
| `emitted_batches` | 已发射的 batch 数 |
| `training_queue_batches` | 训练队列中待消费的 batch 数（仅 `judge_output` 模式） |

## Judge 评分机制

Gateway 使用**延迟触发**的 LLM-as-Judge 评分：

1. Turn N 完成后，Gateway 将 `user(N) + assistant(N)` 暂存为 pending
2. Turn N+1 的用户消息到来时，取出 pending 样本，组装完整评分输入
3. 发送给 Qwen3-32B Judge，使用以下评分维度：

| 维度 | 分数范围 | 说明 |
|------|----------|------|
| `task_completion` | 0-10 | Agent 是否完成了用户意图 |
| `response_quality` | 0-10 | 回答是否准确、有帮助、简洁 |
| `tool_usage` | 0-10 | 工具调用是否必要且正确 |
| `coherence` | 0-10 | 多轮对话是否自然流畅 |
| `overall` | 0-10 | 综合评分 |

- **归一化**：`score = (overall - 5) / 5`，映射到 [-1, 1]
- **容错**：Judge 调用失败时 score 默认 0.0（中性），不阻塞流程
- **投票**：支持多次调用取中位数（`num_votes`），提高评分稳定性

Judge 评分代码位于 `agent-online-rl/gateway/judge_scorer.py`。

## 在线 RL 闭环原理

```
Turn N: user(N) → Gateway → vLLM → assistant(N) 回传
                   ↓
         记录 prompt_ids, response_ids, logprobs
         暂存为 pending_judge
                   ↓
Turn N+1: user(N+1) 到来
                   ↓
         取出 pending → 发送 Judge 打分
         user(N) + assistant(N) + user(N+1) → Qwen3-32B 四维度评分
                   ↓
         写入 JSONL + 推入 pending_samples
         凑够 batch_size → 发射 batch
                   ↓  (judge_output 模式)
         推入训练队列
                   ↓
OnlineTrainingScheduler (每 N 秒轮询)
  │  /v1/gateway/training_queue/pop 拉取 batch
  │  累积样本 >= threshold?
  ▼
verl SFT LoRA 训练 (GPU 2,3)
  │  1. 转换为 parquet (input_ids, labels, reward)
  │  2. torchrun verl SFT LoRA 训练
  │  3. FSDP checkpoint → PEFT adapter 转换
  │  4. 发布到 lora_repo/<user>/<version>/
  │  5. 通知 vLLM 热加载
  ▼
后续请求 → Gateway 检测到该用户有 LoRA → 注入请求 → vLLM 使用新 LoRA 推理
```

## Gateway API

Gateway 兼容 OpenAI API，同时提供管理端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 透传推理 + 轨迹记录（支持 stream） |
| `/healthz` | GET | 健康检查 |
| `/health` | GET | 健康检查（兼容） |
| `/v1/gateway/stats` | GET | Gateway 实时统计 |
| `/v1/gateway/training_queue/pop` | POST | 拉取训练 batch（`{"max_batches": N}`） |
| `/v1/models` | GET | 透传模型列表 |

请求头 `x-session-id` 用于标识会话，Gateway 据此维护逐轮状态。

## 组件依赖

| 组件 | 说明 |
|------|------|
| `agent-core/` | openjiuwen SDK — jiuwenclaw 的底层框架（LLM client、Tool、Workflow 等） |
| `jiuwenclaw/` | AgentServer + Web 前端（`.env` 由脚本自动配置指向 Gateway） |
| `agent-online-rl/gateway/` | Gateway 核心：逐轮记录、延迟 Judge、流式透传、LoRA 注入、Batch 发射 |
| `agent-online-rl/judge/` | Judge 服务端（可独立部署）+ 客户端 |
| `agent-online-rl/scheduler/` | OnlineTrainingScheduler（轮询 Gateway 训练队列） |
| `agent-online-rl/trainer/` | verl SFT LoRA 训练脚本 |
| `agent-online-rl/storage/` | LoRA 仓库管理 |
| `agent-online-rl/inference/` | vLLM 热加载通知 |
| `vLLM` | 推理服务（`--enable-lora`）+ Judge 服务（独立实例） |
| `verl` | SFT LoRA 训练后端 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_online_rl.py` | **在线 RL 闭环启动脚本**（一键拉起全部服务） |
| `train.py` | 离线 GRPO 训练入口（不依赖在线闭环） |
| `generate_data.py` | 离线数据生成（从题库生成 parquet） |
| `tasks.py` | 搜索问答题库 |
| `prompts.py` | Agent 系统提示词模板 |
| `reward.py` | 离线奖励函数 |
| `sample_processing.py` | 从 parquet 提取训练字段 |

## 常见问题

**Q: Judge vLLM (32B) 启动时报 CUDA OOM？**
Qwen3-32B TP=4 在 4×24GB GPU 上内存非常紧张。脚本已设置 `--max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.85` 来适配。如果仍然 OOM，可以进一步降低 `max-model-len`（Judge 场景 4096 就够了）。

**Q: 推理 vLLM 启动报 GPU 内存不足？**
`nvidia-smi` 确认目标 GPU 空闲。如果有残留进程：`lsof -i :18000` 找到 PID 并 `kill -9`。

**Q: Gateway 启动超时（tokenizer 下载失败）？**
Gateway 需要加载 tokenizer。如果网络不通，确保 `--model-path` 指向本地模型目录。脚本会自动将 `model_path` 传递给 Gateway，优先使用本地路径加载 tokenizer。

**Q: 样本采集了但 Judge 没打分？**
Judge 使用延迟奖励机制，需要等待**下一轮**用户消息才会触发打分。查看 Gateway stats 的 `pending_judge_samples` 字段确认是否有暂存样本。

**Q: 样本打分了但没触发训练？**
检查以下几点：
1. Gateway 模式是否为 `judge_output`（只有此模式会推入训练队列）
2. 已打分样本数是否达到 `rollout-batch-size`（默认 8）以发射 batch
3. OnlineTrainingScheduler 累积的样本数是否达到 `threshold`（默认 20）
4. 用 `curl http://127.0.0.1:18080/v1/gateway/stats` 查看 `training_queue_batches`

**Q: 如何只部署 Judge 不部署推理？**
不支持。但可以先独立启动 Judge，然后 `--judge-url` 跳过：
```bash
# 终端 1：独立启动 Judge
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
  --model /data1/models/Qwen/Qwen3-32B --served-model-name Qwen3-32B \
  --port 18001 --tensor-parallel-size 4 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --max-num-seqs 16

# 终端 2：启动闭环（跳过 Judge 启动，秒级就绪）
python run_online_rl.py --judge-url http://127.0.0.1:18001
```

**Q: 远程机器如何访问 Web 前端？**
所有服务已绑定 `0.0.0.0`，直接浏览器访问 `http://服务器IP:5173` 即可。如果防火墙限制，用 SSH 隧道：`ssh -L 5173:127.0.0.1:5173 -L 19000:127.0.0.1:19000 user@HOST`。

**Q: 如何查看历史轨迹的详细 Judge 评分？**
```bash
python3 -c "
import json
with open('records/trajectories.jsonl') as f:
    for line in f:
        r = json.loads(line)
        j = r.get('judge', {})
        d = j.get('details', {})
        print(f\"turn={r['turn_num']} score={j.get('score',0):.2f} \"
              f\"tc={d.get('task_completion')} rq={d.get('response_quality')} \"
              f\"tu={d.get('tool_usage')} co={d.get('coherence')}\")
        print(f\"  reason: {d.get('reason','')[:100]}\")
"
```
