# verl RL Rollout 性能基准测试工具

模拟 verl 训练过程中的 rollout 阶段，使用 vLLM 在线服务模式独立测试不同模型和量化精度的推理性能，并自动生成对比表格。

## 文件说明

```
verl_rollout_bench/
├── README.md                    # 本文档
├── run_vllm_benchmark.sh        # 主控脚本 (拉起 server → benchmark → 采集 profiling → 汇总)
└── summarize_benchmark.py       # 结果分析脚本 (解析 JSON → 对比表格 + CSV + Markdown)
```

运行后生成的目录：

```
(当前工作目录)/
├── benchmark_results/           # benchmark JSON 结果 + 汇总文件
│   ├── qwen3-1.7b_bf16.json
│   ├── qwen3-1.7b_w8a16.json
│   ├── ...
│   ├── summary.txt              # 纯文本对比表格
│   ├── summary.csv              # CSV (可导入 Excel/pandas)
│   └── summary.md               # Markdown (可粘贴到文档/Issue)
├── benchmark_logs/              # Server 和 benchmark 运行日志
│   ├── server_qwen3-1.7b_bf16.log
│   ├── bench_qwen3-1.7b_bf16.log
│   └── ...
└── benchmark_profiles/          # Torch profiling trace 文件 (如开启)
    ├── qwen3-1.7b_bf16/
    ├── qwen3-1.7b_w8a16/
    └── ...
```

## 快速开始

### 前置条件

- Ascend NPU 环境，已安装 vLLM + vllm-ascend
- 模型权重已下载到本地（BF16 原始模型 + 量化后的模型）
- Python 3.8+，已安装 `transformers`、`aiohttp`

### Step 0：修改配置

打开 `run_vllm_benchmark.sh`，修改以下配置项以匹配你的环境：

```bash
# 模型根目录
MODEL_BASE="/data/l50044498/models"

# Ascend NPU 设备
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Profiling 输出目录
PROFILING_BASE_DIR="/home/l00861652/exps/benchmark_profiles"
```

以及 `declare_model_config()` 函数中的模型路径：

```bash
QWEN3_1_7B_PATH_BF16="${MODEL_BASE}/qwen3-1.7b"
QWEN3_1_7B_PATH_W8A16="${MODEL_BASE}/qwen3-1.7b-W8A16"
QWEN3_1_7B_PATH_W8A8="${MODEL_BASE}/qwen3-1.7b-W8A8D"
# ...
```

> **提示**：如果某个模型没有某种量化版本（如 Qwen3-30B-A3B 没有 W8A16），不定义对应的 `PATH_W8A16` 变量即可，脚本会自动跳过该组合。

### Step 1：诊断模式（首次必跑）

先用诊断模式验证环境是否正常：

```bash
cd /path/to/your/workspace
bash benchmarks/verl_rollout_bench/run_vllm_benchmark.sh --diagnostic
```

诊断模式会：
1. 检查模型路径是否存在
2. 检查 vllm CLI 是否可用
3. 只跑 **BF16 + Qwen3-1.7B**，仅 **4 个请求**
4. 验证 benchmark 结果 JSON 是否正确生成
5. 验证 profiling 数据是否成功保存
6. 输出一张迷你汇总表格

**诊断通过后再跑全量实验。**

### Step 2：运行全量实验

```bash
# 全部 8 组实验 (3 量化 x 3 模型 - 1 = 8)，含 profiling
bash benchmarks/verl_rollout_bench/run_vllm_benchmark.sh

# 不采集 profiling（纯跑性能数字，更快）
bash benchmarks/verl_rollout_bench/run_vllm_benchmark.sh --no-profile
```

### Step 3：查看结果

全量实验跑完后，脚本会自动调用 `summarize_benchmark.py` 生成汇总表格并打印到终端。

你也可以**事后单独**重新生成表格：

```bash
python3 benchmarks/verl_rollout_bench/summarize_benchmark.py ./benchmark_results
```

## 完整用法参考

### run_vllm_benchmark.sh

```
用法: run_vllm_benchmark.sh [OPTIONS]

选项:
  --diagnostic          诊断模式: 只跑 BF16+Qwen3-1.7B, 少量请求, 验证 profiling 保存
  --no-profile          不采集 torch profiling (纯跑性能数据)
  --models MODELS       逗号分隔的模型列表
  --quants QUANTS       逗号分隔的量化列表
  --input-len N         输入 prompt 长度 (默认: 128)
  --output-len N        输出生成长度 (默认: 128)
  --num-prompts N       请求数量 (默认: 256)
  --max-concurrency N   最大并发数 (默认: 128)
  --port PORT           服务端口 (默认: 8080)
  --model-base DIR      模型根目录
  -h, --help            显示帮助信息
```

**常用组合示例**：

```bash
# 只跑 Qwen3-1.7B 的 BF16 和 W8A16 对比
bash run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16

# 只跑所有模型的 BF16 基线
bash run_vllm_benchmark.sh --quants bf16

# 加大负载参数 (更接近 verl 实际 rollout 配置)
bash run_vllm_benchmark.sh --input-len 512 --output-len 256 --num-prompts 256

# 用不同的模型根目录
bash run_vllm_benchmark.sh --model-base /data/other_user/models
```

**模型 key 与 `--models` 参数的对应关系**：

| --models 参数值 | 对应模型 |
|:---|:---|
| `qwen3-1.7b` | Qwen3-1.7B |
| `pangu-7b` | openPangu-Embedded-7B-V1.1 |
| `qwen3-30b-a3b` | Qwen3-30B-A3B-Instruct |

**量化 key 与 `--quants` 参数的对应关系**：

| --quants 参数值 | 含义 |
|:---|:---|
| `bf16` | BFloat16 全精度（基线） |
| `w8a16` | 权重 INT8 + 激活 FP16 |
| `w8a8` | 权重 INT8 + 激活 INT8 |

### summarize_benchmark.py

```
用法: summarize_benchmark.py [RESULT_DIR] [OPTIONS]

位置参数:
  RESULT_DIR       benchmark 结果 JSON 文件所在目录 (默认: ./benchmark_results)

选项:
  --csv            输出 CSV 文件 (默认开启)
  --markdown       同时输出 Markdown 格式表格
  --no-print       不输出到终端, 仅写入文件
```

**示例**：

```bash
# 基本用法
python3 summarize_benchmark.py ./benchmark_results

# 生成 Markdown (方便粘贴到 GitHub Issue / 飞书文档)
python3 summarize_benchmark.py ./benchmark_results --markdown

# 静默模式，只写文件
python3 summarize_benchmark.py ./benchmark_results --no-print
```

## 输出表格说明

### 表 1：性能对比表

所有实验的原始指标横向对比。

| 指标 | 含义 | verl rollout 中的作用 |
|:---|:---|:---|
| **Output Tput (tok/s)** | 每秒生成的 output token 数 | **核心指标**。直接决定 rollout 生成速度 |
| Total Tput (tok/s) | 每秒处理的总 token 数 (input + output) | 综合吞吐能力 |
| Req Tput (req/s) | 每秒完成的请求数 | 请求级吞吐 |
| TTFT Mean (ms) | 首 token 平均延迟 | 反映 prefill 阶段效率 |
| TPOT Mean (ms) | 每个 output token 的平均延迟 | 反映 decode 阶段效率 |
| **E2EL P99 (ms)** | 第 99 百分位端到端延迟 | **核心指标**。verl 要等全部请求完成，尾部延迟决定 rollout step 耗时 |

> **为什么 E2EL P99 比 E2EL Mean 更重要？**
>
> verl rollout 的工作模式是：256 个 prompt 同时发给 vLLM，等**全部**完成后才进入 actor update。所以实际耗时取决于**最慢那条请求**，即尾部延迟 (P99/P100)，而非平均值。

### 表 2：量化加速比

以 BF16 为基线，展示量化后每个指标的提升倍数。

- `1.20x ^` 表示量化后比 BF16 快 20%（`^` = 更优）
- `0.85x v` 表示量化后比 BF16 慢 15%（`v` = 更差）

## Profiling 数据分析

### 方式 1：Ascend 工具（推荐用于 NPU）

```python
from torch_npu.profiler.profiler import analyse
analyse(profiler_path="./benchmark_profiles/qwen3-1.7b_bf16")
```

### 方式 2：Perfetto UI（通用）

1. 打开 https://ui.perfetto.dev/
2. 上传 `benchmark_profiles/<model>_<quant>/` 下的 trace 文件
3. 可视化分析各 kernel 耗时分布

### Profiling 注意事项

- Profiling 会**显著拖慢推理速度**，采集的延迟和吞吐数字**不能作为性能参考**
- Profiling 仅用于分析 kernel 级别的耗时分布（比如 GEMM kernel 是否被替换为量化版本）
- 性能对比请看 `summary.txt` 中的数字（非 profiling 模式下采集的）
- 如果只需要性能数字，使用 `--no-profile` 跳过 profiling 可大幅加速实验

## 实验设计背景

本工具模拟 verl GRPO 训练中的 rollout 阶段：

```
verl rollout 工作流:
┌─────────────────────────────────────────────────────────┐
│  256 prompts (max_len=512)                              │
│       │                                                 │
│       ▼                                                 │
│  vLLM Server (n=8 completions/prompt)                   │
│       │                                                 │
│       ▼                                                 │
│  2048 sequences (max_output_len=256)                    │
│       │                                                 │
│       ▼                                                 │
│  等待全部完成 → 进入 actor update                        │
└─────────────────────────────────────────────────────────┘
```

对应到 benchmark 参数：
- `--num-prompts 256` + `--request-rate inf` = 一个 batch 同时发出
- `--max-concurrency 128` = 控制服务端并发
- `--input-len 128 --output-len 128` = 默认测试长度（可按实际调整）
- `--ignore-eos` = 强制生成到 output-len，保证长度一致性

## 故障排查

| 问题 | 排查方法 |
|:---|:---|
| Server 启动失败 | 查看 `benchmark_logs/server_<model>_<quant>.log` |
| Benchmark 请求全部失败 | 查看 `benchmark_logs/bench_<model>_<quant>.log`，检查端口是否正确 |
| Profiling 目录为空 | Server 可能退出太快未完成 flush，增大 `--num-prompts` 或 `--output-len` |
| `set -u` 报错未定义变量 | 检查 `declare_model_config()` 中的模型路径变量名是否拼写正确 |
| 汇总表格显示 N/A | 对应实验可能失败了，检查日志；或 JSON 文件名格式不是 `<model>_<quant>.json` |
