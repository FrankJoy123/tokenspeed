# W4A8 Prefill · QCP8 与 DP8

一台 RTX 6000D 八卡独占，均使用 EP8。每条请求 131,072 输入，其中 126,976 是真实命中的公共前缀，4,096 为互异后缀。每条另算 128 行 replay，并输出 1 token。Pareto 的 label 是整组每步新增 token 除以 8。吞吐只计新增 token。

| Label | 每步请求 | DP8 活跃 attention lane | DP8 每活跃 lane 请求 | QCP8 每步物理行 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1 | 1 | 1 | 4,224 |
| 1K | 2 | 2 | 1 | 8,448 |
| 2K | 4 | 4 | 1 | 16,896 |
| 4K | 8 | 8 | 1 | 33,792 |
| 8K | 16 | 8 | 2 | 67,584 |
| 16K | 32 | 8 | 4 | 135,168 |

QCP8 将每条请求划分到 8 卡。DP8 的空 attention lane 在全局元数据中记零行，仍使用同一 EP8 通信组。

## 启动

先按 [W4A8 环境构建](../../../../deploy/w4a8/setup/README.md)安装运行时。在 [环境示例](../../../../deploy/w4a8/launch/env.example.sh) 中填写本机 checkpoint、venv、CUDA 和网络参数，另存到 `_local/`。使用八卡可见的环境和兼容工具链，实测使用 GCC 12.4。

```bash
source _local/w4a8-prefill.env
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export BENCH="$PWD/.agents/bench/pareto/w4a8/prefill"
export POINT="$PWD/_local/prefill-dp8-2k"
"$VENV/bin/python" "$BENCH/run_point.py" --layout dp8 --label 2048 \
  --model "$MODEL_PATH" --python "$VENV/bin/python" \
  --expert-routing natural --indexer bf16 --profile-steps 0 --out "$POINT"
"$VENV/bin/python" "$BENCH/audit_point.py" --point "$POINT"
```

按表切换布局为 `dp8` 或 `cp8`、选择 `--label`，每个布局和档位用一个新输出目录。脚本创建离线引擎和独立 JIT 缓存，采用 Elastic FP8 dispatch / BF16 combine、capacity 2048，默认保留 autotune。通过 [points.json](points.json) 查询已发布点的 `indexer`、`queue_requests_per_lane`、`max_model_len` 和 snapshot 设置，并以 `--indexer`、`--queue-requests-per-lane`、`--max-model-len`、`--retraction-snapshot-ratio` 传入。snapshot 为 null 时沿用默认值。若复建历史实测点，用该点的 `source_commit` 构建 venv，并用 `--source` 指向实际构建的干净 checkout。脚本从当前分支运行。

## 审核与结果

每档正式窗口前保留一轮完整预热队列。两轮分别 flush、真实 prime，再完整处理队列。审核有效缓存、每步请求数、活跃与零行 lane、连续端点和全部编译日志，按 [SOP](../../SKILL.md) 生成 Pareto。脚本保存启动参数、模型头部与加载制品身份，行数和计时来源可由 `samples/accepted-summary.json` 复核。

在固定 4K 后缀下，QCP8 在 1、2、4 条请求时的最快点步时更短。8 条请求时 DP8 用满八条 lane，并在测试中取得更短步时。QCP8 的 16K 未形成连续满载窗口。菱形表示打开 Native FP4 选项，具体算子由运行时按形状选择。[Pareto](pareto.svg) · [CSV](pareto.csv)
