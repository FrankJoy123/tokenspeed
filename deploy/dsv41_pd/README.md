# DeepSeek V4.1 Flash: SM120 PD deployment files

两个 TS 仓库使用同一套目录，脚本自动定位所在仓库。P 与 D 的代码版本不同，不能交换 checkout。不会连接旧实验目录、自动停止服务或自动覆盖已有虚拟环境。

```bash
cp deploy/dsv41_pd/env.example.sh deploy/dsv41_pd/env.local.sh
# 编辑 ROLE、模型路径、P/D/Gateway 地址、网卡和依赖源码位置。
bash deploy/dsv41_pd/install.sh
bash deploy/dsv41_pd/prepare_artifacts.sh
bash deploy/dsv41_pd/fetch_sources.sh
# P only: bash deploy/dsv41_pd/build_nvshmem.sh
bash deploy/dsv41_pd/install_dependencies.sh
bash deploy/dsv41_pd/check.sh
```

安装需要 Python 3.12、CUDA 13、Rust/Cargo、C++ 编译器及可访问固定版本 wheel 的 package mirror。两个 TS 仓库分别从本地源码编译 engine + scheduler，host binary 随 engine wheel 安装；不引用旧 ts-host 文件。kernel Python 实现直接来自当前 checkout。

**源码与制品：** 同目录 `artifacts/` 保存 P 的三个定制源码快照、32 条原始压测 token 数据；`artifacts.json` 固定 SHA256。它们不含模型权重、旧虚拟环境或访问凭据。`prepare_artifacts.sh` 校验后解包到本目录 `.runtime/`，无需原机器的任何第三方目录。NVSHMEM 使用 `build_nvshmem.sh` 按原构建选项重新编译；DeepEP/FlashInfer 再从解包源码安装。Gateway 从 sources.tsv 中的 Git 仓库拉取固定 commit；DeepGEMM/CUTLASS 优先从公开 Git 固定 SHA 获取，附带本地修复。可以在 `sources.local.tsv` 指定发布后的同等 Git 源替换源码快照。

迁移时需要复制**整个目录，包括被 gitignore 忽略的 artifacts/**；单独 git clone 暂不足以获取 P 的未发布依赖和测试数据。未来发布到制品库/Git LFS 后可替换分发方式。本次新环境已完成依赖编译、模型启动和真实 PD 小请求验证；长上下文精度及性能仍需单独验收。

配置文件不包含旧路径。基础 wheel 版本在 requirements.txt，定制依赖在 dependencies.tsv。P 的源码快照没有完整 Git 来源，故未伪造提交号。配置 `VENV` 为新的路径，安装器拒绝覆盖现有环境。P 的 NVSHMEM 构建需要系统 RDMA 开发库（libibverbs/mlx5 等），不会调用 sudo 自动安装。

启动：在三个独立终端/进程管理器中分别执行（gateway 可以与 D 同机，但建议独立配置和 venv）：

```bash
PD_CONFIG=/your/gateway.env bash deploy/dsv41_pd/launch_gateway.sh
PD_CONFIG=/your/prefill.env bash deploy/dsv41_pd/launch_prefill.sh
PD_CONFIG=/your/decode.env bash deploy/dsv41_pd/launch_decode.sh
```

网关绑定 18500 HTTP、18510 P ZMQ、18511 D ZMQ；两端连接 GATEWAY_HOST。P bootstrap 18598；各节点还需 GPU RDMA/Mooncake 连通。网卡名及 NCCL 通信选项必须按迁移目标拓扑填写。P/D 均使用相同 `DeepSeek-V4.1-Flash-1017` 权重与 tokenizer；模型路径允许不同。`DRY_RUN=1` 只输出命令，不加载模型。

默认 P QCP8/EP8 CUTLASS W4A8 + DeepEP normal，chunk 16384；D DP8/EP8 FlashInfer + DeepEP，chunk 8192，Graph BS1–32，DSpark，host Engram，两端显存比例 .85。CUDA_LAUNCH_BLOCKING 未设置。D 不启用 FP4 Indexer。模型、端口、后端、环境都应以 launch 文件为准。额外 CLI 参数会原样传给 TS。

```bash
bash deploy/dsv41_pd/smoke.sh
bash deploy/dsv41_pd/benchmark.sh
```

先以真实接受率跑 smoke/精度检查。性能模拟必须显式配置 `SIMULATED_ACCEPT_LEN=4` 并重启 D，表示 6 个验证位置包含 anchor 共接受 4 个，draft 接受率约 3/5；不能用于精度评估。

压测引用用户指定 `SESSIONS_FILE`，不复制旧 HDFS 路径：32 条高接受率 LongBench 来源 token 序列，JSON 数组，每项包含 `id` 和 `input_ids`，至少 204800 token，与 1017 tokenizer 匹配。模板中的 SHA 是现有 10 轮数据的 SHA。`prepare_artifacts.sh` 会从同目录制品解包原数据并校验 SHA；不重新生成文本。

C1/2/4/8/16/32，每并发两次完整编译预热，每次先清 P+D KV，再测正式轮。100K prime → 110K–200K 共 10 轮，每次输出 4096；单会话完成立即发下一轮，不等其他会话，不拼接之前输出。prime cached_tokens 必须为 0，后续必须等于上一轮输入长度。缓存清理失败即停止，不重启服务。正式测试应独占 gateway，空闲时才能清缓存。

结果放在 `.runtime/<role>/runs/`：逐请求流式事件与 meta、每轮 mean/p50/p95、workers 监控和 SUMMARY。TPOT=(末输出−首输出)/(输出 token−1)，包含首个 SSE 多 token 的既有约定；SUMMARY 的吞吐是全部正式输出 token / (最后输出−最早输出)，不是 C/TPOT 推算值。prime 不计正式统计，但与异步首轮重叠可能影响首轮 TTFT。

本套件未带旧实验的 monkeypatch batch observer，因此不会把 gateway running 冒充真实 decode BS，也不虚构 KV/接受率数据；这些仍须保留 D 日志/服务端计数。正式性能验收还要检查 P/D 日志的 JIT 编译窗口，warmup 完成不等于证明没有新编译。

首轮 TTFT 分析及 prime 同步对照见 [部署说明](DEPLOYMENT.md#首轮-ttft-与-prime-同步对照)。当前已提交入口仍为原独立推进协议；屏障对照尚未合入，正式结果待完成后补充。

安装、首次启动、gateway 健康恢复、JIT 预热、跨机器迁移和验收边界见 [安装问题记录](INSTALL_TROUBLESHOOTING.md)。失败续装使用 install.sh --resume，仅限本次独立部署环境。
