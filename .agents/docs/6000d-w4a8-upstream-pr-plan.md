# RFC 草案 · RTX 6000D 上的 DSV4.1 Flash W4A8

**建议先把 SM120 能力和 Attention 拆成可独立验证的 PR。** Prefill 首选单请求 Paged 路径和 Native FP4 Indexer，Decode 首选复用 FlashInfer 的双缓存 MLA。两者可以并行评审。Prefill 的 Compact BF16x2 单独提交，重点覆盖多请求仍需 gather 的 8K 单步。Prefill MegaMoE 有较大的收益空间，适合先讨论后端接口。EPLB 放在独立系列，从路由正确性开始。

基于 2026-11-20 检出的 TokenSpeed main `c20ea8c9888075be7287fed95f149b888d5d07af`、本分支 `24f8b04cf1ca53997a93c696341d927c79ac4445` 和 Njuapp `c52f86bfa5a8fc2ee79424a3f8bd933cfe4b0752`。下文的性能数字来自 6000D 上的两个实验分支。移植到该 main 后需要重跑基线，确认取代的是哪个当前路径。[main 快照](https://github.com/lightseekorg/tokenspeed/tree/c20ea8c9888075be7287fed95f149b888d5d07af) · [同口径 Pareto](../bench/pareto/w4a8/optimized/README.md) · [Njuapp 原报告](https://github.com/Njuapp/tokenspeed/blob/c52f86bfa5a8fc2ee79424a3f8bd933cfe4b0752/.agents/6000d-dsv41-opt/docs/OPTIMIZATION_RESULTS.md)

## 先对齐 main 的能力

- main 已合入 DSV4.1 普通 attention DP/EP 的布局修正和 collective CI，也有新版精确 Decode Indexer。沿用当前模型、请求布局和打分/选择接口。6000D 的 Indexer 应提交 SM120 实现与能力注册，并复用 main 已有的流程。[普通 DP/EP #3492](https://github.com/lightseekorg/tokenspeed/pull/3492) · [Indexer #3504](https://github.com/lightseekorg/tokenspeed/pull/3504)
- 上述 main 快照的 V4.1 FlashMLA 能力上限为 10.3，分支的 SM120 FlashInfer 适配和 SM120 Cake MoE 接口属于待移植范围。main 锁定 FlashInfer 0.7.5，分支验证用 0.7.4。先对 main 的依赖做导入、JIT 和 Graph 验证。建议通过现有 kernel registry 注册能力，把 FlashInfer 调用封装在一个适配器内。若所需的计划或启动能力只由 FlashInfer 私有模块提供，优先和库维护者对齐公开接口。[main FlashMLA 注册](https://github.com/lightseekorg/tokenspeed/blob/c20ea8c9888075be7287fed95f149b888d5d07af/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/flash_mla.py) · [CUDA 依赖](https://github.com/lightseekorg/tokenspeed/blob/c20ea8c9888075be7287fed95f149b888d5d07af/tokenspeed-kernel/python/requirements/cuda.txt)
- main 的通用 EPLB 已支持从 `logical_count` 生成放置，也能读取 `physical_to_logical_map`。DSV4.1 预计算路由的 EP 入口仍拒绝 EPLB 和非平凡放置。可以复用现有放置 API，重点解决 DSV4.1 logical Top-K 到物理槽的一次性映射、计数以及迁移。[main MoE 入口](https://github.com/lightseekorg/tokenspeed/blob/c20ea8c9888075be7287fed95f149b888d5d07af/python/tokenspeed/runtime/models/deepseek_v41_moe.py) · [main 参数](https://github.com/lightseekorg/tokenspeed/blob/c20ea8c9888075be7287fed95f149b888d5d07af/python/tokenspeed/runtime/utils/server_args.py)

## 建议提交顺序

| 顺序 | 拟提交标题 | 证据和范围 |
| --- | --- | --- |
| 0 | `feat(dsv41): register SM120 W4A8 inference backends` | 从当前 main 做最小可运行基线。按接口拆出 SM120 注意力与算子能力、W4A8 GEMM 和 EP 传输；只带跑通对应项所需的支持和测试。确认两卡 collective、八卡 DP8 和双机 EP16 后分别开始 A/B。若等价底座先合入，直接基于它投后续 PR。 |
| 1A | `perf(dsv41): read SM120 prefill history from paged KV` | 本分支 DP8 4K 的发布参考 1417.562 ms，Paged 候选 822.498 ms，约少 42%。历史对照包含运行版本与配置差异，同输入单核也支持直接读页。优先覆盖单请求、连续行、无 query 分片和无 DCP partial 的路径。 |
| 1B | `perf(dsv41): enable SM120 native FP4 prefill indexing` | 分支 4K 固定形状 Indexer 对照 1957.718 → 1417.562 ms，约少 27.6%。在同机低负载也有明显差距。复用 main 的长度、无效页和候选选择语义。单独测索引分数和 Top-K，对原 BF16 打分路径保留回退，最终与 Paged 交叉测量。 |
| 1C | `perf(dsv41): use FlashInfer mixed-cache MLA for SM120 decode` | B8 在发布参考 47.179 ms 到集成候选 43.341 ms，整版本少 8.1%。在同一 Cake B8 基线继续扩到 B4/B16/B32，分别少 4.34%、5.20%、4.65%。Njuapp 使用同族核，支持更广的 head 和 SWA 路径，可借鉴它的 split planner。 |
| 2 | `perf(dsv41): pipeline BF16x2 compact attention for SM120 prefill` | 取 Njuapp 的两项 BF16 PV 流水思路。本分支 8K 单步同配置开关 3021.006 → 1798.450 ms，少 40.47%；4K 少 2.39%，8K 两步组少 1.63%。PR 主张限定在 gather 后的多请求慢路径。 |
| 3 | `RFC: SM120 FlashInfer MegaMoE for DSV4.1 W4A8 prefill` | 同输入 L8、4224 行代理的 Elastic 34.125 ms → 同分块 Cake 23.569 ms。整模旧 Paged 参考 822.498 ms、加入 Cake 的版本 705.649 ms，同时改了显存预算。先讨论标准后端接口，首轮只做 Prefill routed MoE，再做增量消融。 |
| 4 | `feat(dsv41): support placement-aware precomputed EP routing` | 在现有通用 EPLB 上补模型支持。Prefill 同机同分块 700.380 → 584.335 ms，支持继续做。EPLB、固定校准表、路由计数融合各自独立，默认策略在质量归因完成后再讨论。 |

[本分支数据与复测口径](../bench/pareto/w4a8/optimized/README.md) · [Njuapp P-only 与 D-only 消融](https://github.com/Njuapp/tokenspeed/blob/c52f86bfa5a8fc2ee79424a3f8bd933cfe4b0752/.agents/6000d-dsv41-opt/docs/OPTIMIZATION_RESULTS.md)

### Attention 评审要点

- **Paged Prefill。** 使用现有的 cache codec、页索引和有效长度，帧内旧 SWA 前缀读完后再按原契约提交当前行。多请求、分片、DCP partial 和不支持的设备走既有路径。报告有序的 P4K 和 P8K 回退数据。基线保持相同显存预算，并记录实际 KV 容量。[分支 Paged 接入](https://github.com/FrankJoy123/tokenspeed/commit/224acacb328153b124218795e40d2eb5f3d815ae)
- **Decode MLA。** 内核入口按设备、cache 格式、head/dim、query 行数、输出和 LSE 契约选择，先覆盖已验证的 24/48/96/192 行。保留小 B 和 DCP partial 的现行路径，并与 main 的 FP8 fused output 能力并存。在同一索引结果和 Graph 条件下比较两种 reader。分支的 B32 多形状实验虽然改善截尾均值，全 rank P95 由 159.538 升至 164.600 ms，扩大默认覆盖前要解释该差异。Njuapp 的 B32 领先可用来选择测试形状和 planner，整引擎差距包含 MoE 及驻留、Graph 与 KV 配置。[本分支形状扩展](https://github.com/FrankJoy123/tokenspeed/commit/009dec34368c541b9e4eca91c77b5eb139e04e08) · [Njuapp FlashInfer 适配](https://github.com/Njuapp/tokenspeed/blob/c52f86bfa5a8fc2ee79424a3f8bd933cfe4b0752/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/dsv41_mixed_decode.py)
- **Compact BF16x2。** 保留 FP32 参考和原量化格式，核对不同累加顺序下的误差。先把相同 workspace 输入的计算核 A/B 做清楚，再包含 gather 测整步。Njuapp 报告从两项 BF16 基础到流水实现，在 2048 行 × 640 键时 2455 → 1345 µs。本分支的 8K 两步档继续使用各自 4K 的最快路径。实现沿用当前布局及能力接口，并保留原作者和来源。[本分支接入](https://github.com/FrankJoy123/tokenspeed/commit/9aca5275127e3cf0f69bd1a93b6e8920f459b586) · [Njuapp kernel](https://github.com/Njuapp/tokenspeed/blob/c52f86bfa5a8fc2ee79424a3f8bd933cfe4b0752/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/_triton/compact_variants.py)

### MoE 和 EPLB 的拆分

**先把 Prefill Cake 作为标准后端接入。** 可借鉴 Njuapp 的按实际行数调度和 FlashInfer 的传输能力探测。在已有 Expert/MegaMoE 结构中接收行数、拓扑、精度和 workspace 预算。所有 rank 对每次 collective 的后端选择一致，覆盖空 rank、小尾块、Graph 和回退。首轮先保持 TP1、EP8、R0 与独立 shared expert。用真实层权重与路线测输出，再跑 DP8 4K 和实际处理 8192 新 token 的整组。以实际行数和能力决定路径，分块预算从设备和 KV 规划得到。[main 后端](https://github.com/lightseekorg/tokenspeed/blob/c20ea8c9888075be7287fed95f149b888d5d07af/python/tokenspeed/runtime/layers/moe/expert.py) · [Njuapp FlashInfer 适配](https://github.com/Njuapp/tokenspeed/blob/c52f86bfa5a8fc2ee79424a3f8bd933cfe4b0752/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/mega_moe_sm120.py)

**再补模型放置语义。** 第一项代码变更在普通 EP 上验证逻辑计数、静态就近副本、权重与 scale 同步迁移及 Graph replay。若后端持有打包权重，下一项单独验证已捕获图中的指针、打包后的原地更新及迁移边界。最后再考虑静态路由和计数共用一核。分支中该融合单项在 B8 为 42.794 → 41.419 ms，约 3.2%。B32 的更大变化来自布局和融合的组合，归因应分别验证。[现有 EPLB 接入](https://github.com/FrankJoy123/tokenspeed/commit/61e0b6ac6b1cde91c91e7540f5fd171bb18a2362) · [路由融合](https://github.com/FrankJoy123/tokenspeed/commit/3b9bcd49c700f37557d673c3c6548abdf591bdf7)

**固定放置沿用现有 schema。** Njuapp 同机 4K 从 676.867 降到 549.906 ms（重建 R8）。同类方法扩到本分支 R16 后，相对已经启用的在线 R16 为 584.335 → 577.838 ms，约 1.1% 的单档结果。若需要提交，可在独立校准工具生成 `physical_to_logical_map`，以留出的路由或真实请求测试泛化。在线 R16 候选 MMMU 210 从 58.5714% 降到 55.2381%，源自整套版本对比。角色隔离一轮仍观察到同源码关闭 EPLB 后的回落，原因待拆，当前证据不足以支持生产默认开启。[同口径成绩与质量](../bench/pareto/w4a8/optimized/README.md#质量与部署范围)

## 留在实验分支的项目

| 特性 | 目前的证据和下一步 |
| --- | --- |
| Decode MegaMoE 与 shared 核内融合 | Njuapp B32 为 89.902 ms，本分支为 114.498 ms，但两者驻留、Graph 上限和 MoE 配置不同。Njuapp C176 的 9.669 → 8.523 ms 同时覆盖多个开关。目前先用 EP16、192 行同权重同路由对照 routed、独立 shared 和核内 shared，再测 B32 与 B8。 |
| Prefill shared 核内融合 | Njuapp 的顺序消融增加约 5% req/s。本分支 L8 4224 行代理，独立流 25.204 ms、核内融合 25.271 ms。保留独立 shared，等覆盖更多路由和整步数据。 |
| grouped score、b12x tile、mHC、MXFP8 scale pack | 本分支当前完整步时收益或估计在 3% 内。b12x 在部分形状有回退。可单独留有数据的微测，避免提高基础 PR 的评审成本。 |
| 固定步数预热、同步放行、硬编码实验布局 | Pareto 的固定接受宽度、224 forward 预热和同步放行都服务于可复测。将其保留在测试配方。把实服务调度当成单独问题，若出现排队或部分 rank 出批，另做真实流量验证。 |
| `wo_a`、draft LM head 的 MXFP8，激活与线路精度变更 | 会改变数值或传输合同，单独准备量化研究和数据集证据。 |

[Njuapp 原报告](https://github.com/Njuapp/tokenspeed/blob/c52f86bfa5a8fc2ee79424a3f8bd933cfe4b0752/.agents/6000d-dsv41-opt/docs/OPTIMIZATION_RESULTS.md) · [本分支曲线与取样](../bench/pareto/w4a8/optimized/README.md)

## 每个 PR 附的证据

1. 用 review 时的 main 与单一候选做对照，固定 GPU 拓扑、依赖、路由请求种子、真实缓存、量化与采样配置。先测算子正确性和延迟，再跑整模关键点。P 先测 4K，影响多请求路径时补 8K 单步与 8K 两步。D 先测 B8 和 B32，并用 B1/B2 检查回退。分别记录截尾步时、P95、每卡吞吐与峰值显存。
2. 对 Attention 报告同输入 BF16/FP32 或 FP64 参考误差、页布局和 Graph 改输入后 replay。对 MoE 报告真实专家权重、逻辑 ID、Top-K 权重及跨 rank 求和，并保留不支持布局的当前行为。对 EPLB 加迁移前后和不同角色的独立对照。
3. 端到端 Longdoc50 和 MMMU 210 使用固定题集、真实接受逻辑，分别记录得分、答案结束率和长度截断。随机 ID 的固定宽度数字用于性能对照。若只完成算子验收，在 PR 描述中明确整模和质量状态。
4. 每个 PR 从 main 的独立分支发起，依赖用 PR 链接说明，保留独立基准和 DCO 签署，更改大型后端接口时先发 RFC。可先给维护者看这份拆分和可复算证据，再按依赖逐个送审。[TokenSpeed 贡献说明](https://github.com/lightseekorg/tokenspeed/blob/c20ea8c9888075be7287fed95f149b888d5d07af/CONTRIBUTING.md)
