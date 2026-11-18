---
name: sm120-w4a8-pareto
description: Run, audit, and plot TokenSpeed W4A8 QCP8 and DP8 Prefill and EP16 Decode Pareto measurements on SM120.
---

# W4A8 Pareto SOP

1. 按 [W4A8 部署配方](../../deploy/w4a8/README.md)安装固定版本并构建。用完整 W4A8 checkpoint，核对源码、加载的原生制品、依赖、模型索引与权重分片头、Indexer 和通信配置。运行目录、日志和缓存放 `_local/`。
2. 使用空闲 GPU。两阶段均保留 TokenSpeed 的默认 replay、R0 自然 MoE 路由与 `temperature=0.3, top_p=0.95, top_k=-1`。每项显式记录所用 Indexer；同档不同配置各自完整跑验收。
3. [Prefill 启动与审核](w4a8/prefill/README.md)。一台八卡运行 DP8/EP8 或 QCP8/EP8。每条请求 126,976 公共前缀加 4,096 互异新 token，原生前缀缓存和 128 行 SWA replay，每条输出 1 token。每布局每档独立起引擎，预热和正式队列各做一次清缓存、真实 prime 和完整队列。正式窗口前至少 5 个符合目标形状的步，取尾部连续 11 个完成端点形成 10 个间隔。
4. [Decode 启动与审核](w4a8/decode/README.md)。两台八卡加入一个 EP16 引擎，所有 B 共用一次真实 prime 与 124,416 token 的缓存前缀，后缀 6,656 token 且跨请求和轮次唯一。每 B 实测 200 个全 rank 共同满载的 draft+verify 预热步。两轮独立正式请求各跳过 100 个全 rank 共同满载完成点，取接下来的连续 51 点形成 50 个间隔。全 rank 实际验证与保留宽度为 6。
5. 把闭合原件交给各目录审核器。检查目标形状、缓存、步连续性、实际宽度、所有 rank 的完成点及窗口内编译。保持完整有效步时间隔与分布，失败或降载的档位记录原因。
6. [生成两张 Pareto 与 CSV](w4a8/README.md)。每点在审核后按步时排序，两端各截 20%，以中间 60% 的均值绘图。Prefill 留 6/10，Decode 合并两轮留 60/100。Decode 用 τ=4.75 换算每用户与每卡吞吐，0.75 是换算假设。公开以脱敏的输入形状、配置、间隔和结果支持重绘。

坐标与完成点方法参考 [DeepInfer a679bb86](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e)。这里采用 TokenSpeed 的缓存、调度及执行器。
