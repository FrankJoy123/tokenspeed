# DSV4.1 Flash / RTX 6000D

本目录提供 SM120 的 QCP8 Prefill、TP1/DP16/EP16 Decode 配方与可重跑测试。

- [部署](deploy/README.md)：W4A8 和 NVFP4 MoE / MXFP4 Engram 两种配置，依赖、补丁、构建与启动。
- [性能](bench/README.md)：DeepInfer 风格的 QCP8/DP8 Prefill、EP16 Decode 当前方法与本轮结果。
- [质量回归](eval/README.md)：Longdoc 固定题集与多模态评测，协议分别标明。

维护约束见 [AGENTS](AGENTS.md)。运行时下载、原始输出和日志保留在 `_local/`。
