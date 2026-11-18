# W4A8 / QCP8 P + EP16 D

MXFP4 专家权重使用运行时 FP8 激活（`--moe-mxfp4-fp8-activation`）；attention/dense 与其余部分按原 checkpoint 的 FP8/BF16 元数据。Engram 保持原格式。P/D 均使用 Elastic FP8 dispatch / BF16 combine。

1. [准备依赖](setup/README.md)，从同一源码构建四个组件。
2. [核对补丁](patches/README.md)，不要混用未标识的本地第三方修改。
3. [启动 P 与两个 D](launch/README.md)，两台 D 的 node rank 为0/1。
4. [启动单机 DP8 Prefill Pareto](launch/README.md#dp8-prefill-pareto)，用[性能方法](../../bench/README.md)和[质量方法](../../eval/README.md)分别验证。
