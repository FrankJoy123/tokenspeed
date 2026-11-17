# FP4 / QCP8 P + EP16 D

使用独立量化 checkpoint：NVFP4 routed MoE（group size 16）及 MXFP4 Engram（block 32，UE8M0 scale，层1/14）。不把原 W4A8 权重改名当作量化权重。当前基础启动仍使用 Elastic FP8 dispatch / BF16 combine；NVFP4/FP8 wire1 需另构建并验证固定补丁依赖。

1. [准备依赖](setup/README.md)，从同一源码构建四个组件。
2. [核对补丁](patches/README.md)，不要混用未标识的本地第三方修改。
3. [启动 P 与两个 D](launch/README.md)，两台 D 的 node rank 为0/1。
4. 用[性能方法](../../bench/README.md)和[质量方法](../../eval/README.md)分别验证。
