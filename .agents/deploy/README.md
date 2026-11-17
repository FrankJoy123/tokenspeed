# SM120 启动

两套配置共用本仓 runtime/kernel/engine/scheduler。每台机器各 8 张 RTX 6000D；Prefill 一台 QCP8/EP8，Decode 两台 TP1/DP16/EP16。

- [W4A8](w4a8/README.md)：MXFP4 专家权重，FP8 专家激活，Elastic FP8 dispatch / BF16 combine。
- [FP4](fp4/README.md)：NVFP4 MoE、MXFP4 Engram；基础配方仍为 Elastic FP8/BF16 transport，wire1另行构建与验收。
- [共用构建](setup/README.md)：在独立 venv 从当前 checkout 重建四个组件，保存实际版本与制品身份。
- [共用启动实现](launch/README.md)：只组装 TokenSpeed 原生命令；各配置目录提供固定入口。

HTTP P/D 接入使用[上游公开配方](../../docs/serving/disaggregation.md)，所有连接地址由使用者配置。本目录不提供另外的服务路由实现。性能测试直接使用离线引擎，见[测试入口](../bench/README.md)。
