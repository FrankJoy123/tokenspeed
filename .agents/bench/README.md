# W4A8 测试

[DeepInfer 风格方法与运行入口](deepinfer/SKILL.md)维护三条曲线：Prefill QCP8、Prefill DP8、W4A8 Decode EP16；[本轮结果](deepinfer/w4a8/README.md)只收录本次新测量。这里测原生满载step。

共同使用R0、自然MoE路由和 `temperature=0.3, top_p=0.95, top_k=-1`，保留TokenSpeed默认算法与Replay。固定源码、依赖、权重、Indexer、通信和native身份。

环境来自[部署入口](../deploy/README.md)。原始请求、日志、trace与中间结果写 `_local/`；公开只放当前方法、脱敏精简数据、版本与结论。
