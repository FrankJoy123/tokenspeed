---
name: sm120-w4a8-benchmark
description: Run and audit W4A8 QCP8/DP8 Prefill and EP16 Decode step benchmarks with real prefix caching.
---

# W4A8 P/D benchmark

1. 按[部署入口](../../deploy/README.md)准备当前源码与组件；核验实际加载路径、源码、native SHA、依赖、权重、Indexer 与通信身份。生产源码未变而复用已验证制品时，记录原构建版本和源码一致性依据。
2. 确认 GPU 空闲。共同使用 R0、自然 MoE 路由、`temperature=0.3, top_p=0.95, top_k=-1`，保留 TokenSpeed 默认算法及 Replay；不要强制均衡。
3. [Prefill](prefill/README.md)：分别测QCP8和DP8，每预算独立引擎，原生 prefix cache，126976 公共前缀加4096互异后缀；真实清KV/prime/完整队列预热，正式再清KV/prime/一轮完整队列。每请求输出1 token，默认128行 encoder SWA replay，新增4096与物理4224行分报；跳至少5个满载step，末尾11端点形成10间隔。
4. [Decode](decode/README.md)：所有B复用一个引擎和公共前缀，124416实际缓存加6656互异后缀；每B预热200个真实满载draft+verify step，两轮正式各跳100个满载完成点再取51端点/50间隔。固定验证/保留宽度6；输出token预算只是结束手段，必须用原始完成点证明step数。
5. 各自审核器独立核验全部rank的连续性、满载、缓存、实际宽度、编译及两轮分布；保留慢点，不强制曲线单调。Decode按τ=1+5×0.75=4.75换算DeepInfer风格吞吐；0.75只是接受率假设。
6. 真实复测完成后，将精简脱敏结果、身份和限制写入 [w4a8](w4a8/README.md)，同步导航。Prefill一轮正式与Decode两轮正式各按自身协议，只收录本轮新测结果。

方法参考固定为公开 [DeepInfer a679bb86ce36e883a98a97733492b686e5b9a08e](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e)，阅读其 `examples/benchmark.py`。这里执行 TokenSpeed，硬件、replay、缓存实现与内部采样器不同；不宣称逐 bit 复刻，也不直接用跨硬件绝对值认定引擎优劣。
