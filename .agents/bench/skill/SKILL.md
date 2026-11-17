---
name: sm120-w4a8-benchmark
description: Run and audit the W4A8 QCP8 Prefill and EP16 Decode CUSTOM step benchmarks.
---

# W4A8 CUSTOM benchmark

1. 按[部署入口](../../deploy/README.md)构建当前 checkout；每台机器激活其自身 venv，核验实际加载路径、源码版本和 native SHA。不要以旧 wheel 的版本号代替构建身份。
2. 确认测试 GPU 空闲，权重及依赖相同。按 [Prefill](prefill/README.md) 或 [Decode](decode/README.md)使用新 `_local/` 输出目录。
3. 保留每条实际缓存、完成长度、输入哈希、原始完成点及编译计数；失败停止。完整预热与两正式轮不能合并成一个短窗口。
4. 用各自审核器独立重算全部 rank。报告原生 ms/step 和逻辑 token 分子，区分 replay、固定接受宽度和换算吞吐。
5. 当前源码 GPU 复测通过后，将精简脱敏结果与限制写入 [w4a8](../w4a8/README.md)，同步导航；历史参考不充作新源码验收。

方法参考固定为公开 [DeepInfer a679bb86ce36e883a98a97733492b686e5b9a08e](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e)，阅读其 `examples/benchmark.py`。这里执行 TokenSpeed，硬件、replay、缓存实现与内部采样器不同；不宣称逐 bit 复刻，也不直接用跨硬件绝对值认定引擎优劣。
