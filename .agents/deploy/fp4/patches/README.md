# FP4 transport 源码补丁

可选 NVFP4 dispatch / FP8 combine 依赖 DeepEP 公开源码：

```text
https://github.com/deepseek-ai/DeepEP.git
01dc3aaac82068020353dce2c302e38153c0bfaa
```

[deepep-nvfp4-fp8-wire1.patch](deepep-nvfp4-fp8-wire1.patch) 添加 wire format 1、缓存布局和尾块/padding 修正；SHA256 见[清单](checksums.json)。NVSHMEM 使用 `nvidia-nvshmem-cu13==3.4.5`，NCCL 开发头与运行库须匹配实际 PyTorch 环境。此补丁作为必要源码交付，不跟踪预编译二进制。

默认 FP4 启动配方使用 Elastic transport；启用 wire1 前需从上述固定源码应用本补丁并本地编译，检查真实多 rank 通信与数值正确性。只切环境变量不等于完成依赖安装或验证。

补丁仅包含通信源码、Python接口和数值测试；构建与启动由本层当前 README 负责，不携带历史部署脚本或过程记录。
