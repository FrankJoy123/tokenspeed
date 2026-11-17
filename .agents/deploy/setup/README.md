# 构建与依赖

要求 Python 3.12、CUDA 13.0（包含 nvcc/ptxas 与开发头文件）、GCC/G++ 12.4、CMake >=3.22、Rust/Cargo >=1.85（本次构建工具链1.99.1）、libclang、libnuma 和 RDMA 开发库。

可使用上游 [CUDA 构建镜像](../../../docker/Dockerfile.nvidia)中固定的公开 runner 基底 `lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.6.18` 准备 Linux/CUDA 工具链；镜像原有依赖仍须由本配方更新并核验，不直接当成当前环境。若宿主默认编译器不同，显式设置 `CC` / `CXX` 为兼容工具链的绝对路径。源码编译需要 OpenSSL 开发头及链接库，使用 sysroot 工具链时显式配置其可见的头文件/库路径。

CUDA/PyTorch 与第三方库精确版本由当前 checkout 的 [CUDA 清单](../../../tokenspeed-kernel/python/requirements/cuda.txt)和[算子依赖](../../../tokenspeed-kernel/python/requirements/cuda-thirdparty.txt)维护。runtime、scheduler、engine、kernel 全部从同一 checkout 构建；不套用旧 runtime 的 scheduler/native 二进制。

```bash
export VENV=/absolute/path/to/venv
export STATE_DIR=/absolute/path/to/run/_local/build
export CUDA_HOME=/usr/local/cuda-13.0
bash .agents/deploy/setup/build.sh
```

`build.sh` 新建独立 venv，安装锁定的 GPU 依赖，构建本仓组件，然后写入 `STATE_DIR/setup/` 的 pip freeze、源码和模块/二进制 SHA256。安装会访问公开包源；已有经过校验的 wheelhouse 可用标准 pip 环境变量配置。原生源码补丁由各精度目录的 patches/ 维护。模型文件需由使用者按许可自行取得；两端完整配置、tokenizer、encoding 和全部权重分片须一致。

构建成功后先运行相关 GPU 正确性检查，再使用新目录做性能和精度回归。重建后的实际加载身份是成绩的一部分。

使用独立 GCC 工具链而宿主 `libstdc++` 较旧时，运行环境显式设置 `TOOLCHAIN_LIBRARY_DIR=/path/to/compiler/lib`；启动入口将该目录加入动态库路径。须能实际导入新 `tokenspeed_engine._host`，不能仅凭编译成功认定运行依赖满足。

SM120 所需 DeepGEMM 扩展由同一个 `build.sh` 自动从[固定公开源码](../w4a8/patches/sources.json)构建，安装为独立 namespace。脚本逐项核对源码 commit、补丁 SHA 和安装版本，保留上游普通 DeepGEMM；运行时按 GPU 架构选择。首次构建需要 Git >=2.28、网络与上述编译工具链，完整源码不随本仓重复分发。
