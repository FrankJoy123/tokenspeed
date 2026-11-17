# SM120 源码依赖

[deepgemm.patch](deepgemm.patch) 基于 [sources.json](sources.json) 的三个完整公开 commit。补丁添加独立包名与 namespace、SM120 必要编译和共享内存屏障修正；普通 DeepGEMM 2.8.2 继续由上游依赖维护。补丁不携带构建机器的运行缓存路径。

使用者运行共用 `setup/build.sh` 即自动获取固定源码、应用补丁、编译安装并核对两包共存，无需另找预制 wheel。SM120 算子自动选择 `deep_gemm_sm120`；其余平台继续使用普通包。实际 wheel/native SHA、源码与依赖版本写入运行目录的 setup/。

`elastic-rdma-tuner.patch` 是固定 `tokenspeed-deepep==2.1.0.post20260810` 的 Python tuner 修正；当前固定通信参数使用显式 SM/QP。校验值见[checksums.json](checksums.json)。
