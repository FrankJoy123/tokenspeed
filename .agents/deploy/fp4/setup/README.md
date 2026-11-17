# FP4 环境

从[共用构建说明](../../setup/README.md)创建独立 venv，构建当前 checkout。该配置的权重格式见[配置入口](../README.md)，额外源码依赖见[补丁清单](../patches/README.md)。

可选 wire1 transport：设置 `NVSHMEM_DIR` / `NCCL_ROOT_DIR` 开发库目录后运行 `bash .agents/deploy/fp4/setup/wire1.sh`。脚本获取固定公开 DeepEP commit，核对并应用版本补丁，在隔离源码目录编译，不替换 Elastic 安装。GPU 多 rank 检查通过后再接入生产角色。
