# W4A8 Pareto

- [Prefill QCP8 / DP8 单机启动](prefill/README.md) · [图](prefill/pareto.svg) · [CSV](prefill/pareto.csv)
- [Decode EP16 双机启动](decode/README.md) · [图](decode/pareto.svg) · [CSV](decode/pareto.csv)

标签与工作量见各图。每个 Prefill 布局在每档保留审核后截尾平均步时最短的配置。Decode 固定采用 [ep16.json](decode/ep16.json) 所记录的 Fast Indexer 正式六档。两张图分别从 10 和 100 个步时间隔中保留中间 6 和 60 个。CSV 同时提供完整样本均值和截尾均值。

在仓库根目录重建。

```bash
python3.12 -m venv _local/pareto-plot
_local/pareto-plot/bin/python -m pip install -r .agents/bench/pareto/w4a8/requirements-plot.txt
_local/pareto-plot/bin/python .agents/bench/pareto/w4a8/plot.py --out "$PWD/_local/pareto-render"
```

提供本地测量文件时，为每个已审核的 Prefill 点传一次 `--prefill-point PATH`，其中 PATH 是 `run_point.py --out` 的目录。通过 `--decode-audit PATH` 读取双机原件生成的正式 Decode 审核 JSON。图与 CSV 写到输出目录的 `prefill/`、`decode/`。

## 节点环境

按 [构建说明](../../../deploy/setup/README.md)使用 Python 3.12、CUDA 13.0 和兼容编译器。编译器来自独立工具链时，将其 C++ 运行库目录与 CUDA 库、所用 venv 的 `torch/lib` 加入动态库搜索路径。用构建时写出的 `pip freeze` 和源码、模块、host 二进制身份核验实际加载版本。

RTX 6000D 测量使用 RoCE。按该节点 GPU 顺序设置网卡并填写实际 GID 与 socket 网口。原测量在启动前配置 `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`、`NCCL_IB_CUDA_SUPPORT=0`、`NVSHMEM_DISABLE_P2P=1`、`NVSHMEM_IB_ENABLE_IBGDA=1`、`NVSHMEM_IBGDA_NIC_HANDLER=gpu` 和 `NVSHMEM_ENABLE_NIC_PE_MAPPING=1`。记录实际网卡绑定与所用通信库版本。
