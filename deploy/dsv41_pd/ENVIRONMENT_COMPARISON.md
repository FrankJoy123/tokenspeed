# P / D 环境比较

以下版本表记录迁移前的环境，不能当作本次新部署的版本清单。本次 P/D 均已创建独立虚拟环境：P Python 3.12.15、D Python 3.12.14；按角色安装固定依赖，并已通过真实 PD 小请求。准确版本以各角色安装日志和依赖清单为准。

核对方式：使用当前 P/D 服务进程的 PYTHONPATH 等环境，查看包 metadata 与实际 import 路径；metadata 相同不代表运行源码相同。

| 组件 | P | D |
| --- | --- | --- |
| Python | 3.12.3 | 3.12.14 |
| PyTorch | 2.14.0+cu130 | 2.14.0+cu130 |
| Transformers | 5.12.0 | 5.12.0 |
| TokenSpeed Triton | 3.9.10.post20261008 | 相同 |
| CUTLASS DSL | 4.8.0 | 相同 |
| cuda-bindings | 13.4.4 | 13.3.1 |
| cuda-core | 1.2.2 | 1.0.1 |
| cuda-tile | 1.7.0 | 1.6.0 |
| FlashInfer | 标记 0.7.2，定制源码覆盖 | 0.7.2 安装包 |
| DeepEP | PCIe PR375 定制 Buffer 接口 | 2.1 wheel，ElasticBuffer 接口 |
| TS engine / scheduler | P checkout 单独编译 | D checkout 单独编译 |

P 部分 CUDA Python 包较新，但并非整体环境更新。P venv 启用了 include-system-site-packages=true，还会访问系统安装的包；直接复制这个环境不能形成独立、完整的依赖锁。

P/D 的 DeepEP 都导入 deep_ep，但 P 的定制 __init__.py 只导出 Buffer、EventOverlap、Config，不提供 D 必需的 ElasticBuffer。两个 pip distribution 即使名字不同也会写入同一模块命名空间，不能视作可并存。FlashInfer、NVSHMEM 和 TS host 也需要按角色分别加载。

建议共享基础镜像、CUDA/PyTorch 版本和下载缓存，分别创建 P/D venv；各自编译 TS 原生组件。技术上可用 PYTHONPATH/LD_PRELOAD 为同一基础环境叠加角色私有依赖，但仍然是两套运行时隔离，必须覆盖 Python 与 .so 加载路径，不能简单安装两个 DeepEP 后切换 ROLE。当前新 D 环境采用不继承 system site-packages 的独立 venv，未尝试统一 P/D 二进制依赖。
