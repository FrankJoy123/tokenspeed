# P/D 安装问题与排查

本文记录独立 P/D 环境安装中已验证的修复方法。所有命令从当前仓库根目录执行，使用本部署的配置与源码，不依赖其他实验目录、个人共享存储路径或既有虚拟环境。

## 路径与环境约定

| 变量 | 用途 |
| --- | --- |
| `REPO_ROOT` / `DEPLOY_DIR` | 当前 checkout / 本部署目录，由部署入口确定 |
| `VENV` | 当前角色独立的 Python 虚拟环境 |
| `STATE_DIR` | 当前角色可写的构建、缓存、工具链及日志目录 |
| `SOURCE_DIR` | 本部署准备的外部依赖源码 |
| `CUDA_HOME` | 所选完整 CUDA 工具链的根目录 |
| `CC` / `CXX` | PyTorch 扩展使用的 C/C++ 编译器 |
| `NVSHMEM_CC` / `NVSHMEM_CXX` | NVSHMEM 专用编译器，可与 PyTorch 扩展不同 |
| `RDMA_INCLUDE_DIR` | 含 infiniband、rdma 子目录的开发头文件根目录 |
| `P_DEEPEP_SOURCE` / `P_FLASHINFER_SOURCE` | P 的定制依赖源码 |
| `P_NVSHMEM_LIB_DIR` | P 的定制 NVSHMEM 库目录 |
| `PD_CONFIG` | 本次角色配置文件；未设置时读取 deploy 内的 env.local.sh |
| `PD_TRITON_CACHE_DIR` | 当前节点的本地 Triton 缓存，可独立于共享 STATE_DIR |
| `P_HOST` / `D_HOST` / `GATEWAY_HOST` | 实际服务地址，不要求 gateway 与某个角色绑定 |
| `EP_NIC_NAME` / `NCCL_IB_HCA` | 目标节点实际可用的通信设备 |

在 `env.local.sh` 中按目标环境设置这些变量；不要照搬另一台机器的配置。P 和 D 使用不同虚拟环境，避免 P 的定制 DeepEP 覆盖 D 的 ElasticBuffer 实现。依赖版本以 `requirements.txt`、`sources.tsv`、`dependencies.tsv` 和补丁文件为准。

## 常见问题与修复

| 现象 | 原因 | 处理及对应入口 |
| --- | --- | --- |
| uv 下载 Python 时 Permission denied | 继承了不可写的全局下载目录 | 将 `UV_PYTHON_INSTALL_DIR` 设为 `"$STATE_DIR/uv-python"`，使用可写状态目录；无需 sudo。 |
| CUDA 版本解析冲突 | 继承的 pip/uv 约束与部署版本不符 | `install.sh` 清除旧约束，使用本部署 requirements.txt；源码 kernel 的嵌套 pip 也须使用同一约束。 |
| 找不到完整 CUDA toolkit | 机器只有驱动或部分运行库 | 显式选择 `CUDA_TOOLKIT_MODE=system` 或 `python`。system 模式配置实际 CUDA_HOME；python 模式在当前 VENV 安装工具链，并将 CUDA_HOME 指向该安装位置。 |
| Python CUDA 包缺少 cuobjdump 或必要库链接 | Python 包布局不同于完整系统 toolkit | `prepare_cuda_tools.sh` 下载固定版本工具并校验 SHA256，在当前环境内补齐工具及链接，不引用旧环境。 |
| 找不到 cargo | engine/scheduler 源码构建需要 Rust | 准备独立 Rust 工具链，配置 CARGO_HOME、RUSTUP_HOME 和 PATH；按各项目固定源码中的工具链要求选择版本，TS 与 gateway 不一定相同。 |
| kernel 要求的 FlashInfer 版本与 checkout 不一致 | 主包安装拉取了发布版 kernel | 按当前源码构建 engine/scheduler、kernel，再安装 TS 主包；不要使用旧 wheel 代替当前源码。 |
| Proton/Triton 或其他基础包被自动升级 | 宽版本约束导致解析结果漂移 | 使用统一版本清单；主包及嵌套安装均受约束，最后执行 pip check。 |
| ATen 提示需要 C++20 | 编译器即使接受 -std=c++20，也未满足 ATen 的宏检查 | 使用 `install_compiler.sh` 准备独立 GCC 12 工具链，并显式指定 CC/CXX 和运行库目录；不得伪造 __cplusplus。 |
| NVSHMEM 找不到 RDMA 头文件，但系统实际有文件 | Conda GCC 的 sysroot 不搜索系统开发头文件 | 给 NVSHMEM 单独选择与系统 RDMA 开发库匹配的编译器；使用新的 NVSHMEM_BUILD_DIR，避免复用旧 CMake 探测缓存。 |
| DeepEP 找不到 mlx5dv.h 或 cuda/std/tuple | RDMA、CCCL 不在扩展编译器的搜索路径中 | P 安装入口加载 `prepare_rdma_headers.sh`，将 RDMA 子目录及 CUDA_HOME 下 CCCL 加入 CPATH；DeepEP 继续使用满足 PyTorch 要求的编译器。 |
| FlashInfer 长时间停留在 Preparing metadata | 普通 wheel 构建复制大量 CUTLASS、CCCL 等源码；共享文件系统元数据操作较慢 | 当前 P 安装采用 editable 模式，保留本部署源码目录。结合构建子进程 I/O 变化和日志判断进度，不能只凭静默判断死锁。 |
| 某依赖构建成功，但续装找不到 wheel | 同次 pip 安装中其他包失败，临时 wheel 缓存被清理 | 分开安装依赖；DeepEP 用 pip wheel 输出到 STATE_DIR/dependency-wheels 下独立子目录，再安装该 wheel，不依赖 pip 临时缓存。 |
| pip check 报 DeepEP 缺少 pynvml | 安装源码使用 --no-deps，未补齐其声明依赖 | requirements.txt 已固定 pynvml==13.0.1；它与 nvidia-ml-py 配合使用，不可仅因同名模块能导入就忽略包元数据依赖。 |
| pip 提示 invalid distribution，出现 ~ 开头残留 | pip 卸载/安装被中断 | 确认安装进程及子进程已停止，只隔离本次环境中已确认的残留，保留备份后续装；不要批量清理未知环境。 |
| 运行中的 shell 出现无关的参数或语法错误 | 安装脚本执行时被改写 | 停止旧执行并保留日志，再运行修复后的脚本；编排可读取脚本快照执行，避免运行期间继续读取变化的文件。 |

## 编译器与头文件配置

使用部署提供的独立 GCC 工具链时：

```bash
bash deploy/dsv41_pd/install_compiler.sh
```

随后在 env.local.sh 中配置：

```bash
export CC="$STATE_DIR/compiler/bin/x86_64-conda-linux-gnu-cc"
export CXX="$STATE_DIR/compiler/bin/x86_64-conda-linux-gnu-c++"
EXTRA_LIBRARY_PATH="$STATE_DIR/compiler/lib"
```

这里的路径属于本部署生成的工具链，不是另一份实验环境。也可将 CC/CXX 指向目标环境已安装且满足要求的编译器。验证其 C++20 宏值至少为 202002L 后再构建扩展；保留对应 libstdc++ 运行库。

NVSHMEM 可通过 NVSHMEM_CC、NVSHMEM_CXX 选择另一套已验证的编译器，并设置：

```bash
NVSHMEM_BUILD_DIR="$STATE_DIR/build-nvshmem-system"
```

更换编译器必须换构建目录或重新生成 CMake 配置。不要修改系统编译器软链接。

DeepEP 的头文件修复只将 RDMA 子目录暴露到 `"$STATE_DIR/rdma-include"`，不把整套系统 libc 头文件混入 Conda sysroot。RDMA_INCLUDE_DIR 可覆盖默认系统搜索位置；如果头文件确实缺失，应安装与目标运行库匹配的开发包。CCCL 需来自当前 CUDA_HOME，不能从另一套 CUDA 随意借用。

## 源码与构建产物保留

P 的 FlashInfer 为 editable 安装，P_FLASHINFER_SOURCE 是运行依赖，安装后不能删除。迁移时重新准备固定版本源码并安装，不应只复制虚拟环境。

DeepEP wheel 保留在 STATE_DIR 下便于排查、续装；复用前仍须核对源码版本、Python ABI、CUDA 和编译器兼容性。临时 wheel 缓存不属于持久产物。

依赖优先采用固定 Git revision。无法使用 Git 来源的定制源码快照，应记录来源并验证清单中的校验和；不能将临时目录中的手工修改作为部署前提。

## 失败续装

基础环境首次安装要求目标 venv 不存在。失败后保留环境和日志，确认配置指向本次独立环境，再执行：

```bash
bash deploy/dsv41_pd/install.sh --resume
```

这只续装基础环境；专用依赖失败应在修复原因后重跑对应的 build_nvshmem.sh 或 install_dependencies.sh，不必重装已验证的无关组件。

同一虚拟环境只能有一个安装写入者。切换机器前，确认旧安装进程及子进程停止。共享存储可见不等于跨机器可直接复用：Python、动态库、GPU 架构和驱动仍须重新验证。

## 安装顺序与验收

1. 配置角色、独立 VENV、可写 STATE_DIR、CUDA 和编译器；准备固定版本源码。
2. 安装基础包，从当前 checkout 构建 engine/scheduler、kernel，再安装 TS 主包。
3. P 构建定制 NVSHMEM，再安装 DeepEP 和 FlashInfer；D 安装固定 DeepGEMM 并应用 DeepEP 补丁。
4. 执行 check.sh、pip check，确认模块真实来源、扩展加载、GPU 架构及 ts-host。
5. 检查 CLI 入口与启动命令，再分别验证模型加载、通信、PD 请求、精度和性能。

已验证：P/D 的源码组件及专用依赖完成安装，模块导入和 pip check 通过；两侧 CLI 入口和启动命令生成通过。D 另通过 8 卡基础 CUDA 小张量计算检查。

启动验证结果：独立 P/D 新环境已完成模型加载，两端在 gateway 中均为健康 Ready。保留 P QCP8/EP8、D DP8/EP8、DSpark 与 D CUDA Graph，不使用模拟接受率。真实小请求“1 + 1”经 P prefill 和 D decode 返回“2”，输入 41 token、输出 18 token，两端日志的请求 ID 一致。D 已完成 CUDA Graph 捕获。此结果验证基础 PD 连通性，不代表长上下文精度或性能验收。

## 首次服务启动补充检查

- 模型缓存必须递归保留模型目录结构，不能只复制顶层权重和 JSON。DeepSeek tokenizer 需要模型目录中的 encoding/encoding.py 等配套文件；缺失时会在 scheduler 初始化阶段退出。复制后应核对权重索引引用的分片、tokenizer 配置及配套代码目录。
- gateway 与 TS 的 Rust 工具链要求可能不同。构建 gateway 时应读取其固定 revision 下的 rust-toolchain.toml，并使用该 channel；不能因为从 TS 目录调用 maturin 就沿用 TS 的工具链。安装入口已显式选择 gateway 的工具链并使用 --locked，避免依赖解析漂移。此次 gateway 指定 Rust 1.98.0，而沿用 Rust 1.95 会导致依赖 MSRV 检查失败。
- 上述修复已用于本次成功的 PD 小请求；首次请求仍触发 sampling 等 JIT，不能将其延迟当作预热后的性能。

## Triton 缓存读取失败

多 rank 启动时曾在共享文件系统缓存目录中出现编译产物 .ptx 缺失，后续伴随 CompiledKernel.module 未初始化错误。可配置 PD_TRITON_CACHE_DIR，将 Triton 缓存放到当前节点可写的本地文件系统，按部署及角色隔离；不删除其他服务的缓存。该配置已接入 common.sh，D 在切换本地缓存后已完成启动预热及 CUDA Graph 捕获。此结果支持该规避措施有效，但不能仅凭错误位置确认底层文件系统根因。

## gateway wheel 打包工具

gateway Rust 编译成功后，maturin 可能因系统 patchelf 版本无法解析而在 auditwheel repair 阶段失败。requirements.txt 已固定 patchelf==0.17.2.4（可执行程序版本 0.17.2），由部署 VENV/bin 优先提供。修复后重跑 gateway 依赖安装可复用已有 Cargo 产物，无需删除构建缓存。

## 首次 JIT 与 worker 健康状态

首次启动会编译 Fused MoE、量化、Sparse MLA、topk 等模块；disable-autotune 不会跳过必要的 JIT 编译。判断进度需检查实际编译子进程及日志，不能只看 GPU 利用率或日志更新时间。不要在编译仍正常运行时反复重启。

固定版本 gateway 在 Pending 探测超过阈值后可将 worker 标为 Failed。未启用 remove-unhealthy-workers 时会继续探测；连续健康探测达到阈值后，Failed 可以恢复 Ready。因此该状态本身不能证明模型进程崩溃。服务验收仍须确认两个 worker 均健康，再提交真实请求并检查返回结果。


## 迁移到另一组机器

1. 获取对应的 P/D checkout 和本部署目录。依赖优先从固定 Git revision 获取；尚未发布的 P 定制源码及压测数据需一并分发校验过的 artifacts，单独 clone 不能替代这些制品。不要复制旧虚拟环境。
2. 每个角色复制 env.example.sh 为自己的配置，填写模型、地址、网卡、工具链及可写目录。模型目录可不同，但两端权重、tokenizer 和配套代码必须一致。
3. 将模型递归缓存到目标磁盘，核对索引引用的全部分片。将编译缓存放在本地磁盘；若 STATE_DIR 位于共享存储，至少单独配置 PD_TRITON_CACHE_DIR。按部署、角色和软件版本隔离缓存，保留成功编译产物。
4. 按 README 安装顺序重新构建各自源码组件和专用依赖。P 的 editable 源码必须保留；D 不应加载 P 的 DeepEP。不要在一个正在运行服务的环境中安装另一角色依赖。
5. Gateway 可放在 D 或独立主机。使用独立 ROLE=gateway 配置，P/D 均连接同一 GATEWAY_HOST。HTTP/ZMQ 端口属于 gateway；模型 KV 通过 Mooncake 在 P/D 间传输，不经过 HTTP 代理。
6. 对照目标拓扑设置通信选项，不直接照搬网卡编号、地址或 NCCL P2P 开关。先验证两端健康及真实小请求，再开展长上下文和性能测试。

源码下载若需要 HTTP(S) 代理，通过当前 shell 注入，不把代理地址或凭据写入仓库。Git SSH 源不自动遵循 http_proxy/https_proxy，须另行保证 SSH 连通及访问权限。服务内网地址应绕过代理；common.sh 根据 P/D/Gateway 地址设置 NO_PROXY，smoke.sh 的 HTTP 客户端不继承代理。

## 首次请求、预热与结果解释

本次成功冒烟的首请求在 P 执行阶段耗时约 77 秒，期间有 sampling 等首次 JIT；这不是模型稳定运行时的 TTFT。启动预热和 CUDA Graph 捕获不能覆盖所有实际请求形状，服务 Ready 也不代表所有请求内核已编译。

性能测试前应使用目标输入长度、并发、输出长度和采样方式预热，并审计实际请求日志中的 JIT。预热后清理 prefix/KV cache 不需要清磁盘编译缓存；重启、删除 JIT cache 和清 KV 是不同操作。仅在服务空闲时使用部署提供的清缓存入口，不用反复重启替代缓存清理。

## 验收证据与排查顺序

分别保存到各角色 STATE_DIR/logs，目录由部署配置决定，不依赖原测试机器：

- checkout revision、未提交修改、依赖版本与实际模块来源；记录配置和启动命令时排除代理口令等凭据。
- P/D/gateway 的完整安装与启动日志、CUDA Graph 捕获结果，以及两端健康状态。
- 冒烟请求参数、HTTP 状态、完整响应及请求 ID；将同一 ID 对应的 P prefill 和 D decode 日志一起保存。
- 是否使用真实 DSpark 接受率、是否存在首次 JIT，以及尚未覆盖的验证范围。

排查按“安装/导入 → 模型完整性 → 启动编译 → worker 健康 → 请求路由与 KV 传输 → 输出”推进。仅凭 HTTP 200、worker 已注册、GPU 显存占用或 pip check 通过都不能代替端到端验收。

本次已验证基础 PD 请求及其真实输出，尚未覆盖迁移后的长上下文精度、多轮 prefix cache、满并发容量和性能。短请求成功也不能证明所有 cache group 在长上下文下的传输正确性。
