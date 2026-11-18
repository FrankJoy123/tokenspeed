# W4A8 Decode · EP16

两台 RTX 6000D 八卡节点组成一个 TP1/DP16/EP16 引擎。每个请求总输入 131,072 token，真实缓存 124,416 token（94.921875%），另有 6,656 个不同后缀 token。整个 sweep 共用一个引擎和一次清缓存、真实 prime。prime 含 256 token 对齐余量，后缀在不同请求、B 和轮次间保持唯一。

正式协议为 R0 自然 MoE 路由，采样 temperature=0.3、top_p=0.95、top_k=-1。B 是每 rank 的请求数，全组为 16B。依次跑 B1、2、4、8、16、32，每 B 预热 200 个全 rank 实际满载 draft+verify step。启动两轮独立正式请求，每轮跳过 100 个全 rank 共同满载完成点，取紧接的 51 个端点形成 50 个间隔。固定验证与保留宽度为 6。输出 token 预算只控制请求退出，满载步数由运行记录证明。

## 准备并冻结配置

先按 [W4A8 环境构建](../../../../deploy/w4a8/setup/README.md)在两台机器安装相同版本和完整 checkpoint。参考 [网络与环境示例](../../../../deploy/w4a8/launch/env.example.sh)，设置本机 venv、CUDA、工具链、8 GPU 到 NIC 的映射和 RoCE GID。模型在两节点应使用同一绝对路径。

本配方启用 SM120 DeepGEMM Full Indexer 与 DeepSelect，并关闭 Native FP4 Indexer 和 FP4 Candidate 两个独立开关。Fast Full 路径对 MXFP4 打包的 query 和 V4.1 缓存行打分。[实测归档](ep16.json)保存源码、依赖与实际 Indexer 身份。

```bash
export BENCH="$PWD/.agents/bench/pareto/w4a8/decode"
export RUN_DIR="$PWD/_local/decode-ep16"
export IDENTITY="$PWD/_local/decode-ep16-identity.json"
export CC="$(command -v gcc)" CXX="$(command -v g++)"
"$VENV/bin/python" "$BENCH/freeze_identity.py" --model "$MODEL_PATH" \
  --node0-nics "$NODE0_NICS" --node1-nics "$NODE1_NICS" \
  --gid-index "$GID_INDEX" --socket-ifname "$SOCKET_IFNAME" \
  --out "$IDENTITY"
```

NODE0_NICS、NODE1_NICS 均为对应节点按可见 GPU 顺序的逗号分隔网卡名。在 leader 生成冻结文件，两端各创建同一个新运行目录，并将文件复制为各自的 RUN_DIR/run-identity.json。复建指定实测提交时，可在冻结命令中用 --source 指向实际构建运行时的干净 checkout，并在两节点设置 RUNTIME_SOURCE 指向各自对应的 checkout。

## 启动双机

先在 leader 运行控制器。D_MASTER_HOST 为两节点可达的 leader 地址。开放 32051 到 32054，分别用于握手、输入、输出和分布式初始化。控制器进入等待 16 个 worker 后，再启动两节点引擎。

```bash
"$VENV/bin/python" "$BENCH/controller.py" --out "$RUN_DIR" \
  --model "$MODEL_PATH" --run-identity "$IDENTITY" \
  --leader "$D_MASTER_HOST" --port-base 32051 --world-size 16 \
  --context 131072 --rounds 2 --batches 1 2 4 8 16 32 \
  --acceptance-mode fixed
```

下面的命令在两台节点各执行一次。分别设置 NODE_RANK=0 或 1、NODE_NICS=本机八卡映射。RUN_DIR、MODEL_PATH 和 D_MASTER_HOST 需要导出。用 --print-env 可预览启动环境。

```bash
export TOKENSPEED_SM120_DEEPEP_NICS="$NODE_NICS" NCCL_IB_HCA="$NODE_NICS"
export NCCL_IB_GID_INDEX="$GID_INDEX" NVSHMEM_IB_GID_INDEX="$GID_INDEX"
export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME" NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="$SOCKET_IFNAME"
bash "$BENCH/engine.sh" w4a8 "$NODE_RANK" decode 256 fixed \
  16 2 "$D_MASTER_HOST" 32051 128 "$VENV" 7200
```

引擎采用 Elastic FP8 dispatch / BF16 combine，capacity 128，shared overlap 开启。检查两台 worker 日志中实际触发的 DeepGEMM Full Indexer。实际 HCA 绑定以 worker 记录为准。

## 审核与出图

控制器完成后，停止两节点引擎并归档各自运行目录。将两份目录拷贝到同一审核机器，leader 的目录排在首位。

```bash
"$VENV/bin/python" "$BENCH/audit.py" --run-dir "$RUN_D0" "$RUN_D1" \
  --out "$PWD/_local/decode-ep16-audit.json"
```

审核器先验证相同的全 rank 原始调度计数及缓存、宽度和编译记录，再取同一 dispatch 的全局满载交集。两轮 100 个原始间隔全部保留。Pareto 用中间 60 个间隔的均值计算。横轴是 4.75 / 步时（秒）得到的等效每用户输出速度，纵轴乘 B 得到等效每 GPU 吞吐。4.75 来自 1+5×0.75 的固定换算假设，真实验证和保留宽度另记为 6。

[原始间隔与身份](ep16.json) · [Pareto](pareto.svg) · [CSV](pareto.csv) · [统一 SOP](../../SKILL.md)
