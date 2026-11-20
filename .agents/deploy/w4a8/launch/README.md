# W4A8 启动

先完成[环境构建](../setup/README.md)，在每台机器准备相同源码和完整匹配的checkpoint。复制 [env.example.sh](env.example.sh) 到 `_local/`，填写绝对路径与可达节点地址后 source；不要直接执行占位示例。

## DP8/EP8 Prefill + DP16/EP16 Decode

一台 RTX 6000D 八卡运行 Prefill，两台八卡共同运行一个 Decode 引擎。两角色均是 TP1，启用在线 EPLB 和静态就近路由。R16 是整个 EP 组的 16 个冗余槽，P 每卡两个，D 每卡一个。角色参数如下。

| 角色 | 启动配置 | EPLB 周期 |
| --- | --- | ---: |
| Prefill | DP8/EP8、外层 4224、Cake 内部 1024、Native FP4 Indexer、Paged 与 Compact Attention | 1000 |
| Decode | DP16/EP16、Fast Indexer、Cake BF16 MLA 多形状、Elastic MoE、静态路由融合 | 10000 |

周期以 forward 计，分四层推进。服务周期与 [Pareto 固定预热](../../../bench/pareto/w4a8/optimized/README.md)分别配置。三个节点应从同一提交构建并核对模型身份。为每个角色设置本机的独立 `STATE_DIR`，在 D 两台机器设置同一 `D_DIST_INIT_ADDR` 并分别选择 `NODE_RANK=0` 和 `NODE_RANK=1`。

```bash
# Prefill 节点
source _local/w4a8-prefill.env
export W4A8_PD_PROFILE=dp8-eplb
DRY_RUN=1 bash .agents/deploy/w4a8/launch/prefill.sh
bash .agents/deploy/w4a8/launch/prefill.sh

# 两台 Decode 节点各自执行
source _local/w4a8-decode.env
export W4A8_PD_PROFILE=dp8-eplb
DRY_RUN=1 bash .agents/deploy/w4a8/launch/decode.sh
bash .agents/deploy/w4a8/launch/decode.sh
```

按[公开 SMG P/D 接入](../../../../docs/serving/disaggregation.md)配置两角色握手端口与 Mooncake。启动后核对 P 八个、D 十六个 rank ready，发送真实文本和图片请求验证 KV 交接与返回，再评估部署质量。该配方的启动参数已通过本机测试。

## QCP8 P + EP16 D

```bash
source _local/w4a8-prefill.env
DRY_RUN=1 bash .agents/deploy/w4a8/launch/prefill.sh
bash .agents/deploy/w4a8/launch/prefill.sh
# 两台D分别source自身环境；NODE_RANK为0和1，D_DIST_INIT_ADDR必须相同。
source _local/w4a8-decode.env
DRY_RUN=1 bash .agents/deploy/w4a8/launch/decode.sh
bash .agents/deploy/w4a8/launch/decode.sh
```

先按[上游 P/D 接入](../../../../docs/serving/disaggregation.md)启动公开SMG，配置P/D握手端口为环境中的 `P_HANDSHAKE_PORT`/`D_HANDSHAKE_PORT`，然后启动P及两台D。两台D共用一个16rank引擎；follower不是独立D服务。此处只说明官方P/D连接方式，不增加路由实现。

启动前自动检查模型元数据、全部被索引权重分片及头部完整性，见[校验范围](../../launch/README.md)。运行后核对加载制品SHA、ready/健康状态，并用真实文本及多图请求验收；命令打印成功或模型能加载都不代表性能/质量通过。GPU网络参数（如 `NCCL_IB_HCA`、`GLOO_SOCKET_IFNAME`、`EP_NIC_NAME`）按现场拓扑设置。

## DP8 Prefill Pareto

在一台独占的 RTX 6000D 八卡节点上从仓库根目录执行。下面是已发布的 **DP8/EP8 per-rank 4K** 档，八条 attention lane 每步各处理一条有 4,096 个新增 token 的请求，整组每步 32K 新增 token。命令直接启动单机离线 Prefill 引擎。

```bash
source _local/w4a8-prefill.env
export CC="$(command -v gcc)" CXX="$(command -v g++)"
BENCH="$PWD/.agents/bench/pareto/w4a8/prefill"
POINT="$PWD/_local/prefill-dp8-4k"
"$VENV/bin/python" "$BENCH/run_point.py" --layout dp8 --label 4096 \
  --model "$MODEL_PATH" --python "$VENV/bin/python" \
  --expert-routing natural --indexer native-fp4 --profile-steps 0 \
  --queue-requests-per-lane 64 --max-model-len 1048576 --out "$POINT"
"$VENV/bin/python" "$BENCH/audit_point.py" --point "$POINT"
```

其他档位各用一个新输出目录，按 [Pareto 实测参数](../../../bench/pareto/w4a8/prefill/points.json)选择 label、Indexer 和队列大小。512、1K、2K 档仍固定每条请求 4K 后缀，每步分别启用 1、2、4 条 DP lane。[完整 Prefill SOP](../../../bench/pareto/w4a8/prefill/README.md)
