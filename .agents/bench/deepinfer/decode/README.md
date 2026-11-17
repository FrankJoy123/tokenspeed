# W4A8 EP16 Decode

`controller.py`通过原生token wire驱动一个常驻TP1/DP16/EP16引擎，`engine.sh`在两台八卡节点分别启动node rank0/1。保留TokenSpeed默认算法与Replay，R0自然MoE路由，采样`temperature=0.3, top_p=0.95, top_k=-1`。

每请求131072输入=124416实际缓存（94.921875%）+6656不同后缀；公共前缀与后缀均真实计算，后缀跨请求/轮次唯一。一次真实prime124672含256对齐保护；B=每rank请求数，扫描1/2/4/8/16/32，全组16B。所有B复用同一引擎与prefix，不清KV、不重新prime。

每B先完成200个真实满载draft+verify预热step；随后两轮独立正式请求，各跳100个真实满载完成点，取51端点形成50间隔，两轮共100间隔。实际验证/保留宽度6。输出预算仅用于结束请求：包含Prefill首输出1、每rank B个suffix启动余量和2步收尾余量；warm预算`1+6×(200+B+2)`，formal预算`1+6×(151+B+2)`。这些余量不计成绩，实际满载步数必须由16rank原始完成点证明，不能按token数推断。

## 冻结身份与启动

先按[部署入口](../../../deploy/w4a8/README.md)构建并显式准备两节点相同源码、依赖、权重、BF16 Indexer及通信配置。每节点使用自身venv；明确`CC`/`CXX`和CUDA13工具链。NIC列表按本机可见GPU顺序填写，不能复制其他机器拓扑。

```bash
export BENCH="$PWD/.agents/bench/deepinfer/decode"
export RUN_DIR="$PWD/_local/decode-run"
export IDENTITY="$PWD/_local/decode-fixed.json"
export CC="$(command -v gcc)" CXX="$(command -v g++)"
# MODEL_PATH必须是两节点相同的绝对路径；填写实际8GPU→NIC映射、GID与网口。
python "$BENCH/freeze_identity.py" --model "$MODEL_PATH"  --node0-nics "$NODE0_NICS" --node1-nics "$NODE1_NICS"  --gid-index "$GID_INDEX" --socket-ifname "$SOCKET_IFNAME" --out "$IDENTITY"
```

将同一`IDENTITY`字节复制到两节点，各自新建`RUN_DIR`并复制为`$RUN_DIR/run-identity.json`。冻结文件绑定source commit/tree/archive、实际模块/native/host/依赖、权重metadata与分片header、Indexer和两机逐卡NIC。权重检查覆盖索引/分片/头部完整性，不冒充全payload字节哈希。

D leader先启动控制器，`D_MASTER_HOST`须两端可访问：

```bash
mkdir -p "$RUN_DIR"
cp "$IDENTITY" "$RUN_DIR/run-identity.json"
python "$BENCH/controller.py" --out "$RUN_DIR" --model "$MODEL_PATH"  --run-identity "$IDENTITY" --leader "$D_MASTER_HOST" --port-base 32051  --world-size 16 --context 131072 --rounds 2 --batches 1 2 4 8 16 32  --acceptance-mode fixed --profile-batches
```

两台D在各自环境中启动，`NODE_RANK`分别0/1，网络变量按冻结身份准备：

```bash
RUN_DIR="$RUN_DIR" MODEL_PATH="$MODEL_PATH"  bash "$BENCH/engine.sh" w4a8 "$NODE_RANK" decode 256 fixed  16 2 "$D_MASTER_HOST" 32051 128 "$VENV" 7200
```

入口固定Elastic FP8/BF16、capacity128、shared-overlap1、BF16 Indexer及fixed6，清除路由模拟、旧host覆盖和其他测试观察器。`--print-env`接同样参数只检查子进程环境。真实启动分别记录源码/runtime/native身份和权重manifest；逐worker启动快照与原始step记录保留实际NIC/HCA和编译计数。

## 审核与报告

观察器只在`DECODE_BENCH_OBSERVE=1`启用，包装原有execute/sync完成回调，不新增GPU同步或profiler。完整结束并退出GPU进程后，将两节点`RUN_DIR`收集到不同目录：

```bash
python "$BENCH/audit.py" --run-dir "$RUN_D0" "$RUN_D1"  --world-size 16 --out "$AUDIT_JSON"
```

审核独立重建所有输入与真实缓存，核验16rank、warm200、两formal100+51、连续满载、宽度6及实际身份。遇窗口内编译、空洞、乱序、降载后重新满载或最终截断日志即拒绝，不向后挑干净窗口。

保留各rank全部50间隔、两轮分布及慢点，报告rank0两轮100间隔合并均值和最慢rank诊断。令`t=ms/step÷1000`，DeepInfer固定宽度换算τ=1+5×0.75=4.75，每用户τ/t、每GPU Bτ/t；0.75是换算假设，不是实测接受率。实际6-token吞吐另记，不能当质量结论。退出测试并清除模拟接受环境后才能做真实DSPARK精度回归。

CPU验证：`python -m unittest discover -s "$BENCH" -p 'test_*.py'`，覆盖真实200步、151端点、完整16rank、缓存/prime/身份、编译与慢点分布。
