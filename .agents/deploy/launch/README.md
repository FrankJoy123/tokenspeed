# 原生启动实现

[engine.sh](engine.sh) 由两个精度目录的 prefill/decode 包装器调用。固定 QCP8/EP8 P、两节点 TP1/DP16/EP16 D；模型与可达地址由环境显式提供。先用 `DRY_RUN=1` 检查可解析的完整命令，再启动真实进程。dry run 也要求真实可读的 checkpoint、venv Python 和 CUDA ptxas，但不启动引擎、不创建运行目录。

[preflight_model.py](preflight_model.py) 在分配GPU前检查配置、tokenizer、encoding、权重索引、全部被索引的分片和safetensors头部长度；拒绝缺片、截断或与精度目录不符的量化元数据。正式启动把元数据SHA、各分片大小/头部SHA写入 `STATE_DIR/setup/model-identity.json`。这不是全权重payload校验；模型加载仍须通过完整tensor覆盖检查。取得checkpoint时另保存公开来源、revision和供应方完整payload校验依据，两端对齐。

运行目录与 JIT 缓存隔离在 `STATE_DIR`。质量启动清除固定接受、模拟专家路由和性能观察环境变量，使用真实 DSPARK 接受率与自然专家路由；固定接受只由[离线性能驱动](../../bench/README.md)设置。额外命令行参数逐项原样传给 TokenSpeed，使用者显式覆盖时须把完整命令与新配置纳入结果身份。

[行为测试](test_launch.py) 使用真实shell包装器及小型本地checkpoint，只替代GPU引擎进程，覆盖参数透传、QCP8/EP16拓扑、必填变量/rank校验、精度约束、真实接受清理与dry run：

```bash
python -m pytest -q .agents/deploy/launch/test_launch.py
```

`TOKENSPEED_SM120_DEEPEP_NICS`记录按可见GPU顺序请求的环境映射；Elastic通过底层NCCL communicator通信，不能仅凭`EP_NIC_NAME`/`NVSHMEM_HCA_LIST`赋值认定逐rank HCA绑定。初始化communicator前显式设置并保存`NCCL_IB_HCA`允许列表，核对本次初始化日志中的实际GIN provider/HCA。可单独使用`NCCL_DEBUG=INFO`及`NCCL_DEBUG_SUBSYS=INIT,NET,ENV`收集启动信息，并记录该观测配置；旧运行若未记录，不倒推其选择。

`TOKENSPEED_HOST_BINARY` 覆盖值会被拒绝；使用当前 venv 中与 engine wheel 配套的原生 host，构建身份记录其实际路径及 SHA256。

HTTP 评测入口使用当前 runtime 锁定的公开 SMG 1.16.0.post20261115。在入口主机上先运行（地址必须可被三个节点访问）：

```bash
"$VENV/bin/python" -m smg launch \
 --pd-disaggregation --backend tokenspeed \
 --prefill "tcp://$GATEWAY_HOST:$P_HANDSHAKE_PORT" \
 --decode "tcp://$GATEWAY_HOST:$D_HANDSHAKE_PORT" \
 --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" \
 --model-alias "deepseek-v41-flash=$MODEL_PATH" \
 --host 0.0.0.0 --port 18500 \
 --reasoning-parser deepseek_v41 --multimodal-tensor-transport inline \
 --mm-per-request-image-limit 160 --max-payload-size 536870912 \
 --worker-wedge-secs 0 --request-timeout-secs 7200 --queue-timeout-secs 7200 \
 --disable-retries --disable-circuit-breaker
```

首次图文shape的冷JIT可能超过SMG默认3秒静默阈值；上述命令关闭该误取消规则，保留7200秒整体请求超时。

然后 P 主机运行所选精度的 `prefill.sh`，两个 D 节点分别以 `NODE_RANK=0/1` 运行同精度 `decode.sh`。检查真实文本和图片请求均完成后再跑固定评测；HTTP端口监听或单独 `/health` 通过不足以验收 P→D。Longdoc50 最多102张原图，上述图数与 body 上限覆盖该固定回归。
