# W4A8 Prefill：QCP8 与 DP8

`run_point.py`从当前环境启动全新的原生离线引擎；`prefill_steady.py`通过WireDriver满队列测量；`audit_point.py`独立重算。两条曲线分别为QCP8/EP8（QCP8、DP1）和DP8/EP8（QCP1、DP8），预算均为4096/8192/16384新增token，不需要Decode或HTTP入口。

每请求131072输入=126976公共prefix＋4096跨请求互异后缀，使用TokenSpeed原生prefix cache。默认额外128行SWA replay，物理行4224与新增4096分别计量。采样固定temperature=0.3、top_p=0.95、top_k=-1，每请求只输出1 token。

每预算独立启动引擎：先flush、真实prime并完成一个完整预热队列，再flush、真实prime并完成一个完整正式队列。正式队列取末尾连续11个满载完成端点形成10个间隔，之前至少5个满载step；保留选中11个step的ID和完成时间、全部10个原始interval_ns，并报告min/p50/p95/max毫秒分布；p50/p95按排序后(n−1)×q位置线性插值，不截慢点。任一间隔非正数，或有新编译、不满载、缓存或完成长度不符即拒绝。完整队列的供给量按预算计算，不套用Decode的200/100/50步数。

预算按每attention DP lane计，物理预算=新增预算×4224/4096。QCP8的全组吞吐除8得到每GPU；DP8逐lane审核并报告rank0计时，保留全部8个lane供偏斜检查。

```bash
export BENCH="$PWD/.agents/bench/deepinfer/prefill"
export CC="$(command -v gcc)" CXX="$(command -v g++)"
# 已核构建为GCC12.4；确认本机版本或选择兼容工具链绝对路径。
python "$BENCH/run_point.py" --layout cp8 --budget 4096 \
  --model "$MODEL_PATH" --python "$VENV/bin/python" \
  --expert-routing natural --indexer bf16 --profile-steps 0 \
  --out "$PWD/_local/prefill-cp8-4k"
python "$BENCH/audit_point.py" --point "$PWD/_local/prefill-cp8-4k"
```

使用新目录依次测8192/16384；DP8将`--layout`改为`dp8`。两条曲线可在两台独占的八卡节点并行，每节点各预算依次执行。脚本将DeepGEMM、DeepEP、Triton、CUDA、CUTE、kernel缓存以及FlashInfer workspace、XDG缓存放入本点输出目录的`cache/`。OOM或注册失败保留失败记录，不补零、删点或静默重试。

入口固定R0自然路由、BF16 Indexer、SM120 DeepEP=1、Elastic FP8/BF16传输、capacity2048、shared-overlap0。保留TokenSpeed默认autotune，由启动阶段选择并缓存适用tactic；正式窗口仍拒绝编译污染。R0/trivial使用默认专家映射，不指定副本分发算法。脚本清除Decode固定接受、路由模拟、旧物理路由、host trace和Indexer kernel override残留，实际选择写入`invocation.json`。

`sitecustomize.py`仅在`PREFILL_BENCH_OBSERVE=1`安装测试观察器，记录现有`ModelExecutionResult.sync`完成点、`ForwardOp.extend_replay_lens`、DP metadata和compile monitor，不增加GPU工作。NORMAL QCP的纯Decode队尾使用已有复制Decode路径。

运行拒绝`TOKENSPEED_HOST_BINARY`覆盖，并保存源码树、实际`ts-host`路径与SHA、runtime与native身份。计时包含Host调度/提交，是原生step墙钟。实际请求、日志、命令与PID均放输出目录；公开结果只放本方案经过审核的脱敏摘要。CPU检查：`python -m unittest discover -s "$BENCH" -p 'test_*.py'`。
