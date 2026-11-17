# QCP8 Prefill CUSTOM

`run_point.py`从当前环境启动全新的NORMAL离线引擎；`prefill_steady.py`通过原生WireDriver满队列测量；`audit_point.py`独立重算。QCP8/EP8为主配置，DP8/EP8可显式选择。测试不需要D或HTTP入口。

当前协议：131072输入=126976公共prefix＋4096互异新增token，每请求额外128行SWA replay，物理行4224。每点一个完整队列预热＋两轮正式cohort，每轮先清KV并真实prime。取末尾连续11个满载完成端点形成10间隔，之前至少5满载step；有新编译/不满载/缓存不符即拒绝。预算为每attention DP lane新增token，物理预算=预算×4224/4096。QCP8的全组吞吐除8得每GPU；DP8按lane计。

```bash
export BENCH="$PWD/.agents/bench/skill/prefill"
export CC="$(command -v gcc)" CXX="$(command -v g++)"
# 已核构建为GCC12.4；确认本机版本或显式选择兼容工具链绝对路径。
python "$BENCH/run_point.py" --layout cp8 --budget 4096 \
  --model "$MODEL_PATH" --python "$VENV/bin/python" \
  --expert-routing natural --indexer bf16 --profile-steps 0 \
  --out "$PWD/_local/prefill-cp8-4k"
python "$BENCH/audit_point.py" --point "$PWD/_local/prefill-cp8-4k"
```

依次以新目录测8192/16384；OOM或注册失败保留失败记录，不补零或删点。高预算32768/65536只有实测成功才发布。`--profile-steps 8`会在两正式轮后增加独立profile cohort，不计正式成绩。

入口显式设置SM120 DeepEP=1、Elastic FP8/BF16传输、capacity2048和shared-overlap0，并清除Decode固定接受、旧物理路由、host trace及Indexer kernel override残留。实际选择写入本次`invocation.json`。R0自然路由和Indexer BF16是当前明确选择；`--expert-routing uniform`调用产品已有模拟模式，不等同历史416物理槽/独立未归一化随机权重。`--indexer native-fp4`显式启用两个Indexer开关，必须另记精度与协议。历史数据使用R32均衡、Indexer两个开关均1、只有一正式轮，不能直接解释为这次源码或性能改善。

`sitecustomize.py`仅在`PREFILL_BENCH_OBSERVE=1`安装测试观察器，记录现有`ModelExecutionResult.sync`完成点、`ForwardOp.extend_replay_lens`、DP metadata和compile monitor，不增加GPU工作。当前源码还须允许NORMAL QCP的纯Decode队尾使用已有复制Decode路径；不能用旧参考快照或旧native包替代当前实现。

两入口拒绝旧`TOKENSPEED_HOST_BINARY`覆盖，并保存当前源码树、实际`ts-host`路径与SHA、runtime与native身份。

计时包含Host调度/提交，是原生step墙钟，不是TTFT或纯kernel时间。实际请求、日志、命令与PID均放输出目录；对外只选取经过审核的脱敏摘要。CPU检查：`python -m unittest discover -s "$BENCH" -p 'test_*.py'`。
