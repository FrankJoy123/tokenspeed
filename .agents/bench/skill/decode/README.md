# EP16 Decode CUSTOM95

`controller.py`通过原生 token wire驱动一个常驻 TP1/DP16/EP16 引擎，`engine.sh`在两台八卡机器分别启动 rank0/rank1。无需平台或私有路由组件。源码/依赖须从当前 checkout构建。

固定131072输入=124416公共prefix＋6656互异尾部；真实prime124672含256对齐保护token。B=每rank请求数，默认扫描1/2/4/8/16/32，全组=16B；B间不清KV、不重新prime、不重启引擎。每B一完整1152输出预热＋两轮独立1152输出正式cohort；正式各跳110个连续满载完成点，取61端点/60间隔。每请求实际验证和保留宽度6，仅用于性能。

在D leader先启动控制器，设置可被两端访问的地址：

```bash
export BENCH="$PWD/.agents/bench/skill/decode"
export RUN_DIR="$PWD/_local/decode-custom95"
python "$BENCH/controller.py" --out "$RUN_DIR" --model "$MODEL_PATH" \
  --leader "$D_MASTER_HOST" --port-base 32051 --world-size 16 \
  --context 131072 --output 1152 --rounds 2 --batches 1 2 4 8 16 32 \
  --acceptance-mode fixed --profile-batches
```

两台D在新shell中显式选择编译器；已核构建为GCC12.4，不能依赖旧venv激活脚本暗中注入路径。例：

```bash
export CC="$(command -v gcc)" CXX="$(command -v g++)"
"$CC" --version
"$CXX" --version
```

若系统编译器不同，改为本机已安装兼容工具链的绝对路径，并记录实际版本。

控制器常驻后，在两台D分别运行（`NODE_RANK`为0或1；`RUN_DIR`各自新建且同用途，`VENV`为本机环境）：

```bash
RUN_DIR="$RUN_DIR" MODEL_PATH="$MODEL_PATH" \
  bash "$BENCH/engine.sh" w4a8 "$NODE_RANK" custom95 256 fixed \
  16 2 "$D_MASTER_HOST" 32051 128 "$VENV" 7200
```

入口显式设置SM120 DeepEP=1、Elastic FP8/BF16传输、capacity128、shared-overlap1、Indexer两个FP4开关0，以及fixed6；清除Prefill观察、旧物理路由模拟与host trace开关。`engine.sh --print-env`接同样参数可只检查实际子进程环境，不启动GPU；正式运行另用新目录。

网络/工具链环境沿[部署配置](../../../deploy/w4a8/README.md)显式准备；一GPU一NIC由部署层配置，测量器不改网络拓扑。源码模块 `ModelExecutor.execute_forward_op`、`ModelExecutionResult.sync` 的当前接口是观察器依赖；`sitecustomize.py`只在 `DECODE_BENCH_OBSERVE=1`时捕获原生已有完成等待，不新增GPU同步或profiler。原始记录保留在每台D的`RUN_DIR/steps-*.jsonl`，同时保存实际EP NIC和NVSHMEM HCA选择；节点按真实拓扑显式传`TOKENSPEED_SM120_DEEPEP_NICS`，不照搬别的机器编号。`inputs.py`的冻结seed字符串确保三轮尾部独立，完整输入hash和实际cache另行核验。

两入口拒绝旧`TOKENSPEED_HOST_BINARY`覆盖；正式启动前保存当前源码树、实际`ts-host`路径与SHA、runtime与native身份。两机完成后将各自`RUN_DIR`复制到控制机不同目录，保留配置和冻结源码：

```bash
python "$BENCH/audit.py" --run-dir "$RUN_D0" "$RUN_D1" \
  --world-size 16 --out "$AUDIT_JSON"
```

审核要求全部16rank、两正式轮/每B、6B真实输入、每请求6输出、连续完成点与窗口内无编译。分数按`step_ms`直接报告；如画DeepInfer风格吞吐，显式假定接受率a=.75、τ=1+5a=4.75：令t=ms/step÷1000，每用户τ/t、每GPU Bτ/t，不能冒称实测接受率。精度回归必须退出本测试进程并清除模拟接受环境，用真实DSPARK启动。

`test_sampling.py`验证真实输入生成及窗口拒绝条件；运行 `python -m unittest discover -s "$BENCH" -p 'test_*.py'`。
