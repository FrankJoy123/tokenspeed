#!/usr/bin/env bash
set -Eeuo pipefail
print_env=0
if [[ "${1:-}" == --print-env ]]; then print_env=1; shift; fi
[[ $# -eq 12 ]] || { echo "Expected variant node_rank label max_batch acceptance_mode world_size nnodes leader port_base capacity venv timeout" >&2; exit 2; }
variant=$1
node_rank=$2
label=$3
max_batch=$4
acceptance_mode=$5
world_size=$6
nnodes=$7
leader=$8
port_base=$9
transport_capacity=${10}
runtime_venv=${11}
distributed_timeout=${12}
point=${RUN_DIR:?Set a fresh absolute per-node RUN_DIR}
[[ "$world_size" == 16 && "$nnodes" == 2 && ( "$node_rank" == 0 || "$node_rank" == 1 ) ]]
[[ "$acceptance_mode" == fixed ]]
source "$runtime_venv/bin/activate"
[[ -z "${TOKENSPEED_HOST_BINARY:-}" ]] || { echo "Clear TOKENSPEED_HOST_BINARY; use this checkout's built host" >&2; exit 2; }
unset TOKENSPEED_HOST_BINARY
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model=${MODEL_PATH:?Set the W4A8 checkpoint path}

export TOKENSPEED_EXPERIMENT_SM120_DEEPEP=1
export TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP=1
export TOKENSPEED_CUDA_ARCH=120a FLASHINFER_CUDA_ARCH_LIST=12.0a
export TOKENSPEED_DISABLE_PLUGINS=qwen_sm120 TOKENSPEED_KERNEL_DISABLE_PLUGINS=sm120
export TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER=0 TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER=0
export TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER=1 TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT=1
unset PREFILL_BENCH_OBSERVE TOKENSPEED_PREFILL_BENCHMARK TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING
unset TOKENSPEED_MOE_ROUTING_SIMULATION TOKENSPEED_TRACE_HOST_STAGES
unset TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK
: "${CC:?Set CC to an explicit executable C compiler}"
: "${CXX:?Set CXX to an explicit executable C++ compiler}"
test -x "$CC"
test -x "$CXX"
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" DECODE_BENCH_OBSERVE=1 DECODE_BENCH_RUN="$point"
export DG_SM120_JIT_CACHE_DIR="$point/cache/deepgemm-sm120"
mkdir -p "$DECODE_BENCH_RUN"
python - "$script_dir" "$point" <<'PYFREEZE'
import hashlib,json,sys
from pathlib import Path
source,point=map(Path,sys.argv[1:])
assert not list(point.glob('steps-*.jsonl')), 'Choose a fresh run directory'
(point/'source-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},indent=2)+'\n')
PYFREEZE
unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
case "$variant" in
 w4a8) export TOKENSPEED_SM120_DEEPEP_WIRE=elastic_fp8_bf16;activation=(--moe-mxfp4-fp8-activation) ;;
 *) exit 2 ;;
esac
if [[ "$acceptance_mode" == fixed ]];then
 export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=6
else exit 2;fi
export TOKENSPEED_SM120_DEEPEP_CAPACITY="$transport_capacity" NVSHMEM_QP_DEPTH=1024
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
if [[ "$print_env" == 1 ]]; then
 python - <<'PYENV'
import json,os
keys=['TOKENSPEED_EXPERIMENT_SM120_DEEPEP','TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP',
      'TOKENSPEED_SM120_DEEPEP_WIRE','TOKENSPEED_SM120_DEEPEP_CAPACITY',
      'TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER','TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT',
      'TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER','TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER',
      'TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN','TOKENSPEED_MOE_ROUTING_SIMULATION',
      'TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING','TOKENSPEED_PREFILL_BENCHMARK',
      'PREFILL_BENCH_OBSERVE','DECODE_BENCH_OBSERVE','TOKENSPEED_TRACE_HOST_STAGES',
      'TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK']
print(json.dumps({key:os.environ.get(key) for key in keys}))
PYENV
 exit 0
fi
python - "$point/run-identity.json" <<'PYINDEXER'
import json, os, sys
from pathlib import Path

identity = json.loads(Path(sys.argv[1]).read_text())["indexer"]
for key in ("TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER", "TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT",
            "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER", "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER"):
    if identity.get(key) != os.environ.get(key):
        raise SystemExit(f"{key} differs from the frozen run identity")
PYINDEXER
"$VIRTUAL_ENV/bin/python" "$script_dir/../../../../deploy/launch/preflight_model.py" \
 --model "$model" --profile w4a8 --output "$point/model-identity.json"
"$VIRTUAL_ENV/bin/python" "$script_dir/../../../../deploy/setup/identity.py" \
 --source "${RUNTIME_SOURCE:-$script_dir/../../../../..}" --output "$point/runtime-identity.json"
exec "$VIRTUAL_ENV/bin/python" -m tokenspeed.runtime.entrypoints.engine \
 --distributed-timeout-seconds "$distributed_timeout" \
 --model "$model" --trust-remote-code --dtype bfloat16 \
 --dense-gemm-backend deepgemm --moe-backend flashinfer "${activation[@]}" \
 --tensor-parallel-size 1 --dense-tp-size 1 --moe-tp-size 1 --attn-head-tp-size 1 \
 --query-context-parallel-size 1 --kv-parallel-size 1 --data-parallel-size "$world_size" --ep-size "$world_size" \
 --world-size "$world_size" --nnodes "$nnodes" --node-rank "$node_rank" --dist-init-addr "$leader:$((port_base+3))" \
 --disaggregation-mode null --data-parallel-address "$leader" --data-parallel-rpc-port "$port_base" \
 --max-model-len 135168 --max-num-seqs "$((max_batch*world_size))" --gpu-memory-utilization 0.85 \
 --engram-host-table --engram-host-table-layout shared --engram-host-table-dir /dev/shm \
 --disable-prefill-graph --disable-kvstore --disable-pdl --force-deterministic-rsag \
 --speculative-algorithm DSPARK --speculative-num-draft-tokens 6 --speculative-num-steps 5 \
 --seed 968340402 --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
 --max-cudagraph-capture-size "$max_batch" --cudagraph-capture-sizes $(python -c 'import sys; print(" ".join(str(x) for x in [1,2,4,8,16,32,64,128,256] if x<=int(sys.argv[1])))' "$max_batch") \
 --weight-loader-prefetch-num-threads 128 --attention-backend deepseek_v41 \
 --comm-fusion-max-num-tokens 4096 --decode-log-interval 100000 --enable-async-logging
