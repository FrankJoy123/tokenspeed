#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$#" -lt 2 ]]; then
 echo "usage: engine.sh w4a8|fp4 prefill|decode [engine args...]" >&2
 exit 2
fi
profile=$1
role=$2
shift 2
: "${VENV:?set VENV}" "${STATE_DIR:?set STATE_DIR}" "${MODEL_PATH:?set MODEL_PATH}"
: "${CUDA_HOME:?set CUDA_HOME}" "${GATEWAY_HOST:?set GATEWAY_HOST}" "${NODE_HOST:?set NODE_HOST}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${TOOLCHAIN_LIBRARY_DIR:+$TOOLCHAIN_LIBRARY_DIR:}$VENV/lib/python3.12/site-packages/torch/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENSPEED_CUDA_ARCH=120a FLASHINFER_CUDA_ARCH_LIST=12.0a
export TOKENSPEED_DISABLE_PLUGINS=qwen_sm120 TOKENSPEED_KERNEL_DISABLE_PLUGINS=sm120
export TOKENSPEED_EXPERIMENT_SM120_DEEPEP=1
export TOKENSPEED_ENABLE_SM120_PREFILL_INDEXER=1
export TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER=0 TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER=0
export TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER=0 TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP=0
if [[ -n "${TOKENSPEED_HOST_BINARY:-}" ]]; then
 echo 'Unset TOKENSPEED_HOST_BINARY: this profile requires the host built with the matched engine wheel' >&2
 exit 2
fi
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_CACHE_DIR="$STATE_DIR/cache/triton" CUDA_CACHE_PATH="$STATE_DIR/cache/cuda"
export CUTE_DSL_CACHE_DIR="$STATE_DIR/cache/cute" TOKENSPEED_KERNEL_CACHE_ROOT="$STATE_DIR/cache/kernel"
export DG_JIT_CACHE_DIR="$STATE_DIR/cache/deepgemm" EP_JIT_CACHE_DIR="$STATE_DIR/cache/deepep"
export DG_SM120_JIT_CACHE_DIR="$STATE_DIR/cache/deepgemm-sm120"
unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN TOKENSPEED_MOE_ROUTING_SIMULATION
unset TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING DECODE_BENCH_OBSERVE DECODE_BENCH_RUN PREFILL_BENCH_OBSERVE
quant=()
case "$profile" in
 w4a8) quant=(--moe-mxfp4-fp8-activation); export TOKENSPEED_SM120_DEEPEP_WIRE=elastic_fp8_bf16 ;;
 fp4) export TOKENSPEED_SM120_DEEPEP_WIRE=elastic_fp8_bf16 ;;
 *) echo 'profile must be w4a8 or fp4' >&2; exit 2 ;;
esac
common=(--model "$MODEL_PATH" --served-model-name deepseek-v41-flash --trust-remote-code
 --dtype bfloat16 --dense-gemm-backend deepgemm --moe-backend flashinfer
 --dense-tp-size 1 --moe-tp-size 1 --attn-head-tp-size 1 --kv-parallel-size 1
 --max-model-len 135168 --max-num-seqs 512 --gpu-memory-utilization 0.85
 --engram-host-table --engram-host-table-layout shared
 --disable-prefill-graph --disable-kvstore --disable-pdl --force-deterministic-rsag
 --attention-backend deepseek_v41 --comm-fusion-max-num-tokens 4096
 --speculative-algorithm DSPARK --speculative-num-draft-tokens 6 --speculative-num-steps 5
 --seed 968340402 --disaggregation-transfer-backend mooncake --disaggregation-layerwise-interval 0
 --data-parallel-address "$GATEWAY_HOST" --zmq-engine-index 0 --host "$NODE_HOST")
case "$role" in
 prefill)
  : "${P_DIST_INIT_ADDR:?set P_DIST_INIT_ADDR}" "${P_HANDSHAKE_PORT:?set P_HANDSHAKE_PORT}"
  export TOKENSPEED_SM120_DEEPEP_CAPACITY=2048
  args=(--tensor-parallel-size 8 --query-context-parallel-size 8 --data-parallel-size 1 --ep-size 8
   --disaggregation-mode prefill --disaggregation-bootstrap-port 18998
   --data-parallel-rpc-port "$P_HANDSHAKE_PORT" --dist-init-addr "$P_DIST_INIT_ADDR"
   --chunked-prefill-size 16640 --max-prefill-tokens 16640 --enable-prefix-caching --port 18501)
  ;;
 decode)
  : "${D_DIST_INIT_ADDR:?set D_DIST_INIT_ADDR}" "${D_HANDSHAKE_PORT:?set D_HANDSHAKE_PORT}" "${NODE_RANK:?set NODE_RANK}"
  [[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || { echo 'NODE_RANK must be 0 or 1' >&2; exit 2; }
  export TOKENSPEED_SM120_DEEPEP_CAPACITY=128
  export TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP=1
  args=(--tensor-parallel-size 1 --query-context-parallel-size 1 --data-parallel-size 16 --world-size 16 --ep-size 16
   --nnodes 2 --node-rank "$NODE_RANK" --disaggregation-mode decode
   --data-parallel-rpc-port "$D_HANDSHAKE_PORT" --dist-init-addr "$D_DIST_INIT_ADDR"
   --chunked-prefill-size 8192 --max-prefill-tokens 8192 --disable-prefix-caching
   --max-cudagraph-capture-size 256 --cudagraph-capture-sizes 1 2 4 8 16 32 64 128 256 --port 18502)
  ;;
 *) echo 'role must be prefill or decode' >&2; exit 2 ;;
esac
test -x "$VENV/bin/python" || { echo "VENV has no executable bin/python" >&2; exit 2; }
test -x "$CUDA_HOME/bin/ptxas" || { echo "CUDA_HOME has no executable bin/ptxas" >&2; exit 2; }
"$VENV/bin/python" "$repo/.agents/deploy/launch/preflight_model.py" --model "$MODEL_PATH" --profile "$profile"
command=("$VENV/bin/python" -m tokenspeed.runtime.entrypoints.engine "${common[@]}" "${quant[@]}" "${args[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
mkdir -p "$STATE_DIR/setup" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$CUTE_DSL_CACHE_DIR" "$DG_JIT_CACHE_DIR" "$EP_JIT_CACHE_DIR"
"$VENV/bin/python" "$repo/.agents/deploy/launch/preflight_model.py" --model "$MODEL_PATH" --profile "$profile" --output "$STATE_DIR/setup/model-identity.json"
exec "${command[@]}"
