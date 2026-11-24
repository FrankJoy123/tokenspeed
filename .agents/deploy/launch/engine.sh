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
pd_profile=${W4A8_PD_PROFILE:-legacy}
case "$profile:$pd_profile" in
 w4a8:qcp8-compact|w4a8:dp8-eplb|w4a8:legacy|fp4:legacy) ;;
 *) echo 'W4A8_PD_PROFILE must be legacy, qcp8-compact, or dp8-eplb (the last two require W4A8)' >&2; exit 2 ;;
esac
quant=()
case "$profile" in
 w4a8) quant=(--moe-mxfp4-fp8-activation); export TOKENSPEED_SM120_DEEPEP_WIRE=elastic_fp8_bf16 ;;
 fp4) export TOKENSPEED_SM120_DEEPEP_WIRE=elastic_fp8_bf16 ;;
 *) echo 'profile must be w4a8 or fp4' >&2; exit 2 ;;
esac
common=(--model "$MODEL_PATH" --served-model-name deepseek-v41-flash --trust-remote-code
 --dtype bfloat16 --dense-gemm-backend deepgemm --moe-backend flashinfer
 --dense-tp-size 1 --moe-tp-size 1 --attn-head-tp-size 1 --kv-parallel-size 1
 --max-model-len 135168
 --engram-host-table --engram-host-table-layout shared
 --disable-prefill-graph --disable-kvstore --force-deterministic-rsag
 --attention-backend deepseek_v41
 --speculative-algorithm DSPARK --speculative-num-draft-tokens 6 --speculative-num-steps 5
 --seed 968340402 --disaggregation-transfer-backend mooncake --disaggregation-layerwise-interval 0
 --data-parallel-address "$GATEWAY_HOST" --zmq-engine-index 0 --host "$NODE_HOST")
gpu_memory=0.85
max_seqs=512
execution=(--disable-pdl --comm-fusion-max-num-tokens 4096)
eplb=()
if [[ "$pd_profile" == dp8-eplb ]]; then
 eplb=(--enable-eplb --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static
  --ep-num-redundant-experts 16 --eplb-rebalance-layers-per-chunk 4)
fi
case "$role" in
 prefill)
  : "${P_DIST_INIT_ADDR:?set P_DIST_INIT_ADDR}" "${P_HANDSHAKE_PORT:?set P_HANDSHAKE_PORT}"
  export TOKENSPEED_SM120_DEEPEP_CAPACITY=2048
  tp=8
  dp=1
  chunk=16640
  placement=()
  if [[ "$pd_profile" == qcp8-compact ]]; then
   chunk=33792
   gpu_memory=0.98
   max_seqs=192
   execution=(--enforce-eager --disable-overlap-schedule --disable-autotune --engram-host-table-dir /dev/shm
    --all2all-backend none --kvstore-write-policy write_through --max-total-tokens 262144
    --retraction-snapshot-ratio 0.0 --distributed-timeout-seconds 7200
    --enable-async-logging --enable-output-logprobs)
   placement_file=${QCP_EXPERT_PLACEMENT:-$repo/.agents/deploy/w4a8/launch/r8-physical-to-logical.json}
   [[ -r "$placement_file" ]] || { echo 'QCP expert placement file is unreadable' >&2; exit 2; }
   placement=(--ep-num-redundant-experts 8 --ep-dispatch-algorithm static --init-expert-location "$placement_file")
   export TOKENSPEED_ENABLE_SM120_PAGED_PREFILL=1 TOKENSPEED_ENABLE_SM120_QCP_PAGED_PREFILL=1
   export TOKENSPEED_SM120_PREFILL_COMPACT_BF16X2=1 TOKENSPEED_SM120_QCP_COMPACT_PREFILL=1
   export TOKENSPEED_SM120_PREFILL_COMPACT_MIN_QUERY_ROWS=528 TOKENSPEED_EXPERIMENT_QCP_CONTIGUOUS=1
   export TOKENSPEED_SM120_PREFILL_CAKE_ROWS=4224 TOKENSPEED_SM120_QCP_CAKE_CED=1
   export TOKENSPEED_ENABLE_SM120_MHC_MIXES=1 TOKENSPEED_SM120_MHC_FUSED=1
   export TOKENSPEED_SM120_FP8_GROUP32_BATCHED=1 TOKENSPEED_PREFILL_INDEX_LOGITS_MB=2048
   export TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER=1 TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER=1
  fi
  if [[ "$pd_profile" == dp8-eplb ]]; then
   tp=1
   dp=8
   chunk=4224
   gpu_memory=0.95
   export TOKENSPEED_ENABLE_SM120_PAGED_PREFILL=1 TOKENSPEED_SM120_PREFILL_COMPACT_BF16X2=1
   export TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER=1 TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER=1
   export TOKENSPEED_SM120_PREFILL_CAKE_ROWS=1024 TOKENSPEED_SM120_EPLB_FUSED_RECORD=0
   eplb+=(--eplb-rebalance-num-iterations 1000)
  fi
  args=(--tensor-parallel-size "$tp" --query-context-parallel-size "$tp" --data-parallel-size "$dp" --ep-size 8
   --disaggregation-mode prefill --disaggregation-bootstrap-port 18998
   --data-parallel-rpc-port "$P_HANDSHAKE_PORT" --dist-init-addr "$P_DIST_INIT_ADDR"
   --chunked-prefill-size "$chunk" --max-prefill-tokens "$chunk" --enable-prefix-caching --port 18501
   "${placement[@]}")
  ;;
 decode)
  : "${D_DIST_INIT_ADDR:?set D_DIST_INIT_ADDR}" "${D_HANDSHAKE_PORT:?set D_HANDSHAKE_PORT}" "${NODE_RANK:?set NODE_RANK}"
  [[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || { echo 'NODE_RANK must be 0 or 1' >&2; exit 2; }
  export TOKENSPEED_SM120_DEEPEP_CAPACITY=128
  export TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP=1
  capture_sizes=(1 2 4 8 16 32 64 128 256)
  if [[ "$pd_profile" != legacy ]]; then
   export TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT=1 TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER=1
   export TOKENSPEED_EXPERIMENT_SM120_FUSED_SCALE_PACK=1 TOKENSPEED_EXPERIMENT_SM120_QUANT_PACK=1
   export TOKENSPEED_SM120_CAKE_DECODE_MLA=1 TOKENSPEED_SM120_CAKE_DECODE_MLA_WIDE=1
   export TOKENSPEED_SM120_EPLB_FUSED_RECORD=1
  fi
  if [[ "$pd_profile" == dp8-eplb ]]; then
   export TOKENSPEED_SM120_PREFILL_CAKE_ROWS=0
   eplb+=(--eplb-rebalance-num-iterations 10000)
  fi
  if [[ "$pd_profile" == qcp8-compact ]]; then
   max_seqs=128
   capture_sizes=(1 2 4 8 16)
   execution+=(--engram-host-table-dir /dev/shm)
   eplb=(--enable-eplb --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static
    --ep-num-redundant-experts 16 --eplb-rebalance-num-iterations 2048
    --eplb-rebalance-layers-per-chunk 4)
  fi
  args=(--tensor-parallel-size 1 --query-context-parallel-size 1 --data-parallel-size 16 --world-size 16 --ep-size 16
   --nnodes 2 --node-rank "$NODE_RANK" --disaggregation-mode decode
   --data-parallel-rpc-port "$D_HANDSHAKE_PORT" --dist-init-addr "$D_DIST_INIT_ADDR"
   --chunked-prefill-size 8192 --max-prefill-tokens 8192 --disable-prefix-caching
   --max-cudagraph-capture-size "${capture_sizes[-1]}" --cudagraph-capture-sizes "${capture_sizes[@]}" --port 18502)
  ;;
 *) echo 'role must be prefill or decode' >&2; exit 2 ;;
esac
test -x "$VENV/bin/python" || { echo "VENV has no executable bin/python" >&2; exit 2; }
test -x "$CUDA_HOME/bin/ptxas" || { echo "CUDA_HOME has no executable bin/ptxas" >&2; exit 2; }
"$VENV/bin/python" "$repo/.agents/deploy/launch/preflight_model.py" --model "$MODEL_PATH" --profile "$profile"
command=("$VENV/bin/python" -m tokenspeed.runtime.entrypoints.engine "${common[@]}" "${quant[@]}" "${args[@]}"
 --gpu-memory-utilization "$gpu_memory" --max-num-seqs "$max_seqs" "${execution[@]}" "${eplb[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
mkdir -p "$STATE_DIR/setup" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$CUTE_DSL_CACHE_DIR" "$DG_JIT_CACHE_DIR" "$EP_JIT_CACHE_DIR"
"$VENV/bin/python" "$repo/.agents/deploy/launch/preflight_model.py" --model "$MODEL_PATH" --profile "$profile" --output "$STATE_DIR/setup/model-identity.json"
exec "${command[@]}"
