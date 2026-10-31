#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
[[ "$ROLE" == prefill ]] || { echo 'ROLE must be prefill' >&2; exit 2; }
require MODEL_PATH P_HOST GATEWAY_HOST P_DEEPEP_SOURCE P_FLASHINFER_SOURCE P_NVSHMEM_LIB_DIR
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 for model_file in config.json tokenizer_config.json model.safetensors.index.json encoding/encoding.py; do
  [[ -f "$MODEL_PATH/$model_file" ]] || { echo "Incomplete model cache: missing $model_file under MODEL_PATH; copy the complete model directory recursively" >&2; exit 2; }
 done
 [[ -d "$MODEL_PATH" && -d "$P_DEEPEP_SOURCE" && -d "$P_FLASHINFER_SOURCE" && -f "$P_NVSHMEM_LIB_DIR/libnvshmem_host.so.3" ]] || { echo 'Missing P model/custom source/NVSHMEM; see dependencies.tsv' >&2; exit 2; }
fi
export PYTHONPATH="$P_DEEPEP_SOURCE:$P_FLASHINFER_SOURCE:$PYTHONPATH"
export LD_PRELOAD="$P_NVSHMEM_LIB_DIR/libnvshmem_host.so.3"
export LD_LIBRARY_PATH="$P_NVSHMEM_LIB_DIR:$LD_LIBRARY_PATH"
export TOKENSPEED_DISABLE_PLUGINS=qwen_sm120 TOKENSPEED_KERNEL_DISABLE_PLUGINS=sm120
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES=triton_sm120_mhc_mixes
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_PRE=triton_mhc_pre
export TS_DEEPEP_ALLOW_MNNVL=0 TS_DEEPEP_NORMAL_TRANSPORT=rdma TS_DEEPEP_RDMA_BUFFER_BYTES=536870912
unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
run "$VENV/bin/python" -m tokenspeed.cli serve --headless \
 --model "$MODEL_PATH" --served-model-name deepseek-v41-flash --trust-remote-code \
 --tensor-parallel-size 8 --query-context-parallel-size 8 --kv-parallel-size 1 \
 --dense-tp-size 1 --moe-tp-size 1 --attn-head-tp-size 1 --ep-size 8 --dtype bfloat16 \
 --moe-backend flashinfer_cutlass_sm120_deepep --moe-mxfp4-fp8-activation \
 --all2all-backend deepep --deepep-mode normal --moe-combine-order rank --init-expert-location trivial \
 --max-model-len "$MAX_MODEL_LEN" --chunked-prefill-size "$P_CHUNK_SIZE" --max-num-seqs "$MAX_NUM_SEQS" \
 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --engram-host-table --engram-host-table-layout shared \
 --speculative-algorithm DSPARK --disable-prefill-graph --disable-kvstore --disable-pdl --force-deterministic-rsag \
 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-layerwise-interval 0 \
 --disaggregation-bootstrap-port 18598 --data-parallel-address "$GATEWAY_HOST" --data-parallel-rpc-port 18510 \
 --zmq-engine-index 0 --host "$P_HOST" --port 18501 --dist-init-addr "$P_HOST:19400" \
 --disable-autotune --enable-prefix-caching --enable-log-request-stats --decode-log-interval 1 \
 --enable-metrics --metrics-port 18521 "$@"
