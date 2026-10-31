#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
[[ "$ROLE" == decode ]] || { echo 'ROLE must be decode' >&2; exit 2; }
require MODEL_PATH D_HOST GATEWAY_HOST EP_NIC_NAME NCCL_IB_HCA D_DEEPGEMM_SOURCE
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 for model_file in config.json tokenizer_config.json model.safetensors.index.json encoding/encoding.py; do
  [[ -f "$MODEL_PATH/$model_file" ]] || { echo "Incomplete model cache: missing $model_file under MODEL_PATH; copy the complete model directory recursively" >&2; exit 2; }
 done
 [[ -d "$MODEL_PATH" && -d "$D_DEEPGEMM_SOURCE" ]] || { echo 'Missing model or SM120 DeepGEMM source' >&2; exit 2; }
fi
export TOKENSPEED_KERNEL_OVERRIDE_TRANSFORM_HADAMARD_TRANSFORM=triton_hadamard_transform_128
export TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES=triton_mhc_mixes
export TOKENSPEED_EXPERIMENT_SM120_DEEPEP=1 TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER=0
# DeepGEMM is installed from sources.tsv into this venv, including headers.
if [[ -n "${SIMULATED_ACCEPT_LEN:-}" ]]; then
 export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN="$SIMULATED_ACCEPT_LEN"
else
 unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
fi
sizes=(); for ((i=1;i<=MAX_NUM_SEQS;i++)); do sizes+=("$i"); done
run "$VENV/bin/python" -m tokenspeed.cli serve --headless --dense-gemm-backend deepgemm \
 --model "$MODEL_PATH" --served-model-name deepseek-v41-flash --trust-remote-code \
 --tensor-parallel-size 1 --data-parallel-size 8 --world-size 8 --ep-size 8 \
 --moe-backend flashinfer --moe-mxfp4-fp8-activation --dtype bfloat16 --seed 968340402 \
 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --chunked-prefill-size "$D_CHUNK_SIZE" \
 --decode-log-interval 1 --max-cudagraph-capture-size "$MAX_NUM_SEQS" --cudagraph-capture-sizes "${sizes[@]}" \
 --speculative-algorithm DSPARK --disable-prefill-graph --disable-kvstore --engram-host-table \
 --engram-host-table-layout shared --disable-pdl --force-deterministic-rsag \
 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --disable-weight-loader-prefetch-checkpoints \
 --attention-backend deepseek_v41 --comm-fusion-max-num-tokens 4096 \
 --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-layerwise-interval 0 \
 --disable-prefix-caching --disaggregation-ib-device "$NCCL_IB_HCA" \
 --data-parallel-address "$GATEWAY_HOST" --data-parallel-rpc-port 18511 --zmq-engine-index 0 \
 --host "$D_HOST" --port 18347 --dist-init-addr 127.0.0.1:13580 "$@"
