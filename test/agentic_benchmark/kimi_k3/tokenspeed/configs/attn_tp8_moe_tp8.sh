#!/usr/bin/bash

set -euo pipefail

(( $# == 0 )) || { echo "This fixed recipe accepts no positional arguments" >&2; exit 2; }

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --served-model-name nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --moe-tp-size 8 \
    --max-model-len 1048576 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.92 \
    --seed 0 \
    --disable-cuda-graph-padding \
    --max-cudagraph-capture-size 1024 \
    --trust-remote-code \
    --language-model-only \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lightseekorg/kimi-k3-eagle3-mla \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --drafter-attention-backend tokenspeed_mla \
    --sampling-backend greedy \
    --synthetic-acceptance-length 3.7 \
    --stream-interval 10 \
    --disable-kvstore \
    --enable-metrics \
    --metrics-host 0.0.0.0 \
    --metrics-port 8010 \
    --reasoning-parser passthrough \
    --tool-call-parser passthrough \
    --host 0.0.0.0 \
    --port 8000 \
    --engine-startup-timeout 7200
