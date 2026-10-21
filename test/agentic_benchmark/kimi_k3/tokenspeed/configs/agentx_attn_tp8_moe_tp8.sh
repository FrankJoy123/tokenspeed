#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Fixed EAGLE3 AgentX full-trace recipe; checkpoint paths and AL are inputs.
set -euo pipefail

(( $# == 0 )) || { echo "Use the named recipe inputs; trailing flags are unsupported" >&2; exit 2; }

COMMAND=(ts serve
    --model "${MODEL_DIR:?model checkpoint}" --served-model-name "$MODEL_NAME"
    --attn-tp-size 8 --moe-tp-size 8
    --max-model-len 1048576 --max-num-seqs 16
    --gpu-memory-utilization 0.92 --seed 0
    --disable-cuda-graph-padding --max-cudagraph-capture-size 1024
    --trust-remote-code --language-model-only
    --attention-backend tokenspeed_mla --kda-backend cutedsl_kda
    --moe-backend flashinfer_trtllm --kv-cache-dtype fp8
    --mamba-ssm-dtype bfloat16 --prefix-granularity 128
    --chunked-prefill-size 8192 --max-prefill-tokens 8192
    --speculative-algorithm EAGLE3
    --speculative-num-steps 3 --speculative-num-draft-tokens 4
    --speculative-draft-model-path "${DRAFT_DIR:?draft checkpoint}"
    --speculative-eagle-topk 1 --drafter-attention-backend tokenspeed_mla
    --sampling-backend greedy --synthetic-acceptance-length "${AL_TARGET:?fixed acceptance length}"
    --mm-encoder-tp-mode data --reasoning-parser passthrough --tool-call-parser passthrough
    --stream-interval 10 --enable-log-request-stats
    --disable-kvstore --engine-startup-timeout 7200
    --host 0.0.0.0 --port "$API_PORT"
    --enable-metrics --metrics-host 0.0.0.0 --metrics-port "$METRICS_PORT"
)
# One command, with expanded arguments, is both recorded and executed.
if [[ "${SLURM_PROCID:-0}" = 0 ]]; then
    printf "%q " "${COMMAND[@]}" > "${RUN_ROOT:?output directory}/server-command.txt"
    printf "\n" >> "$RUN_ROOT/server-command.txt"
fi
printf "%q " "${COMMAND[@]}"
printf "\n"
exec "${COMMAND[@]}"
