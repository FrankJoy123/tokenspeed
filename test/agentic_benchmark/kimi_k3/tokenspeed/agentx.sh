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

set -eo pipefail

# Share input validation with the launcher so invalid settings fail before startup.
validate_agentx_values() {
    local variable
    for variable in "$@"; do
        [[ -n "${!variable:-}" ]] || { echo "Missing $variable" >&2; return 2; }
        # The pinned native replay builder splits its command on whitespace.
        [[ "${!variable}" != *[[:space:]]* ]] || { echo "$variable must not contain whitespace" >&2; return 2; }
        export "$variable"
    done
}

validate_agentx_config() {
    validate_agentx_values CLIENT_ROOT TOKENIZER_PATH MODEL_NAME API_PORT CONCURRENCY \
        DURATION BENCHMARK_GRACE_SECONDS REQUEST_TIMEOUT_SECONDS CLIENT_SEED \
        AIPERF_WARMUP_REQUESTS_PER_LANE AGENTIC_WARMUP_GRACE_PERIOD \
        AIPERF_TRACE_IDLE_GAP_CAP_SECONDS AIPERF_FAILED_REQUEST_THRESHOLD \
        AIPERF_EXTRA_INPUTS AIPERF_UNSAFE_OVERRIDE AGENTX_PROTOCOL ACCEPTANCE_MODE \
        SPEC_DECODING SPEC_STEPS PRECISION MAX_MODEL_LEN MAX_NUM_SEQS SERVER_METRICS \
        INFERENCEX_WORKSPACE || return "$?"
    case "$AGENTX_PROTOCOL" in
        weka-062126-256k) export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k ;;
        weka-062126-full) export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126 ;;
        *) echo "Unknown AGENTX_PROTOCOL: $AGENTX_PROTOCOL" >&2; return 2 ;;
    esac
    case "$SERVER_METRICS" in
        true) validate_agentx_values METRICS_PORT || return "$?" ;;
        false) ;;
        *) echo "SERVER_METRICS must be true or false" >&2; return 2 ;;
    esac
    case "$ACCEPTANCE_MODE" in
        natural) [[ -z "${AL_TARGET:-}" ]] || { echo "AL_TARGET must be empty for natural acceptance" >&2; return 2; } ;;
        synthetic) validate_agentx_values AL_TARGET || return "$?" ;;
        *) echo "ACCEPTANCE_MODE must be natural or synthetic" >&2; return 2 ;;
    esac
    local variable context
    for variable in MAX_MODEL_LEN MAX_NUM_SEQS SPEC_STEPS API_PORT; do
        [[ "${!variable}" =~ ^[1-9][0-9]{0,8}$ ]] || { echo "$variable must be a positive integer" >&2; return 2; }
    done
    [[ "$API_PORT" -le 65535 ]] || { echo "Invalid API_PORT" >&2; return 2; }
    context=262144
    if [[ "$AGENTX_PROTOCOL" = weka-062126-full ]]; then context=1048576; fi
    [[ "$MAX_MODEL_LEN" = "$context" ]] || { echo "MAX_MODEL_LEN must be $context for $AGENTX_PROTOCOL" >&2; return 2; }
    case "$SPEC_DECODING" in
        eagle3|dspark) ;;
        *) echo "SPEC_DECODING must be eagle3 or dspark" >&2; return 2 ;;
    esac
    if [[ "$ACCEPTANCE_MODE" = synthetic ]]; then
        [[ "$AL_TARGET" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
            awk -v al="$AL_TARGET" -v steps="$SPEC_STEPS" 'BEGIN { exit !(al >= 1 && al <= steps + 1) }' ||
            { echo "AL_TARGET must be between 1 and SPEC_STEPS + 1" >&2; return 2; }
    fi
    if [[ "$SERVER_METRICS" = true ]]; then
        [[ "$METRICS_PORT" =~ ^[1-9][0-9]{0,4}$ ]] &&
            [[ "$METRICS_PORT" -le 65535 && "$METRICS_PORT" != "$API_PORT" ]] ||
            { echo "METRICS_PORT must be a valid port different from API_PORT" >&2; return 2; }
    fi
    case "$AIPERF_UNSAFE_OVERRIDE" in
        true|false) ;;
        *) echo "AIPERF_UNSAFE_OVERRIDE must be true or false; got '$AIPERF_UNSAFE_OVERRIDE'" >&2; return 2 ;;
    esac
}

# Sourcing exposes validation only; replay runs when this script is executed.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
validate_agentx_config
validate_agentx_values RESULT_DIR EP_SIZE AIPERF_SERVER_URL
mkdir "$RESULT_DIR"
export INFMAX_CONTAINER_WORKSPACE="$INFERENCEX_WORKSPACE"
export AIPERF_RUNTIME_DIR="$CLIENT_ROOT"
export PYTHONPATH="$INFMAX_CONTAINER_WORKSPACE"
export MODEL="$MODEL_NAME" SERVED_MODEL_NAME="$MODEL_NAME" MODEL_PREFIX=kimik3
export PORT="$API_PORT" CONC="$CONCURRENCY"
export FRAMEWORK=tokenspeed RUNNER_TYPE=gb300 PRECISION SPEC_DECODING
export TP=8 PP_SIZE=1 PCP_SIZE=1 DCP_SIZE=1 IS_MULTINODE=true DISAGG=false KV_OFFLOADING=none
export AIPERF_PYTHON_VERSION=3.11 AIPERF_EXPERIMENTAL_FAST=0
export AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
export AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=3600
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD="$AIPERF_FAILED_REQUEST_THRESHOLD"
if [[ "$SERVER_METRICS" = true ]]; then
    validate_agentx_values AIPERF_SERVER_METRICS_URLS
    export AIPERF_REQUIRED_SERVER_METRIC_PREFIX=tokenspeed:
else
    export AIPERF_SERVER_METRICS_URLS= AIPERF_REQUIRED_SERVER_METRIC_PREFIX=
fi
export ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 EVAL_ONLY=false
export AGENTIC_OUTPUT_DIR="$RESULT_DIR" RESULT_FILENAME=inferencex_result
export AIPERF_DATASET_WEKA_PARALLEL_WORKERS=2 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export AGENTX_CONTEXT_LENGTH="$MAX_MODEL_LEN"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
AIPERF_DEPS_READY=1
resolve_trace_source
# The native library clears inherited context; restore the explicit protocol cap.
export MAX_MODEL_LEN
MAX_MODEL_LEN="$AGENTX_CONTEXT_LENGTH"
MODEL="$TOKENIZER_PATH" build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --benchmark-grace-period $BENCHMARK_GRACE_SECONDS --request-timeout-seconds $REQUEST_TIMEOUT_SECONDS --random-seed $CLIENT_SEED"
if [[ "$SERVER_METRICS" = false ]]; then REPLAY_CMD+=" --no-server-metrics"; fi

write_agentic_result_json() {
    # Upstream's multi-node report shape describes PD disaggregation. These two
    # nodes are one aggregate TP8 worker: use native aggregate accounting (8 GPUs).
    # Keep physical topology separately without rewriting native metrics or JSON.
    printf '%s\n' '{"physical_nodes":2,"gpus_per_node":4,"num_gpus":8,"deployment":"aggregate","official_inferencex_result":false}' \
        > "$RESULT_DIR/deployment.json"
    if [[ "$SERVER_METRICS" = true ]]; then
        [[ -s "$RESULT_DIR/aiperf_artifacts/server_metrics_export.json" ]] ||
            { echo "Missing or empty server_metrics_export.json" >&2; return 2; }
    fi
    IS_MULTINODE=false "$AIPERF_PYTHON" -m infx.results.agentic.process_agentic_result
}
run_agentic_replay_and_write_outputs "$RESULT_DIR"
