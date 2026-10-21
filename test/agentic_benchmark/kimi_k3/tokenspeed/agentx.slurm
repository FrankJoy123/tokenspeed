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

# Run via sbatch with a native client, or from the login host in a held job.
# All paths must be absolute and visible where they are used. See agentx.md.
# Shared configs own the fixed server recipe; this script owns deployment and replay.
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch or an existing allocation}"
: "${HF_HUB_CACHE:?shared mounted HF Hub cache}" "${RUN_MODE:?smoke or formal}" \
  "${CLIENT_ROOT:?prepared client venv root}" "${INFERENCEX_WORKSPACE:?InferenceX checkout}"
case "$RUN_MODE" in
    smoke|formal) ;;
    *) echo 'RUN_MODE must be smoke or formal' >&2; exit 2 ;;
esac

# Check the complete sweep before starting any GPU work.
read -r -a CONFIGS <<< "${CONFIG_NAMES:?server configurations}"
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo 'CONFIG_NAMES must not be empty' >&2; exit 2; }
declare -A seen_config=()
for config in "${CONFIGS[@]}"; do
    case "$config" in
        attn_tp8_moe_tp8|attn_tp8_moe_ep8) ;;
        *) echo "Unknown configuration: $config" >&2; exit 2 ;;
    esac
    [[ -z "${seen_config[$config]:-}" ]] || { echo "Duplicate configuration: $config" >&2; exit 2; }
    seen_config[$config]=1
done
read -r -a CONCURRENCY_VALUES <<< "${CONCURRENCIES:?concurrency points}"
[[ ${#CONCURRENCY_VALUES[@]} -gt 0 ]] || { echo 'CONCURRENCIES must not be empty' >&2; exit 2; }
declare -A seen_concurrency=()
for concurrency in "${CONCURRENCY_VALUES[@]}"; do
    case "$concurrency" in
        1|2|4|8|16) ;;
        *) echo "Unsupported concurrency: $concurrency" >&2; exit 2 ;;
    esac
    [[ -z "${seen_concurrency[$concurrency]:-}" ]] || {
        echo "Duplicate concurrency: $concurrency" >&2; exit 2;
    }
    seen_concurrency[$concurrency]=1
done

SERVER_PID=
CLIENT_PID=
CLIENT_SIGNAL=TERM

finish() {
    local result=$?
    # Stop and reap the client before releasing its server. GNU timeout forwards
    # the signal to the client's process group and retains its kill-after bound.
    trap '' INT TERM
    trap - EXIT
    if [[ -n "$CLIENT_PID" ]]; then
        kill -s "$CLIENT_SIGNAL" "$CLIENT_PID" 2>/dev/null || true
        wait "$CLIENT_PID" 2>/dev/null || true
        CLIENT_PID=
    fi
    stop_server
    if [[ -d "$RUN_ROOT" ]]; then printf '%s\n' "$result" > "$RUN_ROOT/client-exit-code.txt"; fi
    exit "$result"
}

cancel_steps() {
    # Never cancel the held allocation or another invocation's steps.
    while read -r step name; do
        if [[ "$name" = "$STEP_NAME" ]]; then
            scancel "$step" || true
        fi
    done < <(squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j')
}

stop_server() {
    cancel_steps
    if [[ -n "$SERVER_PID" ]]; then
        # Signal the tracked launch even before its Slurm step is registered.
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
    # Catch a step that registered while its launch process was being stopped.
    cancel_steps
}

launch_server() {
    scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation.txt"
    mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o '%N')")
    [[ "${#NODES[@]}" = 2 ]] || { echo "K3 TP8 requires two four-GPU nodes" >&2; exit 2; }
    HEAD="${NODES[0]}"
    BASE_URL="http://${HEAD}:8000"
    AIPERF_SERVER_METRICS_URLS="http://${HEAD}:8010/metrics"

    # Bound startup (7500 s), each client (8400 s), drains and cleanup.
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HEAD="$HEAD" RUN_ROOT="$RUN_ROOT" SERVER_VENV="$SERVER_VENV" \
    timeout --signal=TERM --kill-after=45 "$((7500 + (8400 + DRAIN_TIMEOUT) * ${#CONCURRENCY_VALUES[@]} + 120))" \
        srun --jobid="$SLURM_JOB_ID" --overlap --nodes="${#NODES[@]}" --ntasks-per-node=1 \
        --cpus-per-task=64 --gpus-per-node=4 \
        --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,HF_HUB_CACHE \
        --container-image="$CONTAINER_IMAGE" --container-name="agentx_${SLURM_JOB_ID}_$$" \
        --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$SOURCE_ROOT" \
        --container-remap-root --no-container-mount-home --container-writable \
        --job-name="$STEP_NAME" --mpi=pmix --kill-on-bad-exit=1 \
        bash -c '
            set -euo pipefail
            if [[ -n "$SERVER_VENV" ]]; then source "$SERVER_VENV/bin/activate"; fi
            exec bash -x "$1"
        ' agentx-server "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/configs/${CONFIG}.sh" \
        > "$RUN_ROOT/server.log" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    local start=$SECONDS
    until curl --fail --silent --max-time 5 --output /dev/null "$BASE_URL/readiness"; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Server step exited before readiness' >&2; exit 1; }
        (( SECONDS - start < 7500 )) || { echo 'Server readiness timeout' >&2; exit 1; }
        sleep 5
    done
    printf '%s\n' "$BASE_URL" > "$RUN_ROOT/endpoint.txt"
}

run_benchmark() {
    # Run the prepared native client on this host, outside the server container.
    mkdir "$RUN_ROOT/c${CONCURRENCY}"
    case "$CONFIG" in
        attn_tp8_moe_tp8) EP_SIZE=1 ;;
        attn_tp8_moe_ep8) EP_SIZE=8 ;;
    esac
    AIPERF_SERVER_URL="$BASE_URL" AIPERF_SERVER_METRICS_URLS="$AIPERF_SERVER_METRICS_URLS" \
    RESULT_DIR="$RUN_ROOT/c${CONCURRENCY}/client" CONCURRENCY="$CONCURRENCY" EP_SIZE="$EP_SIZE" \
    timeout --signal=INT --kill-after=45 8400 \
        bash -se -o pipefail > "$RUN_ROOT/c${CONCURRENCY}/client.log" 2>&1 <<'CLIENT' &
# InferenceX command-builder inputs: duration and warmup count per lane.
case "${RUN_MODE:?smoke or formal}" in
    smoke) DURATION=180; AIPERF_WARMUP_REQUESTS_PER_LANE=1 ;;
    formal) DURATION=3600; AIPERF_WARMUP_REQUESTS_PER_LANE=10 ;;
    *) echo 'RUN_MODE must be smoke or formal' >&2; exit 2 ;;
esac
# InferenceX helper inputs: source directory, prepared client venv and loader.
mkdir "$RESULT_DIR"
INFMAX_CONTAINER_WORKSPACE="$INFERENCEX_WORKSPACE"
AIPERF_RUNTIME_DIR="$CLIENT_ROOT"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
AIPERF_DEPS_READY=1  # InferenceX: reuse the prepared venv; do not install during a run.
WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126 resolve_trace_source
# The native loader selects the workload; do not impose a client context cap.
unset MAX_MODEL_LEN
# AIPerf reads this in its worker processes too: always replay recorded turns.
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
# InferenceX build_replay_cmd reads these shell variables and builds AIPerf CLI flags.
MODEL=nvidia/Kimi-K3-NVFP4 SERVED_MODEL_NAME=nvidia/Kimi-K3-NVFP4 CONC="$CONCURRENCY" \
FRAMEWORK=tokenspeed AIPERF_EXPERIMENTAL_FAST=0 AIPERF_UNSAFE_OVERRIDE=false \
AGENTIC_WARMUP_GRACE_PERIOD=1800 \
AIPERF_TRACE_IDLE_GAP_CAP_SECONDS=300 AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.1 \
AIPERF_EXTRA_INPUTS='{"ignore_eos":true,"temperature":1.0,"top_p":0.95,"top_k":20}' \
build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --benchmark-grace-period 30 --request-timeout-seconds 21600"

# Override only the native reporting hook for aggregate topology accounting.
write_agentic_result_json() {
    # Upstream's multi-node report shape describes PD disaggregation. These two
    # nodes are one aggregate TP8 worker: use native aggregate accounting (8 GPUs).
    # Keep physical topology separately without rewriting native metrics or JSON.
    printf '%s\n' '{"physical_nodes":2,"gpus_per_node":4,"num_gpus":8,"deployment":"aggregate","official_inferencex_result":false}' \
        > "$RESULT_DIR/deployment.json"
    [[ -s "$RESULT_DIR/aiperf_artifacts/server_metrics_export.json" ]] ||
        { echo "Missing or empty server_metrics_export.json" >&2; return 2; }
    # InferenceX process_agentic_result reads these environment fields for metadata/output.
    MODEL=nvidia/Kimi-K3-NVFP4 MODEL_PREFIX=kimik3 FRAMEWORK=tokenspeed RUNNER_TYPE=gb300 PRECISION=fp4 \
    SPEC_DECODING=eagle3 CONC="$CONCURRENCY" TP=8 EP_SIZE="$EP_SIZE" \
    PP_SIZE=1 PCP_SIZE=1 DCP_SIZE=1 IS_MULTINODE=false DISAGG=false \
    KV_OFFLOAD_BACKEND= KV_OFFLOAD_BACKEND_METADATA= TOTAL_CPU_DRAM_GB=0 KV_OFFLOADING=none \
    RESULT_DIR="$RESULT_DIR" AGENTIC_OUTPUT_DIR="$RESULT_DIR" RESULT_FILENAME=inferencex_result \
    "$AIPERF_PYTHON" -m infx.results.agentic.process_agentic_result
}
# InferenceX replay/validation controls; power collection is explicitly disabled.
# AIPerf reads the dataset-worker/header settings; Python and libraries read
# PYTHONPATH and thread limits. These are process settings, not result metadata.
AIPERF_FAILED_REQUEST_THRESHOLD=0.1 AIPERF_REQUIRED_SERVER_METRIC_PREFIX=tokenspeed: \
ENABLE_AGENTX_POWER=0 \
PYTHONPATH="$INFMAX_CONTAINER_WORKSPACE" \
AIPERF_DATASET_WEKA_PARALLEL_WORKERS=2 TOKENIZERS_PARALLELISM=false \
AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
run_agentic_replay_and_write_outputs "$RESULT_DIR"
CLIENT
    CLIENT_PID=$!
    local result=0
    wait "$CLIENT_PID" || result=$?
    CLIENT_PID=
    printf '%s\n' "$result" > "$RUN_ROOT/c${CONCURRENCY}/client-exit-code.txt"
    return "$result"
}

# A completed client may leave server work after tail cancellation. Do not let
# that work overlap the next point. Missing counters fail closed at the timeout.
wait_for_drain() {
    local start=$SECONDS
    while true; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Server exited while draining' >&2; return 1; }
        if curl --fail --silent --max-time 5 "$AIPERF_SERVER_METRICS_URLS" > "$RUN_ROOT/c${CONCURRENCY}/drain-metrics.txt" &&
            awk '
                /^tokenspeed:num_requests_(running|waiting)(\{|[[:space:]])/ {
                    if ($0 ~ /^tokenspeed:num_requests_running/) running=1
                    else waiting=1
                    sample=$0
                    sub(/^tokenspeed:num_requests_(running|waiting)(\{.*\})?[[:space:]]+/, "", sample)
                    count=split(sample, fields, /[[:space:]]+/)
                    # Labels may contain spaces; optional timestamps follow the value.
                    if (count > 2 || fields[1] !~ /^[+]?[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/ ||
                        (count == 2 && fields[2] !~ /^-?[0-9]+$/)) invalid=1
                    else if (fields[1]+0 != 0) busy=1
                }
                END { exit !(running && waiting && !invalid && !busy) }
            ' "$RUN_ROOT/c${CONCURRENCY}/drain-metrics.txt"; then
            return 0
        fi
        (( SECONDS - start < DRAIN_TIMEOUT )) || { echo 'Server drain timeout' >&2; return 1; }
        sleep 1
    done
}

# Keep the configuration sweep beside the launch lifecycle, like agentic_bench.slurm.
(( $# == 0 )) || { echo 'Use the named recipe inputs; trailing flags are unsupported' >&2; exit 2; }
SWEEP_ROOT=$RUN_ROOT
mkdir "$SWEEP_ROOT"
for CONFIG in "${CONFIGS[@]}"; do
    RUN_ROOT="$SWEEP_ROOT/$CONFIG"
    STEP_NAME="agentx_server_${SLURM_JOB_ID}_$$_${CONFIG}"
    trap finish EXIT
    trap 'CLIENT_SIGNAL=INT; exit 130' INT
    trap 'CLIENT_SIGNAL=TERM; exit 143' TERM
    mkdir "$RUN_ROOT"
    launch_server
    wait_for_ready
    for CONCURRENCY in "${CONCURRENCY_VALUES[@]}"; do
        run_benchmark
        wait_for_drain
    done
    stop_server
    printf '0\n' > "$RUN_ROOT/client-exit-code.txt"
    trap - EXIT INT TERM
done
