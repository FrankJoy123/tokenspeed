# Kimi K3 AgentX with InferenceX

`agentx.slurm` loops over the existing TP8/EP8 configs, starts a K3 server,
waits for readiness, runs native InferenceX AgentX replay, saves its outputs,
and stops the server. It uses two four-GPU GB300 nodes as one aggregate TP8
instance. The shared server configs own topology and kernel backends; `SERVER_ARGS` in
`agentx.slurm` holds AgentX engine tuning. Arguments after the script override
that tuning (for example, checkpoint paths).
The launcher appends `--port "$API_PORT"` last so serving and readiness use the
same port; any caller-supplied port is overridden. Client settings are validated
before server startup.


The client stack is `agentx.sh → InferenceX → AIPerf → HTTP inference server`.
InferenceX defines the experiment protocol and aggregates results; AIPerf loads
and replays traces, schedules requests and records measurements. The official
[command builder](https://github.com/SemiAnalysisAI/InferenceX/blob/ebedc2bbac4ff3e51ea9acb1f95093d017162809/benchmarks/benchmark_lib.sh#L3073-L3077)
invokes `aiperf profile --scenario inferencex-agentx-mvp` with streaming enabled.
This launcher uses that same path, without EvalScope.

## Prepare once

Prepare the client on the host running the launcher; its architecture must match
that host. The GPU container and client environments are separate. No packages
are installed during the benchmark.

```bash
export CLIENT_ROOT=/shared/agentx-client
mkdir "$CLIENT_ROOT"
git clone https://github.com/SemiAnalysisAI/InferenceX.git "$CLIENT_ROOT/inferencex"
git -C "$CLIENT_ROOT/inferencex" checkout ebedc2bbac4ff3e51ea9acb1f95093d017162809
git clone https://github.com/SemiAnalysisAI/aiperf.git "$CLIENT_ROOT/aiperf"
git -C "$CLIENT_ROOT/aiperf" checkout 754356e9a39acc6cc6afb242d123bb57c3fb6f75
# Apply the two public commits from AIPerf #1540 to this pinned client.
for commit in 556293ec52a7fd57f0221255ecf2de2f6a9eb363 e84e5edd1ab6f660eb5d9eb7e3c5d24124f3b6a7; do
  curl --fail --location "https://github.com/ai-dynamo/aiperf/commit/$commit.patch" \
    --output "$CLIENT_ROOT/$commit.patch"
  git -C "$CLIENT_ROOT/aiperf" apply --check "$CLIENT_ROOT/$commit.patch"
  git -C "$CLIENT_ROOT/aiperf" apply "$CLIENT_ROOT/$commit.patch"
done
python3.11 -m venv "$CLIENT_ROOT/venv"
source "$CLIENT_ROOT/venv/bin/activate"
python -m pip install -e "$CLIENT_ROOT/aiperf" \
  'transformers==4.57.1' 'numpy>=1.24' 'pandas>=2.0' 'aiohttp>=3.10' \
  'xlsxwriter>=3.2.1' 'tqdm>=4.66' 'datasets>=4.7' tiktoken matplotlib \
  'huggingface_hub[cli]>=0.25' urllib3 requests
python -m pip check
python -m pip freeze --all > "$CLIENT_ROOT/requirements-resolved.txt"
```

[AIPerf #1540](https://github.com/ai-dynamo/aiperf/pull/1540) fixes child-branch
cleanup when sending stops: unfinished join bookkeeping could otherwise keep
replay waiting until grace expires after HTTP requests have completed. The
commands above apply the two public commits to the pinned client; no separately
prepared local patch is required. This launcher does not vendor AIPerf changes.
Reuse the resolved client environment across comparisons. The installation
example resolves dependencies once; version ranges are not a dependency lock.

Keep client sources, resolved dependencies, model checkpoints and the trace
cache unchanged across comparisons. The protocol selects either the public
`semianalysisai/cc-traces-weka-062126` corpus or its `-256k` variant.
Credentials may be needed. Prepare a dedicated HF cache and pin its `main` ref
in the external experiment setup: the native public loader resolves `main`, so
merely downloading a revision does not pin subsequent replay. The launcher does
not verify dataset revisions or force offline mode. For a frozen offline run,
pre-download the tokenizer and all native loader assets into this same cache,
then export `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` from the experiment driver.

```bash
export HF_HUB_CACHE="$CLIENT_ROOT/hf-cache"
export TRACE_REVISION=23f152f6f0f9399a85901b89a6458def0ef16729
hf download --repo-type dataset semianalysisai/cc-traces-weka-062126 \
  --revision "$TRACE_REVISION"
TRACE_CACHE="$HF_HUB_CACHE/datasets--semianalysisai--cc-traces-weka-062126"
mkdir -p "$TRACE_CACHE/refs"
printf '%s' "$TRACE_REVISION" > "$TRACE_CACHE/refs/main"
```

For `weka-062126-256k`, prepare the `-256k` repository and its own pinned
revision instead. Do not reuse the full-corpus revision for another repository.
The pinned native command builder splits arguments on whitespace: use
whitespace-free client/output/tokenizer paths and compact JSON extra inputs.

## Run

Use a dedicated allocation and a server image compatible with the selected
TokenSpeed checkout. All server paths, including the checkout and output,
must be mounted into the container. Set `SERVER_VENV` empty if the image already
provides `ts`. The client runs on the host invoking this script.

```bash
export SOURCE_ROOT=/shared/tokenspeed
export CONFIG_NAMES="attn_tp8_moe_tp8"
export CONTAINER_IMAGE=/shared/server.sqsh CONTAINER_MOUNTS=/shared:/shared
export SERVER_VENV=/opt/server-venv
export TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR=/shared/frozen-autotune-cache
export MODEL_DIR=/shared/k3 DRAFT_DIR=/shared/eagle3
export MODEL_NAME=moonshotai/Kimi-K3 TOKENIZER_PATH=/shared/pinned-k3-tokenizer
export CLIENT_PYTHON="$CLIENT_ROOT/venv/bin/python"
export INFERENCEX_WORKSPACE="$CLIENT_ROOT/inferencex"
export AGENTX_PROTOCOL=weka-062126-256k MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=16
export SPEC_DECODING=eagle3 SPEC_STEPS=3 PRECISION=fp4
export ACCEPTANCE_MODE=synthetic AL_TARGET=3.7
export SERVER_METRICS=true METRICS_PORT=8010
export CONCURRENCIES="1" API_PORT=8000 CPUS_PER_TASK=64
export DRAIN_TIMEOUT=1800
export READINESS_PATH=/readiness READINESS_TIMEOUT=7500 CLIENT_TIMEOUT=8400
export RUN_ROOT=/shared/new-agentx-output

# Smoke uses one point. For formal runs, use the settings below with a new RUN_ROOT.
export DURATION=180 AIPERF_WARMUP_REQUESTS_PER_LANE=1 AIPERF_UNSAFE_OVERRIDE=true
export BENCHMARK_GRACE_SECONDS=30 AGENTIC_WARMUP_GRACE_PERIOD=1800
export REQUEST_TIMEOUT_SECONDS=21600 CLIENT_SEED=42
export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS=300 AIPERF_FAILED_REQUEST_THRESHOLD=0.1
export AIPERF_EXTRA_INPUTS='{"ignore_eos":true,"temperature":1.0,"top_p":0.95,"top_k":20}'

SERVER_ARGS=(
  --model "$MODEL_DIR" --served-model-name "$MODEL_NAME"
  --speculative-draft-model-path "$DRAFT_DIR"
  --disable-kvstore
)
export SLURM_JOB_ID=YOUR_HELD_JOB_ID
bash "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm" "${SERVER_ARGS[@]}"
```

For the formal sweep, set these explicitly before invoking the same launcher:

```bash
export RUN_ROOT=/shared/new-agentx-formal-output
export CONCURRENCIES="1 2 4 8 16"
export DURATION=3600 AIPERF_WARMUP_REQUESTS_PER_LANE=10 AIPERF_UNSAFE_OVERRIDE=false
```

The server checkpoint in this recipe is K3 MXFP4; `fp4` is the native report
category. Pin model, EAGLE3 draft and tokenizer snapshots separately. The fixed-AL
recipe uses the greedy server backend even though the native scenario sends
sampling parameters. Record effective sampling, not just the HTTP parameters.

Alternatively, submit with a client environment available on the allocated node:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
  --exclusive --partition=YOUR_PARTITION --time=16:00:00 --no-requeue --export=ALL \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm" "${SERVER_ARGS[@]}"
```

`RUN_ROOT` must be new and its parent must exist. Each config starts a fresh
server and executes `CONCURRENCIES` in the supplied order. A failed point or
failed drain stops the sweep. Each point gets `c<concurrency>/client/` under its
config directory; use a new output root for a retry. Duplicate CC values are
rejected; run repetitions under another output root.
Multiple points require `SERVER_METRICS=true`. Between points the launcher waits
until both running and waiting server request counters are present and zero;
`DRAIN_TIMEOUT` is separate from client grace. The client uses native per-replay
prefix isolation, not a physical KV-cache flush. The service, allocator and
compiled/autotuned state remain alive across points. Keep point order and this
initialization policy consistent across engine comparisons. C1/C2/C4 mean 1/2/4 root replay lanes; child agents can add concurrent
HTTP requests. Keep the selected shared config's TP8/EP8 topology. `PRECISION` must describe
the actual checkpoint, and `DRAFT_DIR` must match `SPEC_DECODING`.
Extra server arguments must not change the topology used for reporting.
The launcher owns context, sequence capacity, speculative method/steps, AL,
and metrics arguments; conflicting trailing flags fail before startup, including
`--speculative-config`, `--speculative-eagle-topk` and underscore aliases.
Drain checks require valid, nonnegative running/waiting counters and handle
spaces in labels; malformed or missing counters cannot count as idle.

Grace is the maximum drain wait after sending stops, not a fixed sleep.
`CLIENT_TIMEOUT` covers preparation, warmup and export for each point; it is a
separate outer limit. Request an allocation long enough for the complete sweep. Allocation holding and multi-allocation scheduling remain external.
Cleanup handles interrupted preflight/server launches even before their Slurm
steps are registered. Only this invocation's steps are cancelled.

## Outputs and scope

Each `RUN_ROOT/<config>/` contains the server log, the server command,
allocation and GPU information, endpoint details, and an exit-code file.
Each `c<concurrency>/` contains a client log, exit code and drain metrics. Its
`client/` contains the native request exports, phase logs, summary and
`inferencex_result.json`. Native replay and validation failures propagate to the
launcher; partial outputs are retained.

`agentx.sh` calls native `resolve_trace_source`, `build_replay_cmd`, and
`run_agentic_replay_and_write_outputs`. The pinned native report's multi-node
shape describes PD disaggregation. For this aggregate TP8 service, report
processing uses the native aggregate shape (eight GPUs); `deployment.json`
records the two physical nodes separately. Native metric values and JSON are
not rewritten. These are local reproductions, not official InferenceX results.
Power collection is disabled. Server counters are explicitly selected with
`SERVER_METRICS=true|false`; unavailable values are not zero. With metrics enabled,
preflight checks once per server startup that API and metrics ports can be bound
on the allocated nodes, before the concurrency loop.
This catches existing listeners, including non-HTTP services; it does not reserve
the ports against races with other launches. The launcher enables the engine
endpoint and checks HTTP reachability after readiness, retaining the response in `server-metrics-readiness.txt`. Replay must
produce a nonempty `aiperf_artifacts/server_metrics_export.json`; a missing or empty
file fails the run. Native replay validation remains in place. The launcher does
not add counter-content checks, measured-AL thresholds or a custom acceptance
policy. Those belong to the external experiment auditor. Optional upstream
backend adapters determine which counters appear in native summaries.

Full experiment auditing, version/hash verification, grace-timeout diagnosis,
cross-run acceptance policy and spreadsheet updates belong to the external experiment
tools. Exit zero means native replay/validation succeeded, not that a run passed
an additional comparison audit. Smoke is not a formal performance measurement.

Run a GPU smoke for each selected layout before starting a formal benchmark.

## Explicit protocol settings

| Setting | Values and meaning |
| --- | --- |
| `AGENTX_PROTOCOL` | `weka-062126-256k` or `weka-062126-full`; selects the native public loader |
| `MAX_MODEL_LEN` | Positive server context limit; the example uses 1048576. No client context cap is injected |
| `CONCURRENCIES` | Ordered, unique positive root-lane counts, e.g. `1 2 4 8 16` |
| `DRAIN_TIMEOUT` | Maximum server-drain seconds after each point; missing counters cannot count as drained |
| `MAX_NUM_SEQS` | Explicit service admission capacity, independent of root-lane `CONCURRENCY` |
| `ACCEPTANCE_MODE` | `natural` (empty/unset `AL_TARGET`) or `synthetic` (finite target required) |
| `AL_TARGET` | Synthetic accepted length including the verification token; between 1 and `SPEC_STEPS + 1` |
| `SPEC_DECODING`, `SPEC_STEPS` | `eagle3` or `dspark`, and explicit positive draft steps |
| `SERVER_METRICS` | `true` requires a separate `METRICS_PORT`; `false` disables scraping explicitly |
| `INFERENCEX_WORKSPACE` | Directory containing `benchmarks/benchmark_lib.sh` |

Synthetic acceptance adds `--sampling-backend greedy` and
`--synthetic-acceptance-length`. Natural acceptance does not force a sampling
backend. Fixed-AL measurements describe controlled execution efficiency, not
natural speculative acceptance performance. The launcher checks only that the
target is numeric and between 1 and `SPEC_STEPS + 1`. Even when metrics are
enabled, an external auditor must verify measured AL before calling it verified.

`server-command.txt` retains the model/draft paths and effective server
arguments. Trace hashes, revision checks, version manifests and measured-AL
acceptance checks belong to the external experiment tooling, not this launcher.
Freeze checkpoints, tokenizer, trace and client versions there before comparing
runs. Reusing an existing output root is rejected.

Interpret native **output** throughput separately from total (input + output)
throughput. For eight GPUs, output tok/s/GPU = system output tok/s / 8.
Interactivity is a streaming speed metric; do not substitute end-to-end request
latency or label fixed-AL results as natural acceptance.

Engine tuning values are written directly in `agentx.slurm`'s `SERVER_ARGS`.
Context capacity, sequence capacity, speculative decoding and acceptance settings
remain explicit inputs so the same launcher supports the comparison recipes
above; metrics remain opt-in. These protocol-owned values take precedence over
server tuning and cannot be overridden through trailing flags.
