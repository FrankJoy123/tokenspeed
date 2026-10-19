# Kimi K3 AgentX with InferenceX

`agentx.slurm` loops over the existing TP8/EP8 configs, starts a K3 server,
waits for readiness, runs native InferenceX AgentX replay, saves its outputs,
and stops the server. It uses two four-GPU GB300 nodes as one aggregate TP8
instance. Arguments after the script are appended to the existing server config.
The launcher appends `--port "$API_PORT"` last so serving and readiness use the
same port; any caller-supplied port is overridden. Client settings are validated
before server startup.

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
git -C "$CLIENT_ROOT/aiperf" checkout 89b21867872a5bbc4b0676bf5005c404da5e9f94
python3.11 -m venv "$CLIENT_ROOT/venv"
source "$CLIENT_ROOT/venv/bin/activate"
python -m pip install -e "$CLIENT_ROOT/aiperf" \
  'transformers==4.57.1' 'numpy>=1.24' 'pandas>=2.0' 'aiohttp>=3.10' \
  'xlsxwriter>=3.2.1' 'tqdm>=4.66' 'datasets>=4.7' tiktoken matplotlib \
  'huggingface_hub[cli]>=0.25' urllib3 requests
python -m pip check
python -m pip freeze --all > "$CLIENT_ROOT/requirements-resolved.txt"
```

Keep client sources, resolved dependencies, model checkpoints and the trace
cache unchanged across comparisons. The protocol selects either the public
`semianalysisai/cc-traces-weka-062126` corpus or its `-256k` variant.
Credentials may be needed. Prepare a dedicated HF cache and pin its `main` ref
before running: the native public loader resolves `main`, so merely downloading
a revision does not pin subsequent replay. The launcher verifies the ref and
snapshot, then runs offline. Pre-download the tokenizer and all native loader
assets into this same cache as well.

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
export CONFIG_NAMES="attn_tp8_moe_tp8 attn_tp8_moe_ep8"
export CONTAINER_IMAGE=/shared/server.sqsh CONTAINER_MOUNTS=/shared:/shared
export SERVER_VENV=/opt/server-venv
export MODEL_DIR=/shared/k3 DRAFT_DIR=/shared/eagle3
export MODEL_NAME=kimi-k3 TOKENIZER_PATH="$MODEL_DIR"
export CLIENT_PYTHON="$CLIENT_ROOT/venv/bin/python"
export INFERENCEX_WORKSPACE="$CLIENT_ROOT/inferencex"
export AGENTX_PROTOCOL=weka-062126-full MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=2
export SPEC_DECODING=eagle3 SPEC_STEPS=3 PRECISION=fp4
export ACCEPTANCE_MODE=synthetic AL_TARGET=3.7
export SERVER_METRICS=true METRICS_PORT=8010
export CONCURRENCY=1 API_PORT=8000 CPUS_PER_TASK=64
export READINESS_PATH=/readiness READINESS_TIMEOUT=7500 CLIENT_TIMEOUT=8400
export RUN_ROOT=/shared/new-agentx-output

# Smoke. For formal runs: DURATION=3600, warmup requests=10, unsafe override=false.
export DURATION=180 AIPERF_WARMUP_REQUESTS_PER_LANE=1 AIPERF_UNSAFE_OVERRIDE=true
export BENCHMARK_GRACE_SECONDS=30 AGENTIC_WARMUP_GRACE_PERIOD=1800
export REQUEST_TIMEOUT_SECONDS=21600 CLIENT_SEED=42
export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS=300 AIPERF_FAILED_REQUEST_THRESHOLD=0.1
export AIPERF_EXTRA_INPUTS='{"ignore_eos":true,"temperature":1.0,"top_p":0.95,"top_k":20}'

SERVER_ARGS=(
  --model "$MODEL_DIR" --served-model-name "$MODEL_NAME"
  --speculative-draft-model-path "$DRAFT_DIR"
  --gpu-memory-utilization 0.9 --seed 42
  --engine-startup-timeout 7200 --disable-kvstore
)
export SLURM_JOB_ID=YOUR_HELD_JOB_ID
bash "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm" "${SERVER_ARGS[@]}"
```

Alternatively, submit with a client environment available on the allocated node:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
  --exclusive --partition=YOUR_PARTITION --time=12:00:00 --no-requeue --export=ALL \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm" "${SERVER_ARGS[@]}"
```

`RUN_ROOT` must be new and its parent must exist. Each config starts a fresh
server; a failed point stops the sweep. Use a new output root for each concurrency
or retry. C1/C2/C4 mean 1/2/4 root replay lanes; child agents can add concurrent
HTTP requests. Keep the selected shared config's TP8/EP8 topology. `PRECISION` must describe
the actual checkpoint, and `DRAFT_DIR` must match `SPEC_DECODING`.
Extra server arguments must not change the topology used for reporting.
The launcher owns context, sequence capacity, speculative method/steps, AL,
and metrics arguments; conflicting trailing flags fail before startup.

Grace is the maximum drain wait after sending stops, not a fixed sleep.
`CLIENT_TIMEOUT` also covers preparation, warmup and export; it is a separate
outer limit. Allocation holding and multi-allocation scheduling remain external.
Cleanup handles interrupted preflight/server launches even before their Slurm
steps are registered. Only this invocation's steps are cancelled.

## Outputs and scope

Each `RUN_ROOT/<config>/` contains server/client logs, the server command,
allocation and GPU information, endpoint details, and an exit-code file.
Its `client/` contains the native request exports, phase logs, summary and
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
the launcher enables the engine endpoint and checks the required Prometheus
counters after readiness. Replay must export a parseable
`aiperf_artifacts/server_metrics_export.json` with token activity from that endpoint.
A missing, malformed, empty or wrong-endpoint export fails the run. Synthetic runs
also require measured `1 + accepted_draft_tokens / draft_steps` within 0.1 of the
AL target. The metric window may include warmup; this is a configuration check,
not a new performance metric. Raw native metrics remain unchanged. Optional
upstream backend adapters determine which counters also appear in native summaries.

Full experiment auditing, version/hash verification, grace-timeout diagnosis,
cross-run acceptance policy and spreadsheet updates belong to the external experiment
tools. Exit zero means native replay/validation succeeded, not that a run passed
an additional comparison audit. Smoke is not a formal performance measurement.

Run a GPU smoke for each selected layout before starting a formal benchmark.

CPU lifecycle and input-validation checks run in the Lint workflow without GPUs:

```bash
python -m unittest discover -s test/agentic_benchmark/tests \
  -p test_k3_inferencex_agentx.py -v
```

## Explicit protocol settings

| Setting | Values and meaning |
| --- | --- |
| `AGENTX_PROTOCOL` | `weka-062126-256k` or `weka-062126-full`; selects the native public loader |
| `MAX_MODEL_LEN` | Exactly 262144 or 1048576 respectively; applied to both server and replay |
| `MAX_NUM_SEQS` | Explicit service admission capacity, independent of root-lane `CONCURRENCY` |
| `ACCEPTANCE_MODE` | `natural` (empty/unset `AL_TARGET`) or `synthetic` (finite target required) |
| `AL_TARGET` | Synthetic accepted length including the verification token; between 1 and `SPEC_STEPS + 1` |
| `SPEC_DECODING`, `SPEC_STEPS` | `eagle3` or `dspark`, and explicit positive draft steps |
| `SERVER_METRICS` | `true` requires a separate `METRICS_PORT`; `false` disables scraping explicitly |
| `INFERENCEX_WORKSPACE` | Directory containing `benchmarks/benchmark_lib.sh`; old checkouts used `inferencex-e2e/` |
| `TRACE_REVISION`, `HF_HUB_CACHE` | Prepared offline dataset snapshot; the local `refs/main` must match |

Synthetic acceptance adds `--sampling-backend greedy` and
`--synthetic-acceptance-length`. Natural acceptance does not force a sampling
backend. Fixed-AL measurements describe controlled execution efficiency, not
natural speculative acceptance performance. With metrics disabled, the target is
recorded but measured AL is unavailable; do not call it verified.

Each point saves `protocol.json` with the effective protocol, trace revision/hash,
client Git commits and dirty flags, precision, speculation, AL target and replay
settings. `server-command.txt` retains explicit model/draft paths and overrides.
These records expose configuration differences; they do not prove checkpoint
identity or validate generation quality. Prepare frozen checkpoints and tokenizer
snapshots externally. Reusing an existing output root is rejected.

Interpret native **output** throughput separately from total (input + output)
throughput. For eight GPUs, output tok/min/GPU = system output tok/s × 60 / 8.
Interactivity is a streaming speed metric; do not substitute end-to-end request
latency or label fixed-AL results as natural acceptance.

## Changes adopted from the shared benchmark suite

- [#2284](https://github.com/lightseekorg/tokenspeed/pull/2284): synchronize the
  shared TP8/EP8 config capacity to 64. AgentX still passes `MAX_NUM_SEQS`
  explicitly, so adopting the shared default does not silently change a protocol.
  Its replicated SWE-smith data, 4K initial prompts, 500-token outputs and EvalScope
  summary keys are specific to that workload and are not substituted for AgentX.
- [#2265](https://github.com/lightseekorg/tokenspeed/pull/2265): keep output
  throughput units explicit. Its Qwen performance gates and four-GPU divisor do
  not apply to this eight-GPU K3 launcher.
- P/D-specific changes from #2284 and MegaMoE EP16 changes from
  [#2285](https://github.com/lightseekorg/tokenspeed/pull/2285) do not apply to this
  aggregate TP8/EP8 launcher.

The new protocol/metrics paths have CPU regression tests. Run GPU smoke on the
exact candidate commit before claiming end-to-end validation. Full experiment
scheduling, GPU allocation retention and spreadsheet generation remain external.
