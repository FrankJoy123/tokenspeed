# Kimi K3 AgentX with InferenceX

This is a fixed aggregate benchmark on two four-GPU GB300 nodes (8 GPUs total).
It supports TP8/MoE TP8 or TP8/MoE EP8, EAGLE3 steps3, and the
native full trace (`weka-062126-full`). It does not launch PD workers or a separate PD router.

## Launcher structure

`agentx.slurm` manages the benchmark lifecycle:

- `launch_server` resolves the allocation and endpoints, then calls the selected
  `configs/attn_tp8_moe_tp8.sh` or `configs/attn_tp8_moe_ep8.sh`
  inside the GPU container.
- `run_benchmark` runs native InferenceX in a fresh client shell per concurrency point.
- Readiness, drain and cleanup keep the server lifecycle around the sweep.

The shared configs contain the HF model IDs, topology and EAGLE3 settings.
They append `"$@"` to `ts serve`; `launch_server` passes the fixed AgentX overrides
explicitly, including context length, sequence capacity, text-only execution,
synthetic AL and metrics. Repeated scalar flags use the last supplied value.
Calling the configs without extra arguments preserves the existing benchmark recipe.
The launcher uses `bash -x` to record the expanded server command in `server.log`.
There is no separate AgentX server script.

InferenceX defines the replay protocol and aggregates results. Its native
`build_replay_cmd` invokes AIPerf (`aiperf profile --scenario inferencex-agentx-mvp`),
which loads AgentX traces, schedules HTTP requests and records measurements.

## Prepare once

Prepare the client on the host running the launcher; its architecture must match
that host. The GPU container and client environments are separate. No packages
are installed during the benchmark.

```bash
: "${BENCH_ROOT:?set a shared benchmark directory}"
CLIENT_ROOT="$BENCH_ROOT/agentx-client"
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

Keep the client environment, model, tokenizer and full-trace revision fixed
across comparisons. The native loader reads
`semianalysisai/cc-traces-weka-062126`; prepare its cache before running.
Use whitespace-free client and output paths: the native command
builder splits arguments on whitespace.

## Run

Use a dedicated allocation, a compatible frozen server image, and mounted paths.
Set `BENCH_ROOT` to a shared mounted directory before using these commands.
No personal directory is embedded in the scripts or examples. No dependencies
are installed during a run. The server environment must support the selected
speculative strategy and synthetic acceptance length.

```bash
: "${BENCH_ROOT:?set a shared benchmark directory}"
SOURCE_ROOT="$BENCH_ROOT/tokenspeed"
CLIENT_ROOT="$BENCH_ROOT/agentx-client"

SOURCE_ROOT="$SOURCE_ROOT" CLIENT_ROOT="$CLIENT_ROOT" \
INFERENCEX_WORKSPACE="$CLIENT_ROOT/inferencex" \
CONTAINER_IMAGE="$BENCH_ROOT/server.sqsh" CONTAINER_MOUNTS="$BENCH_ROOT:$BENCH_ROOT" \
SERVER_VENV="" \
MODEL_NAME=moonshotai/Kimi-K3 \
TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR="$BENCH_ROOT/autotune-cache" \
CONFIG_NAMES=attn_tp8_moe_tp8 AL_TARGET=3.7 \
RUN_MODE=smoke CONCURRENCIES=1 \
API_PORT=8000 METRICS_PORT=8010 DRAIN_TIMEOUT=1800 \
RUN_ROOT="$BENCH_ROOT/new-agentx-smoke" SLURM_JOB_ID=YOUR_HELD_JOB_ID \
bash "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm"
```

`SERVER_VENV=""` uses `ts` installed in the image. Settings are passed only to
this invocation; no exports are needed in the invoking shell. Fixed values are written at their
call sites. Names such as `FRAMEWORK=tokenspeed` and `TP=8` are required inputs
to native InferenceX functions/result processing, not recipe selectors.

For a formal run, use the same command with `RUN_MODE=formal`, `CONCURRENCIES="1 2 4 8 16"` and a new
`RUN_ROOT="$BENCH_ROOT/new-agentx-formal"`.

EAGLE3 is fixed at three draft steps and four draft tokens. The server commands
use `nvidia/Kimi-K3-NVFP4` and `lightseekorg/kimi-k3-eagle3-mla` directly;
the client loads its tokenizer from `nvidia/Kimi-K3-NVFP4` too.
There are no model-path inputs or external strategy selector.
Prepare these repositories in the Hugging Face cache before running, and record
the resolved revisions with each comparison; a model ID alone does not pin a revision.
The full-trace loader is fixed in `agentx.slurm`.
The model checkpoint must be K3 FP4 and the tokenizer must match it. AL is a
synthetic control, not a measured natural acceptance rate.

To allocate nodes instead of using a held job, keep the same command-local
settings, omit `SLURM_JOB_ID`, and replace the final `bash` line with:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
  --exclusive --partition=YOUR_PARTITION --time=16:00:00 --no-requeue --export=ALL \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm"
```

There is no external `SERVER_ARGS` and trailing flags are rejected. Change AgentX overrides in `launch_server` when deliberately defining a different recipe.
PD receiver/routing options belong in the separate PD benchmark scripts, not here.

## Fixed settings

The shared engine command plus the explicit `launch_server` overrides set: context 1048576,
max sequences 16, memory utilization 0.92, FP8 KV, BF16 SSM, prefix granularity
128, prefill limits 8192, graph capture limit 1024, stream interval 10, seed 0,
greedy synthetic acceptance, metrics enabled and KV offloading disabled.

The pinned InferenceX `build_replay_cmd` explicitly adds `--random-seed 42`.
`run_benchmark` in `agentx.slurm` supplies tail grace
30 s, warmup grace 1800 s, idle gap cap 300 s, request timeout 21600 s and error threshold 10%.
`smoke` sends for 180 s with one warmup request per lane; `formal` sends for
3600 s with ten warmup requests per lane. Both use the native trace loader and
command builder without a client context cap. The benchmark points are C1/C2/C4/C8/C16. These count root lanes;
child agents can create additional concurrent HTTP requests.

The inline client comments distinguish native InferenceX shell inputs,
InferenceX result metadata, and AIPerf/process environment settings. Power
collection is explicitly disabled; power-only arguments are omitted. Result
paths are passed directly to the result processor, which does not read
`IS_AGENTIC`. Recorded assistant responses are fixed with
`AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0`. Explicit fast-mode,
unsafe-override and Dynamo-header switches
remain so inherited environment settings cannot change the fixed protocol.

## Lifecycle and outputs

The launcher checks the required client inputs, run mode, and the complete
configuration and concurrency lists (including duplicates) before starting GPU work.
Each selected configuration starts one server, waits for `/readiness`, then runs
the ordered concurrency points. The launcher drains running and waiting requests
after each point before continuing. Missing or invalid counters never mean idle;
`DRAIN_TIMEOUT` bounds this wait separately from client grace. There is no separate
bind probe, GPU preflight job, `/v1/models` probe or metrics-readiness probe.
Use an exclusively allocated node pair and dedicated API/metrics ports. Startup
is bounded at 7500 s and each client at 8400 s; these fixed limits are written
directly at their call sites.

The native client isolates prefixes per replay; this is not a physical KV-cache
flush. The service and compiled/autotuned state remain alive across points.
Keep the ordering and initialization policy consistent across comparisons.
Failure or interruption stops the sweep and cleans up only its server steps,
not the allocation. Allocation retention and multi-job scheduling stay outside.

`RUN_ROOT` must be new and its parent must exist. Each `<config>/` saves the
expanded **actual `ts serve` command** in `server.log`, the server log,
allocation, endpoint and exit code. Each `c<N>/` saves its client log, exit code,
drain counters and `client/` containing native exports and `inferencex_result.json`.
The client requires a nonempty metrics export. It does not rewrite native metrics.
Native aggregate reporting uses eight GPUs; `deployment.json` records two nodes.
Output tok/s/GPU is system output tok/s divided by eight.

Detailed auditing, measured-AL validation, revision manifests and spreadsheet
reporting remain external. Run GPU smoke on the exact candidate before claiming
end-to-end validation; CPU lifecycle tests alone cannot establish performance.
