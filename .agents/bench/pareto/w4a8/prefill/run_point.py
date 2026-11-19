# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run one fresh W4A8 saturated Prefill point in the selected Python environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def benchmark_environment(inherited, *, script_dir, expert_routing, indexer):
    """Choose measured algorithms explicitly and remove incompatible observations."""
    if inherited.get("TOKENSPEED_HOST_BINARY"):
        raise ValueError(
            "Clear TOKENSPEED_HOST_BINARY; benchmark the host built with this checkout"
        )
    env = dict(inherited)
    env.pop("TOKENSPEED_HOST_BINARY", None)
    for name in [
        "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN",
        "DECODE_BENCH_OBSERVE",
        "DECODE_BENCH_RUN",
        "TOKENSPEED_PREFILL_BENCHMARK",
        "TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING",
        "TOKENSPEED_TRACE_HOST_STAGES",
        "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK",
        "TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER",
        "TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT",
    ]:
        env.pop(name, None)
    if expert_routing != "natural":
        raise ValueError("Prefill requires natural expert routing")
    env.pop("TOKENSPEED_MOE_ROUTING_SIMULATION", None)
    for name in [
        "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER",
        "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER",
    ]:
        env[name] = "1" if indexer == "native-fp4" else "0"
    env["TOKENSPEED_EXPERIMENT_SM120_DEEPEP"] = "1"
    env["TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP"] = "0"
    env["TOKENSPEED_CUDA_ARCH"] = "120a"
    env["FLASHINFER_CUDA_ARCH_LIST"] = "12.0a"
    env["TOKENSPEED_DISABLE_PLUGINS"] = "qwen_sm120"
    env["TOKENSPEED_KERNEL_DISABLE_PLUGINS"] = "sm120"
    env["PREFILL_BENCH_OBSERVE"] = "1"
    env["TOKENSPEED_SM120_DEEPEP_WIRE"] = "elastic_fp8_bf16"
    env["TOKENSPEED_SM120_DEEPEP_CAPACITY"] = "2048"
    env["PYTHONPATH"] = str(script_dir) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def benchmark_cache_environment(inherited, *, out):
    env = dict(inherited)
    cache = out / "cache"
    env.update(
        {
            "DG_JIT_CACHE_DIR": str(cache / "deepgemm"),
            "DG_SM120_JIT_CACHE_DIR": str(cache / "deepgemm-sm120"),
            "EP_JIT_CACHE_DIR": str(cache / "deepep"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "CUTE_DSL_CACHE_DIR": str(cache / "cute"),
            "TOKENSPEED_KERNEL_CACHE_ROOT": str(cache / "kernel"),
            "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer-workspace"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
    )
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["dp8", "cp8"], required=True)
    parser.add_argument(
        "--label",
        type=int,
        choices=[512, 1024, 2048, 4096, 8192, 16384],
        required=True,
        help="New tokens per step averaged over eight GPUs",
    )
    parser.add_argument("--queue-requests-per-lane", type=int)
    parser.add_argument("--max-model-len", type=int, default=135168)
    parser.add_argument("--retraction-snapshot-ratio", type=float)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="Optional checkout used to build the runtime"
    )
    parser.add_argument("--expert-routing", choices=["natural"], required=True)
    parser.add_argument("--redundant-experts", type=int, default=0)
    parser.add_argument("--expert-placement", default="trivial")
    parser.add_argument("--ep-dispatch", choices=["static", "dynamic"])
    parser.add_argument("--eplb-warmup-forwards", type=int, default=0)
    parser.add_argument("--eplb-interval", type=int)
    parser.add_argument("--eplb-layers-per-chunk", type=int)
    parser.add_argument("--indexer", choices=["bf16", "native-fp4"], required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--profile-steps", type=int, required=True)
    parser.add_argument("--cache-seed", type=Path)
    parser.add_argument("--flashinfer-workspace", type=Path)
    args = parser.parse_args()
    if args.profile_steps < 0:
        parser.error("--profile-steps must be nonnegative")
    if args.eplb_warmup_forwards and (
        args.eplb_warmup_forwards < 200
        or not args.eplb_interval
        or not args.eplb_layers_per_chunk
        or not args.ep_dispatch
    ):
        parser.error(
            "EPLB requires at least 200 warmup forwards, an interval, layers per chunk, and replica dispatch"
        )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip(), "GPUs are occupied"
    if args.cache_seed is not None:
        cache_seed = args.cache_seed.resolve(strict=True)
        shutil.copytree(
            cache_seed,
            out / "cache",
            ignore=shutil.ignore_patterns("*.lock", "*lockfile*"),
        )
        (out / "cache-seed.json").write_text(
            json.dumps({"source": str(cache_seed)}, indent=2) + "\n"
        )
    script_dir = Path(__file__).resolve().parent
    agents = script_dir.parents[3]
    source = args.source.resolve() if args.source else agents.parent
    cp = 8 if args.layout == "cp8" else 1
    dp = 8 // cp
    group_requests = args.label // 512
    active = min(group_requests, dp)
    requests_per_lane = group_requests // active
    physical = requests_per_lane * 4224
    count = args.queue_requests_per_lane
    if count is None:
        count = 20 * requests_per_lane
    if count < 16 * requests_per_lane:
        parser.error("Each active lane needs at least 16 target steps of requests")
    max_model_len = args.max_model_len
    engine_args = [
        "--model",
        str(args.model.resolve()),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--dense-gemm-backend",
        "deepgemm",
        "--moe-backend",
        "flashinfer",
        "--moe-mxfp4-fp8-activation",
        "--tensor-parallel-size",
        str(cp),
        "--data-parallel-size",
        str(dp),
        "--query-context-parallel-size",
        str(cp),
        "--kv-parallel-size",
        "1",
        "--dense-tp-size",
        "1",
        "--moe-tp-size",
        "1",
        "--attn-head-tp-size",
        "1",
        "--ep-size",
        "8",
        "--disaggregation-mode",
        "null",
        "--max-model-len",
        str(max_model_len),
        "--chunked-prefill-size",
        str(physical),
        "--max-prefill-tokens",
        str(physical),
        "--max-num-seqs",
        str(count * dp),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--engram-host-table",
        "--engram-host-table-layout",
        "shared",
        "--engram-host-table-dir",
        "/dev/shm",
        "--enforce-eager",
        "--disable-prefill-graph",
        "--disable-pdl",
        "--force-deterministic-rsag",
        "--all2all-backend",
        "none",
        "--enable-prefix-caching",
        "--kvstore-write-policy",
        "write_through",
        "--language-model-only",
        "--ep-num-redundant-experts",
        str(args.redundant_experts),
        "--init-expert-location",
        args.expert_placement,
        "--seed",
        "336846783",
        "--dist-init-addr",
        f"127.0.0.1:{31000 + requests_per_lane + (20 if cp == 8 else 0)}",
        "--distributed-timeout-seconds",
        "7200",
        "--enable-async-logging",
        "--enable-output-logprobs",
    ]
    if args.ep_dispatch:
        engine_args.extend(("--ep-dispatch-algorithm", args.ep_dispatch))
    if args.eplb_warmup_forwards:
        engine_args.extend(
            (
                "--enable-eplb",
                "--expert-distribution-recorder-mode",
                "stat",
                "--eplb-rebalance-num-iterations",
                str(args.eplb_interval),
                "--eplb-rebalance-layers-per-chunk",
                str(args.eplb_layers_per_chunk),
            )
        )
    if args.retraction_snapshot_ratio is not None:
        engine_args.extend(
            ["--retraction-snapshot-ratio", str(args.retraction_snapshot_ratio)]
        )
    env = benchmark_environment(
        os.environ,
        script_dir=script_dir,
        expert_routing=args.expert_routing,
        indexer=args.indexer,
    )
    env = benchmark_cache_environment(env, out=out)
    if args.flashinfer_workspace is not None:
        env["FLASHINFER_WORKSPACE_BASE"] = str(
            args.flashinfer_workspace.resolve(strict=True)
        )
    for name in ("CC", "CXX"):
        value = env.get(name)
        if not value or not Path(value).is_file() or not os.access(value, os.X_OK):
            parser.error(f"Set {name} to an explicit executable compiler path")
    subprocess.run(
        [
            str(args.python.absolute()),
            str(agents / "deploy/setup/identity.py"),
            "--source",
            str(source),
            "--output",
            str(out / "runtime-identity.json"),
        ],
        env=env,
        check=True,
        timeout=120,
    )
    subprocess.run(
        [
            str(args.python.absolute()),
            str(agents / "deploy/launch/preflight_model.py"),
            "--model",
            str(args.model.resolve()),
            "--profile",
            "w4a8",
            "--output",
            str(out / "model-identity.json"),
        ],
        env=env,
        check=True,
        timeout=180,
    )
    log = out / "engine.log"
    command = [
        str(args.python.absolute()),
        str(script_dir / "prefill_steady.py"),
        "--out",
        str(out / "samples"),
        "--engine-log",
        str(log),
        "--active-lanes",
        str(active),
        "--queue-requests-per-lane",
        str(count),
        "--prefix-tokens",
        "126976",
        "--new-tokens",
        "4096",
        "--computed-rows-per-request",
        "4224",
        "--prefill-rows-per-rank",
        str(physical),
        "--rounds",
        "1",
        "--extra-experts",
        str(args.redundant_experts),
        "--eplb-warmup-forwards",
        str(args.eplb_warmup_forwards),
        "--expert-routing",
        args.expert_routing,
        "--warmup-steps",
        "5",
        "--measure-steps",
        "10",
        "--seed",
        "336846783",
        "--profile-steps",
        str(args.profile_steps),
        "--",
        *engine_args,
    ]
    # Paths are local recovery evidence. Only a reviewed summary is published.
    record = {
        "command": command,
        "layout": args.layout,
        "new_tokens_per_active_lane": requests_per_lane * 4096,
        "label_new_tokens_per_gpu_per_step": active * requests_per_lane * 512,
        "active_attention_lanes": active,
        "expert_routing": args.expert_routing,
        "indexer": args.indexer,
        "selected_environment": {
            k: env[k]
            for k in [
                "TOKENSPEED_EXPERIMENT_SM120_DEEPEP",
                "TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP",
                "TOKENSPEED_SM120_DEEPEP_WIRE",
                "TOKENSPEED_SM120_DEEPEP_CAPACITY",
                "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER",
                "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER",
                "PREFILL_BENCH_OBSERVE",
            ]
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in script_dir.glob("*.py")
        },
    }
    (out / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
    with log.open("x") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        (out / "pid.json").write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "starttime": Path(f"/proc/{process.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19],
                    "epoch": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        rc = process.wait()
    (out / "terminal.json").write_text(
        json.dumps({"returncode": rc, "epoch": time.time()}, indent=2) + "\n"
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
