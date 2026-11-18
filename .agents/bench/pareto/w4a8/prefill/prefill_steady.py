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

"""Saturated Prefill windows, following DeepInfer's offline benchmark contract.

Reference: deepseek-ai/DeepInfer@a679bb86, examples/benchmark.py and sweep.
Execution uses TokenSpeed. Distinct suffixes retain a fixed prefix hit without
an internal sequence-fork API. Report useful new rows and computed/replayed
rows separately; timing is the host-observed completion interval, not TTFT.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

MARKER = "PREFILL_BENCHMARK "


def read_samples(path: Path) -> list[dict]:
    samples = []
    for line in path.read_text(errors="replace").splitlines():
        if MARKER not in line:
            continue
        payload = re.sub(r"\x1b\[[0-9;]*m", "", line.split(MARKER, 1)[1])
        samples.append(json.loads(payload))
    return samples


def steady_window(
    samples: list[dict],
    *,
    request_prefix: str,
    rows: int,
    dp_size: int,
    warmup_steps: int,
    measure_steps: int,
    active_lanes: int,
) -> dict:
    global_rows = [rows] * active_lanes + [0] * (dp_size - active_lanes)
    records = [
        s
        for s in samples
        if s["request_ids"]
        and all(r.startswith(request_prefix) for r in s["request_ids"])
    ]
    records.sort(key=lambda s: s["step"])
    runs, run = [], []
    for record in records:
        full = (
            record["global_decodes"] == [0] * dp_size
            and record["num_extends"] == len(record["request_ids"])
            and record["prefill_rows"] == rows
            and record["global_rows"] == global_rows
        )
        if not full or (run and record["step"] != run[-1]["step"] + 1):
            if run:
                runs.append(run)
            run = []
        if full:
            run.append(record)
    if run:
        runs.append(run)
    if not runs or len(runs[-1]) < warmup_steps + measure_steps + 1:
        raise ValueError(
            "not enough consecutive, globally full Prefill steps after warmup"
        )
    selected = runs[-1][-(measure_steps + 1) :]
    if any(s["serving_compiles"] is None for s in selected):
        raise ValueError("compile monitor unavailable in the steady window")
    if any(s["serving_compiles"] != selected[0]["serving_compiles"] for s in selected):
        raise ValueError("kernel compilation inside the measured steady window")
    completed_ns = [s["completed_ns"] for s in selected]
    interval_ns = [end - start for start, end in zip(completed_ns, completed_ns[1:])]
    if any(value <= 0 for value in interval_ns):
        raise ValueError("completion timestamps must be strictly increasing")
    elapsed = sum(interval_ns) / 1e9
    ordered = sorted(interval_ns)

    def quantile_ms(q):
        # Linear interpolation over all intervals, with positions (n - 1) * q.
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return (ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)) / 1e6

    measured = selected[1:]
    computed = sum(r["prefill_rows"] for r in measured)
    replay = sum(r["replay_rows"] for r in measured)
    useful = sum(r["new_rows"] for r in measured)
    if computed != useful + replay:
        raise ValueError("computed/new/replayed row accounting disagrees")
    return {
        "window_steps": measure_steps,
        "window_seconds": elapsed,
        "step_latency_ms": elapsed / measure_steps * 1000,
        "step_ids": [s["step"] for s in selected],
        "endpoint_completed_ns": completed_ns,
        "interval_ns": interval_ns,
        "interval_ms_distribution": {
            "min": ordered[0] / 1e6,
            "p50": quantile_ms(0.5),
            "p95": quantile_ms(0.95),
            "max": ordered[-1] / 1e6,
        },
        "computed_rows": computed,
        "new_rows": useful,
        "replay_rows": replay,
        "computed_tokens_s_gpu": computed / elapsed,
        "new_tokens_s_gpu": useful / elapsed,
        "first_step": selected[0]["step"],
        "last_step": selected[-1]["step"],
        "first_completed_ns": selected[0]["completed_ns"],
        "last_completed_ns": selected[-1]["completed_ns"],
        "full_run_steps": len(runs[-1]),
        "warmup_steps": warmup_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-log", type=Path, required=True)
    parser.add_argument("--active-lanes", type=int, required=True)
    parser.add_argument("--queue-requests-per-lane", type=int, required=True)
    parser.add_argument("--prefix-tokens", type=int, required=True)
    parser.add_argument("--new-tokens", type=int, required=True)
    parser.add_argument("--computed-rows-per-request", type=int, required=True)
    parser.add_argument("--prefill-rows-per-rank", type=int, required=True)
    parser.add_argument("--rounds", type=int, choices=[1], required=True)
    parser.add_argument("--extra-experts", type=int, choices=[0], required=True)
    parser.add_argument("--expert-routing", choices=["natural"], required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--measure-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-steps", type=int, required=True)
    args, engine_args = parser.parse_known_args()
    if engine_args[:1] != ["--"]:
        parser.error("separate engine arguments with --")
    from transformers import AutoTokenizer

    from tokenspeed.runtime.engine import zmq_wire
    from tokenspeed.runtime.engine.io_struct import (
        FlushCacheReqInput,
        MsgpackEncoder,
        ProfileReq,
        ProfileReqType,
    )
    from tokenspeed.runtime.entrypoints.wire_driver import (
        Completion,
        WireDriver,
        generate_request,
    )
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    server = prepare_server_args(engine_args[1:])
    dp = server.mapping.attn.dp_size
    cp = server.mapping.attn.qcp_size
    assert server.mapping.attn.tp_size == cp
    assert server.disaggregation_mode == "null"
    assert not server.disable_overlap_schedule and server.enable_async_logging
    assert server.speculative_algorithm is None and not server.enable_mixed_batch
    assert (
        not server.enable_eplb and server.ep_num_redundant_experts == args.extra_experts
    )
    assert server.enable_kvstore and server.kvstore_write_policy == "write_through"
    assert (
        server.max_prefill_tokens
        == server.chunked_prefill_size
        == args.prefill_rows_per_rank
    )
    assert os.environ.get("PREFILL_BENCH_OBSERVE") == "1"
    assert os.environ.get("TOKENSPEED_MOE_ROUTING_SIMULATION") is None
    assert args.prefill_rows_per_rank % args.computed_rows_per_request == 0
    assert args.computed_rows_per_request >= args.new_tokens > 0
    assert args.rounds == 1 and args.warmup_steps >= 5 and args.measure_steps == 10
    assert (args.prefix_tokens, args.new_tokens, args.computed_rows_per_request) == (
        126976,
        4096,
        4224,
    )
    active = args.active_lanes
    assert 1 <= active <= dp
    per_rank = args.queue_requests_per_lane
    total_requests = per_rank * active
    assert (
        per_rank * args.computed_rows_per_request
        >= (args.warmup_steps + args.measure_steps + 1) * args.prefill_rows_per_rank
    )
    assert server.max_num_seqs >= per_rank * dp
    args.out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")

    def stage(name, **fields):
        value = {"stage": name, **fields}
        save("state.json", value)
        print(json.dumps(value), flush=True)

    config = get_config(server.model, trust_remote_code=server.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(server.model, trust_remote_code=True)
    vocab_size = config.vocab_size
    sampling = {
        "max_new_tokens": 1,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": -1,
        "ignore_eos": True,
    }
    SamplingParams(**sampling).verify(vocab_size)
    excluded = set(tokenizer.added_tokens_decoder)
    vocab = [i for i in range(vocab_size) if i not in excluded]
    rng = random.Random(args.seed)
    prefix = rng.choices(vocab, k=args.prefix_tokens)
    # Different suffixes prevent a completed request from extending another's cache hit.
    suffixes = [rng.choices(vocab, k=args.new_tokens) for _ in range(total_requests)]
    save(
        "configuration.json",
        {
            "engine_args": engine_args[1:],
            "benchmark": vars(args)
            | {"out": str(args.out), "engine_log": str(args.engine_log)},
            "dp": dp,
            "requests_per_rank": per_rank,
            "requests_per_round": total_requests,
            "active_attention_lanes": active,
            "output_tokens": 1,
            "eplb": False,
            "redundant_experts": args.extra_experts,
            "routing": args.expert_routing,
            "routing_scope": "Natural expert routing without redundant expert replicas or routing simulation.",
            "cp": cp,
            "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            "suffix_pool_sha256": hashlib.sha256(
                json.dumps(suffix_hashes).encode()
            ).hexdigest(),
            "sampling": sampling | {"top_logprobs": 5},
            "differences_from_upstream": "TokenSpeed executor, explicit replica count, distinct suffixes instead of internal sequence fork, 128-row SWA prefix replay; no DeepInfer internal sampler_force_topk=1024; report both physical and logical rows.",
        },
    )
    driver = None
    try:
        stage("loading")
        driver = WireDriver(
            engine_args[1:],
            num_engines=dp,
            routing="round_robin",
            startup_timeout_s=7200,
            env=os.environ.copy(),
        )
        save("engines.json", driver.engines())

        def request(rid, ids, lane):
            r = generate_request(
                ids,
                sampling,
                data_parallel_rank=lane,
                return_logprob=True,
                top_logprobs_num=5,
            )
            r.rid = rid
            return r

        async def queued_round(tag, *, profile=False):
            encoder = MsgpackEncoder()
            results = {}
            finished = set()

            async def submit():
                for i, suffix in enumerate(suffixes):
                    r = request(f"{tag}-request-{i}", prefix + suffix, i % active)
                    results[r.rid] = Completion()
                    await driver.frontend.generate(
                        [bytes(f) for f in encoder.encode(r)]
                    )

            async def receive():
                while len(finished) < len(suffixes):
                    for rid, chunk in await driver.frontend.next_outputs():
                        if isinstance(chunk, Exception):
                            raise chunk
                        if rid not in results:
                            raise RuntimeError(f"unexpected output {rid}")
                        results[rid].add(chunk)
                        if chunk["finish_reason"] is not None:
                            if rid in finished:
                                raise RuntimeError(f"duplicate completion {rid}")
                            finished.add(rid)

            async def capture():
                count = 0
                last_step = None
                deadline = time.monotonic() + 1800
                with args.engine_log.open() as stream:
                    stream.seek(0, 2)
                    while count < 5:
                        assert (
                            time.monotonic() < deadline
                        ), "No five full Prefill steps for bounded profile"
                        position = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith("\n"):
                            stream.seek(position)
                            await asyncio.sleep(0.05)
                            continue
                        if MARKER not in line:
                            continue
                        sample = json.loads(
                            re.sub(r"\x1b\[[0-9;]*m", "", line.split(MARKER, 1)[1])
                        )
                        if sample["rank"] != 0:
                            continue
                        full = (
                            sample["request_ids"]
                            and all(
                                x.startswith(tag + "-request-")
                                for x in sample["request_ids"]
                            )
                            and sample["global_decodes"] == [0] * dp
                            and sample["num_extends"] == len(sample["request_ids"])
                            and sample["prefill_rows"] == args.prefill_rows_per_rank
                            and sample["global_rows"]
                            == [args.prefill_rows_per_rank] * active
                            + [0] * (dp - active)
                        )
                        if not full:
                            count = 0
                            last_step = None
                            continue
                        if last_step is not None and sample["step"] != last_step + 1:
                            count = 0
                        count += 1
                        last_step = sample["step"]
                request = ProfileReq(
                    type=ProfileReqType.START_PROFILE,
                    output_dir=str(args.out / "traces"),
                    num_steps=args.profile_steps,
                    activities=["CPU", "GPU"],
                    with_stack=False,
                    record_shapes=False,
                    profile_by_stage=False,
                    profile_id=tag,
                )
                replies = await driver.frontend.control(
                    [zmq_wire.encode_control(request)]
                )
                decoded = [zmq_wire.decode_control_output(x[0]) for x in replies]
                save(
                    "profile-rpc.json",
                    {
                        "after_rank0_step": last_step,
                        "full_steps": count,
                        "replies": [
                            {"success": x.success, "message": x.message}
                            for x in decoded
                        ],
                    },
                )
                assert len(decoded) == dp and all(x.success for x in decoded)

            if profile:
                await asyncio.gather(capture(), submit(), receive())
            else:
                await asyncio.gather(submit(), receive())
            return results

        summaries = []
        for iteration in range(0, args.rounds + 1 + bool(args.profile_steps)):
            profiling = iteration > args.rounds
            tag = f"steady-s{args.prefill_rows_per_rank}-" + (
                "profile" if profiling else f"r{iteration}"
            )
            stage("priming", round=iteration)
            deadline = time.monotonic() + 60
            attempts = 0
            while not all(v.success for v in driver.control(FlushCacheReqInput())):
                attempts += 1
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "engine/cache work did not drain before flush within 60s"
                    )
                if attempts == 1:
                    stage("waiting-for-idle-cache", round=iteration)
                time.sleep(0.1)
            stage("priming", round=iteration)
            primes = driver.generate(
                [request(f"{tag}-prime-{lane}", prefix, lane) for lane in range(active)]
            )
            assert len(primes) == active and all(
                p.finish_reason != "abort" and p.completion_tokens == 1 for p in primes
            )
            save(f"r{iteration}-primes.json", [dataclasses.asdict(p) for p in primes])
            stage("queued-prefill", round=iteration, requests=total_requests)
            results = asyncio.run(queued_round(tag, profile=profiling))
            assert len(results) == total_requests
            for rid, result in results.items():
                assert result.finish_reason != "abort", (
                    rid,
                    dataclasses.asdict(result),
                )
                assert (
                    result.prompt_tokens,
                    result.cached_tokens,
                    result.completion_tokens,
                ) == (args.prefix_tokens + args.new_tokens, args.prefix_tokens, 1), (
                    rid,
                    dataclasses.asdict(result),
                )
            save(
                f"r{iteration}-requests.json",
                {rid: dataclasses.asdict(r) for rid, r in results.items()},
            )
            if iteration == 0:
                stage("warmup-complete", requests=len(results))
                continue
            if profiling:
                traces = list((args.out / "traces").glob("*.trace.json.gz"))
                assert len(traces) == 8, ("Expected all8 GPU profiles", len(traces))
                save(
                    "trace-inventory.json",
                    [
                        {
                            "file": p.name,
                            "bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        }
                        for p in traces
                    ],
                )
                continue
            by_rank = defaultdict(list)
            for sample in read_samples(args.engine_log):
                by_rank[sample["rank"]].append(sample)
            assert all(rank in by_rank for rank in range(active)), sorted(by_rank)
            windows = {
                rank: steady_window(
                    records,
                    request_prefix=tag + "-request-",
                    rows=args.prefill_rows_per_rank,
                    dp_size=dp,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    active_lanes=active,
                )
                for rank, records in by_rank.items()
                if rank < active
            }
            assert all(
                w["computed_rows"] == args.prefill_rows_per_rank * args.measure_steps
                for w in windows.values()
            )
            # DeepInfer reports DP rank 0; retain all ranks for a skew audit.
            summary = {
                "round": iteration,
                "prefill_rows_per_rank": args.prefill_rows_per_rank,
                "requests": len(results),
                "rank0": windows[0]
                | {
                    "new_tokens_s_gpu": windows[0]["new_tokens_s_gpu"] * active / 8,
                    "computed_tokens_s_gpu": windows[0]["computed_tokens_s_gpu"]
                    * active
                    / 8,
                },
                "cp": cp,
                "non_cached_tokens": args.new_tokens,
                "prefill_batch_size": args.prefill_rows_per_rank
                // args.computed_rows_per_request
                * args.new_tokens,
                "ranks": windows,
                "errors": 0,
            }
            summaries.append(summary)
            save(f"r{iteration}-window.json", summary)
            save("summary.json", summaries)
            print(json.dumps(summary), flush=True)
        stage("complete", rounds=args.rounds, requests=total_requests * args.rounds)
    except BaseException as error:
        stage("failed", error=repr(error))
        raise
    finally:
        if driver is not None:
            driver.shutdown()


if __name__ == "__main__":
    main()
