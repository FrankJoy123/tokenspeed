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

"""One persistent Decode engine, 200 full warm steps and two formal cohorts."""

import argparse
import asyncio
import dataclasses
import json
import os
import time
from pathlib import Path

from analysis import (
    BATCHES,
    CONTEXT,
    FORMAL_INTERVALS,
    FORMAL_SKIP,
    SAMPLING,
    WARM_STEPS,
    WORLD_SIZE,
    output_budget,
)
from audit import collect_node, validate_run_identity
from inputs import common_prompt, digest, request_tokens
from tokenspeed_engine import FrontendListener
from transformers import AutoTokenizer

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    MsgpackEncoder,
    PauseSchedulerReqInput,
    RebalanceExpertsReqInput,
    ResumeSchedulerReqInput,
)
from tokenspeed.runtime.entrypoints.wire_driver import Completion, generate_request


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ["out", "model", "leader"]:
        parser.add_argument(
            "--" + name, required=True, type=Path if name == "out" else str
        )
    for name in ["context", "rounds", "world-size", "port-base"]:
        parser.add_argument("--" + name, required=True, type=int)
    parser.add_argument("--run-identity", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument("--acceptance-mode", choices=["fixed"], required=True)
    args = parser.parse_args()
    eplb_warmup_forwards = int(os.environ.get("PARETO_EPLB_WARMUP_FORWARDS", "0"))
    if eplb_warmup_forwards and eplb_warmup_forwards < 200:
        parser.error("EPLB requires at least 200 forwards after its load snapshot")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "state.json").exists(), "Choose a fresh output directory"
    assert (
        args.context == CONTEXT and args.rounds == 2 and args.world_size == WORLD_SIZE
    )
    assert args.batches and len(set(args.batches)) == len(args.batches)
    assert all(batch in BATCHES for batch in args.batches)
    import hashlib

    fixed = validate_run_identity(json.loads(args.run_identity.read_text()))
    assert (
        fixed["model"]["checkpoint"] == args.model
    ), "Checkpoint differs from frozen identity"
    (out / "run-identity.json").write_bytes(args.run_identity.read_bytes())
    save(
        out / "config.json",
        {
            "protocol": "decode",
            "protocol_version": 1,
            "batches": args.batches,
            "world_size": WORLD_SIZE,
            "rounds": 2,
            "context": CONTEXT,
            "sampling": SAMPLING,
            "warm_steps": WARM_STEPS,
            "formal_skip": FORMAL_SKIP,
            "formal_intervals": FORMAL_INTERVALS,
            "actual_width": 6,
            "eplb_warmup_forwards": eplb_warmup_forwards,
            "run_identity_sha256": hashlib.sha256(
                args.run_identity.read_bytes()
            ).hexdigest(),
            "output_budgets": {
                f"b{batch}-r{cohort}": output_budget(batch, cohort)
                for batch in args.batches
                for cohort in (0, 1, 2)
            },
        },
    )
    save(
        out / "source-sha256.json",
        {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
    )

    def state(stage, **fields):
        value = {"stage": stage, "time": time.time(), **fields}
        save(out / "state.json", value)
        print(json.dumps(value), flush=True)

    try:
        listener = FrontendListener.bind(
            handshake_address=f"tcp://{args.leader}:{args.port_base}",
            input_address=f"tcp://{args.leader}:{args.port_base + 1}",
            output_address=f"tcp://{args.leader}:{args.port_base + 2}",
            first_engine_index=0,
            num_engines=args.world_size,
            routing="round_robin",
            handshake_timeout_s=7200,
            send_timeout_s=120,
        )
        pending = listener.accept()
        save(out / "listener-ready.json", {"handshake": listener.handshake_address})
        state("waiting-for-16-engines")
        frontend = pending.wait()
        assert len(list(frontend.engines())) == 16
        save(out / "engines.json", list(frontend.engines()))
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        excluded = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
        vocab = [i for i in range(tokenizer.vocab_size) if i not in excluded]
        prompt = common_prompt(vocab, args.context)
        prefix = prompt[:-6656]
        prime_prompt = prefix + prompt[-6656:-6400]
        save(
            out / "input-identity.json",
            {
                "seed": 37,
                "vocab_size": tokenizer.vocab_size,
                "excluded_token_ids": sorted(excluded),
                "context": args.context,
                "prefix_tokens": len(prefix),
                "prefix_sha256": digest(prefix),
                "prime_prompt_tokens": len(prime_prompt),
                "prime_prompt_sha256": digest(prime_prompt),
                "alignment_guard_tokens": 256,
                "suffix_tokens": 6656,
                "suffix_key": "DECODE:37:cohort:batch:rank:index",
                "cohort_input_uniqueness": 3 * args.world_size * sum(args.batches),
                "routing": "natural",
                "fixed_acceptance": 6,
                "engine_restarts_between_batches": 0,
                "cache_flushes_between_batches": 0,
            },
        )
        all_suffix_hashes = set()
        encoder = MsgpackEncoder()
        sampling = SAMPLING | {"ignore_eos": True}

        async def control(req):
            replies = await frontend.control([zmq_wire.encode_control(req)])
            assert all(zmq_wire.decode_control_output(r[0]).success for r in replies)

        async def generate(tag, batch, output, prime=False, cohort=1, seed=37):
            results = {}
            finished = set()
            input_hashes = {}
            full_hashes = {}
            await control(PauseSchedulerReqInput(mode="keep"))

            async def submit():
                for j in range(batch):
                    for rank in range(args.world_size):
                        ids = (
                            prime_prompt
                            if prime
                            else request_tokens(
                                prompt, vocab, batch, rank, j, cohort=cohort, seed=seed
                            )
                        )
                        rid = f"{tag}-lane{rank}-r{j}"
                        if not prime:
                            h = digest(ids[-6656:])
                            full = digest(ids)
                            assert h not in all_suffix_hashes
                            all_suffix_hashes.add(h)
                            input_hashes[rid] = h
                            full_hashes[rid] = full
                        req = generate_request(
                            ids,
                            sampling | {"max_new_tokens": output},
                            data_parallel_rank=rank,
                        )
                        req.rid = rid
                        results[rid] = Completion()
                        await frontend.generate([bytes(f) for f in encoder.encode(req)])
                await control(ResumeSchedulerReqInput())

            async def receive():
                while len(finished) < args.world_size * batch:
                    for rid, chunk in await frontend.next_outputs():
                        if isinstance(chunk, Exception):
                            raise chunk
                        results[rid].add(chunk)
                        if chunk["finish_reason"] is not None:
                            assert rid not in finished
                            finished.add(rid)

            await asyncio.gather(submit(), receive())
            save(
                out / f"{tag}-requests.json",
                {rid: dataclasses.asdict(x) for rid, x in results.items()},
            )
            assert all(
                x.finish_reason != "abort" and x.completion_tokens == output
                for x in results.values()
            )
            if not prime:
                assert len(set(input_hashes.values())) == args.world_size * batch
                assert all(
                    x.prompt_tokens == args.context and x.cached_tokens == len(prefix)
                    for x in results.values()
                ), "Public prefix evicted, suffix accidentally cached, or prefix length changed"
                save(
                    out / f"{tag}-input-hashes.json",
                    {
                        "prefix_sha256": digest(prefix),
                        "suffix_sha256_by_request": input_hashes,
                        "input_sha256_by_request": full_hashes,
                    },
                )
            return results

        # One flush/real prime only. A256-token alignment guard keeps the
        # entire reusable prefix inside completed cache blocks.
        state("priming-common-prefix", prefix_tokens=len(prefix))
        t = time.monotonic()
        asyncio.run(control(FlushCacheReqInput()))
        asyncio.run(generate("common-prime", 1, 1, prime=True))
        save(
            out / "common-prime-timing.json",
            {
                "seconds": time.monotonic() - t,
                "flush_count": 1,
                "real_prime_count_per_rank": 1,
            },
        )
        summaries = []
        eplb_before_formal = None
        engine_log = out.parent / "engine.log"
        for batch in args.batches:
            for cohort in [0, 1, 2]:
                budget = output_budget(batch, cohort)
                state(
                    "warmup-running" if cohort == 0 else "formal-running",
                    batch=batch,
                    round=cohort,
                    requests=args.world_size * batch,
                )
                t = time.monotonic()
                results = asyncio.run(
                    generate(
                        f"b{batch}-r{cohort}",
                        batch,
                        budget["max_new_tokens"],
                        cohort=cohort,
                    )
                )
                generation_seconds = time.monotonic() - t
                time.sleep(0.5)
                node = collect_node(out, args.world_size, live=True)
                assert node["ranks"] == list(range(8))
                observations = node["warmups" if cohort == 0 else "windows"][
                    f"b{batch}-r{cohort}"
                ]
                assert all(
                    "error" not in row for row in observations.values()
                ), observations
                windows = {} if cohort == 0 else observations
                row = {
                    "batch_per_rank": batch,
                    "round": cohort,
                    "warmup_round": cohort == 0,
                    "formal_intervals": 0 if cohort == 0 else FORMAL_INTERVALS,
                    "warmup_observations": observations if cohort == 0 else {},
                    "local_observed_ranks": node["ranks"],
                    "global_16rank_audit_pending": True,
                    "termination_budget": budget,
                    "windows": windows,
                    "context_tokens": args.context,
                    "cached_tokens_min": min(x.cached_tokens for x in results.values()),
                    "generation_wall_seconds": generation_seconds,
                    "cohort_wall_seconds": time.monotonic() - t,
                    "output_tokens": budget["max_new_tokens"],
                    "semantic_scope": "Raw warm cohort requires 200 full steps; each independent formal cohort discards100 full points then measures50 intervals from51 endpoints; token budgets are termination margins only",
                }
                save(out / f"b{batch}-r{cohort}-point.json", row)
                summaries.append(row)
                save(out / "summary.json", summaries)
                state("point-complete", batch=batch, round=cohort)
                if eplb_warmup_forwards and eplb_before_formal is None:
                    start_log_offset = engine_log.stat().st_size
                    state("eplb-warmup", batch=batch, forwards=eplb_warmup_forwards)
                    asyncio.run(control(RebalanceExpertsReqInput()))
                    asyncio.run(
                        generate(
                            "eplb-warm",
                            batch,
                            eplb_warmup_forwards * 6 + 6,
                            cohort=0,
                            seed=369,
                        )
                    )
                    eplb_before_formal = engine_log.stat().st_size
                    with engine_log.open("rb") as log:
                        log.seek(start_log_offset)
                        assert b"Expert rebalance 1 applied" in log.read(
                            eplb_before_formal - start_log_offset
                        )
                    save(
                        out / "eplb-warmup.json",
                        {
                            "start_log_offset": start_log_offset,
                            "formal_log_offset": eplb_before_formal,
                            "requested_forwards": eplb_warmup_forwards,
                        },
                    )
        calibration_requests = args.batches[0] if eplb_warmup_forwards else 0
        assert len(all_suffix_hashes) == args.world_size * (
            3 * sum(args.batches) + calibration_requests
        )
        if eplb_before_formal is not None:
            with engine_log.open("rb") as log:
                log.seek(eplb_before_formal)
                assert (
                    b"Expert rebalance" not in log.read()
                ), "EPLB moved during formal sampling"
        state(
            "complete",
            measurements=2 * len(args.batches),
            world_size=args.world_size,
        )
    except BaseException as error:
        state("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
