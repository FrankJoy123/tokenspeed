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

"""CUSTOM95: one engine/prefix, full warmup, two independent formal cohorts."""

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path

from audit import collect_node
from inputs import common_prompt, digest, request_tokens
from tokenspeed_engine import FrontendListener
from transformers import AutoTokenizer

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    MsgpackEncoder,
    PauseSchedulerReqInput,
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
    for name in ["context", "output", "rounds", "world-size", "port-base"]:
        parser.add_argument("--" + name, required=True, type=int)
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument("--profile-batches", nargs="*", type=int, required=True)
    parser.add_argument("--acceptance-mode", choices=["fixed"], required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "state.json").exists(), "Choose a fresh output directory"
    save(
        out / "config.json",
        {
            "batches": args.batches,
            "world_size": args.world_size,
            "output": args.output,
            "protocol": "CUSTOM95",
        },
    )
    import hashlib

    save(
        out / "source-sha256.json",
        {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
    )
    assert (
        args.output == 1152
        and args.context == 131072
        and args.rounds == 2
        and args.world_size == 16
        and not args.profile_batches
    )
    assert (
        args.batches
        and len(set(args.batches)) == len(args.batches)
        and all(b in (1, 2, 4, 8, 16, 32) for b in args.batches)
    )

    def state(stage, **fields):
        value = {"stage": stage, "time": time.time(), **fields}
        save(out / "state.json", value)
        print(json.dumps(value), flush=True)

    try:
        listener = FrontendListener.bind(
            handshake_address=f"tcp://{args.leader}:{args.port_base}",
            input_address=f"tcp://{args.leader}:{args.port_base+1}",
            output_address=f"tcp://{args.leader}:{args.port_base+2}",
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
                "suffix_key": "r1 legacy NEW-W4A8:37:batch:rank:index; r0/r2 CUSTOM-OFFICIAL-SAMPLING:37:cohort:batch:rank:index",
                "cohort_input_uniqueness": 3 * args.world_size * sum(args.batches),
                "routing": "natural",
                "fixed_acceptance": 6,
                "engine_restarts_between_batches": 0,
                "cache_flushes_between_batches": 0,
            },
        )
        all_suffix_hashes = set()
        encoder = MsgpackEncoder()
        sampling = {"temperature": 0, "top_p": 1, "top_k": -1, "ignore_eos": True}

        async def control(req):
            replies = await frontend.control([zmq_wire.encode_control(req)])
            assert all(zmq_wire.decode_control_output(r[0]).success for r in replies)

        async def generate(tag, batch, output, prime=False, cohort=1):
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
                                prompt, vocab, batch, rank, j, cohort=cohort
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
        for batch in args.batches:
            for cohort in [0, 1, 2]:
                state(
                    "warmup-running" if cohort == 0 else "formal-running",
                    batch=batch,
                    round=cohort,
                    requests=args.world_size * batch,
                )
                t = time.monotonic()
                results = asyncio.run(
                    generate(f"b{batch}-r{cohort}", batch, args.output, cohort=cohort)
                )
                generation_seconds = time.monotonic() - t
                time.sleep(0.5)
                node = collect_node(out, args.world_size)
                assert node["ranks"] == list(range(8))
                windows = {} if cohort == 0 else node["windows"][f"b{batch}-r{cohort}"]
                assert all("error" not in row for row in windows.values()), windows
                row = {
                    "batch_per_rank": batch,
                    "round": cohort,
                    "warmup_round": cohort == 0,
                    "formal_intervals": 0 if cohort == 0 else 60,
                    "windows": windows,
                    "context_tokens": args.context,
                    "cached_tokens_min": min(x.cached_tokens for x in results.values()),
                    "generation_wall_seconds": generation_seconds,
                    "cohort_wall_seconds": time.monotonic() - t,
                    "output_tokens": args.output,
                    "semantic_scope": "Full1152 warm r0 plus two independent full1152 formal request cohorts r1/r2; each formal discards110 full points and measures60 intervals",
                }
                save(out / f"b{batch}-r{cohort}-point.json", row)
                summaries.append(row)
                save(out / "summary.json", summaries)
                state("point-complete", batch=batch, round=cohort)
        assert len(all_suffix_hashes) == 3 * args.world_size * sum(args.batches)
        state(
            "complete",
            measurements=2 * len(args.batches),
            world_size=args.world_size,
            profile_batches=[],
        )
    except BaseException as error:
        state("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
