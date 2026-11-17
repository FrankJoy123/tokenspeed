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

"""Formal windows: independent r1/r2 request cohorts, each110 warm full points+60 intervals."""

import argparse
import json
import re
from pathlib import Path

from analysis import window


def strict_window(samples, prefix, batch, warmup, steps, world_size):
    # Count warmup and endpoint supply only when every request really advanced6.
    guarded = [
        s
        | {
            "all_decode": s["all_decode"]
            and s["input_rows"] == 6 * batch
            and s["output_lengths"] == [6] * batch
        }
        for s in samples
    ]
    return window(
        guarded, prefix, batch, warmup=warmup, steps=steps, world_size=world_size
    )


def validate_fixed_width(samples, result, batch):
    selected = [
        s
        for s in samples
        if result["first_ns"] <= s["completed_ns"] <= result["last_ns"]
    ]
    if len(selected) != 61:
        raise RuntimeError("Expected61 endpoints per60 intervals")
    if any(
        s["input_rows"] != 6 * batch or s["output_lengths"] != [6] * batch
        for s in selected
    ):
        raise RuntimeError("Every request must verify/keep exactly6 tokens")


def collect_node(root, world_size):
    root = Path(root)
    by_rank = {}
    tags = set()
    seen = set()
    for path in root.glob("steps-*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.endswith("}"):
                continue
            s = json.loads(line)
            key = (s["rank"], s["step"])
            if key in seen:
                raise RuntimeError("Duplicate rank/step")
            seen.add(key)
            by_rank.setdefault(s["rank"], []).append(s)
            if s["request_ids"]:
                tag = s["request_ids"][0].split("-lane", 1)[0]
                if re.fullmatch(r"b(?:1|2|4|8|16|32)-r[12]", tag):
                    tags.add(tag)
    output = {}
    for tag in sorted(tags, key=lambda x: int(x.split("-")[0][1:])):
        batch = int(tag.split("-")[0][1:])
        rows = {}
        for rank, samples in by_rank.items():
            try:
                result = strict_window(
                    sorted(samples, key=lambda s: s["step"]),
                    tag + "-lane",
                    batch,
                    warmup=110,
                    steps=60,
                    world_size=world_size,
                )
                validate_fixed_width(samples, result, batch)
            except RuntimeError as error:
                result = {"error": str(error)}
            rows[rank] = result
        output[tag] = rows
    return {
        "windows": output,
        "ranks": sorted(by_rank),
        "source_sha256": json.loads((root / "source-sha256.json").read_text()),
    }


def audit_inputs(root, config):
    from inputs import common_prompt, digest, request_tokens

    identity = json.loads((root / "input-identity.json").read_text())
    excluded = set(identity["excluded_token_ids"])
    vocab = [i for i in range(identity["vocab_size"]) if i not in excluded]
    prompt = common_prompt(vocab, context=identity["context"], seed=identity["seed"])
    assert digest(prompt[:-6656]) == identity["prefix_sha256"]
    seen = set()
    count = 0
    for batch in config["batches"]:
        for cohort in [0, 1, 2]:
            tag = f"b{batch}-r{cohort}"
            requests = json.loads((root / f"{tag}-requests.json").read_text())
            hashes = json.loads((root / f"{tag}-input-hashes.json").read_text())
            assert len(requests) == batch * config["world_size"]
            for rank in range(config["world_size"]):
                for index in range(batch):
                    rid = f"{tag}-lane{rank}-r{index}"
                    row = requests[rid]
                    assert (
                        row["prompt_tokens"] == 131072
                        and row["cached_tokens"] == 124416
                    )
                    assert (
                        row["completion_tokens"] == 1152
                        and row["finish_reason"] != "abort"
                    )
                    values = request_tokens(
                        prompt, vocab, batch, rank, index, cohort=cohort
                    )
                    suffix = digest(values[-6656:])
                    full = digest(values)
                    assert suffix not in seen
                    seen.add(suffix)
                    count += 1
                    assert hashes["suffix_sha256_by_request"][rid] == suffix
                    assert hashes["input_sha256_by_request"][rid] == full
    return {
        "complete_requests": count,
        "unique_suffixes": len(seen),
        "every_cached_tokens": 124416,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    windows = {}
    ranks = []
    for root in args.run_dir:
        node = collect_node(root, args.world_size)
        ranks += node["ranks"]
        for tag, rows in node["windows"].items():
            bucket = windows.setdefault(tag, {})
            assert not bucket.keys() & rows.keys()
            bucket.update(rows)
    assert sorted(ranks) == list(range(args.world_size))
    config = json.loads((args.run_dir[0] / "config.json").read_text())
    assert len(windows) == 2 * len(config["batches"])
    for rows in windows.values():
        assert set(rows) == set(range(args.world_size))
        assert all(
            "error" not in row and row["actual_kept_tokens_per_request_step"] == 6
            for row in rows.values()
        )
    input_audit = audit_inputs(args.run_dir[0], config)
    args.out.write_text(
        json.dumps({"windows": windows, "input_audit": input_audit}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "all_ranks": len(ranks),
                "all_tags": len(windows),
                "formal_intervals_per_batch": 120,
            }
        )
    )


if __name__ == "__main__":
    main()
