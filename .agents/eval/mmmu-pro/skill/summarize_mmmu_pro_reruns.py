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

"""Rebuild requests and rescore three official MMMU-Pro subset runs."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import mmmu_pro

PROTOCOL_FIELDS = (
    "model",
    "concurrency",
    "repeats",
    "prompt_style",
    "max_tokens",
    "reasoning_effort",
    "temperature",
    "seed",
    "timeout",
    "manifest_sha256",
    "script_sha256",
    "sample_ids",
    "scorer_revision",
    "scorer_sha256",
    "automatic_request_retries",
)


def audit(directory, protocol, data, manifest):
    samples = {s["id"]: s for s in manifest["samples"]}
    scorer = mmmu_pro.official_scorer(data, manifest)
    rows = [
        json.loads(line)
        for line in (directory / "responses.jsonl").read_text().splitlines()
    ]
    if protocol["repeats"] != 1:
        raise ValueError("Use one complete subset per run")
    if len(rows) != len(samples) or {r["id"] for r in rows} != set(samples):
        raise ValueError(f"Incomplete or duplicate run: {directory}")
    correct = truncated = 0
    for result in rows:
        if not result["ok"] or result["repeat"] != 0:
            raise ValueError(f"Request failed or unexpected repeat: {directory}")
        sample = samples[result["id"]]
        payload = dict(
            model=protocol["model"],
            messages=mmmu_pro.messages(
                sample, data, prompt_style=protocol["prompt_style"]
            ),
            temperature=protocol["temperature"],
            max_tokens=protocol["max_tokens"],
            stream=False,
            seed=protocol["seed"],
            reasoning_effort=protocol["reasoning_effort"],
        )
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False).encode()
        ).hexdigest()
        if digest != result["request_sha256"]:
            raise ValueError(f"Request body differs: {directory}, {sample['id']}")
        choice = result["response"]["choices"][0]
        if (
            choice["finish_reason"] != result["finish_reason"]
            or (choice["message"].get("content") or "") != result["content"]
        ):
            raise ValueError(f"Saved answer differs: {directory}, {sample['id']}")
        recomputed = mmmu_pro.score_result(
            result, sample, scorer, seed=protocol["seed"]
        )
        if any(result[key] != value for key, value in recomputed.items()):
            raise ValueError(f"Saved score differs: {directory}, {sample['id']}")
        correct += recomputed["correct"]
        truncated += result["finish_reason"] == "length"
    saved = json.loads((directory / "summary.json").read_text())
    if (
        saved["correct"] != correct
        or saved["output_truncated"] != truncated
        or saved["errors"]
    ):
        raise ValueError(f"Saved summary differs: {directory}")
    return dict(correct=correct, total=len(samples), truncated=truncated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({r.resolve() for r in args.runs}) != 3:
        raise ValueError("Three distinct run directories are required")
    protocols = [json.loads((r / "protocol.json").read_text()) for r in args.runs]
    for field in PROTOCOL_FIELDS:
        if any(p[field] != protocols[0][field] for p in protocols[1:]):
            raise ValueError(f"Run protocols differ in {field}")
    manifest_path = args.data / "manifest.json"
    if mmmu_pro.sha(manifest_path.read_bytes()) != protocols[0]["manifest_sha256"]:
        raise ValueError("Manifest differs from executed runs")
    manifest = json.loads(manifest_path.read_text())
    if [s["id"] for s in manifest["samples"]] != protocols[0]["sample_ids"]:
        raise ValueError("Every run must cover the full frozen subset")
    results = [
        audit(d, p, args.data, manifest)
        for d, p in zip(args.runs, protocols, strict=True)
    ]
    values = [r["correct"] / r["total"] * 100 for r in results]
    summary = dict(
        accuracy_per_run_percent=values,
        accuracy_mean_percent=statistics.mean(values),
        accuracy_sample_std_percentage_points=statistics.stdev(values),
        accuracy_min_percent=min(values),
        accuracy_max_percent=max(values),
        truncated_per_run=[r["truncated"] for r in results],
        rebuilt_request_hashes=sum(r["total"] for r in results),
        response_file_sha256=[
            mmmu_pro.sha((r / "responses.jsonl").read_bytes()) for r in args.runs
        ],
        execution_identity="Compare before/after build and model identity separately",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    mmmu_pro.write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
