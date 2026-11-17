#!/usr/bin/env python3
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

"""Audit three complete frozen mmmu-pro-210 runs and summarize their variation."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import statistics
from pathlib import Path

PROTOCOL_FIELDS = (
    "dataset",
    "model",
    "client_sha256",
    "concurrency",
    "max_tokens",
    "reasoning_effort",
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "automatic_request_retries",
    "timeout",
    "count",
)


def validate_protocols(protocols):
    if len(protocols) != 3:
        raise ValueError("Exactly three runs are required")
    if any(p["count"] != 210 for p in protocols):
        raise ValueError("Every run must cover all 210 questions")
    for key in PROTOCOL_FIELDS:
        if any(p[key] != protocols[0][key] for p in protocols[1:]):
            raise ValueError(f"Run protocols differ in {key}")


def score_statistics(scores):
    if len(scores) != 3 or any(
        s["total"] != 210 or s["failed_requests"] for s in scores
    ):
        raise ValueError(
            "Expected three complete scores with denominator 210 and no request failures"
        )
    correct = [s["correct"] for s in scores]
    if any(not 0 <= n <= 210 for n in correct):
        raise ValueError("Correct count outside the fixed denominator")
    return {
        "runs": 3,
        "questions_per_run": 210,
        "correct_per_run": correct,
        "correct_mean": statistics.mean(correct),
        "correct_sample_std": statistics.stdev(correct),
        "correct_min": min(correct),
        "correct_max": max(correct),
        "accuracy_mean_percent": statistics.mean(correct) / 210 * 100,
        "accuracy_min_percent": min(correct) / 210 * 100,
        "accuracy_max_percent": max(correct) / 210 * 100,
        "failed_requests_per_run": [s["failed_requests"] for s in scores],
        "truncated_per_run": [s["truncated"] for s in scores],
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs=3, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len({r.resolve() for r in args.runs}) != 3:
        raise ValueError("Three distinct run directories are required")
    protocols = [json.loads((r / "protocol.json").read_text()) for r in args.runs]
    validate_protocols(protocols)
    manifest = protocols[0]["dataset"]
    if (
        digest(args.data) != manifest["dataset_sha256"]
        or digest(args.reference) != manifest["reference_scorer_sha256"]
    ):
        raise ValueError("Data or original scorer differs from the frozen manifest")
    spec = importlib.util.spec_from_file_location(
        "mmmu_pro_210_reference", args.reference
    )
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    with contextlib.redirect_stdout(io.StringIO()):
        data = ref.load_dataset(str(args.data))
    if [r["instance_id"] for r in data] != manifest["sample_ids"]:
        raise ValueError("Dataset IDs differ from the manifest")
    scores, per_run, request_hashes = [], [], []
    for directory, protocol in zip(args.runs, protocols, strict=True):
        rows = [
            json.loads(line)
            for line in (directory / "responses.jsonl").read_text().splitlines()
        ]
        rows.sort(key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(210)) or [
            r["instance_id"] for r in rows
        ] != manifest["sample_ids"]:
            raise ValueError(f"Incomplete or duplicate run: {directory}")
        hashes = []
        for row, item in zip(rows, data, strict=True):
            payload = ref.build_request(
                row["index"],
                item,
                protocol["url"].rstrip("/") + "/v1/chat/completions",
                protocol["model"],
                protocol["max_tokens"],
                0,
                1,
                1,
                use_base=False,
                openai_format=True,
            )
            payload = {k: v for k, v in payload.items() if not k.startswith("_")}
            payload.update(
                seed=protocol["seed"], reasoning_effort=protocol["reasoning_effort"]
            )
            rebuilt = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False).encode()
            ).hexdigest()
            if rebuilt != row["request_sha256"]:
                raise ValueError(f"Request body differs: {directory}, {row['index']}")
            hashes.append(rebuilt)
        request_hashes.append(hashes)
        adapted = [{**r, "item": data[r["index"]], "use_base": False} for r in rows]
        with contextlib.redirect_stdout(io.StringIO()):
            score = ref.evaluate(adapted)
            individual = [ref.evaluate([r]) for r in adapted]
        score.pop("wrong_cases", None)
        reported = json.loads((directory / "summary.json").read_text())
        for key in ("total", "correct", "failed_requests", "truncated"):
            if score[key] != reported[key]:
                raise ValueError(f"Saved score differs in {key}: {directory}")
        scores.append(score)
        per_run.append(individual)
    if any(h != request_hashes[0] for h in request_hashes[1:]):
        raise ValueError("Request hashes differ between reruns")
    questions = [
        {
            "instance_id": item["instance_id"],
            "correct": [r[i]["correct"] for r in per_run],
            "truncated": [r[i]["truncated"] for r in per_run],
            "failed": [r[i]["failed_requests"] for r in per_run],
        }
        for i, item in enumerate(data)
    ]
    summary = {
        "label": args.label,
        **score_statistics(scores),
        "stable_correct_questions": sum(q["correct"] == [1, 1, 1] for q in questions),
        "stable_wrong_questions": sum(q["correct"] == [0, 0, 0] for q in questions),
        "variable_correctness_questions": sum(
            len(set(q["correct"])) > 1 for q in questions
        ),
        "independently_rebuilt_request_hashes": 630,
        "identical_request_hashes_between_runs": 210,
        "execution_identity": "Review per-run identity evidence separately; protocol equality does not prove identical builds",
        "run_directories": [str(r) for r in args.runs],
        "response_file_sha256": [digest(r / "responses.jsonl") for r in args.runs],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "per-question.json").write_text(
        json.dumps(questions, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
