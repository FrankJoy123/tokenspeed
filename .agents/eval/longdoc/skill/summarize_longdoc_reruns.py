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

"""Audit and summarize three complete runs of a frozen DocQA subset."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import longdoc
from longdoc_scoring import eval_docqa_score, parse_output

PROTOCOL_FIELDS = (
    "model",
    "manifest_sha256",
    "image_inventory_sha256",
    "client_sha256",
    "scorer_sha256",
    "sample_ids",
    "concurrency",
    "max_tokens",
    "temperature",
    "top_p",
    "seed",
    "reasoning_effort",
    "timeout_seconds",
    "automatic_request_retries",
    "metric",
)


def validate_protocols(protocols):
    if len(protocols) != 3:
        raise ValueError("Exactly three runs are required")
    count = len(protocols[0]["sample_ids"])
    if count == 0:
        raise ValueError("The frozen subset must contain unique questions")
    for protocol in protocols:
        if (
            len(protocol["sample_ids"]) != count
            or len(set(protocol["sample_ids"])) != count
        ):
            raise ValueError(
                f"Every run must contain the same {count} unique questions"
            )
    for field in PROTOCOL_FIELDS:
        if any(p[field] != protocols[0][field] for p in protocols[1:]):
            raise ValueError(f"Run protocols differ in {field}")


def score_statistics(scores, expected_questions):
    if len(scores) != 3:
        raise ValueError("Expected three complete runs without errors")
    count = expected_questions
    if count <= 0 or any(
        s["expected"] != count or s["completed"] != count or s["errors"] or s["skipped"]
        for s in scores
    ):
        raise ValueError(
            f"Expected three complete {count}-question runs without errors"
        )
    values = [s["doc_qa_score"] * 100 for s in scores]
    if any(not 0 <= value <= 100 for value in values):
        raise ValueError("Score outside [0, 100]")
    return {
        "runs": 3,
        "questions_per_run": count,
        "doc_qa_per_run_percent": values,
        "doc_qa_mean_percent": statistics.mean(values),
        "doc_qa_sample_std_percentage_points": statistics.stdev(values),
        "doc_qa_min_percent": min(values),
        "doc_qa_max_percent": max(values),
        "strict_doc_qa_per_run_percent": [
            s["strict_doc_qa_score"] * 100 for s in scores
        ],
        "errors_per_run": [s["errors"] for s in scores],
        "truncated_per_run": [s["output_truncated"] for s in scores],
        "wall_seconds_per_run": [s["wall_seconds"] for s in scores],
    }


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({r.resolve() for r in args.runs}) != 3:
        raise ValueError("Three distinct run directories are required")
    protocols = [read(r / "protocol.json") for r in args.runs]
    validate_protocols(protocols)
    protocol = protocols[0]
    if sha(args.manifest) != protocol["manifest_sha256"]:
        raise ValueError("Manifest differs from the executed runs")
    if sha(args.data / "images-sha256.json") != protocol["image_inventory_sha256"]:
        raise ValueError("Image inventory differs from the executed runs")
    if (
        sha(Path(longdoc.__file__)) != protocol["client_sha256"]
        or sha(Path(longdoc.__file__).with_name("longdoc_scoring.py"))
        != protocol["scorer_sha256"]
    ):
        raise ValueError("Client or scorer differs from the executed runs")
    rows = longdoc.load_rows(args.data, read(args.manifest))
    if [r["id"] for r in rows] != protocol["sample_ids"]:
        raise ValueError("Selected IDs differ from the executed runs")
    inventory = read(args.data / "images-sha256.json")
    hashes = {}
    for row in rows:
        payload = {
            "model": protocol["model"],
            "messages": longdoc.messages(row, args.data, inventory),
            "temperature": protocol["temperature"],
            "top_p": protocol["top_p"],
            "seed": protocol["seed"],
            "max_tokens": protocol["max_tokens"],
            "reasoning_effort": protocol["reasoning_effort"],
            "stream": False,
        }
        hashes[row["id"]] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False).encode()
        ).hexdigest()
    summaries, per_run = [], []
    for directory in args.runs:
        responses = [
            json.loads(line)
            for line in (directory / "responses.jsonl").read_text().splitlines()
        ]
        indexed = {r["id"]: r for r in responses}
        if len(responses) != len(indexed) or set(indexed) != set(hashes):
            raise ValueError(f"Incomplete or duplicate run: {directory}")
        rescored, strict, truncated = [], [], 0
        for row in rows:
            result = indexed[row["id"]]
            if not result["ok"] or result["request_sha256"] != hashes[row["id"]]:
                raise ValueError(
                    f"Request failed or input differs: {directory}, {row['id']}"
                )
            choice = result["response"]["choices"][0]
            content = choice["message"].get("content") or ""
            finish = choice["finish_reason"]
            parsed = parse_output("Answer:" + content, prefix="Answer:")
            score = float(
                eval_docqa_score(row["answer"], parsed or "", row["answer_format"])
            )
            strict_score = 0.0 if finish == "length" else score
            if (
                content != result["content"]
                or finish != result["finish_reason"]
                or not math.isclose(score, result["score"], abs_tol=1e-12)
                or not math.isclose(strict_score, result["strict_score"], abs_tol=1e-12)
            ):
                raise ValueError(
                    f"Saved response or score differs: {directory}, {row['id']}"
                )
            rescored.append(score)
            strict.append(strict_score)
            truncated += finish == "length"
        saved = read(directory / "summary.json")
        if (
            not math.isclose(
                statistics.mean(rescored), saved["doc_qa_score"], abs_tol=1e-12
            )
            or not math.isclose(
                statistics.mean(strict), saved["strict_doc_qa_score"], abs_tol=1e-12
            )
            or truncated != saved["output_truncated"]
        ):
            raise ValueError(f"Saved summary differs: {directory}")
        summaries.append(saved)
        per_run.append(rescored)
    questions = [
        {"id": row["id"], "scores": [run[index] for run in per_run]}
        for index, row in enumerate(rows)
    ]
    summary = {
        **score_statistics(summaries, len(rows)),
        "stable_score_questions": sum(len(set(q["scores"])) == 1 for q in questions),
        "variable_score_questions": sum(len(set(q["scores"])) > 1 for q in questions),
        "rebuilt_request_hashes": len(rows),
        "response_request_hashes_verified": len(rows) * len(args.runs),
        "independently_rescored_responses": len(rows) * len(args.runs),
        "response_file_sha256": [sha(r / "responses.jsonl") for r in args.runs],
        "execution_identity": "Compare per-run before/after identity evidence separately",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    longdoc.write_json(args.output / "summary.json", summary)
    longdoc.write_json(args.output / "per-question.json", questions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
