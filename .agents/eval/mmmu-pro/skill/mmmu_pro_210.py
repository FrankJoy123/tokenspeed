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

"""Run the frozen mmmu-pro-210 set with its original prompt and scorer."""

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], required=True
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timeout", type=int, required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if (
        min(
            args.concurrency,
            args.max_tokens,
            args.timeout,
            args.limit if args.limit is not None else 1,
        )
        <= 0
    ):
        p.error("Run bounds must be positive")
    manifest = json.loads(args.manifest.read_text())
    if (
        digest(args.data) != manifest["dataset_sha256"]
        or digest(args.reference) != manifest["reference_scorer_sha256"]
    ):
        raise ValueError("Dataset or reference scorer differs from the frozen manifest")
    spec = importlib.util.spec_from_file_location(
        "mmmu_pro_210_reference", args.reference
    )
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rows = ref.load_dataset(str(args.data))
    if len(rows) != 210 or [r["instance_id"] for r in rows] != manifest["sample_ids"]:
        raise ValueError("210 IDs differ from the frozen set")
    if args.limit:
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "dataset": manifest,
        "client_sha256": digest(Path(__file__)),
        "identity": json.loads(args.identity.read_text()),
        "url": args.url,
        "model": args.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "automatic_request_retries": 0,
        "timeout": args.timeout,
        "count": len(rows),
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output / "mmmu_pro_210.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "reference_mmmu_eval.py").write_bytes(args.reference.read_bytes())

    def request(pair):
        i, row = pair
        payload = ref.build_request(
            i,
            row,
            args.url.rstrip("/") + "/v1/chat/completions",
            args.model,
            args.max_tokens,
            0,
            1,
            1,
            use_base=False,
            openai_format=True,
        )
        payload = {k: v for k, v in payload.items() if not k.startswith("_")}
        payload.update(seed=args.seed, reasoning_effort=args.reasoning_effort)
        raw = json.dumps(payload, ensure_ascii=False).encode()
        result = {
            "index": i,
            "instance_id": row["instance_id"],
            "request_sha256": hashlib.sha256(raw).hexdigest(),
            "image_count": len(row["img_urls"]),
            "started": time.time(),
        }
        start = time.monotonic()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(
                args.url.rstrip("/") + "/v1/chat/completions",
                data=raw,
                headers={"Content-Type": "application/json"},
            )
            with opener.open(req, timeout=args.timeout) as response:
                content = response.read()
            result.update(
                status="success",
                response=json.loads(content),
                response_sha256=hashlib.sha256(content).hexdigest(),
            )
        except Exception as error:
            result.update(status="error", error=repr(error))
            if isinstance(error, urllib.error.HTTPError):
                result["error_body"] = error.read().decode(errors="replace")
        result["seconds"] = time.monotonic() - start
        return result

    results = []
    start = time.monotonic()
    with (
        (args.output / "responses.jsonl").open("w") as out,
        concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool,
    ):
        futures = [pool.submit(request, pair) for pair in enumerate(rows)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            results.append(result)
            print(
                f"{len(results)}/{len(rows)} {result['instance_id']} {result['status']}",
                flush=True,
            )
    adapted = [
        {**r, "item": rows[r["index"]], "use_base": False}
        for r in sorted(results, key=lambda r: r["index"])
    ]
    summary = ref.evaluate(adapted)
    summary.pop("wrong_cases", None)
    summary.update(
        wall_seconds=time.monotonic() - start,
        expected=len(rows),
        finished=len(results),
        metric="unchanged original check_answer/evaluate; truncated outputs score zero",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
