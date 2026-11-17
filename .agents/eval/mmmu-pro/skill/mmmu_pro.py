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

"""Freeze official MMMU-Pro images and evaluate an HTTP endpoint."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import mimetypes
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET = "MMMU/MMMU_Pro"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def select_rows(rows: list[dict], *, count: int, multi_min: int) -> list[dict]:
    """Choose multi-image coverage and subject diversity without using answers."""
    candidates = sorted(rows, key=lambda r: (r["subject"], r["id"]))
    images = lambda r: sum(r.get(f"image_{i}") is not None for i in range(1, 8))
    selected = [r for r in candidates if images(r) > 1][:multi_min]
    if len(selected) != multi_min:
        raise ValueError("Candidate pool has too few multi-image questions")
    ids = {r["id"] for r in selected}
    subjects = {r["subject"] for r in selected}
    for r in candidates:
        if len(selected) == count:
            break
        if images(r) and r["id"] not in ids and r["subject"] not in subjects:
            selected.append(r)
            ids.add(r["id"])
            subjects.add(r["subject"])
    for r in candidates:
        if len(selected) == count:
            break
        if images(r) and r["id"] not in ids:
            selected.append(r)
            ids.add(r["id"])
    if len(selected) != count:
        raise ValueError("Not enough distinct image questions")
    return sorted(selected, key=lambda r: r["id"])


def prepare(args) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision + ""):
        raise ValueError("Use an exact dataset commit")
    if not re.fullmatch(r"[0-9a-f]{40}", args.scorer_revision):
        raise ValueError("Use an exact official scorer commit")
    if (
        not 0 <= args.multi_image_min <= args.count <= args.candidate_rows
        or args.count < 1
    ):
        raise ValueError("Require 0 <= multi-image-min <= count <= candidate-rows")
    args.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(
            {"http": args.proxy, "https": args.proxy} if args.proxy else {}
        )
    )

    def fetch(url):
        with opener.open(url, timeout=90) as response:
            return response.read()

    import pyarrow.parquet as pq

    config_path = urllib.parse.quote(args.config, safe="")
    metadata = json.loads(
        fetch(
            f"https://huggingface.co/api/datasets/{DATASET}/tree/{args.revision}/{config_path}"
        )
    )
    parts = sorted(
        (f for f in metadata if f["path"].endswith(".parquet")), key=lambda f: f["path"]
    )
    if len(args.parquet) > len(parts):
        raise ValueError("Too many parquet parts")
    all_rows, sources = [], []
    for local, remote in zip(args.parquet, parts):
        digest = sha(local.read_bytes())
        if digest != remote["lfs"]["oid"]:
            raise ValueError(f"Parquet does not match official revision: {local}")
        sources.append(dict(path=remote["path"], sha256=digest))
        for batch in pq.ParquetFile(local).iter_batches(batch_size=100):
            all_rows.extend(batch.to_pylist())
            if len(all_rows) >= args.candidate_rows:
                break
        if len(all_rows) >= args.candidate_rows:
            break
    if len(all_rows) < args.candidate_rows:
        raise ValueError("Not enough candidate rows in the provided parquet files")
    all_rows = all_rows[: args.candidate_rows]
    selected = select_rows(all_rows, count=args.count, multi_min=args.multi_image_min)
    (args.output / "images").mkdir()
    samples = []
    for row in selected:
        sample = {k: row[k] for k in ("id", "subject", "question", "answer")}
        sample["options"] = ast.literal_eval(row["options"])
        sample["images"] = {}
        for i in range(1, 8):
            image = row.get(f"image_{i}")
            if image is None:
                continue
            raw = image["bytes"]
            suffix = ".png" if raw.startswith(b"\x89PNG") else ".jpg"
            relative = f"images/{row['id']}-{i}{suffix}"
            (args.output / relative).write_bytes(raw)
            sample["images"][str(i)] = dict(
                path=relative, sha256=sha(raw), mime=mimetypes.guess_type(relative)[0]
            )
        samples.append(sample)
        print(f"frozen {sample['id']} images={len(sample['images'])}", flush=True)
    scorer_files = {}
    for path in ["mmmu-pro/evaluate.py", "mmmu-pro/prompts.yaml", "LICENSE"]:
        raw = fetch(
            f"https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/{args.scorer_revision}/{path}"
        )
        name = Path(path).name
        (args.output / name).write_bytes(raw)
        scorer_files[name] = sha(raw)
    write_json(
        args.output / "manifest.json",
        dict(
            dataset=DATASET,
            revision=args.revision,
            config=args.config,
            split="test",
            candidate_rows=args.candidate_rows,
            selection="multi-image quota, then subject diversity, sorted IDs",
            multi_image_min=args.multi_image_min,
            parquet_sources=sources,
            scorer_revision=args.scorer_revision,
            scorer_files=scorer_files,
            samples=samples,
        ),
    )


def messages(sample: dict, root: Path, *, prompt_style: str) -> list[dict]:
    """Preserve image/text order; append any image not referenced by a placeholder."""
    content, used = [], set()

    def image_part(index):
        image = sample["images"][index]
        raw = (root / image["path"]).read_bytes()
        if sha(raw) != image["sha256"]:
            raise ValueError("Frozen image checksum changed")
        used.add(index)
        return dict(
            type="image_url",
            image_url=dict(
                url=f"data:{image['mime']};base64,{base64.b64encode(raw).decode()}"
            ),
        )

    for part in re.split(r"(<image\s+\d+>)", sample["question"]):
        match = re.fullmatch(r"<image\s+(\d+)>", part)
        if match:
            content.append(image_part(match.group(1)))
        elif part:
            content.append(dict(type="text", text=part))
    for index in sample["images"]:
        if index not in used:
            content.append(image_part(index))
    options = "\n".join(
        f"({chr(65 + i)}) {answer}" for i, answer in enumerate(sample["options"])
    )
    # Exact official standard-setting instructions from prompts.yaml.
    instruction = (
        "Answer with the option letter from the given choices directly."
        if prompt_style == "direct"
        else "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
    )
    content.append(dict(type="text", text=f"\n{options}\n\n{instruction}"))
    return [dict(role="user", content=content)]


def official_scorer(root: Path, manifest: dict):
    path = root / "evaluate.py"
    if sha(path.read_bytes()) != manifest["scorer_files"]["evaluate.py"]:
        raise ValueError("Official scorer checksum changed")
    spec = importlib.util.spec_from_file_location("official_mmmu_pro", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_result(result: dict, sample: dict, scorer, *, seed: int) -> dict:
    """Use the pinned official parser; exclude errors and truncated answers."""
    if not result["ok"]:
        return dict(correct=False, parse_random_fallback=False)
    index2ans, choices = scorer.get_multi_choice_info(sample["options"])
    state = random.getstate()
    try:
        random.seed(f"{seed}:{sample['id']}")
        before = random.getstate()
        prediction = scorer.parse_multi_choice_response(
            result["content"], choices, index2ans
        )
        fallback = before != random.getstate()
    finally:
        random.setstate(state)
    return dict(
        prediction=prediction,
        correct=(
            result["finish_reason"] != "length"
            and scorer.eval_multi_choice(sample["answer"], prediction)
        ),
        parse_random_fallback=fallback,
    )


def run(args) -> None:
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scorer = official_scorer(args.data, manifest)
    samples = manifest["samples"]
    if args.limit is not None:
        samples = samples[: args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    script_source = Path(__file__).read_bytes()
    (args.output / "client.py").write_bytes(script_source)
    write_json(
        args.output / "protocol.json",
        dict(
            endpoint=args.url,
            identity=json.loads(args.identity.read_text()),
            model=args.model,
            concurrency=args.concurrency,
            repeats=args.repeats,
            prompt_style=args.prompt_style,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            temperature=0,
            seed=args.seed,
            timeout=args.timeout,
            manifest_sha256=sha(manifest_path.read_bytes()),
            script_sha256=sha(script_source),
            automatic_request_retries=0,
            sample_ids=[s["id"] for s in samples],
            scorer_revision=manifest["scorer_revision"],
            scorer_sha256=manifest["scorer_files"]["evaluate.py"],
            scoring="official parser with deterministic per-ID random fallback; failures count incorrect",
        ),
    )
    lock = threading.Lock()
    active = peak = 0

    def request(sample, repeat):
        nonlocal active, peak
        payload = dict(
            model=args.model,
            messages=messages(sample, args.data, prompt_style=args.prompt_style),
            temperature=0,
            max_tokens=args.max_tokens,
            stream=False,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
        )
        data = json.dumps(payload, ensure_ascii=False).encode()
        rid = f"mmmu-pro-{args.output.name}-{repeat}-{sample['id']}"
        started = time.time()
        tick = time.perf_counter()
        with lock:
            active += 1
            peak = max(peak, active)
            outstanding = active
        result = dict(
            id=sample["id"],
            repeat=repeat,
            request_id=rid,
            request_sha256=sha(data),
            image_count=len(sample["images"]),
            started=started,
            client_outstanding_at_start=outstanding,
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(
                args.url.rstrip("/") + "/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json", "X-Request-ID": rid},
            )
            with opener.open(req, timeout=args.timeout) as response:
                raw = response.read()
                result.update(
                    status=response.status,
                    response_sha256=sha(raw),
                    response=json.loads(raw),
                )
            body = result["response"]
            choice = body["choices"][0]
            result.update(
                content=choice["message"].get("content") or "",
                finish_reason=choice["finish_reason"],
                usage=body.get("usage", {}),
                ok=True,
            )
        except Exception as error:
            result.update(ok=False, error=repr(error))
            if isinstance(error, urllib.error.HTTPError):
                result["error_body"] = error.read().decode(errors="replace")
        finally:
            result["ended"] = time.time()
            result["latency_seconds"] = time.perf_counter() - tick
            with lock:
                active -= 1
        return result

    started = time.time()
    results = []
    with (args.output / "responses.jsonl").open("w") as output, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = [
            pool.submit(request, sample, repeat)
            for repeat in range(args.repeats)
            for sample in samples
        ]
        by_id = {s["id"]: s for s in samples}
        for future in as_completed(futures):
            result = future.result()
            result.update(
                score_result(result, by_id[result["id"]], scorer, seed=args.seed)
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            results.append(result)
            print(
                f"{len(results)}/{len(futures)} id={result['id']} ok={result['ok']} latency={result['latency_seconds']:.2f}s",
                flush=True,
            )
    elapsed = time.time() - started
    good = [r for r in results if r["ok"]]
    latencies = sorted(r["latency_seconds"] for r in good)
    summary = dict(
        total=len(results),
        completed=len(good),
        errors=len(results) - len(good),
        correct=sum(r["correct"] for r in results),
        accuracy=sum(r["correct"] for r in results) / len(results),
        unparsed=sum(r["parse_random_fallback"] for r in good),
        correct_without_random_fallback=sum(
            r["correct"] and not r["parse_random_fallback"] for r in results
        ),
        output_truncated=sum(r.get("finish_reason") == "length" for r in results),
        wall_seconds=elapsed,
        completed_requests_per_second=len(good) / elapsed,
        client_peak_outstanding=peak,
        client_peak_is_not_gpu_batch_size=True,
        latency_mean=statistics.mean(latencies) if latencies else None,
        latency_p50=statistics.median(latencies) if latencies else None,
        latency_p95=(
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            if latencies
            else None
        ),
        completion_tokens=sum(
            r.get("usage", {}).get("completion_tokens", 0) for r in good
        ),
        finish_reasons={
            reason: sum(r.get("finish_reason") == reason for r in good)
            for reason in sorted({r["finish_reason"] for r in good})
        },
    )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["errors"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--parquet", type=Path, nargs="+", required=True)
    prep.add_argument("--revision", required=True)
    prep.add_argument("--scorer-revision", required=True)
    prep.add_argument("--config", choices=["standard (10 options)"], required=True)
    prep.add_argument("--count", type=int, required=True)
    prep.add_argument("--candidate-rows", type=int, required=True)
    prep.add_argument("--multi-image-min", type=int, required=True)
    prep.add_argument("--proxy")
    runner = commands.add_parser("run")
    runner.add_argument("--data", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--url", required=True)
    runner.add_argument("--identity", type=Path, required=True)
    runner.add_argument("--model", required=True)
    runner.add_argument("--concurrency", type=int, required=True)
    runner.add_argument("--repeats", type=int, required=True)
    runner.add_argument("--prompt-style", choices=["direct", "cot"], required=True)
    runner.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], required=True
    )
    runner.add_argument("--max-tokens", type=int, required=True)
    runner.add_argument("--seed", type=int, required=True)
    runner.add_argument("--timeout", type=float, required=True)
    runner.add_argument("--limit", type=int)
    args = parser.parse_args()
    if (
        args.command == "run"
        and min(
            args.concurrency,
            args.repeats,
            args.max_tokens,
            args.timeout,
            1 if args.limit is None else args.limit,
        )
        <= 0
    ):
        parser.error("Run bounds must be positive")
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
