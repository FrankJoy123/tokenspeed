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

"""Prepare and evaluate the frozen MMLongBench DocQA K128 subset."""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path

from longdoc_scoring import eval_docqa_score, parse_output

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "ref" / "longdoc-k128-regression50.json"
)
USER_TEMPLATE = (
    "You are given a document with text and images, and a question. Answer the "
    "question as concisely as you can, using a single phrase or sentence if "
    "possible. If the question cannot be answered based on the information in "
    "the article, write 'Not answerable.' Write your answer in the following "
    "format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def row_digest(row):
    return digest(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )


def load_rows(root, manifest):
    source = (root / "mmlongdoc_K128.jsonl").read_bytes()
    if digest(source) != manifest["source_sha256"]:
        raise ValueError("Dataset file differs from the frozen source")
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate source IDs")
    selected = []
    for item in manifest["samples"]:
        row = by_id[item["id"]]
        if row_digest(row) != item["row_sha256"]:
            raise ValueError(f"Changed frozen row: {item['id']}")
        selected.append(row)
    if len(selected) != manifest["selection"]["count"]:
        raise ValueError("Frozen subset size disagrees")
    return selected


def prepare(args):
    manifest = json.loads(args.manifest.read_text())
    args.data.mkdir(parents=True, exist_ok=True)
    source = args.data / "mmlongdoc_K128.jsonl"
    if not source.exists():
        url = (
            "https://huggingface.co/datasets/"
            f"{manifest['dataset']}/resolve/{manifest['dataset_revision']}/"
            f"{manifest['source_path']}"
        )
        source.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    rows = load_rows(args.data, manifest)
    paths = sorted({image for row in rows for image in row["page_list"]})
    images = manifest["images"]

    def download(name):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe image path")
        path = args.data / "images" / name
        if path.is_file():
            content = path.read_bytes()
            # Existing files must have a previously verified inventory entry.
            if existing.get(name) == digest(content):
                return name, digest(content)
        url = (
            f"https://huggingface.co/datasets/{images['repo']}/resolve/"
            f"{images['revision']}/{name}"
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read()
                if not content.startswith(b"\xff\xd8"):
                    raise ValueError("Expected original JPEG image")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_suffix(".partial").write_bytes(content)
                path.with_suffix(".partial").replace(path)
                return name, digest(content)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    inventory = args.data / "images-sha256.json"
    existing = json.loads(inventory.read_text()) if inventory.exists() else {}
    verified = {}
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for name, sha in pool.map(download, paths):
            verified[name] = sha
            if len(verified) % 100 == 0:
                write_json(inventory, {**existing, **verified})
                print(f"images {len(verified)}/{len(paths)}", flush=True)
    write_json(inventory, dict(sorted(verified.items())))
    if digest(inventory.read_bytes()) != images["inventory_sha256"]:
        raise ValueError("Image inventory differs from the frozen subset")
    write_json(args.data / "subset-manifest.json", manifest)
    print(f"ready: {len(rows)} questions, {len(verified)} images", flush=True)


def messages(row, root, inventory):
    image_names = row["page_list"]
    context = "\n\n".join(
        f"Document {Path(name).parent.name[:15]}: <image>" for name in image_names
    )
    question = (
        f"Based on Document {row['doc_name'][:15]}, answer the following question. "
        + row["question"]
    )
    text = USER_TEMPLATE.format(context=context, question=question)
    segments = text.split("<image>")
    if len(segments) != len(image_names) + 1:
        raise ValueError("Image placeholders disagree with the page list")
    content = []
    for index, name in enumerate(image_names):
        content.append({"type": "text", "text": segments[index]})
        raw = (root / "images" / name).read_bytes()
        if digest(raw) != inventory[name]:
            raise ValueError(f"Changed image: {name}")
        mime = mimetypes.guess_type(name)[0]
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(raw).decode(),
                    "detail": "auto",
                },
            }
        )
    content.append({"type": "text", "text": segments[-1]})
    return [{"role": "user", "content": content}]


def run(args):
    manifest = json.loads(args.manifest.read_text())
    rows = load_rows(args.data, manifest)
    if args.limit:
        rows = rows[: args.limit]
    inventory_path = args.data / "images-sha256.json"
    if digest(inventory_path.read_bytes()) != manifest["images"]["inventory_sha256"]:
        raise ValueError("Image inventory differs from the frozen subset")
    inventory = json.loads(inventory_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        {
            "endpoint": args.url,
            "model": args.model,
            "manifest_sha256": digest(args.manifest.read_bytes()),
            "image_inventory_sha256": digest(inventory_path.read_bytes()),
            "client_sha256": digest(Path(__file__).read_bytes()),
            "scorer_sha256": digest(
                Path(__file__).with_name("longdoc_scoring.py").read_bytes()
            ),
            "sample_ids": [row["id"] for row in rows],
            "concurrency": args.concurrency,
            "temperature": 0,
            "top_p": 1,
            "seed": 968340402,
            "reasoning_effort": "none",
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
            "automatic_request_retries": 0,
            "metric": "frozen MMLongBench deterministic doc_qa; failures score zero; length truncation also reported separately with strict score zero",
            "author_length_is_not_model_token_count": True,
            "identity": json.loads(args.identity.read_text()),
        },
    )
    for name in ["longdoc.py", "longdoc_scoring.py"]:
        (args.output / name).write_bytes(Path(__file__).with_name(name).read_bytes())

    def request(row):
        result = {
            "id": row["id"],
            "image_count": len(row["page_list"]),
            "author_input_tokens": row["length"],
            "started": time.time(),
        }
        started = time.perf_counter()
        try:
            payload = {
                "model": args.model,
                "messages": messages(row, args.data, inventory),
                "temperature": 0,
                "top_p": 1,
                "seed": 968340402,
                "max_tokens": args.max_tokens,
                "reasoning_effort": "none",
                "stream": False,
            }
            body = json.dumps(payload, ensure_ascii=False).encode()
            result["request_sha256"] = digest(body)
            req = urllib.request.Request(
                args.url.rstrip("/") + "/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Request-ID": f"longdoc-{args.output.name}-{row['id']}",
                },
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=args.timeout) as response:
                raw = response.read()
                result.update(
                    ok=True, response_sha256=digest(raw), response=json.loads(raw)
                )
            choice = result["response"]["choices"][0]
            result["content"] = choice["message"].get("content") or ""
            result["finish_reason"] = choice["finish_reason"]
            # Upstream eval_api prepends system_template before post_process.
            result["parsed"] = parse_output(
                "Answer:" + result["content"], prefix="Answer:"
            )
            result["score"] = float(
                eval_docqa_score(
                    row["answer"], result["parsed"] or "", row["answer_format"]
                )
            )
            result["strict_score"] = (
                result["score"] if choice["finish_reason"] != "length" else 0.0
            )
            result["usage"] = result["response"].get("usage", {})
        except Exception as error:
            result.update(ok=False, score=0.0, strict_score=0.0, error=repr(error))
            if isinstance(error, urllib.error.HTTPError):
                result["error_body"] = error.read().decode(errors="replace")
        result["ended"] = time.time()
        result["seconds"] = time.perf_counter() - started
        return result

    results = []
    started = time.time()
    iterator = iter(rows)
    with (
        (args.output / "responses.jsonl").open("w") as out,
        concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool,
    ):
        pending = {
            pool.submit(request, row)
            for row in [next(iterator, None) for _ in range(args.concurrency)]
            if row is not None
        }
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                result = future.result()
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    f"{len(results)}/{len(rows)} {result['id']} ok={result['ok']} score={result['score']:.3f}",
                    flush=True,
                )
                if sum(not r["ok"] for r in results) < 3:
                    row = next(iterator, None)
                    if row is not None:
                        pending.add(pool.submit(request, row))
    summary = {
        "expected": len(rows),
        "finished": len(results),
        "completed": sum(r["ok"] for r in results),
        "errors": sum(not r["ok"] for r in results),
        "skipped": len(rows) - len(results),
        "doc_qa_score": sum(r["score"] for r in results) / len(rows),
        "strict_doc_qa_score": sum(r["strict_score"] for r in results) / len(rows),
        "output_truncated": sum(r.get("finish_reason") == "length" for r in results),
        "wall_seconds": time.time() - started,
        "actual_prompt_tokens": {
            "min": min(
                (
                    r.get("usage", {}).get("prompt_tokens", 0)
                    for r in results
                    if r["ok"]
                ),
                default=None,
            ),
            "max": max(
                (
                    r.get("usage", {}).get("prompt_tokens", 0)
                    for r in results
                    if r["ok"]
                ),
                default=None,
            ),
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["errors"] or summary["skipped"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    r = commands.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--identity", type=Path, required=True)
    r.add_argument("--model", default="deepseek-v41-flash")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--max-tokens", type=int, required=True)
    r.add_argument("--limit", type=int)
    args = parser.parse_args()
    if (
        args.command == "run"
        and min(
            args.max_tokens,
            args.concurrency,
            args.timeout,
            args.limit if args.limit is not None else 1,
        )
        <= 0
    ):
        parser.error("Run bounds must be positive")
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
