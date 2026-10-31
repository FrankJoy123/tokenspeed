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

"""Real PD request latency; preserve raw streaming events and failed batches."""
import os
import concurrent.futures
import datetime
import json
import statistics
import threading
import time
from pathlib import Path

import requests

ROOT = Path(os.environ["BENCH_OUTPUT"]).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_TOKENS = int(os.environ["OUTPUT_TOKENS"])
URL = os.environ["BENCH_URL"]


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def client():
    session = requests.Session()
    session.trust_env = False
    return session


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def collect(index, ids, barrier, path):
    row = dict(index=index, input_tokens=len(ids), events=[], output_ids=[])
    payload = dict(model="deepseek-v41-flash", input_ids=ids, stream=True,
                   sampling_params=dict(temperature=0, seed=37,
                                        max_new_tokens=OUTPUT_TOKENS, ignore_eos=True))
    try:
        with client() as session:
            if barrier is not None:
                barrier.wait(timeout=60)
            row["start"] = time.time()
            done = False
            with session.post(URL + "/generate", json=payload, stream=True,
                              timeout=(30, 7200)) as response:
                row["http_status"] = response.status_code
                if response.status_code != 200:
                    raise RuntimeError(response.text[:4000])
                for line in response.iter_lines(chunk_size=None):
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        done = True
                        break
                    event = json.loads(raw)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    meta = event["meta_info"]
                    n = meta["completion_tokens"]
                    previous = row["events"][-1][1] if row["events"] else 0
                    if n < previous:
                        raise ValueError("Nonmonotonic completion count")
                    if n > previous:
                        row["events"].append([time.time(), n])
                        row["output_ids"].extend(event.get("output_ids", []))
                    row["meta"] = meta
                    row["output_text"] = event.get("text", "")
            row["end"] = time.time()
            if not done:
                raise ValueError("Missing stream completion")
            meta = row["meta"]
            if meta["prompt_tokens"] != len(ids) or meta["completion_tokens"] != OUTPUT_TOKENS:
                raise ValueError(f"Incorrect token counts: {meta}")
            first, last = row["events"][0][0], row["events"][-1][0]
            row.update(success=True, ttft_s=first-row["start"],
                       tpot_ms=1000*(last-first)/(OUTPUT_TOKENS-1),
                       e2e_s=row["end"]-row["start"],
                       first_event_tokens=row["events"][0][1])
    except Exception as error:
        row.update(success=False, error=repr(error), end=time.time())
    write(path, row)
    return row


def wait_ready():
    with client() as session:
        for _ in range(180):
            try:
                state = session.get(URL + "/workers", timeout=10).json()
                workers = state.get("workers", [])
                if len(workers) == 2 and all(w["is_healthy"] for w in workers):
                    write(ROOT / "ready-workers.json", state)
                    return
            except Exception:
                pass
            time.sleep(10)
    raise TimeoutError("Services did not become healthy in 30 minutes")

