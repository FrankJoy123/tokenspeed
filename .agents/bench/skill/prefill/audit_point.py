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

"""Recompute two formal windows and all-rank compile audit from raw records."""

import argparse
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--point", type=Path, required=True)
args = parser.parse_args()
point = args.point / "samples"
root = args.point
module_spec = importlib.util.spec_from_file_location(
    "steady", Path(__file__).with_name("prefill_steady.py")
)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
config = json.loads((point / "configuration.json").read_text())
terminal = json.loads((root / "terminal.json").read_text())
assert terminal["returncode"] == 0
assert json.loads((point / "state.json").read_text())["stage"] == "complete"
assert config["routing"] in ("natural", "uniform") and config["redundant_experts"] == 0
cp = config["cp"]
dp = config["dp"]
physical = config["benchmark"]["prefill_rows_per_rank"]
budget = physical // 4224 * 4096
assert cp * dp == 8
text = (root / "engine.log").read_text(errors="replace")
samples = module.read_samples(root / "engine.log")
by_rank = defaultdict(list)
for sample in samples:
    by_rank[sample["rank"]].append(sample)
assert sorted(by_rank) == list(range(dp))
for iteration in (0, 1, 2):
    rows = json.loads((point / f"r{iteration}-requests.json").read_text())
    assert len(rows) == config["requests_per_round"]
    assert all(
        (x["prompt_tokens"], x["cached_tokens"], x["completion_tokens"])
        == (131072, 126976, 1)
        and x["finish_reason"] != "abort"
        for x in rows.values()
    )
trials = []
for iteration in (1, 2):
    tag = f"steady-s{physical}-r{iteration}-request-"
    windows = {
        rank: module.steady_window(
            values,
            request_prefix=tag,
            rows=physical,
            dp_size=dp,
            warmup_steps=5,
            measure_steps=10,
        )
        for rank, values in by_rank.items()
    }
    assert all(
        x["new_rows"] == budget * 10 and x["replay_rows"] == budget // 4096 * 128 * 10
        for x in windows.values()
    )
    endpoints = {}
    for line in text.splitlines():
        if module.MARKER not in line:
            continue
        value = json.loads(
            re.sub(r"\x1b\[[0-9;]*m", "", line.split(module.MARKER, 1)[1])
        )
        if value["rank"] == 0 and value["step"] in (
            windows[0]["first_step"],
            windows[0]["last_step"],
        ):
            stamp = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
            assert stamp
            endpoints[value["step"]] = datetime.strptime(
                stamp.group(), "%Y-%m-%d %H:%M:%S"
            )
    assert len(endpoints) == 2
    lo = min(endpoints.values()) - timedelta(seconds=1)
    hi = max(endpoints.values()) + timedelta(seconds=1)
    for line in text.splitlines():
        if not any(
            term in line.lower()
            for term in ["while serving", "building jit", "jit compiled"]
        ):
            continue
        stamp = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
        assert (
            stamp is None
            or not lo <= datetime.strptime(stamp.group(), "%Y-%m-%d %H:%M:%S") <= hi
        ), line
    for window in windows.values():
        window["lane_new_tokens_s"] = window["new_tokens_s_gpu"]
        window["lane_computed_rows_s"] = window["computed_tokens_s_gpu"]
        window["new_tokens_s_gpu"] /= cp
        window["computed_tokens_s_gpu"] /= cp
    trials.append(
        {
            "round": iteration,
            "rank0": windows[0],
            "ranks": windows,
            "worker_compile_log_audit": {
                "messages": 0,
                "padded_wall_start": str(lo),
                "padded_wall_end": str(hi),
            },
        }
    )
elapsed = sum(x["rank0"]["window_seconds"] for x in trials)
record = {
    "label": root.name,
    "budget_new_tokens": budget,
    "physical_rows": physical,
    "cp": cp,
    "dp": dp,
    "step_ms": elapsed / 20 * 1000,
    "new_tokens_s_gpu": budget * 20 / elapsed / cp,
    "computed_rows_s_gpu": physical * 20 / elapsed / cp,
    "trials": trials,
    "prefix_sha256": config["prefix_sha256"],
    "benchmark_sha256": hashlib.sha256(
        (Path(__file__).with_name("prefill_steady.py")).read_bytes()
    ).hexdigest(),
    "scope": "Host completion windows, independent last11 full endpoints per round, every request/cache/physical row and all-worker compile-log audit. Not TTFT or pure CUDA kernel timing.",
}
(point / "accepted-summary.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k not in ("trials", "scope")}))
