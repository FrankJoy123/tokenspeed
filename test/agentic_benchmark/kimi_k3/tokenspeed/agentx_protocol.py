# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Validate explicit replay settings and retain evidence beside native outputs."""

import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def trace_snapshot(env):
    suffix = "-256k" if env["AGENTX_PROTOCOL"] == "weka-062126-256k" else ""
    cache = Path(env["HF_HUB_CACHE"]) / (
        "datasets--semianalysisai--cc-traces-weka-062126" + suffix
    )
    revision = env["TRACE_REVISION"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("TRACE_REVISION must be a full dataset commit")
    if (cache / "refs/main").read_text().strip() != revision:
        raise ValueError(
            "Dataset cache refs/main differs from TRACE_REVISION; prepare a dedicated pinned cache"
        )
    path = cache / "snapshots" / revision / "traces.jsonl"
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing pinned trace: {path}")
    return path


def validate(env):
    for key in ("MAX_MODEL_LEN", "MAX_NUM_SEQS", "SPEC_STEPS"):
        if not env[key].isdigit() or int(env[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    context = 262144 if env["AGENTX_PROTOCOL"] == "weka-062126-256k" else 1048576
    if int(env["MAX_MODEL_LEN"]) != context:
        raise ValueError(
            f"MAX_MODEL_LEN must be {context} for {env['AGENTX_PROTOCOL']}"
        )
    if env["SPEC_DECODING"] not in ("eagle3", "dspark"):
        raise ValueError("SPEC_DECODING must be eagle3 or dspark")
    if env["ACCEPTANCE_MODE"] == "synthetic":
        al = float(env["AL_TARGET"])
        if not math.isfinite(al) or not 1 <= al <= int(env["SPEC_STEPS"]) + 1:
            raise ValueError(
                "AL_TARGET must be finite and between 1 and SPEC_STEPS + 1"
            )
    if env["SERVER_METRICS"] == "true":
        port = int(env["METRICS_PORT"])
        if not 1 <= port <= 65535 or port == int(env["API_PORT"]):
            raise ValueError(
                "METRICS_PORT must be a valid port different from API_PORT"
            )
    trace_snapshot(env)


def required_metrics(env):
    names = ["tokenspeed:prompt_tokens", "tokenspeed:generation_tokens"]
    if env["ACCEPTANCE_MODE"] == "synthetic":
        names += [
            "tokenspeed:spec_decode_num_drafts",
            "tokenspeed:spec_decode_num_accepted_tokens",
        ]
    return names


def check_endpoint(env):
    url = env["AIPERF_SERVER_METRICS_URLS"]
    with urllib.request.urlopen(url, timeout=10) as response:
        text = response.read().decode()
    # Counters may not have labelled samples until the first request. Require
    # their Prometheus declarations before replay; exports must contain samples.
    for name in required_metrics(env):
        if not any(
            line.startswith(
                ("# HELP " + name + " ", "# TYPE " + name + " ", name + "_total")
            )
            for line in text.splitlines()
        ):
            raise ValueError(f"Missing metric {name} at {url}")


def metric_total(metrics, name, endpoint):
    series = metrics[name]["series"]
    values = [s["stats"]["total"] for s in series if s.get("endpoint_url") == endpoint]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"Missing or invalid samples for {name} at {endpoint}")
    return sum(values)


def check_export(env):
    root = Path(env["RESULT_DIR"])
    path = root / "aiperf_artifacts/server_metrics_export.json"
    metrics = json.loads(path.read_text())["metrics"]
    totals = {
        name: metric_total(metrics, name, env["AIPERF_SERVER_METRICS_URLS"])
        for name in required_metrics(env)
    }
    if (
        totals["tokenspeed:prompt_tokens"] <= 0
        or totals["tokenspeed:generation_tokens"] <= 0
    ):
        raise ValueError("Server metrics export contains no request token activity")
    report = {"status": "passed", "artifact": str(path), "totals": totals}
    if env["ACCEPTANCE_MODE"] == "synthetic":
        steps = totals["tokenspeed:spec_decode_num_drafts"]
        if steps <= 0:
            raise ValueError("Cannot validate AL without measured draft steps")
        al = 1 + totals["tokenspeed:spec_decode_num_accepted_tokens"] / steps
        report["measured_acceptance_length"] = al
        report["target_acceptance_length"] = float(env["AL_TARGET"])
        if abs(al - float(env["AL_TARGET"])) > 0.1:
            raise ValueError(f"Measured AL {al} differs from target {env['AL_TARGET']}")
    (root / "server-metrics-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def git_identity(path):
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def write_manifest(env):
    trace = trace_snapshot(env)
    with trace.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    keys = (
        "AGENTX_PROTOCOL",
        "WEKA_LOADER_OVERRIDE",
        "TRACE_REVISION",
        "TOKENIZER_PATH",
        "MODEL_NAME",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "PRECISION",
        "SPEC_DECODING",
        "SPEC_STEPS",
        "ACCEPTANCE_MODE",
        "CONCURRENCY",
        "DURATION",
        "BENCHMARK_GRACE_SECONDS",
        "CLIENT_SEED",
        "SERVER_METRICS",
        "AIPERF_WARMUP_REQUESTS_PER_LANE",
        "AGENTIC_WARMUP_GRACE_PERIOD",
        "AIPERF_EXTRA_INPUTS",
    )
    report = {k: env[k] for k in keys}
    report.update(
        al_target=(
            float(env["AL_TARGET"]) if env["ACCEPTANCE_MODE"] == "synthetic" else None
        ),
        trace_sha256=digest,
        inferencex=git_identity(env["INFERENCEX_WORKSPACE"]),
        aiperf=git_identity(Path(env["CLIENT_ROOT"]) / "aiperf"),
    )
    (Path(env["RESULT_DIR"]) / "protocol.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    actions = {
        "validate": validate,
        "metrics-endpoint": check_endpoint,
        "metrics-export": check_export,
        "manifest": write_manifest,
    }
    try:
        actions[sys.argv[1]](os.environ)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"AgentX protocol check failed: {error}", file=sys.stderr)
        sys.exit(2)
