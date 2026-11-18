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

"""Audit the first continuous full-load Decode points without repairing raw data."""

import math
import statistics

CONTEXT = 131072
CACHED_PREFIX = 124416
SUFFIX = 6656
WORLD_SIZE = 16
WIDTH = 6
WARM_STEPS = 200
FORMAL_SKIP = 100
FORMAL_INTERVALS = 50
FORMAL_ENDPOINTS = FORMAL_INTERVALS + 1
ACCEPTANCE_ASSUMPTION = 0.75
TAU = 1 + 5 * ACCEPTANCE_ASSUMPTION
BATCHES = (1, 2, 4, 8, 16, 32)
SAMPLING = {"temperature": 0.3, "top_p": 0.95, "top_k": -1}
FAST_INDEXER_ENV = {
    "TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER": "1",
    "TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT": "1",
    "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER": "0",
    "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER": "0",
}


def output_budget(batch, cohort):
    """A termination budget; only raw observations establish completed full steps.

    Allow one output from Prefill, one startup step per independently prefilling
    suffix on a rank, and two terminal steps. This margin is not measured work.
    """
    if batch not in BATCHES or cohort not in (0, 1, 2):
        raise ValueError("unsupported batch or cohort")
    target = WARM_STEPS if cohort == 0 else FORMAL_SKIP + FORMAL_ENDPOINTS
    return {
        "required_full_points": target,
        "prefill_first_output_tokens": 1,
        "startup_step_allowance": batch,
        "terminal_step_allowance": 2,
        "max_new_tokens": 1 + WIDTH * (target + batch + 2),
    }


def distribution(values):
    if not values:
        raise RuntimeError("empty interval distribution")
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": ordered[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "percentile_method": "linear interpolation of all raw interval values",
    }


def full_prefix(samples, prefix, batch, world_size):
    if world_size != WORLD_SIZE:
        raise ValueError("this protocol requires all 16 ranks")
    run = []
    ended = False
    for sample in samples:
        ids = sample["request_ids"]
        if not any(rid.startswith(prefix) for rid in ids):
            continue
        if any(not rid.startswith(prefix) for rid in ids):
            raise RuntimeError("mixed request cohorts")
        expected = {f"{prefix}{sample['rank']}-r{index}" for index in range(batch)}
        valid = (
            sample["all_decode"] is True
            and sample["num_extends"] == 0
            and sample["global_decodes"] == [batch] * world_size
            and sample["global_batch"] == [batch] * world_size
            and len(ids) == batch
            and set(ids) == expected
            and sample["input_rows"] == WIDTH * batch
            and sample["output_lengths"] == [WIDTH] * batch
            and sample["prefill_lengths"] == [CONTEXT] * batch
        )
        if not valid:
            if run:
                ended = True
            continue
        if ended:
            raise RuntimeError("full-load sequence resumed after a gap or partial step")
        if run and sample["step"] != run[-1]["step"] + 1:
            raise RuntimeError("nonconsecutive raw dispatch sequence")
        if run and sample["completed_ns"] <= run[-1]["completed_ns"]:
            raise RuntimeError("nonincreasing raw completion timestamps")
        count = sample["compiles"]
        if type(count) is not int or count < 0:
            raise RuntimeError("compile monitor unavailable or invalid")
        if run and count < run[-1]["compiles"]:
            raise RuntimeError("compile counter reset")
        if type(sample["completed_ns"]) is not int or sample["completed_ns"] <= 0:
            raise RuntimeError("invalid raw completion timestamp")
        run.append(sample)
    return run


def align_global_dispatches(by_rank):
    """Verify raw counter alignment before using a step as a cross-rank key.

    No rank offset, timestamp alignment or duration participates in alignment.
    Every archived worker must cover the same uninterrupted counter from zero,
    with identical gathered batch metadata at every dispatch.
    """
    if set(by_rank) != set(range(WORLD_SIZE)):
        raise RuntimeError("global selection requires all 16 ranks exactly once")
    indexed = {}
    reference_steps = None
    for rank, samples in by_rank.items():
        steps = [sample["step"] for sample in samples]
        if not steps or steps != list(range(steps[-1] + 1)):
            raise RuntimeError(f"rank {rank} missing/reordered global dispatch counter")
        if reference_steps is None:
            reference_steps = steps
        if steps != reference_steps:
            raise RuntimeError("different global dispatch counter coverage")
        indexed[rank] = {sample["step"]: sample for sample in samples}
    for step in reference_steps:
        reference = indexed[0][step]
        for rank in range(WORLD_SIZE):
            if any(
                indexed[rank][step][key] != reference[key]
                for key in ("global_batch", "global_decodes")
            ):
                raise RuntimeError(f"global metadata differs at dispatch {step}")
    return indexed


def global_full_prefix(indexed, prefix, batch):
    """Intersection of actual full points on all ranks, at identical raw steps.

    Per-rank validation rejects gaps and a later resumption even if a later
    segment is long or faster. The intersection then fixes one shared window.
    """
    runs = {
        rank: full_prefix(list(rows.values()), prefix, batch, WORLD_SIZE)
        for rank, rows in indexed.items()
    }
    common = set.intersection(
        *({sample["step"] for sample in run} for run in runs.values())
    )
    steps = sorted(common)
    if not steps or any(b != a + 1 for a, b in zip(steps, steps[1:])):
        raise RuntimeError("missing or nonconsecutive all-rank full-load sequence")
    return {rank: [rows[step] for step in steps] for rank, rows in indexed.items()}


def warmup(samples, prefix, batch, *, world_size):
    run = full_prefix(samples, prefix, batch, world_size)
    if len(run) < WARM_STEPS:
        raise RuntimeError(
            f"need {WARM_STEPS} consecutive full warm steps; observed {len(run)}"
        )
    checked = run[:WARM_STEPS]
    return {
        "required_full_points": WARM_STEPS,
        "observed_full_points": len(run),
        "checked_steps": [point["step"] for point in checked],
        "first_ns": checked[0]["completed_ns"],
        "last_ns": checked[-1]["completed_ns"],
        "compile_first": checked[0]["compiles"],
        "compile_last": checked[-1]["compiles"],
        "compile_delta": checked[-1]["compiles"] - checked[0]["compiles"],
        "actual_kept_width": WIDTH,
    }


def window(samples, prefix, batch, warmup, steps, *, world_size):
    run = full_prefix(samples, prefix, batch, world_size)
    needed = warmup + steps + 1
    if len(run) < needed:
        raise RuntimeError(
            f"need {needed} consecutive full points; observed {len(run)}"
        )
    selected = run[warmup:needed]
    counts = [sample["compiles"] for sample in selected]
    if any(count != counts[0] for count in counts):
        raise RuntimeError("compilation inside measured window")
    intervals = [
        b["completed_ns"] - a["completed_ns"] for a, b in zip(selected, selected[1:])
    ]
    if any(value <= 0 for value in intervals):
        raise RuntimeError("invalid raw completion interval")
    elapsed = sum(intervals) / 1e9
    accepted = WIDTH * batch * steps
    return {
        "steps": steps,
        "seconds": elapsed,
        "step_s": steps / elapsed,
        "step_ms": elapsed / steps * 1000,
        "interval_ns": intervals,
        "interval_ms_distribution": distribution([v / 1e6 for v in intervals]),
        "endpoint_steps": [s["step"] for s in selected],
        "discarded_full_points": warmup,
        "observed_full_points": len(run),
        "accepted_tokens": accepted,
        "actual_kept_tokens_per_request_step": WIDTH,
        "actual_tokens_s_gpu": accepted / elapsed,
        "assumed_acceptance": ACCEPTANCE_ASSUMPTION,
        "assumed_tau": TAU,
        "deepinfer_tokens_s_user": TAU * steps / elapsed,
        "deepinfer_tokens_s_gpu": batch * TAU * steps / elapsed,
        "context_min": min(s.get("context_min", 0) for s in selected[1:]),
        "context_max": max(s.get("context_max", 0) for s in selected[1:]),
        "first_ns": selected[0]["completed_ns"],
        "last_ns": selected[-1]["completed_ns"],
        "compile_count": counts[-1],
        "measured_compile_delta": 0,
        "cohort_compile_delta": run[-1]["compiles"] - run[0]["compiles"],
        "global_batch": world_size * batch,
    }
