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

"""Select consecutive, globally full Decode completion intervals."""


def window(samples, prefix, batch, warmup, steps, *, world_size):
    if world_size not in (8, 16, 32):
        raise ValueError("unsupported world size")
    run = []
    for sample in samples:
        if "step" not in sample:
            raise RuntimeError("missing dispatch sequence id")
        valid = (
            sample["all_decode"]
            and sample["num_extends"] == 0
            and sample["global_decodes"] == [batch] * world_size
            and len(sample["request_ids"]) == batch
            and all(r.startswith(prefix) for r in sample["request_ids"])
        )
        if not valid:
            run = []
            continue
        if run and sample["step"] != run[-1]["step"] + 1:
            run = []
        run.append(sample)
        if len(run) >= warmup + steps + 1:
            selected = run[warmup : warmup + steps + 1]
            counts = [s["compiles"] for s in selected]
            if any(c is None for c in counts):
                raise RuntimeError("compile monitor unavailable")
            if any(c != counts[0] for c in counts):
                raise RuntimeError("compilation inside steady window")
            elapsed = (selected[-1]["completed_ns"] - selected[0]["completed_ns"]) / 1e9
            if elapsed <= 0:
                raise RuntimeError("invalid completion interval")
            accepted = sum(sum(s["output_lengths"]) for s in selected[1:])
            return {
                "steps": steps,
                "seconds": elapsed,
                "step_s": steps / elapsed,
                "step_ms": elapsed / steps * 1000,
                "accepted_tokens": accepted,
                "actual_kept_tokens_per_request_step": accepted / (batch * steps),
                "actual_tokens_s_gpu": accepted / elapsed,
                "context_min": min((s.get("context_min", 0) for s in selected[1:])),
                "context_max": max((s.get("context_max", 0) for s in selected[1:])),
                "first_ns": selected[0]["completed_ns"],
                "last_ns": selected[-1]["completed_ns"],
                "compile_count": counts[-1],
                "global_batch": world_size * batch,
            }
    raise RuntimeError(
        f"no full DP{world_size} window for {prefix}: need {warmup + steps + 1} consecutive samples"
    )
