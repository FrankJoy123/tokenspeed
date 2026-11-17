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

"""Render the three accepted current curves without fitting or removing points."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open() as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"No accepted points: {path}")
    return result


def main():
    prefill = {name: rows(ROOT / "prefill" / f"{name}.csv") for name in ("cp8", "dp8")}
    decode = rows(ROOT / "decode" / "ep16.csv")
    for name, points in prefill.items():
        if [int(p["budget_new_tokens"]) for p in points] != [4096, 8192, 16384]:
            raise ValueError(f"Incomplete {name} curve")
    if [int(p["B"]) for p in decode] != [1, 2, 4, 8, 16, 32]:
        raise ValueError("Incomplete EP16 curve")
    plt.rcParams.update(
        {"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "tokenspeed-w4a8"}
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for name, label in (("cp8", "QCP8 / EP8"), ("dp8", "DP8 / EP8")):
        points = prefill[name]
        axes[0].plot(
            [int(p["budget_new_tokens"]) / 1024 for p in points],
            [float(p["new_tokens_s_gpu"]) for p in points],
            marker="o",
            label=label,
        )
    axes[0].set(
        title="W4A8 Prefill",
        xlabel="New-token budget / attention DP lane (Ki tokens)",
        ylabel="New tokens / s / GPU",
        xticks=[4, 8, 16],
        ylim=(0, None),
    )
    axes[0].legend()
    batch = [int(p["B"]) for p in decode]
    rates = [float(p["tokens_s_gpu"]) for p in decode]
    axes[1].plot(batch, rates, marker="o", label="EP16, two-round mean step time")
    for key, marker, label in (
        ("round1_step_ms", "x", "Round 1"),
        ("round2_step_ms", "+", "Round 2"),
    ):
        axes[1].scatter(
            batch,
            [b * 4.75 * 1000 / float(p[key]) for b, p in zip(batch, decode)],
            marker=marker,
            s=40,
            alpha=0.75,
            label=label,
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set(
        title="W4A8 Decode EP16",
        xlabel="Requests / attention DP rank (B)",
        ylabel="Converted tokens / s / GPU",
        xticks=batch,
        xticklabels=batch,
        ylim=(0, None),
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "R0 natural MoE routing | 131072 input tokens | TokenSpeed defaults"
    )
    figure.text(
        0.5,
        -0.025,
        "Decode conversion: tau = 4.75 (assumed acceptance 0.75; not measured acceptance). "
        "Points and slow intervals retained.",
        ha="center",
        fontsize=8,
    )
    for suffix in ("svg", "png"):
        target = ROOT / f"curves.{suffix}"
        figure.savefig(
            target,
            dpi=180,
            bbox_inches="tight",
            metadata={"Date": None} if suffix == "svg" else None,
        )
        if suffix == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
    plt.close(figure)


if __name__ == "__main__":
    main()
