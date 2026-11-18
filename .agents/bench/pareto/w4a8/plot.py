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

"""Draw the standalone W4A8 Pareto plots from audited step intervals."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

HERE = Path(__file__).resolve().parent
TAU = 4.75
BATCHES = (1, 2, 4, 8, 16, 32)


def read(path):
    return json.loads(path.read_text())


def timing(intervals, expected):
    if len(intervals) != expected or not all(
        math.isfinite(value) and value > 0 for value in intervals
    ):
        raise ValueError(f"Expected {expected} positive audited intervals")
    ordered = sorted(intervals)
    cut = expected // 5
    retained = ordered[cut:-cut]
    return {
        "intervals_total": expected,
        "discarded_each_tail": cut,
        "intervals_used": len(retained),
        "raw_mean_step_ms": statistics.fmean(ordered),
        "trimmed_mean_step_ms": statistics.fmean(retained),
    }


def prefill_input(directory):
    sample = read(directory / "samples/accepted-summary.json")
    launch = read(directory / "invocation.json")
    identity = read(directory / "runtime-identity.json")
    return {
        "layout": "QCP8" if launch["layout"] == "cp8" else "DP8",
        "label": sample["new_tokens_per_gpu_rank_per_step"],
        "indexer": launch["indexer"],
        "source_commit": identity["source_commit"],
        "interval_ms": [
            value / 1e6 for value in sample["trials"][0]["rank0"]["interval_ns"]
        ],
    }


def prefill_rows(points):
    best = {}
    for point in points:
        label = point["label"]
        layout = point["layout"]
        requests = label // 512
        key = layout, label
        step = timing(point["interval_ms"], 10)
        row = {
            "layout": layout,
            "label_new_tokens_per_gpu_per_step": label,
            "requests_per_step": requests,
            "active_attention_lanes": 1 if layout == "QCP8" else min(requests, 8),
            "new_tokens_per_request": 4096,
            "physical_rows_per_group_step": requests * 4224,
            **step,
            "new_tokens_s_gpu": 1000 * label / step["trimmed_mean_step_ms"],
            "indexer": point["indexer"],
            "source_commit": point["source_commit"],
        }
        if (
            key not in best
            or step["trimmed_mean_step_ms"] < best[key]["trimmed_mean_step_ms"]
        ):
            best[key] = row
    return [best[key] for key in sorted(best)]


def decode_rows(audit):
    published = "measured_source" in audit
    batches = sorted(map(int, audit["batches"]))
    identity = audit["identity"]["indexer"] if published else audit["indexer"]
    fast = all(
        identity.get(key) == "1"
        for key in (
            "TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER",
            "TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT",
        )
    )
    if not fast:
        raise ValueError("Expected the Fast Indexer EP16 measurement")
    if published:
        config = audit["configuration"]
        expected = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": -1,
            "warm_steps": 200,
            "formal_skip": 100,
            "formal_intervals_per_round": 50,
            "rounds": 2,
        }
        if any(
            config[key] != value for key, value in expected.items()
        ) or batches != list(BATCHES):
            raise ValueError("Expected the full formal EP16 measurement")
    elif (
        audit["formal_intervals_per_rank_per_batch"] != 100
        or audit["assumed_tau"] != TAU
    ):
        raise ValueError("Expected the formal EP16 audit")
    rows = []
    for batch in batches:
        intervals = []
        for round_ in (1, 2):
            window = audit["windows"][f"b{batch}-r{round_}"]
            rank = window["ranks"]["0"] if published else window["0"]
            values = rank["interval_ns"]
            if len(values) != 50:
                raise ValueError("Each formal Decode round must provide 50 intervals")
            intervals.extend(value / 1e6 for value in values)
        step = timing(intervals, 100)
        per_user = 1000 * TAU / step["trimmed_mean_step_ms"]
        rows.append(
            {
                "requests_per_rank": batch,
                "requests_per_step": 16 * batch,
                "indexer": "fast",
                **step,
                "equivalent_tokens_s_user": per_user,
                "equivalent_tokens_s_gpu": batch * per_user,
            }
        )
    return rows


def write_csv(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pareto.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def canvas(title, subtitle, *, bottom):
    fig, ax = plt.subplots(figsize=(11, 7.4), facecolor="#f7f9fc")
    fig.subplots_adjust(left=0.105, right=0.96, top=0.815, bottom=bottom)
    fig.text(0.105, 0.936, title, fontsize=27, weight="bold", color="#19243b")
    fig.text(0.105, 0.884, subtitle, fontsize=11, color="#4b5871")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aeb8ca")
    ax.tick_params(colors="#35425b", length=4)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(color="#dbe3ef", alpha=0.65)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, directory):
    path = directory / "pareto.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def plot_prefill(rows, directory):
    fig, ax = canvas(
        "W4A8 Prefill",
        "RTX 6000D  /  8 GPUs  /  128K context  /  fixed 4K new tokens per request",
        bottom=0.32,
    )
    for layout, color in (("DP8", "#2457ac"), ("QCP8", "#087a70")):
        points = [p for p in rows if p["layout"] == layout]
        ax.plot(
            [p["trimmed_mean_step_ms"] for p in points],
            [p["new_tokens_s_gpu"] for p in points],
            color=color,
            lw=2.0,
            label=f"{layout} + EP8",
        )
        for p in points:
            label = p["label_new_tokens_per_gpu_per_step"]
            x, y = p["trimmed_mean_step_ms"], p["new_tokens_s_gpu"]
            fast = p["indexer"] == "native-fp4"
            ax.scatter(
                x,
                y,
                marker="D" if fast else "o",
                s=57,
                color=color,
                edgecolor="white",
                zorder=4,
            )
            short = label < 4096 and layout == "DP8"
            ax.annotate(
                f"{label // 1024}K" if label >= 1024 else str(label),
                (x, y),
                xytext=(-10, 0) if short else (0, 10),
                textcoords="offset points",
                va="center" if short else "bottom",
                ha="right" if short else "center",
                color=color,
            )
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], linestyle="none", marker="D", color="#596680"),
            Line2D([], [], linestyle="none", marker="o", color="#596680"),
        ]
    )
    ax.legend(
        handles,
        labels + ["Native FP4 enabled", "Default Indexer"],
        ncol=1,
        loc="upper right",
        frameon=False,
    )
    ax.set(
        xlabel="Prefill step time (ms, 20% trimmed mean)",
        ylabel="New input tokens / s / GPU",
        xlim=(0, 8650),
        ylim=(0, 3380),
    )
    fig.text(
        0.105,
        0.208,
        "Point label = total new tokens per step / 8 GPUs.  4K new + 128 replay rows per request.",
        fontsize=10.5,
        weight="bold",
        color="#19243b",
    )
    grid = (
        ("Label", ("512", "1K", "2K", "4K", "8K", "16K")),
        ("Requests / step", (1, 2, 4, 8, 16, 32)),
        ("DP8 active lanes", (1, 2, 4, 8, 8, 8)),
    )
    table = "\n".join(
        f"{heading:<18}" + "".join(f"{value:>7}" for value in values)
        for heading, values in grid
    )
    fig.text(
        0.105,
        0.171,
        table,
        family="monospace",
        fontsize=9,
        color="#35425b",
        va="top",
        linespacing=1.45,
    )
    fig.text(
        0.105,
        0.081,
        "DP8 divides the requests evenly across active lanes. Below 4K, each active lane has one full request.\n"
        "QCP8 shards every request across all 8 GPUs. Each point keeps the middle 6 of 10 step intervals.",
        fontsize=9.6,
        color="#4b5871",
        va="top",
        linespacing=1.6,
    )
    save(fig, directory)


def plot_decode(rows, directory):
    fig, ax = canvas(
        "W4A8 Decode",
        "RTX 6000D  /  EP16  /  Fast Indexer  /  128K input context",
        bottom=0.26,
    )
    xs = [p["equivalent_tokens_s_user"] for p in rows]
    ys = [p["equivalent_tokens_s_gpu"] for p in rows]
    ax.plot(
        xs,
        ys,
        marker="o",
        color="#643c9d",
        lw=2.2,
        markeredgecolor="white",
        markersize=8,
    )
    for row, x, y in zip(rows, xs, ys, strict=True):
        ax.annotate(
            f"B{row['requests_per_rank']}",
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            va="bottom",
            ha="center",
            color="#643c9d",
            weight="bold",
        )
    ax.set(
        xlabel="Equivalent output tokens / s / user",
        ylabel="Equivalent output tokens / s / GPU",
        xlim=(0, max(xs) * 1.13),
        ylim=(0, max(ys) * 1.14),
    )
    fig.text(
        0.105,
        0.162,
        "B = simultaneous requests per rank. B8 = 8 per rank, 128 across EP16.",
        fontsize=11,
        weight="bold",
        color="#19243b",
    )
    fig.text(
        0.105,
        0.124,
        "Per user = 4.75 / step time in seconds. Per GPU = B × per-user rate. 4.75 is a conversion assumption.",
        fontsize=10,
        color="#4b5871",
    )
    fig.text(
        0.105,
        0.087,
        "20% trimmed mean uses 60 of 100 intervals pooled from two rounds. Measured verify width: 6.\n"
        "Temperature 0.3 / top-p 0.95 / top-k -1. Warm 200 steps; skip 100 and measure 50 per round.",
        fontsize=9.6,
        color="#4b5871",
        va="top",
        linespacing=1.6,
    )
    save(fig, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--prefill-point", action="append", type=Path)
    parser.add_argument("--decode-audit", type=Path)
    args = parser.parse_args()
    plt.rcParams.update(
        {"svg.hashsalt": "w4a8-pareto-trimmed", "font.size": 10.5, "axes.labelpad": 9}
    )
    points = (
        [prefill_input(path) for path in args.prefill_point]
        if args.prefill_point
        else read(HERE / "prefill/points.json")
    )
    prefill = prefill_rows(points)
    decode = decode_rows(read(args.decode_audit or HERE / "decode/ep16.json"))
    for name, rows, plot in (
        ("prefill", prefill, plot_prefill),
        ("decode", decode, plot_decode),
    ):
        directory = args.out / name
        write_csv(directory, rows)
        plot(rows, directory)


if __name__ == "__main__":
    main()
