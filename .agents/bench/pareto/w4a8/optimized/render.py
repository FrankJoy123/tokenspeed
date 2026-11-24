"""Overlay audited candidate points on the published TokenSpeed Pareto curves."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFILL_LABELS = (512, 1024, 2048, 4096, 8192)
DECODE_LABELS = (1, 2, 4, 8, 16, 32)
CAKE_PREFILL_SERIES = "Paged Prefill + Cake DP8"
COMPACT_PREFILL_SERIES = "Compact BF16x2 + Cake DP8"
OPTIMIZED_COLOR = "#25816b"
EPLB_COLOR = "#8b4d89"
EPLB_DECODE_SERIES = "EPLB + fused routing EP16"
EPLB_PREFILL_SERIES = "EPLB R16 static DP8"
NJUAPP_PREFILL_SERIES = "Njuapp (same precision, no placement file)"
NJUAPP_R8_PREFILL_SERIES = "Njuapp (rebuilt R8 placement, DP8)"
NJUAPP_DECODE_SERIES = "Njuapp (same precision)"
NJUAPP_COLOR = "#b14f58"
NJUAPP_R8_COLOR = "#7b263c"
CAKE_PREFILL_STEADY_RUNS = (
    "dp8-512-d88-c2048-u95-native-fp4",
    "dp8-1k-d88-c2048-u95-native-fp4",
    "dp8-2k-d88-c2048-u95-native-fp4",
    "integrated-p4k-d88-c2048-u95",
)
CLIENT_8K_SERIES = {
    "dp8-8k-dp-scheduler-pause-enabled-d3": CAKE_PREFILL_SERIES,
    "p8k-aligned-compact-9aca52751-d3": COMPACT_PREFILL_SERIES,
    "p8k-aligned-static-r16-c1024-d2": EPLB_PREFILL_SERIES,
    "p-njuapp-8k-aligned-1120": NJUAPP_PREFILL_SERIES,
    "p-njuapp-r8-8k-aligned-1120": NJUAPP_R8_PREFILL_SERIES,
}


def load_plot(bench):
    spec = importlib.util.spec_from_file_location(
        "tokenspeed_pareto_plot", bench / "plot.py"
    )
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    return plot


def line(ax, rows, order, label_field, x, y, *, label, color, dashed=False):
    points = {row[label_field]: row for row in rows}
    missing = {x: float("nan"), y: float("nan")}
    ordered = [points.get(key, missing) for key in order]
    ax.plot(
        [row[x] for row in ordered],
        [row[y] for row in ordered],
        label=label,
        color=color,
        linestyle="--" if dashed else "-",
        linewidth=2,
        zorder=2,
    )


def njuapp_curve(
    ax,
    rows,
    labels,
    label_field,
    x,
    y,
    series,
    label_text,
    *,
    color=NJUAPP_COLOR,
    marker="P",
):
    if not rows:
        return
    if len(rows) > 1:
        line(ax, rows, labels, label_field, x, y, label=series, color=color)
    ax.scatter(
        [row[x] for row in rows],
        [row[y] for row in rows],
        marker=marker,
        s=85,
        color=color,
        edgecolor="white",
        label=series if len(rows) == 1 else None,
        zorder=8,
    )
    for row in rows:
        ax.annotate(
            label_text(row[label_field]),
            (row[x], row[y]),
            xytext=(7, 10),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )


def write_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_chart(plot, fig, path):
    fig.savefig(path, metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    plot.plt.close(fig)


def current_prefill(plot, curves, directory):
    fig, ax = plot.canvas(
        "W4A8 Prefill Pareto",
        "RTX 6000D · EP8 · 124K cached + 4K new tokens per request",
        bottom=0.24,
    )
    for curve in curves:
        points = sorted(curve["points"], key=lambda point: point["new_tokens_per_gpu"])
        ax.plot(
            [point["trimmed_step_ms"] for point in points],
            [point["new_tokens_per_s_per_gpu"] for point in points],
            label=curve["series"],
            color=curve["color"],
            linewidth=2,
            marker="o",
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=2,
        )
    ax.set(xlabel="Prefill time (ms)", ylabel="New input tokens / s / GPU")
    ax.margins(x=0.07, y=0.13)
    fig.subplots_adjust(top=0.75)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.015),
        ncols=2,
        frameon=False,
        borderaxespad=0,
        fontsize=9,
    )
    fig.text(
        0.105,
        0.075,
        "Workload units are new tokens per GPU.\n"
        "Four-point curves connect 512, 1K, 2K and 4K in load order. Latency uses 20% trimmed step time.\n"
        "nv uses the public DP8 and QCP8 launch recipes with the local Pareto workload.",
        fontsize=10,
        color="#4b5871",
    )
    save_chart(plot, fig, directory / "prefill.svg")


def prefill(plot, historical, candidates, cohorts, directory, *, full_plot):
    label_field = "label_new_tokens_per_gpu_per_step"
    plot_labels = PREFILL_LABELS if full_plot else (4096, 8192)
    x, y = "processing_time_ms", "new_tokens_s_gpu"
    cohort_rows = [
        row for row in cohorts if row["new_tokens_per_gpu_per_cohort"] in PREFILL_LABELS
    ]
    selected = {}
    for row in cohort_rows:
        key = (row["series"], row["new_tokens_per_gpu_per_cohort"])
        row["processing_time_ms"] = row["mean_complete_cohort_ms"]
        if row["measurement"] == "scheduler_release_to_complete":
            if row["all_ranks_aligned"]:
                selected[key] = row
        else:
            selected.setdefault(key, row)
    if not full_plot:
        for row in cohort_rows:
            series = CLIENT_8K_SERIES.get(row["run"])
            if series is not None and row["all_ranks_aligned"]:
                selected[series, 8192] = dict(
                    row,
                    processing_time_ms=row["mean_client_complete_ms"],
                    new_tokens_s_gpu=row["client_new_tokens_s_gpu"],
                    measurement="client_submit_to_complete",
                )
    for row in (*historical, *candidates):
        row.update(
            processing_time_ms=row["trimmed_mean_step_ms"],
            measurement="steady_trimmed_20pct",
        )
    fig, ax = plot.canvas(
        "W4A8 DP8 Prefill",
        "RTX 6000D · EP8 · Each request: 124K cached + 4K new tokens",
        bottom=0.24,
    )
    groups = (
        (
            "Published DP8",
            [
                dict(row, series="Published DP8")
                for row in historical
                if row["layout"] == "DP8" and row[label_field] in PREFILL_LABELS
            ],
            "#6c7890",
            True,
        ),
        (
            "Paged Prefill DP8",
            [
                row
                for row in candidates
                if row["series"] == "Paged Prefill DP8"
                and row[label_field] in PREFILL_LABELS
            ],
            "#985d18",
            False,
        ),
        (
            CAKE_PREFILL_SERIES,
            [
                row
                for row in candidates
                if row["series"] == CAKE_PREFILL_SERIES
                and row[label_field] in PREFILL_LABELS
            ],
            "#2457ac",
            False,
        ),
    )
    legend_labels = {
        "Published DP8": "Published",
        "Paged Prefill DP8": "Attn opt.",
        CAKE_PREFILL_SERIES: "Attn opt. + MegaMoE",
        COMPACT_PREFILL_SERIES: "Compact + MegaMoE",
    }
    cake_steady = {
        row[label_field]: row
        for row in candidates
        if row["run"] in CAKE_PREFILL_STEADY_RUNS
    }
    compact_runs = {"p4k-compact-9aca52751-d3"}
    if full_plot:
        compact_runs.add("p8k-single-compact-9aca52751-d4")
    compact = (
        COMPACT_PREFILL_SERIES,
        [row for row in candidates if row["run"] in compact_runs],
        OPTIMIZED_COLOR,
        False,
    )
    split_points = []
    for series, rows, color, dashed in (*groups, compact):
        optimized = series == COMPACT_PREFILL_SERIES
        series_labels = PREFILL_LABELS if series == CAKE_PREFILL_SERIES else plot_labels
        rows = [row for row in rows if row[label_field] in series_labels]
        if series == CAKE_PREFILL_SERIES and not full_plot:
            rows = [
                cake_steady.get(row[label_field], row)
                for row in rows
                if row[label_field] != 8192
            ]
        curve = {row[label_field]: row for row in rows}
        for (split_series, size), row in selected.items():
            if split_series == series and size in series_labels:
                if full_plot or size not in curve:
                    split_points.append((size, row, color))
                curve.setdefault(size, {**row, label_field: size})
        if not curve:
            continue
        line(
            ax,
            curve.values(),
            series_labels,
            label_field,
            x,
            y,
            label=legend_labels[series],
            color=color,
            dashed=dashed,
        )
        for row in rows:
            mark = "D" if row["indexer"] == "native-fp4" else "o"
            size = row[label_field]
            ax.scatter(
                row[x],
                row[y],
                marker=mark,
                s=99 if optimized else 56,
                facecolor="none" if optimized else color,
                edgecolor=color if optimized else "white",
                linewidth=2 if optimized else 1,
                zorder=7 if optimized else 5,
                gid=f"prefill-compact-steady-{size}" if optimized else None,
            )
            value = f"{size // 1024}K" if size >= 1024 else str(size)
            ax.annotate(
                value,
                (row[x], row[y]),
                color=color,
                xytext=(-9, 8) if optimized else (6, -10 if dashed else 8),
                textcoords="offset points",
                fontsize=9,
                ha="right" if optimized else "left",
                va="center",
            )
    for size, row, color in split_points:
        optimized = row["series"] == COMPACT_PREFILL_SERIES
        ax.scatter(
            row[x],
            row[y],
            marker="s",
            s=131 if optimized else 85,
            facecolor="none" if optimized else color,
            edgecolor=color if optimized else "white",
            linewidth=2 if optimized else 1,
            zorder=7 if optimized else 6,
            gid=f"prefill-compact-client-{size}" if optimized else None,
        )
        ax.annotate(
            f"{size // 1024}K",
            (row[x], row[y]),
            color=color,
            xytext=(-9, 8) if optimized else (8, 8),
            textcoords="offset points",
            fontsize=9,
            ha="right" if optimized else "left",
        )
    for series, color, marker in (
        (NJUAPP_PREFILL_SERIES, NJUAPP_COLOR, "P"),
        (NJUAPP_R8_PREFILL_SERIES, NJUAPP_R8_COLOR, "X"),
    ):
        upstream = [row for row in candidates if row["series"] == series]
        if not full_plot:
            upstream = [row for row in upstream if row[label_field] != 8192]
            if client_8k := selected.get((series, 8192)):
                upstream.append(dict(client_8k, **{label_field: 8192}))
        njuapp_curve(
            ax,
            upstream,
            PREFILL_LABELS,
            label_field,
            x,
            y,
            series,
            lambda size: f"{size // 1024}K" if size >= 1024 else str(size),
            color=color,
            marker=marker,
        )
    eplb = [row for row in candidates if row["series"] == EPLB_PREFILL_SERIES]
    if not full_plot and (client_8k := selected.get((EPLB_PREFILL_SERIES, 8192))):
        eplb = [row for row in eplb if row[label_field] != 8192]
        eplb.append(dict(client_8k, **{label_field: 8192}))
    if eplb:
        if len(eplb) > 1:
            line(
                ax,
                eplb,
                PREFILL_LABELS,
                label_field,
                x,
                y,
                label=EPLB_PREFILL_SERIES,
                color=EPLB_COLOR,
            )
        for row in eplb:
            size = row[label_field]
            ax.scatter(
                row[x],
                row[y],
                marker=(
                    "s" if row["measurement"] == "client_submit_to_complete" else "^"
                ),
                s=85,
                facecolor="white",
                edgecolor=EPLB_COLOR,
                linewidth=1.7,
                label=EPLB_PREFILL_SERIES if len(eplb) == 1 else None,
                zorder=8,
                gid=f"prefill-eplb-{size}",
            )
            ax.annotate(
                f"{size // 1024}K" if size >= 1024 else str(size),
                (row[x], row[y]),
                color=EPLB_COLOR,
                xytext=(7, -12),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set(
        xlabel="Prefill time (ms)",
        ylabel="New input tokens / s / GPU",
    )
    ax.margins(x=0.07, y=0.13)
    ax.legend(loc="upper right", frameon=False)
    gated = any(
        row["measurement"] == "scheduler_release_to_complete"
        for _, row, _ in split_points
    )
    split_note = (
        "Squares use time from scheduler resume to last output."
        if gated
        else "Squares use time from submission to last output."
    )
    marker_note = (
        "Circles use BF16 Indexer and diamonds use Native FP4 Indexer. Both show trimmed steady step time."
        if full_plot
        else "Diamonds show 20% trimmed steady step time with Native FP4 Indexer."
    )
    if eplb:
        marker_note += " Triangles show EPLB steady steps."
    fig.text(
        0.105,
        0.075,
        f"Labels are new tokens per GPU.\n{marker_note}\n{split_note}",
        fontsize=10,
        color="#4b5871",
    )
    save_chart(plot, fig, directory / "prefill-historical.svg")
    plot_series = {series for series, _, _, _ in groups}
    write_csv(
        directory / "prefill.csv",
        [row for _, rows, _, _ in groups for row in rows]
        + [row for row in candidates if row["series"] not in plot_series]
        + cohort_rows,
    )


def decode(plot, historical, candidates, directory):
    label_field = "requests_per_rank"
    x, y = "equivalent_tokens_s_user", "equivalent_tokens_s_gpu"
    fig, ax = plot.canvas("W4A8 Decode", "RTX 6000D · EP16 · 128K input", bottom=0.22)
    historical = [dict(row, series="Published EP16") for row in historical]
    adopted = [row for row in candidates if row["series"] == "d88a3d1ea EP16"]
    wide = [row for row in candidates if row["series"] == "Cake MLA wider EP16"]
    eplb = [row for row in candidates if row["series"] == EPLB_DECODE_SERIES]
    quant = [
        row for row in candidates if row["series"] == "Direct quant and scale pack"
    ]
    line(
        ax,
        historical,
        DECODE_LABELS,
        label_field,
        x,
        y,
        label="Published EP16",
        color="#6c7890",
        dashed=True,
    )
    line(
        ax,
        adopted,
        DECODE_LABELS,
        label_field,
        x,
        y,
        label="d88a3d1ea EP16",
        color="#2457ac",
    )
    if wide:
        line(
            ax,
            wide,
            DECODE_LABELS,
            label_field,
            x,
            y,
            label="Cake MLA wider shapes",
            color=OPTIMIZED_COLOR,
        )
    for row in historical:
        ax.scatter(
            row[x], row[y], s=74, facecolor="white", edgecolor="#6c7890", zorder=3
        )
        ax.annotate(
            f"B{row[label_field]}",
            (row[x], row[y]),
            xytext=(-7, 10),
            textcoords="offset points",
            ha="right",
            color="#596680",
            fontsize=9,
        )
    for row in quant:
        ax.scatter(
            row[x],
            row[y],
            marker="s",
            s=55,
            facecolor="white",
            edgecolor="#985d18",
            label=row["series"],
            zorder=4,
        )
    sparse = [row for row in adopted if row["selected_backend"] == "sparse"]
    cake = [row for row in adopted if row["selected_backend"] == "cake_dual"]
    if sparse:
        ax.scatter(
            [row[x] for row in sparse],
            [row[y] for row in sparse],
            s=37,
            color="#2457ac",
            edgecolor="white",
            label="Sparse MLA",
            zorder=5,
        )
    if cake:
        ax.scatter(
            [row[x] for row in cake],
            [row[y] for row in cake],
            marker="*",
            s=140,
            color="#2457ac",
            edgecolor="white",
            label="Cake dual-cache MLA at B8",
            zorder=6,
        )
    for row in wide:
        batch = row[label_field]
        ax.scatter(
            row[x],
            row[y],
            s=96,
            facecolor="none",
            edgecolor=OPTIMIZED_COLOR,
            linewidth=2,
            zorder=7,
            gid=f"decode-cake-wide-b{batch}",
        )
        ax.annotate(
            f"B{batch}",
            (row[x], row[y]),
            color=OPTIMIZED_COLOR,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )
    upstream = [row for row in candidates if row["series"] == NJUAPP_DECODE_SERIES]
    njuapp_curve(
        ax,
        upstream,
        DECODE_LABELS,
        label_field,
        x,
        y,
        NJUAPP_DECODE_SERIES,
        lambda batch: f"B{batch}",
    )
    if eplb:
        if len(eplb) > 1:
            line(
                ax,
                eplb,
                DECODE_LABELS,
                label_field,
                x,
                y,
                label=EPLB_DECODE_SERIES,
                color=EPLB_COLOR,
            )
        ax.scatter(
            [row[x] for row in eplb],
            [row[y] for row in eplb],
            marker="^",
            s=85,
            facecolor="white",
            edgecolor=EPLB_COLOR,
            linewidth=1.7,
            label=EPLB_DECODE_SERIES if len(eplb) == 1 else None,
            zorder=8,
            gid="decode-eplb",
        )
        for row in eplb:
            ax.annotate(
                f"B{row[label_field]}",
                (row[x], row[y]),
                color=EPLB_COLOR,
                xytext=(7, -12),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set(
        xlabel="Equivalent output tokens / s / user",
        ylabel="Equivalent output tokens / s / GPU",
    )
    ax.margins(x=0.11, y=0.14)
    ax.legend(loc="lower left", frameon=False)
    fig.text(
        0.105,
        0.09,
        "20% trimmed mean of two formal rounds. Per-user throughput uses the fixed τ=4.75 conversion.",
        fontsize=10,
        color="#4b5871",
    )
    save_chart(plot, fig, directory / "decode.svg")
    write_csv(directory / "decode.csv", historical + candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench",
        type=Path,
        required=True,
        help="Published .agents/bench/pareto/w4a8 directory",
    )
    parser.add_argument("--points", type=Path, default=HERE / "points.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--full-prefill-plot",
        action="store_true",
        help="Show all Prefill workload sizes and both Cake 8K measurements",
    )
    args = parser.parse_args()
    plot = load_plot(args.bench)
    points = json.loads(args.points.read_text())
    plot.plt.rcParams["svg.hashsalt"] = "tokenspeed-pareto-cake"
    args.out.mkdir(parents=True, exist_ok=True)
    current = json.loads((HERE / "prefill-current.json").read_text())
    current_prefill(plot, current["curves"], args.out)
    prior_prefill = plot.prefill_rows(plot.read(args.bench / "prefill/points.json"))
    new_prefill = []
    for point in points["prefill"]:
        row = plot.prefill_rows([point])[0]
        row.update(
            **{
                key: point[key]
                for key in (
                    "series",
                    "run",
                    "gpu_memory_utilization",
                    "selected_backend",
                    "source_is_clean",
                    "tracked_diff_sha256",
                    "cake_rows",
                    "queue_requests_per_lane",
                    "resident_sequences",
                    "max_model_len",
                    "max_total_tokens",
                    "retraction_snapshot_ratio",
                )
            }
        )
        if "native_fp4_warm_rank0_verified" in point:
            row.update(
                native_fp4_warm_rank0_verified=point["native_fp4_warm_rank0_verified"],
                native_fp4_warm_rank0_logit_stack_samples=point[
                    "native_fp4_warm_rank0_logit_stack_samples"
                ],
            )
        new_prefill.append(row)
    prefill(
        plot,
        prior_prefill,
        new_prefill,
        points.get("cohorts", []),
        args.out,
        full_plot=args.full_prefill_plot,
    )
    old_decode = plot.read(args.bench / "decode/ep16.json")
    prior_decode = [
        dict(row, source_commit=old_decode["measured_source"]["commit"])
        for row in plot.decode_rows(old_decode)
    ]
    new_decode = []
    for point in points["decode"]:
        batch = point["batch"]
        row = plot.timing(point["interval_ms"], 100)
        rate = 1000 * plot.TAU / row["trimmed_mean_step_ms"]
        row.update(
            requests_per_rank=batch,
            requests_per_step=batch * 16,
            indexer=(
                "upstream-grouped"
                if point["series"] == NJUAPP_DECODE_SERIES
                else "fast"
            ),
            equivalent_tokens_s_user=rate,
            equivalent_tokens_s_gpu=batch * rate,
            **{
                key: point[key]
                for key in ("source_commit", "series", "run", "selected_backend")
            },
        )
        new_decode.append(row)
    decode(plot, prior_decode, new_decode, args.out)


if __name__ == "__main__":
    main()
