"""Build review figures for the complete Bicameral W1 ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "target": "#16697A",
    "control": "#E09F3E",
    "random": "#7A7A7A",
    "positive": "#2A9D8F",
    "negative": "#C8553D",
    "cluster0": "#355070",
    "cluster1": "#EAAC8B",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(axis, *, zero: bool = True):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    if zero:
        axis.axhline(0, color="#3A3A3A", linewidth=0.9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--phase-b", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_a = load(args.phase_a)
    phase_b = load(args.phase_b)
    generation = load(args.generation)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)

    # Panel A: Phase-A margin ladder.
    axis = axes[0, 0]
    families = ["l0a", "l0b", "l0c", "l0d", "l0g"]
    x = np.arange(len(families))
    width = 0.34
    own = [phase_a["phase_a"]["families"][family]["pooled_seed_mean"] for family in families]
    shuffled = [
        np.mean(phase_a["phase_a"]["families"][family]["shuffle_seed_means"])
        for family in families
    ]
    axis.bar(x - width / 2, own, width, color=COLORS["target"], label="Target")
    axis.bar(x + width / 2, shuffled, width, color=COLORS["control"], label="Shuffled")
    axis.set_xticks(x, [label.upper() for label in families])
    axis.set_ylabel("Teacher-token margin change")
    axis.set_title("A  Phase A selects student correction delta (L0c)", loc="left")
    axis.legend(frameon=False, ncols=2, loc="upper left")
    style_axis(axis)

    # Panel B: Phase-B granularity and residual read.
    axis = axes[0, 1]
    labels = ["L1\ncluster", "L2\nglobal", "L3\nother cluster", "Best L6\nresidual"]
    values = []
    lows = []
    highs = []
    for seed in ("0", "1"):
        cells = phase_b["cells"][seed]
        best_l6 = max(
            (payload for name, payload in cells.items() if name.startswith("l6_")),
            key=lambda payload: payload["mean"],
        )
        local = [cells["l1"], cells["l2"], cells["l3"], best_l6]
        values.append([payload["mean"] for payload in local])
        lows.append([payload["ci_low"] for payload in local])
        highs.append([payload["ci_high"] for payload in local])
    values_array = np.asarray(values)
    for seed, offset, marker in ((0, -0.08, "o"), (1, 0.08, "s")):
        yerr = np.vstack(
            [values_array[seed] - np.asarray(lows[seed]), np.asarray(highs[seed]) - values_array[seed]]
        )
        axis.errorbar(
            np.arange(4) + offset,
            values_array[seed],
            yerr=yerr,
            fmt=marker,
            color=COLORS["target"] if seed == 0 else COLORS["negative"],
            capsize=3,
            label=f"Seed {seed}",
        )
    axis.set_xticks(np.arange(4), labels)
    axis.set_ylabel("Teacher-token margin change")
    axis.set_title("B  Global mean wins margins; residual directions fail", loc="left")
    axis.legend(frameon=False, ncols=2, loc="upper right")
    style_axis(axis)

    # Panel C: generation correct counts.
    axis = axes[1, 0]
    parent_pairs = ["l0a_vs_shuffle", "l0c_vs_shuffle", "l1_vs_shuffle", "l2_vs_shuffle", "l3_vs_shuffle"]
    parent_labels = ["L0a", "L0c", "L1", "L2", "L3"]
    parent = np.asarray(
        [[generation["pairs"][str(seed)][name]["parent_correct"] for name in parent_pairs] for seed in (0, 1)]
    )
    control = np.asarray(
        [[generation["pairs"][str(seed)][name]["control_correct"] for name in parent_pairs] for seed in (0, 1)]
    )
    x = np.arange(len(parent_labels))
    axis.bar(x - width / 2, parent.mean(axis=0), width, color=COLORS["target"], label="Target")
    axis.bar(x + width / 2, control.mean(axis=0), width, color=COLORS["control"], label="Matched control")
    for seed, marker in ((0, "o"), (1, "s")):
        axis.scatter(x - width / 2, parent[seed], color="white", edgecolor="#1F1F1F", marker=marker, s=28, zorder=4)
        axis.scatter(x + width / 2, control[seed], color="white", edgecolor="#1F1F1F", marker=marker, s=28, zorder=4)
    random_mean = np.mean(
        [generation["pairs"][str(seed)]["l0a_vs_random"]["control_correct"] for seed in (0, 1)]
    )
    axis.axhline(random_mean, color=COLORS["random"], linestyle="--", linewidth=1.4, label="Random mean")
    axis.set_xticks(x, parent_labels)
    axis.set_ylabel("Correct rows of 461")
    axis.set_title("C  Margin gains do not transfer to population targets", loc="left")
    axis.legend(frameon=False, ncols=3, loc="upper right")
    style_axis(axis, zero=False)

    # Panel D: paired net rows against matched control.
    axis = axes[1, 1]
    net = np.asarray(
        [[generation["pairs"][str(seed)][name]["net_rows"] for name in parent_pairs] for seed in (0, 1)]
    )
    for seed, offset, marker in ((0, -0.08, "o"), (1, 0.08, "s")):
        colors = [COLORS["positive"] if value > 0 else COLORS["negative"] if value < 0 else COLORS["random"] for value in net[seed]]
        axis.scatter(x + offset, net[seed], c=colors, marker=marker, s=58, edgecolor="white", linewidth=0.7, label=f"Seed {seed}")
    axis.set_xticks(x, parent_labels)
    axis.set_ylabel("Fixes minus regressions")
    axis.set_title("D  Only oracle-assisted L0a has a large control contrast", loc="left")
    axis.legend(frameon=False, ncols=2, loc="upper right")
    style_axis(axis)

    figure.suptitle(
        "Bicameral W1: causal targets move margins, but population targets are not answer-grade",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "svg"):
        figure.savefig(args.output_dir / f"paper2_bicameral_w1_summary.{suffix}", dpi=180)
    plt.close(figure)

    # Separate cluster-composition figure makes the task-family split explicit.
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    composition = phase_b["cluster_extension"]["0"]["battery_composition"]
    batteries = ["arc_challenge", "arc_easy", "gsm8k", "mbpp", "mmlu", "tier1"]
    cluster_ids = ["0", "1"]
    bottom = np.zeros(2)
    palette = ["#355070", "#6D597A", "#2A9D8F", "#E9C46A", "#E76F51", "#8D99AE"]
    for battery, color in zip(batteries, palette):
        counts = np.asarray(
            [composition[cluster]["battery_counts"].get(battery, 0) for cluster in cluster_ids]
        )
        axis.bar(["Cluster 0", "Cluster 1"], counts, bottom=bottom, label=battery.replace("_", " "), color=color)
        bottom += counts
    axis.set_ylabel("DEV-2 rows")
    axis.set_title("Frozen k=2 extension is primarily a task-family partition", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncols=3, loc="upper left")
    style_axis(axis, zero=False)
    for suffix in ("png", "svg"):
        figure.savefig(args.output_dir / f"paper2_bicameral_w1_cluster_composition.{suffix}", dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
