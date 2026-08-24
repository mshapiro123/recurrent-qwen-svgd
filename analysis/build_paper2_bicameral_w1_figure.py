"""Build the W1 ladder diagnostic figure from receipt-backed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("l0a", "l0b", "l0c", "l0d", "l0g")
FAMILY_LABELS = {
    "l0a": "CE gradient",
    "l0b": "14B state",
    "l0c": "Margin gradient",
    "l0d": "Teacher-forced delta",
    "l0g": "Neighbor centroid",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def receipt_cells(receipts: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for receipt in receipts:
        for cell in receipt["cells"]:
            result.setdefault(str(cell["arm"]), []).append(cell)
    return result


def _mean_interval(cells: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    means = [float(cell["mean"]) for cell in cells]
    lows = [float(cell["ci_low"]) for cell in cells]
    highs = [float(cell["ci_high"]) for cell in cells]
    return float(np.mean(means)), float(np.mean(lows)), float(np.mean(highs))


def build_figure(
    *,
    seed_receipts: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    output: Path,
) -> None:
    cells = receipt_cells(seed_receipts)
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    figure.patch.set_facecolor("#f7f7f5")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d9ddd8", linewidth=0.8, alpha=0.75)

    # Panel A: registered Phase-A cells.
    arm_order = [
        "l0a",
        "l5_a",
        "l0b",
        "l5_b",
        "l0c",
        "l5_c",
        "l0d",
        "l5_d",
        "l0g",
        "l5_g",
        "l4",
    ]
    labels = ["L0a", "shuffle", "L0b", "shuffle", "L0c", "shuffle", "L0d", "shuffle", "L0g", "shuffle", "random"]
    means, lower, upper = [], [], []
    colors = []
    for arm in arm_order:
        mean, low, high = _mean_interval(cells[arm])
        means.append(mean)
        lower.append(mean - low)
        upper.append(high - mean)
        colors.append("#167f83" if arm.startswith("l0") else ("#cc6b32" if arm.startswith("l5") else "#70757a"))
    x = np.arange(len(arm_order))
    axes[0, 0].bar(x, means, color=colors, width=0.75)
    axes[0, 0].errorbar(x, means, yerr=[lower, upper], fmt="none", ecolor="#202426", capsize=2, linewidth=1)
    axes[0, 0].axhline(0.0, color="#202426", linewidth=1)
    axes[0, 0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0, 0].set_ylabel("Mean teacher-token margin delta")
    axes[0, 0].set_title("A. Registered target families and controls", loc="left", fontweight="bold")

    # Panel B: paired own-row advantage over the matched shuffled target.
    seed_payloads = list(diagnostics["seeds"].values())
    advantage_means, advantage_low, advantage_high = [], [], []
    for family in FAMILIES:
        summaries = [seed["families"][family]["versus_shuffle"]["paired_advantage"] for seed in seed_payloads]
        mean = float(np.mean([cell["mean"] for cell in summaries]))
        low = float(np.mean([cell["ci_low"] for cell in summaries]))
        high = float(np.mean([cell["ci_high"] for cell in summaries]))
        advantage_means.append(mean)
        advantage_low.append(mean - low)
        advantage_high.append(high - mean)
    y = np.arange(len(FAMILIES))
    axes[0, 1].barh(y, advantage_means, color="#225ea8", height=0.62)
    axes[0, 1].errorbar(advantage_means, y, xerr=[advantage_low, advantage_high], fmt="none", ecolor="#202426", capsize=2)
    axes[0, 1].axvline(0.0, color="#202426", linewidth=1)
    axes[0, 1].set_yticks(y, [FAMILY_LABELS[name] for name in FAMILIES])
    axes[0, 1].set_xlabel("Paired margin advantage over shuffle")
    axes[0, 1].set_title("B. Row-specific information beyond global bias", loc="left", fontweight="bold")

    # Panel C: response across baseline-margin quartiles for the two positive candidates.
    for family, color, marker in (("l0a", "#167f83", "o"), ("l0c", "#7a4eab", "s")):
        by_seed = [seed["families"][family]["baseline_margin_quartiles"] for seed in seed_payloads]
        values = [float(np.mean([rows[index]["mean_delta"] for rows in by_seed])) for index in range(4)]
        axes[1, 0].plot(range(1, 5), values, color=color, marker=marker, linewidth=2, label=FAMILY_LABELS[family])
    axes[1, 0].axhline(0.0, color="#202426", linewidth=1)
    axes[1, 0].set_xticks(range(1, 5), ["lowest", "Q2", "Q3", "highest"])
    axes[1, 0].set_xlabel("Baseline teacher-margin quartile")
    axes[1, 0].set_ylabel("Mean margin delta")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_title("C. Effect persists beyond marginal rows", loc="left", fontweight="bold")

    # Panel D: L0c target-specific advantage by battery, with tiny batteries disclosed.
    battery_values: dict[str, list[float]] = {}
    battery_rows: dict[str, int] = {}
    for seed in seed_payloads:
        by_battery = seed["families"]["l0c"]["versus_shuffle"]["by_battery"]
        for battery, cell in by_battery.items():
            battery_values.setdefault(battery, []).append(float(cell["mean"]))
            battery_rows[battery] = int(cell["rows"])
    battery_order = sorted(battery_values, key=lambda name: (-battery_rows[name], name))
    battery_means = [float(np.mean(battery_values[name])) for name in battery_order]
    battery_labels = [f"{name}\n(n={battery_rows[name]})" for name in battery_order]
    bx = np.arange(len(battery_order))
    axes[1, 1].bar(bx, battery_means, color="#6b8e23", width=0.68)
    axes[1, 1].axhline(0.0, color="#202426", linewidth=1)
    axes[1, 1].set_xticks(bx, battery_labels, rotation=35, ha="right")
    axes[1, 1].set_ylabel("L0c paired advantage over shuffle")
    axes[1, 1].set_title("D. Margin-gradient specificity by battery", loc="left", fontweight="bold")

    seed_label = ", ".join(f"seed {receipt['seed']}" for receipt in seed_receipts)
    figure.suptitle(
        f"Bicameral W1 Phase-A diagnostics ({seed_label}; DEV-2; score-only)",
        fontsize=15,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".svg"), facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_summary", type=Path, action="append", required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_figure(
        seed_receipts=[load_json(path) for path in args.seed_summary],
        diagnostics=load_json(args.diagnostics),
        output=args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
