"""Build the three-panel Option B result figure used by the strategy handoff."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/stage5/stage5_paper2_phase2_option_b_20260807/summary.json"
AUDIT = (
    ROOT
    / "outputs/stage5/stage5_paper2_phase2_option_b_bootstrap_audit_20260808/summary.json"
)
OUTPUT = ROOT / "docs/figures/paper2_option_b_final_20260808"


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    pairs = {int(row["seed"]): row for row in source["pairs"]}
    audits = {int(row["seed"]): row for row in audit["seeds"]}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "svg.hashsalt": "paper2-option-b-final-20260808",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    full_colors = {0: "#087E8B", 1: "#D1495B"}
    control_colors = {0: "#5D6770", 1: "#9A7B4F"}

    ax = axes[0]
    for seed, pair in pairs.items():
        steps = np.asarray([row["step"] for row in pair["curve"]]) / 1000.0
        ax.plot(
            steps,
            [row["full_eal"] for row in pair["curve"]],
            color=full_colors[seed],
            linewidth=2,
            label=f"Full, seed {seed}",
        )
        ax.plot(
            steps,
            [row["control_eal"] for row in pair["curve"]],
            color=control_colors[seed],
            linewidth=1.7,
            linestyle="--",
            label=f"Control, seed {seed}",
        )
    ax.axvline(4, color="#222222", linewidth=1, linestyle=":")
    ax.text(
        4.25,
        0.97,
        "fresh-data\nsplice",
        transform=ax.get_xaxis_transform(),
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax.set_title("A. Accepted-length trajectories")
    ax.set_xlabel("Training updates (thousands)")
    ax.set_ylabel("Expected accepted length")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[1]
    for seed, pair in pairs.items():
        steps = np.asarray([row["step"] for row in pair["curve"]]) / 1000.0
        gains = 100.0 * np.asarray([row["relative_full_gain"] for row in pair["curve"]])
        ax.plot(steps, gains, color=full_colors[seed], linewidth=2, label=f"Seed {seed}")
        endpoint = audits[seed]["endpoint_relative_full_gain"]
        lower, upper = np.asarray(endpoint["document_bootstrap_95_ci"]) * 100.0
        ax.errorbar(
            [20],
            [endpoint["estimate"] * 100.0],
            yerr=[[endpoint["estimate"] * 100.0 - lower], [upper - endpoint["estimate"] * 100.0]],
            color=full_colors[seed],
            capsize=3,
            marker="o",
        )
    ax.axhline(1.0, color="#222222", linewidth=1, linestyle=":", label="1% endpoint target")
    ax.axvline(4, color="#222222", linewidth=1, linestyle=":")
    ax.set_title("B. Full-system advantage")
    ax.set_xlabel("Training updates (thousands)")
    ax.set_ylabel("Relative gain over control (%)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    labels: list[str] = []
    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    colors: list[str] = []
    for seed in (0, 1):
        for label, key in (
            (f"S{seed} full", "full_second_half_exposure_slope_eal_per_1000"),
            (f"S{seed} gap", "writeback_gap_second_half_slope_eal_per_1000"),
        ):
            value = audits[seed][key]
            lower, upper = value["document_bootstrap_95_ci"]
            labels.append(label)
            estimates.append(value["estimate"])
            lower_errors.append(value["estimate"] - lower)
            upper_errors.append(upper - value["estimate"])
            colors.append(full_colors[seed] if label.endswith("full") else control_colors[seed])
    positions = np.arange(len(labels))
    ax.bar(positions, estimates, color=colors, width=0.68, alpha=0.9)
    ax.errorbar(
        positions,
        estimates,
        yerr=[lower_errors, upper_errors],
        fmt="none",
        color="#202020",
        capsize=4,
        linewidth=1,
    )
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_title("C. Late slopes with document-bootstrap CIs")
    ax.set_ylabel("EAL per 1,000 updates")
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.98,
        0.98,
        "Full slopes exclude zero;\ngap slopes do not",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    figure.suptitle(
        "Option B: fresh data sustains learning and writeback adds a small replicated increment",
        fontsize=12,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT.with_suffix(".svg")
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
