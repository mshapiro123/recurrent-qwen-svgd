"""Build the P3.4 executed-lock calibration summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / (
    "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/receipts/"
    "p34_task_guardrail_calibration.json"
)
LOCK = ROOT / "training/paper2_phase3_p34_preregistration.json"
OUTPUT = ROOT / "docs/figures/p34_executed_lock_calibration_summary_20260813"


def power_curve(result: dict) -> tuple[list[float], list[float]]:
    candidates = result["tier_s"]["all_power_candidates"]
    points = sorted(
        (
            abs(float(value["true_mean_difference"])) * 100,
            float(value["estimated_action_probability"]) * 100,
        )
        for value in candidates.values()
    )
    return [point[0] for point in points], [point[1] for point in points]


def main() -> None:
    task = json.loads(TASK.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    colors = ["#2878B5", "#D95F02"]
    labels = ["Empirical autocorrelation", "Conservative upper-95"]
    for entry, color, label in zip(task["sensitivity_band"], colors, labels):
        x, y = power_curve(entry["result"])
        axes[0].plot(x, y, marker="o", markersize=3.5, linewidth=2, color=color, label=label)
    axes[0].axhline(99, color="#3A3A3A", linewidth=1, linestyle="--", label="99% power target")
    axes[0].axvline(5.5, color="#D95F02", linewidth=1, linestyle=":")
    axes[0].scatter([5.5], [99.822], color="#D95F02", s=42, zorder=5)
    axes[0].annotate(
        "Bound Tier-S threshold\n5.5 points, 99.82% power",
        xy=(5.5, 99.822),
        xytext=(6.0, 72),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9,
    )
    axes[0].set(title="Task guardrail detection power", xlabel="Sustained true accuracy drop (points)", ylabel="Campaign detection probability (%)")
    axes[0].set_xlim(0, 9.3)
    axes[0].set_ylim(-2, 104)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)

    weight_sets = lock["loss_share_contract"]["scalar_weights_by_seed"]
    configurations = ["seed_0_main", "seed_1_main", "seed_0_slot"]
    display = ["Seed 0 main", "Seed 1 main", "Seed 0 slot"]
    losses = ["kl", "aim", "ce", "gate", "preserve", "slot"]
    x = np.arange(len(losses))
    width = 0.24
    palette = ["#2878B5", "#6A9F58", "#D95F02"]
    for index, (configuration, label, color) in enumerate(zip(configurations, display, palette)):
        values = [weight_sets[configuration].get(loss, np.nan) for loss in losses]
        axes[1].bar(x + (index - 1) * width, values, width, label=label, color=color)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, ["KL", "Aim", "CE", "Gate", "Preserve", "Slot"])
    axes[1].set(title="KL-normalized static loss weights", ylabel="Scalar weight (log scale)")
    axes[1].grid(axis="y", which="both", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[1].text(
        0.02,
        0.03,
        "Exact target shares are identical across main seeds;\nweights differ because raw gradient scales differ.",
        transform=axes[1].transAxes,
        fontsize=8.5,
        va="bottom",
    )

    fig.suptitle("P3.4 executed-lock calibrations", fontsize=15, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(svg_path)
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
