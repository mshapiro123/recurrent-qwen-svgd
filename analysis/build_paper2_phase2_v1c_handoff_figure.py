"""Build the Phase-2 V1b/V1c radius-extension handoff figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage5/stage5_paper2_phase2_prewindow_20260731"
FIGURE = ROOT / "docs/figures/paper2_phase2_v1c_radius_extension_20260802"


def main() -> None:
    v1b = json.loads((RUN / "v1b/summary.json").read_text(encoding="utf-8"))
    v1c = json.loads((RUN / "v1c/summary.json").read_text(encoding="utf-8"))
    summaries = (v1b, v1c)
    cells = {
        float(c_value): value
        for summary in summaries
        for c_value, value in summary["results"]["pooled"]["oracle_help"].items()
    }
    preserve = {
        float(c_value): value
        for summary in summaries
        for c_value, value in summary["results"]["pooled"]["preserve_control"].items()
    }
    c_values = sorted(cells)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.55))
    colors = {"prediction": "#6B7280", "pair": "#2563EB", "flip": "#D94841"}

    axes[0].plot(
        c_values,
        [100 * cells[c]["first_order_predicted_pair_cross_rate"] for c in c_values],
        marker="o",
        color=colors["prediction"],
        label="First-order pair prediction",
    )
    axes[0].plot(
        c_values,
        [100 * cells[c]["realized_pair_cross_rate"] for c in c_values],
        marker="s",
        color=colors["pair"],
        label="Realized pair crossing",
    )
    axes[0].plot(
        c_values,
        [100 * cells[c]["realized_teacher_flip_rate"] for c in c_values],
        marker="^",
        color=colors["flip"],
        label="Teacher-token top-1 flip",
    )
    axes[0].set(title="A. Reach grows with radius", xlabel="Tube constant c", ylabel="Oracle-help positions (%)")
    axes[0].set_ylim(0, 72)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")

    axes[1].plot(
        c_values,
        [100 * cells[c]["collateral_hurt_rate"] for c in c_values],
        marker="o",
        color="#D97706",
        label="Oracle-help collateral hurt",
    )
    axes[1].plot(
        c_values,
        [100 * preserve[c]["collateral_hurt_rate"] for c in c_values],
        marker="s",
        color="#7C3AED",
        label="Preserve-control collateral hurt",
    )
    axes[1].set(title="B. Collateral remains rare", xlabel="Tube constant c", ylabel="Collateral hurt rate (%)")
    axes[1].grid(axis="y", alpha=0.2)
    right = axes[1].twinx()
    right.spines["top"].set_visible(False)
    right.plot(
        c_values,
        [100 * preserve[c]["target_preservation_rate"] for c in c_values],
        marker="^",
        linestyle="--",
        color="#111827",
        label="Target retained",
    )
    right.set_ylabel("Preserve target retained (%)")
    right.set_ylim(99.75, 100.02)
    handles, labels = axes[1].get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    axes[1].legend(handles + handles2, labels + labels2, frameon=False, fontsize=7.2, loc="upper left")

    quartiles = v1c["results"]["by_first_order_distance_quantile"]["oracle_help"]
    names = ["Q1\nclosest", "Q2", "Q3", "Q4\nfarthest"]
    flip_rates = [100 * quartiles[f"q{i}"]["results"]["0.15"]["realized_teacher_flip_rate"] for i in range(1, 5)]
    axes[2].bar(names, flip_rates, color=["#15803D", "#4D9B62", "#D4A72C", "#9CA3AF"])
    axes[2].set(title="C. Compatibility is strongly structured", ylabel="Teacher-token flips at c=0.15 (%)", ylim=(0, 105))
    axes[2].grid(axis="y", alpha=0.2)
    for index, value in enumerate(flip_rates):
        axes[2].text(index, value + 2.5, f"{value:.1f}%", ha="center", fontsize=8)

    figure.suptitle(
        "Phase-2 V1b/V1c: bounded writeback reaches more positions at larger radii, with a measurable safety tradeoff",
        fontsize=11,
        y=1.03,
    )
    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(FIGURE.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
