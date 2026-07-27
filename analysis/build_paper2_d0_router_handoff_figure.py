"""Build the D0 routing-feasibility handoff figure from landed receipts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage5/stage5_paper2_d0_20260726"
OUTPUT = ROOT / "docs/figures/paper2_d0_router_handoff_20260727"

INK = "#18212b"
BLUE = "#276FBF"
GOLD = "#C7922D"
ORANGE = "#D66A2C"
OLIVE = "#6E7F32"
GREY = "#8A96A3"
LIGHT = "#DDE3E8"


def load(name: str) -> dict:
    return json.loads((RUN / name / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    oracle = load("router_oracle_audit")
    probe = load("router_probe")
    runtime = probe["runtime_hardware_sensitivity"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": LIGHT,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.86, wspace=0.14, hspace=0.28)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "D0 depth-routing feasibility",
        fontsize=18,
        fontweight="bold",
        x=0.03,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.03,
        0.945,
        "Teacher agreement on 53,389 calibration rejections; frozen-floor labels are primary",
        fontsize=10.5,
        color="#52606d",
        ha="left",
    )

    # A: fixed depth and oracle ceiling.
    ax = axes[0, 0]
    depths = np.arange(1, 7)
    for teacher, color, label in (("teacher_7b", BLUE, "7B target"), ("teacher_14b_sensitivity", GOLD, "14B sensitivity")):
        section = oracle[teacher]
        values = [100 * section["fixed_depth"][str(depth)]["accuracy"] for depth in depths]
        ax.plot(depths, values, color=color, marker="o", linewidth=2.2, label=label)
        ax.axhline(100 * section["oracle_any_depth"]["accuracy"], color=color, linestyle="--", linewidth=1.4)
    ax.set_title("Fixed-depth agreement and oracle ceiling", loc="left", fontweight="bold")
    ax.set_xlabel("Forced loops")
    ax.set_ylabel("Agreement (%)")
    ax.set_xticks(depths)
    ax.set_ylim(0, 35)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.text(3.3, 21.4, "7B oracle 20.8%", color=BLUE, fontsize=9)
    ax.text(3.3, 32.2, "14B oracle 31.7%", color=GOLD, fontsize=9)

    # B: pre-loop probe.
    ax = axes[0, 1]
    labels = ["Any later benefit", "Loop-2 benefit"]
    primary = [
        probe["preloop"]["any_extra_depth"]["hidden_plus_structure"]["test_auroc"],
        probe["preloop"]["loop2_decision"]["hidden_plus_structure"]["test_auroc"],
    ]
    runtime_values = [
        runtime["preloop"]["any_extra_depth"]["hidden_plus_structure"]["test_auroc"],
        runtime["preloop"]["loop2_decision"]["hidden_plus_structure"]["test_auroc"],
    ]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, primary, width, color=BLUE, edgecolor=INK, linewidth=0.5, label="Frozen floor")
    ax.bar(x + width / 2, runtime_values, width, color="white", edgecolor=ORANGE, linewidth=1.5, label="L4 sensitivity")
    ax.axhline(0.60, color=INK, linestyle="--", linewidth=1.2, label="Viability AUROC 0.60")
    ax.set_title("Pre-loop deployable signal", loc="left", fontweight="bold")
    ax.set_ylabel("Held-out AUROC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.48, 0.66)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for index, value in enumerate(primary):
        ax.text(index - width / 2, value + 0.004, f"{value:.3f}", ha="center", fontsize=9)

    # C: sequential signal by observed loop.
    ax = axes[1, 0]
    loops = np.arange(1, 6)
    combined = [probe["sequential"]["per_loop"][str(loop)]["combined_test_auroc"] for loop in loops]
    scalar = [probe["sequential"]["per_loop"][str(loop)]["scalar_test_auroc"] for loop in loops]
    runtime_combined = [runtime["sequential"]["per_loop"][str(loop)]["combined_test_auroc"] for loop in loops]
    ax.plot(loops, combined, color=BLUE, marker="o", linewidth=2.2, label="State + scalars, frozen floor")
    ax.plot(loops, scalar, color=OLIVE, marker="s", linewidth=1.8, label="Scalars only, frozen floor")
    ax.plot(loops, runtime_combined, color=ORANGE, marker="o", linestyle="--", linewidth=1.5, label="State + scalars, L4")
    ax.axhline(0.5, color=GREY, linewidth=1)
    ax.set_title("Signal emerges after computation", loc="left", fontweight="bold")
    ax.set_xlabel("Decision after loop")
    ax.set_ylabel("Held-out AUROC")
    ax.set_xticks(loops)
    ax.set_ylim(0.48, 0.92)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")

    # D: actual sequential policy versus fixed depth.
    ax = axes[1, 1]
    fixed = probe["sequential"]["test_fixed_depth_accuracy"]
    ax.plot(
        depths,
        [100 * fixed[str(depth)] for depth in depths],
        color=INK,
        marker="o",
        linewidth=1.8,
        label="Fixed depth",
    )
    primary_frontier = probe["sequential"]["frontier"]
    runtime_frontier = runtime["sequential"]["frontier"]
    ax.plot(
        [point["test"]["mean_loops"] for point in primary_frontier],
        [100 * point["test"]["accuracy"] for point in primary_frontier],
        color=BLUE,
        marker="s",
        linewidth=2.2,
        label="Sequential router, frozen floor",
    )
    ax.plot(
        [point["test"]["mean_loops"] for point in runtime_frontier],
        [100 * point["test"]["accuracy"] for point in runtime_frontier],
        color=ORANGE,
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label="Sequential router, L4",
    )
    ax.axhline(100 * probe["sequential"]["test_oracle_any_depth_accuracy"], color=GOLD, linestyle=":", linewidth=2, label="Test oracle")
    ax.set_title("Predictive signal does not yet yield utility", loc="left", fontweight="bold")
    ax.set_xlabel("Mean executed loops")
    ax.set_ylabel("Agreement (%)")
    ax.set_xlim(0.9, 6.1)
    ax.set_ylim(0, 24)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
