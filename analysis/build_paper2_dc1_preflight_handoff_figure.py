"""Build the DC1 preflight handoff figure from landed receipts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage5/stage5_paper2_dc1_preflight_20260729"
OUTPUT = ROOT / "docs/figures/paper2_dc1_preflight_handoff_20260730"

INK = "#18212b"
BLUE = "#276FBF"
GOLD = "#C7922D"
ORANGE = "#D66A2C"
OLIVE = "#6E7F32"
RED = "#B54747"
GREY = "#8A96A3"
LIGHT = "#DDE3E8"


def load(path: str) -> dict:
    return json.loads((RUN / path).read_text(encoding="utf-8"))


def main() -> None:
    dc1 = load("dc1_p/summary.json")
    numerics = load("rg4_rg11/summary.json")
    scale_rows = dc1["scale_interpolation"]["rows"]
    raw_multiple = (
        dc1["scale_interpolation"]["raw_hidden_rms"]
        / dc1["scale_interpolation"]["embedding_rms"]
    )
    scale_x = np.array([1, 3, 10, 30, 100, 300, raw_multiple], dtype=float)
    scale_labels = ["1x", "3x", "10x", "30x", "100x", "300x", "raw\n603x"]

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
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.86, wspace=0.18, hspace=0.32)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "DC1 composite-interface preflight",
        fontsize=18,
        fontweight="bold",
        x=0.03,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.03,
        0.945,
        "Fresh 500k-token DEV-C; 50,108 teacher-scored positions; no training; EVAL-C untouched",
        fontsize=10.5,
        color="#52606d",
        ha="left",
    )

    # A: feedback scale versus teacher agreement.
    ax = axes[0, 0]
    after_accuracy = np.array(
        [100 * row["transition"]["after_accuracy"] for row in scale_rows]
    )
    baseline = 100 * scale_rows[0]["transition"]["before_accuracy"]
    ax.plot(scale_x, after_accuracy, color=BLUE, marker="o", linewidth=2.2)
    ax.axhline(baseline, color=INK, linestyle="--", linewidth=1.4, label=f"k=0 baseline {baseline:.1f}%")
    ax.set_xscale("log")
    ax.set_xticks(scale_x, scale_labels)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Fed-state RMS relative to embedding RMS")
    ax.set_ylabel("Teacher agreement after one append (%)")
    ax.set_title("Raw feedback is least harmful, not safe", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.annotate(
        "raw 51.9%",
        xy=(scale_x[-1], after_accuracy[-1]),
        xytext=(230, 61),
        arrowprops={"arrowstyle": "-", "color": BLUE},
        color=BLUE,
        fontsize=9,
    )

    # B: help/harm asymmetry across scale.
    ax = axes[0, 1]
    helps = np.array([row["transition"]["helps"] for row in scale_rows])
    hurts = np.array([row["transition"]["hurts"] for row in scale_rows])
    ax.plot(scale_x, helps, color=OLIVE, marker="o", linewidth=2.0, label="Helps")
    ax.plot(scale_x, hurts, color=RED, marker="o", linewidth=2.0, label="Hurts")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(scale_x, scale_labels)
    ax.set_xlabel("Fed-state RMS relative to embedding RMS")
    ax.set_ylabel("Positions (log scale)")
    ax.set_title("Every untrained scale has more harms than helps", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, loc="center right")
    ax.text(scale_x[-1], hurts[-1] * 1.15, "6.88 harms/help", ha="right", color=RED, fontsize=9)

    # C: attention distribution from the appended slot.
    ax = axes[1, 0]
    groups = ["prelude", "recurrent", "coda"]
    general = [100 * dc1["slot_attention_profile"]["general"]["by_layer_group"][g]["prefix_mass"] for g in groups]
    code = [100 * dc1["slot_attention_profile"]["code"]["by_layer_group"][g]["prefix_mass"] for g in groups]
    x = np.arange(len(groups))
    width = 0.34
    ax.bar(x - width / 2, general, width, color=BLUE, edgecolor=INK, linewidth=0.5, label="General")
    ax.bar(x + width / 2, code, width, color=GOLD, edgecolor=INK, linewidth=0.5, label="Code")
    ax.set_xticks(x, [name.title() for name in groups])
    ax.set_ylim(75, 95)
    ax.set_ylabel("Attention mass on original prefix (%)")
    ax.set_title("The appended slot continues to read the prefix", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    # D: RG-11 precision policy.
    ax = axes[1, 1]
    policies = [
        ("fp32 master +\nbf16 autocast", "fp32_master_bf16_autocast", ORANGE),
        ("full bf16", "full_bf16", GREY),
        ("full fp32", "full_fp32", BLUE),
    ]
    loops = np.array([1, 2, 3])
    for label, key, color in policies:
        values = [
            numerics["rg11_precision_policies"][key]["by_horizontal_steps"][str(k)]["minimum_cosine"]
            for k in loops
        ]
        ax.plot(loops, values, marker="o", linewidth=2.0, color=color, label=label)
    ax.axhline(0.99, color=INK, linestyle="--", linewidth=1.3, label="Required 0.99")
    ax.set_xticks(loops)
    ax.set_ylim(0.95, 1.003)
    ax.set_xlabel("Horizontal append steps k")
    ax.set_ylabel("Minimum gradient cosine to fp32")
    ax.set_title("Only full fp32 satisfies RG-11", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    ax.text(0.02, 0.94, "RG-4: PASS", transform=ax.transAxes, color=OLIVE, fontweight="bold", fontsize=9)

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
