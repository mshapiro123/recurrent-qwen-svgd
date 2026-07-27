"""Build the Paper Two D0 result handoff figure from landed receipts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage5/stage5_paper2_d0_20260726"
OUTPUT = ROOT / "docs/figures/paper2_d0_result_handoff_20260727"

INK = "#18212b"
BLUE = "#276FBF"
GOLD = "#C7922D"
ORANGE = "#D66A2C"
OLIVE = "#6E7F32"
GREY = "#8A96A3"
LIGHT = "#DDE3E8"


def load(relative_path: str) -> dict:
    return json.loads((RUN / relative_path).read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return 100.0 * value


def main() -> None:
    natural = load("eval/natural_summary.json")
    teacher_shift = load("eval/teacher_shift_signature.json")

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
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.86, wspace=0.20, hspace=0.32)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "D0 adaptive-depth pilot result",
        fontsize=18,
        fontweight="bold",
        x=0.03,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.03,
        0.945,
        "199,529 evaluation tokens; final-step EMA; teacher-forced next-token agreement",
        fontsize=10.5,
        color="#52606d",
        ha="left",
    )

    # A: overall agreement before and after adaptive execution.
    ax = axes[0, 0]
    overall = [
        pct(natural["initial_loop1"]["by_stratum"]["pooled"]["all"]["plain_all"]["accuracy"]),
        pct(natural["trained_loop1"]["by_stratum"]["pooled"]["all"]["loop1_all"]["accuracy"]),
        pct(natural["trained_adaptive"]["by_stratum"]["pooled"]["all"]["adaptive_all"]["accuracy"]),
    ]
    labels = ["Plain drafter", "Trained, loop 1", "Trained, adaptive"]
    colors = [GREY, BLUE, ORANGE]
    bars = ax.bar(labels, overall, color=colors, edgecolor=INK, linewidth=0.5)
    ax.set_title("Adaptive execution lowers total agreement", loc="left", fontweight="bold")
    ax.set_ylabel("Teacher agreement (%)")
    ax.set_ylim(68.5, 73.5)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    for bar, value in zip(bars, overall, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.10, f"{value:.2f}%", ha="center", fontsize=9)

    # B: the trained substrate has forced-depth capacity that routing does not capture.
    ax = axes[0, 1]
    pooled = natural["trained_adaptive"]["by_stratum"]["pooled"]["all"]
    rejected = [
        pct(pooled["loop1_rejected"]["accuracy"]),
        pct(pooled["adaptive_rejected"]["accuracy"]),
        pct(pooled["loop4_rejected"]["accuracy"]),
    ]
    labels = ["Loop 1", "Self-halted", "Forced loop 4"]
    colors = [BLUE, ORANGE, GOLD]
    bars = ax.bar(labels, rejected, color=colors, edgecolor=INK, linewidth=0.5)
    ax.set_title("Self-halting captures little forced-depth response", loc="left", fontweight="bold")
    ax.set_ylabel("Agreement on baseline rejections (%)")
    ax.set_ylim(0, 14)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    for bar, value in zip(bars, rejected, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}%", ha="center", fontsize=9)

    # C: adaptive speculative-decoding simulation is worse at every gamma.
    ax = axes[1, 0]
    gammas = [2, 4, 8]
    simulation = natural["speculative_decoding_simulation"]
    plain = [pct(simulation[str(gamma)]["plain"]["acceptance_rate"]) for gamma in gammas]
    adaptive = [pct(simulation[str(gamma)]["adaptive"]["acceptance_rate"]) for gamma in gammas]
    x = np.arange(len(gammas))
    width = 0.36
    ax.bar(x - width / 2, plain, width, color=BLUE, edgecolor=INK, linewidth=0.5, label="Plain")
    ax.bar(x + width / 2, adaptive, width, color=ORANGE, edgecolor=INK, linewidth=0.5, label="Adaptive")
    ax.set_title("Speculative-decoding acceptance also declines", loc="left", fontweight="bold")
    ax.set_xlabel("Draft window gamma")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_xticks(x, [str(gamma) for gamma in gammas])
    ax.set_ylim(20, 66)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    for index, (plain_value, adaptive_value) in enumerate(zip(plain, adaptive, strict=True)):
        ax.text(index - width / 2, plain_value + 0.7, f"{plain_value:.1f}", ha="center", fontsize=8.5)
        ax.text(index + width / 2, adaptive_value + 0.7, f"{adaptive_value:.1f}", ha="center", fontsize=8.5)

    # D: both teachers retain a loop-2 peak after training.
    ax = axes[1, 1]
    depths = np.arange(1, 7)
    for key, color, label in (
        ("teacher_7b_own_rejections", BLUE, "7B own rejections"),
        ("teacher_14b_own_rejections", GOLD, "14B own rejections"),
    ):
        curve = [pct(value) for value in teacher_shift[key]["agreement_curve"]]
        ax.plot(depths, curve, marker="o", linewidth=2.2, color=color, label=label)
    ax.set_title("Teacher-specific depth demand still peaks at loop 2", loc="left", fontweight="bold")
    ax.set_xlabel("Forced loops")
    ax.set_ylabel("Agreement on each teacher's rejections (%)")
    ax.set_xticks(depths)
    ax.set_ylim(0, 18)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")

    fig.text(
        0.03,
        0.025,
        "Guardrails passed: natural accepted-position drop 0.69 points (limit 1.0); T1 mechanism 1,005/1,024 with exact control.",
        fontsize=9.5,
        color="#52606d",
        ha="left",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    svg_path = OUTPUT.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.with_suffix(".png"))
    print(svg_path)


if __name__ == "__main__":
    main()
