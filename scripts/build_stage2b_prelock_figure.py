"""Build the Stage 2B pre-campaign calibration and power figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("docs/figures/stage2b_prelock_calibration_power_20260818")


def main() -> None:
    weights = {
        "Seed 0": [0.2805809114, 0.6206603068, 0.0987587818],
        "Seed 1": [0.2236681677, 0.6617089091, 0.1146229232],
    }
    labels = ["CE", "Forward KL", "Monotonicity"]
    colors = ["#2E7D32", "#3465A4", "#D97706"]
    discordance = np.array([0.10, 0.20, 0.30])
    power = np.array([0.6650581629, 0.4026811736, 0.3285655850])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), constrained_layout=True)

    left = axes[0]
    y = np.arange(len(weights))
    offset = np.zeros(len(weights))
    for index, (label, color) in enumerate(zip(labels, colors)):
        values = np.array([row[index] for row in weights.values()])
        left.barh(y, values, left=offset, color=color, height=0.5, label=label)
        for row, value, start in zip(y, values, offset):
            left.text(start + value / 2, row, f"{value:.1%}", ha="center", va="center", color="white")
        offset += values
    left.set_yticks(y, list(weights))
    left.set_xlim(0, 1)
    left.set_xlabel("Registered independent-gradient share")
    left.set_title("A. Seed-specific calibrated loss weights", loc="left", fontweight="bold")
    left.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    left.grid(axis="x", color="#D8D8D8", linewidth=0.7)
    left.set_axisbelow(True)

    right = axes[1]
    right.plot(discordance * 100, power * 100, color="#7A3E9D", marker="o", linewidth=2.2)
    right.axhline(80, color="#B42318", linestyle="--", linewidth=1.2, label="80% planning target")
    for x, value in zip(discordance * 100, power * 100):
        right.annotate(f"{value:.1f}%", (x, value), xytext=(0, 8), textcoords="offset points", ha="center")
    right.set_xlim(7, 33)
    right.set_ylim(0, 100)
    right.set_xticks([10, 20, 30])
    right.set_xlabel("Assumed paired discordance")
    right.set_ylabel("Modeled one-sided power")
    right.set_title("B. DEV-2 power for +30 / 2,048 rows", loc="left", fontweight="bold")
    right.grid(color="#D8D8D8", linewidth=0.7)
    right.set_axisbelow(True)
    right.legend(frameon=False, loc="lower left")

    figure.suptitle("Stage 2B pre-campaign lock diagnostics", fontsize=14, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
