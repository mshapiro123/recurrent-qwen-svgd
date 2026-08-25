"""Plot the registered W2-prime Phase-D desk results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"seed 0": "#2878B5", "seed 1": "#D95F02"}
FEATURES = (("fs1_md", "FS-1"), ("fs2_prime", "FS-2 prime"))


def interval_error(item: dict[str, float], center_key: str) -> tuple[float, float]:
    center = float(item[center_key])
    return center - float(item["ci_low"]), float(item["ci_high"]) - center


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    parser.add_argument("--output_svg", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    width = 0.34
    positions = np.arange(len(FEATURES))

    for panel, metric, title, gate in (
        (axes[0, 0], "conditional", "A. Held-out conditional cosine", 0.30),
        (axes[0, 1], "risk", "B. Hemispheric relative risk reduction", 0.05),
    ):
        for seed_index, seed in enumerate((0, 1)):
            centers = []
            lower = []
            upper = []
            for feature, _label in FEATURES:
                result = payload["seeds"][str(seed)]["targets"]["l0a"]["feature_sets"][feature]
                if metric == "conditional":
                    item = result["fit"]["conditional_cosine"]["pooled"]
                    center_key = "mean"
                else:
                    item = result["hemispheric_relative_risk_reduction"]
                    center_key = "relative_risk_reduction"
                centers.append(float(item[center_key]))
                lo, hi = interval_error(item, center_key)
                lower.append(lo)
                upper.append(hi)
            offset = (seed_index - 0.5) * width
            panel.bar(
                positions + offset,
                centers,
                width,
                yerr=np.array([lower, upper]),
                capsize=4,
                color=COLORS[f"seed {seed}"],
                label=f"Seed {seed}",
                alpha=0.9,
            )
        panel.axhline(gate, color="#333333", linestyle="--", linewidth=1.2, label="Gate")
        panel.axhline(0, color="#777777", linewidth=0.7)
        panel.set_xticks(positions, [label for _name, label in FEATURES])
        panel.set_title(title, loc="left", fontweight="bold")
        panel.grid(axis="y", alpha=0.2)

    axes[0, 0].legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axes[0, 0].set_ylabel("Mean cosine")
    axes[0, 1].set_ylabel("Relative risk reduction")

    nuisance = axes[1, 0]
    for seed_index, seed in enumerate((0, 1)):
        values = [
            payload["seeds"][str(seed)]["targets"]["l0a"]["feature_sets"][feature]["fit"]
            ["rank8_nuisance_deflated_cosine"]["mean"]
            for feature, _label in FEATURES
        ]
        nuisance.bar(
            positions + (seed_index - 0.5) * width,
            values,
            width,
            color=COLORS[f"seed {seed}"],
            alpha=0.9,
        )
    nuisance.axhline(0, color="#777777", linewidth=0.7)
    nuisance.set_xticks(positions, [label for _name, label in FEATURES])
    nuisance.set_ylabel("Mean residual cosine")
    nuisance.set_title("C. Rank-8 nuisance-deflated cosine", loc="left", fontweight="bold")
    nuisance.grid(axis="y", alpha=0.2)

    site_ax = axes[1, 1]
    markers = {"a_to_b": "o", "b_to_a": "s"}
    linestyles = {0: "-", 1: "--"}
    sites = [8, 12, 16, 18]
    for seed in (0, 1):
        table = payload["seeds"][str(seed)]["targets"]["l0a"]["d4_site_screen"]
        for direction in ("a_to_b", "b_to_a"):
            values = [
                table[str(site)][direction]["incremental_relative_risk_reduction"]
                ["relative_risk_reduction"]
                for site in sites
            ]
            site_ax.plot(
                sites,
                values,
                marker=markers[direction],
                linestyle=linestyles[seed],
                color=COLORS[f"seed {seed}"],
                label=f"Seed {seed}, {direction.replace('_', ' ')}",
            )
    site_ax.axhline(0.05, color="#333333", linestyle=":", linewidth=1.2, label="5% reference")
    site_ax.axhline(0, color="#777777", linewidth=0.7)
    site_ax.set_xticks(sites)
    site_ax.set_xlabel("Layer/site")
    site_ax.set_ylabel("Incremental risk reduction")
    site_ax.set_title("D. Cross-hemisphere incremental value by site", loc="left", fontweight="bold")
    site_ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
    site_ax.grid(alpha=0.2)

    fig.suptitle(
        "Bicameral W2-prime desk gate: nominal cosine passes, hemisphere advantage does not replicate",
        fontsize=14,
        fontweight="bold",
    )
    for path in (args.output_png, args.output_svg):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180, bbox_inches="tight")
    fig.savefig(args.output_svg, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
