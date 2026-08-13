"""Build the P3.4 sampled-depth preflight comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    arms = [
        ("main_seed_0", "Main seed 0", "main"),
        ("main_seed_1", "Main seed 1", "main"),
        ("slot_seed_0", "Slot seed 0", "slot"),
    ]
    colors = {"old": "#C84B31", "target": "#247BA0"}
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), sharex=True)
    for axis, (key, title, target_key) in zip(axes, arms):
        old = summary["arms"][key]["old_weight_preflight"]
        target = summary["target_shares"][target_key]
        names = [name for name in ("kl", "aim", "ce", "gate", "slot", "preserve") if name in old]
        y = np.arange(len(names))
        axis.barh(y + 0.18, [100 * old[name] for name in names], 0.34,
                  label="Old weights, actual depth mix", color=colors["old"])
        axis.barh(y - 0.18, [100 * target[name] for name in names], 0.34,
                  label="Registered target", color=colors["target"])
        axis.set_yticks(y, [name.upper() for name in names])
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.grid(axis="x", color="#D9D9D5", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_xlabel("Gradient share (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915),
                  ncol=2, frameon=False)
    figure.suptitle("P3.4 locked weights were calibrated on the wrong depth estimator",
                    y=0.985, fontsize=14, fontweight="bold")
    figure.text(0.5, 0.025,
                "Bars show the pre-optimizer read under the registered 0.1/0.2/0.3/0.4 depth mixture.",
                ha="center", fontsize=9, color="#4A4A46")
    figure.tight_layout(rect=(0, 0.07, 1, 0.82))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_prefix.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
