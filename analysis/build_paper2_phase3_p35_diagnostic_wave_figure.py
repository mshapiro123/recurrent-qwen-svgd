"""Render the P3.5 amplitude and depth-discrimination decision surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build(amplitude: dict[str, Any], depth: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {0: "#0072B2", 1: "#D55E00"}
    ceilings = [0.02, 0.05, 0.08, 0.11]
    for seed in (0, 1):
        values = [
            amplitude["conditions"][f"seed_{seed}_ceiling_{str(value).replace('.', 'p')}"]["net_rows"]
            for value in ceilings
        ]
        axes[0].plot(ceilings, values, marker="o", color=colors[seed], label=f"seed {seed}")
    selected = amplitude.get("selected_ceiling_under_preregistered_rule")
    if selected is not None:
        axes[0].axvline(float(selected), color="#555555", linestyle=":", linewidth=1.4)
    axes[0].axhline(0, color="#999999", linewidth=0.8)
    axes[0].set(title="A. Amplitude surface", xlabel="Gate ceiling", ylabel="Net correct rows (of 1,024)")
    axes[0].legend(frameon=False)

    ks = list(range(1, 7))
    for seed in (0, 1):
        registered = [depth["cells"][f"seed_{seed}_k_{k}"]["pooled"]["correct"] for k in range(1, 5)]
        extension = [registered[-1], *[
            depth["cells"][f"seed_{seed}_k_{k}"]["pooled"]["correct"] for k in (5, 6)
        ]]
        axes[1].plot(range(1, 5), registered, marker="o", color=colors[seed], label=f"seed {seed}")
        axes[1].plot(range(4, 7), extension, marker="o", linestyle="--", color=colors[seed], alpha=0.75)
    axes[1].axvline(4, color="#555555", linestyle=":", linewidth=1.4)
    axes[1].text(4.08, axes[1].get_ylim()[0], "clamped extension", va="bottom", fontsize=8, color="#555555")
    axes[1].set(title="B. Accuracy versus recurrent depth", xlabel="K", ylabel="Correct rows (of 1,024)", xticks=ks)
    axes[1].legend(frameon=False)

    transitions = ["2-1", "3-2", "4-3", "5-4", "6-5"]
    width = 0.34
    positions = list(range(len(transitions)))
    for offset, seed in ((-width / 2, 0), (width / 2, 1)):
        values = [
            depth["marginal_improvement"][f"seed_{seed}_k_{to}_minus_k_{to-1}"]["pooled"]["net_rows"]
            for to in range(2, 7)
        ]
        axes[2].bar([position + offset for position in positions], values, width=width, color=colors[seed], label=f"seed {seed}")
    axes[2].axhline(0, color="#999999", linewidth=0.8)
    axes[2].axvline(2.5, color="#555555", linestyle=":", linewidth=1.4)
    axes[2].set(title="C. Marginal value of another loop", xlabel="K transition", ylabel="Paired net rows", xticks=positions, xticklabels=transitions)
    axes[2].legend(frameon=False)

    fig.suptitle("P3.5 no-training diagnostic wave", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(read_json(args.amplitude), read_json(args.depth), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
