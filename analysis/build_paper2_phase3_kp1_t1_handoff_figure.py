"""Render the KP-1 target audit and amended T1 state-geometry results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build(summary: dict[str, Any], gap_rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    grouped: dict[str, list[int]] = defaultdict(list)
    for row in gap_rows:
        grouped[str(row["battery"])].append(int(row["gold_first_token_id"]))
    batteries = ["arc_challenge", "arc_easy", "mmlu", "gsm8k", "mbpp", "tier1"]
    labels = ["ARC-C", "ARC-E", "MMLU", "GSM8K", "MBPP", "Tier-1"]
    dominance = []
    unique = []
    for battery in batteries:
        counts = Counter(grouped[battery])
        dominance.append(max(counts.values()) / sum(counts.values()))
        unique.append(len(counts))
    colors = ["#0072B2" if value >= 4 else "#D55E00" for value in unique]
    positions = np.arange(len(labels))
    axes[0].barh(positions, dominance, color=colors)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.08)
    axes[0].set_xlabel("Dominant target-token share")
    axes[0].set_title("A. Locked target degeneracy")
    for index, (share, count) in enumerate(zip(dominance, unique)):
        axes[0].text(
            min(share + 0.02, 1.01),
            index,
            f"{share:.0%}; {count} unique",
            va="center",
            fontsize=8,
        )

    colors = {0: "#0072B2", 1: "#D55E00"}
    for seed in (0, 1):
        item = summary["t1"]["stability"][f"p34_seed_{seed}_step_4000"]
        values = [item["core_depth_to_k4"][f"k{k}"]["mean"] for k in range(1, 5)]
        axes[1].plot(range(1, 5), values, marker="o", color=colors[seed], label=f"seed {seed}")
    axes[1].set(
        title="B. Core state approaches K=4",
        xlabel="Forced recurrent depth K",
        ylabel="Mean cosine to K=4 core state",
        xticks=[1, 2, 3, 4],
        ylim=(0.68, 1.01),
    )
    axes[1].legend(frameon=False)

    order = [
        "p34_seed_0_step_4000",
        "p35_seed_0_ema_step_4400",
        "p34_seed_1_step_4000",
        "p35_seed_1_ema_step_4400",
    ]
    short = ["P3.4 s0", "P3.5 s0", "P3.4 s1", "P3.5 s1"]
    matrix = np.eye(4)
    cross = summary["t1"]["cross_checkpoint_fingerprints"]
    for left_index, left in enumerate(order):
        for right_index in range(left_index + 1, len(order)):
            right = order[right_index]
            key = f"{left}__vs__{right}"
            if key not in cross:
                key = f"{right}__vs__{left}"
            value = cross[key]["core_44_cell_fingerprint"]["mean"]
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    image = axes[2].imshow(matrix, vmin=-0.1, vmax=1.0, cmap="RdYlBu")
    axes[2].set_xticks(range(4), short, rotation=35, ha="right")
    axes[2].set_yticks(range(4), short)
    axes[2].set_title("C. Cross-checkpoint core cosine")
    for row in range(4):
        for column in range(4):
            axes[2].text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < 0.25 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[2], shrink=0.82)

    fig.suptitle("KP-1 target audit and amended T1 state extraction", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gap-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(read_json(args.summary), read_jsonl(args.gap_rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
