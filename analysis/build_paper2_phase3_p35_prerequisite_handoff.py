"""Build the public P3.5 prerequisite summary and diagnostic figure."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_sign_pvalue(fixes: int, regressions: int) -> float:
    discordant = fixes + regressions
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(fixes, regressions) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]], *, bootstrap_seed: int) -> dict[str, Any]:
    fresh = np.asarray([bool(row["fresh_correct"]) for row in rows], dtype=np.int8)
    carried = np.asarray([bool(row["carried_correct"]) for row in rows], dtype=np.int8)
    delta = carried - fresh
    fixes = int(((fresh == 0) & (carried == 1)).sum())
    regressions = int(((fresh == 1) & (carried == 0)).sum())
    changed = np.asarray([bool(row["later_token_changed"]) for row in rows], dtype=bool)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = delta[rng.integers(0, len(rows), size=(20_000, len(rows)))].mean(axis=1)
    return {
        "rows": len(rows),
        "fresh_correct": int(fresh.sum()),
        "fresh_accuracy": float(fresh.mean()),
        "carried_correct": int(carried.sum()),
        "carried_accuracy": float(carried.mean()),
        "carried_minus_fresh_rows": int(delta.sum()),
        "carried_minus_fresh_points": float(100.0 * delta.mean()),
        "paired_bootstrap_95ci_points": [
            float(100.0 * np.quantile(bootstrap, 0.025)),
            float(100.0 * np.quantile(bootstrap, 0.975)),
        ],
        "fixes": fixes,
        "regressions": regressions,
        "discordant_rows": fixes + regressions,
        "exact_paired_sign_pvalue_two_sided": exact_sign_pvalue(fixes, regressions),
        "later_token_changed_rows": int(changed.sum()),
        "later_token_changed_rate": float(changed.mean()),
        "changed_rows_with_correctness_transition": int(
            (changed & (fresh != carried)).sum()
        ),
        "nontrivial_reanchors": int(sum(int(row["on_the_fly_reanchors"]) for row in rows)),
    }


def build_figure(summary: dict[str, Any], output_prefix: Path) -> None:
    labels = ["GSM8K", "MBPP", "Pooled"]
    keys = ["gsm8k", "mbpp", "pooled"]
    fresh = [100 * summary[key]["fresh_accuracy"] for key in keys]
    carried = [100 * summary[key]["carried_accuracy"] for key in keys]
    changed = [100 * summary[key]["later_token_changed_rate"] for key in keys]
    x = np.arange(len(labels))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    width = 0.34
    axes[0].bar(x - width / 2, fresh, width, label="Fresh scratch", color="#176B87")
    axes[0].bar(x + width / 2, carried, width, label="Carried scratch", color="#C65D43")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("A. Task accuracy")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].spines[["top", "right"]].set_visible(False)
    for index, key in enumerate(keys):
        axes[0].text(
            index,
            max(fresh[index], carried[index]) + 1.2,
            f"n={summary[key]['rows']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].bar(x, changed, 0.52, color="#5A6B73")
    axes[1].set_ylabel("Rows with changed continuation (%)")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("B. Carry changed later tokens")
    axes[1].spines[["top", "right"]].set_visible(False)
    for index, key in enumerate(keys):
        row = summary[key]
        axes[1].text(
            index,
            changed[index] + 0.8,
            f"{row['later_token_changed_rows']}/{row['rows']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("P3.5 no-training persistence probe (seed 0, DEV only)", fontsize=12)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(re.sub(r"[ \t]+$", "", svg, flags=re.MULTILINE), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--figure_prefix", type=Path, required=True)
    args = parser.parse_args()

    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    rows = read_jsonl(args.rows)
    by_battery = {
        battery: [row for row in rows if row["battery"] == battery]
        for battery in ("gsm8k", "mbpp")
    }
    summary = {
        "kind": "paper2_phase3_p35_prerequisite_public_summary_v1",
        "status": "complete_no_training",
        "serving_oracle_cache": bindings["serving_oracle_cache"],
        "persistence": {
            "gsm8k": summarize(by_battery["gsm8k"], bootstrap_seed=20260815),
            "mbpp": summarize(by_battery["mbpp"], bootstrap_seed=20260816),
            "pooled": summarize(rows, bootstrap_seed=20260817),
            "scope": "seed 0, DEV only, fresh scratch versus controlled cross-token carry",
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_figure(summary["persistence"], args.figure_prefix)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
