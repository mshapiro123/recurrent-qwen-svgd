"""Analyze the locked Paper Two Stage 2A T3 memory screen receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ARMS = ("t3a", "t3b", "shuffled", "random")
BATTERIES = ("arc_challenge", "arc_easy", "gsm8k", "mbpp", "mmlu", "tier1")
ARM_LABELS = {
    "t3a": "Concept memory",
    "t3b": "Literal n-gram",
    "shuffled": "Shuffled values",
    "random": "Random values",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sign_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def flatten(values: Iterable[Iterable[float]]) -> list[float]:
    return [item for group in values for item in group]


def paired_comparison(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("paired comparison item IDs differ")
    left_only = {
        item_id
        for item_id in left
        if left[item_id]["augmented_correct"] and not right[item_id]["augmented_correct"]
    }
    right_only = {
        item_id
        for item_id in left
        if right[item_id]["augmented_correct"] and not left[item_id]["augmented_correct"]
    }
    prediction_changes = sum(left[item_id]["prediction"] != right[item_id]["prediction"] for item_id in left)
    return {
        "left_correct": sum(row["augmented_correct"] for row in left.values()),
        "right_correct": sum(row["augmented_correct"] for row in right.values()),
        "left_only_correct": len(left_only),
        "right_only_correct": len(right_only),
        "left_minus_right_rows": len(left_only) - len(right_only),
        "exact_two_sided_p": exact_sign_p(len(left_only), len(right_only)),
        "prediction_changed_rows": prediction_changes,
    }


def event_sets(rows: dict[str, dict[str, Any]]) -> tuple[set[str], set[str]]:
    fixes = {
        item_id
        for item_id, row in rows.items()
        if not row["base_correct"] and row["augmented_correct"]
    }
    regressions = {
        item_id
        for item_id, row in rows.items()
        if row["base_correct"] and not row["augmented_correct"]
    }
    return fixes, regressions


def telemetry(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    scalar_fields = (
        "answer_token_margin_minimum",
        "memory_compatibility_gate_mean",
        "memory_retrieval_entropy_mean",
        "memory_retrieval_score_mean",
        "position_gate_mean",
        "realized_writeback_ratio_mean",
    )
    result = {}
    for field in scalar_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[f"mean_{field}"] = mean(values) if values else None
    result["maximum_realized_writeback_ratio"] = max(
        float(row.get("realized_writeback_ratio_max", 0.0)) for row in rows
    )
    return result


def build_figure(analysis: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = ["t3a_seed_0", "t3a_seed_1", "t3b_seed_0", "t3b_seed_1", "shuffled_seed_0", "random_seed_0"]
    labels = ["Concept\nS0", "Concept\nS1", "N-gram\nS0", "N-gram\nS1", "Shuffled\nS0", "Random\nS0"]
    colors = ["#2F6B5F", "#2F6B5F", "#C47A2C", "#C47A2C", "#7B8794", "#A84D46"]
    deltas = [analysis["conditions"][key]["pooled"]["delta_rows"] for key in conditions]
    fixes = [analysis["conditions"][key]["pooled"]["fixes"] for key in conditions]
    regressions = [analysis["conditions"][key]["pooled"]["regressions"] for key in conditions]

    fig = plt.figure(figsize=(13.2, 8.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.4, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(conditions))
    ax.bar(x, deltas, color=colors, width=0.68)
    ax.axhline(0, color="#30363D", linewidth=0.8)
    ax.axhspan(-3, 3, color="#DDE3E8", alpha=0.45, zorder=-1)
    ax.axhline(8, color="#8A2D2D", linestyle="--", linewidth=1.1)
    for index, value in enumerate(deltas):
        ax.text(index, value + (0.45 if value >= 0 else -0.75), f"{value:+d}", ha="center", va="center", fontsize=9)
    ax.scatter([0.5, 2.5], [5.5, 7.0], marker="D", s=46, color="#20262D", zorder=4, label="Two-seed mean")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Net correct rows vs frozen base (of 1,024)")
    ax.set_title("A. Registered DEV-only endpoints", loc="left", fontweight="bold")
    ax.text(5.45, 8.25, "T3a mean threshold", color="#8A2D2D", ha="right", va="bottom", fontsize=8)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    width = 0.36
    ax.bar(x - width / 2, fixes, width, color="#397D54", label="Fixes")
    ax.bar(x + width / 2, [-value for value in regressions], width, color="#B94E48", label="Regressions")
    ax.axhline(0, color="#30363D", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Changed rows")
    ax.set_title("B. Net effects are differences of larger flows", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, :])
    matrix = np.array(
        [
            [analysis["conditions"][condition]["by_battery"][battery]["delta_rows"] for battery in BATTERIES]
            for condition in conditions
        ]
    )
    limit = max(1, int(np.abs(matrix).max()))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            ax.text(column_index, row_index, f"{value:+d}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(BATTERIES)), [value.replace("_", " ").upper() for value in BATTERIES])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("C. Battery-level net correct-row changes", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.02)
    colorbar.set_label("Net rows")

    fig.suptitle("Stage 2A memory screen: positive content arms, flat controls, threshold miss", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.06,
        0.01,
        "DEV only; same 1,024 rows in every condition. CONFIRM and EVAL-E remained sealed. Seed repeats are not treated as independent rows.",
        fontsize=9,
        color="#4B5563",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-base", type=Path, required=True)
    args = parser.parse_args()

    aggregate = read_json(args.receipt_root / "receipts" / "summary.json")
    conditions: dict[str, Any] = {}
    rows_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        seeds = (0, 1) if arm in {"t3a", "t3b"} else (0,)
        for seed in seeds:
            key = f"{arm}_seed_{seed}"
            evaluation = args.receipt_root / "private" / key / "evaluation"
            summary = read_json(evaluation / "summary.json")
            rows = read_jsonl(evaluation / "dev_rows.jsonl")
            rows_by_condition[key] = {str(row["item_id"]): row for row in rows}
            conditions[key] = {
                "arm": arm,
                "seed": seed,
                "pooled": summary["pooled"],
                "by_battery": summary["by_battery"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "dev_rows_sha256": summary["dev_rows_sha256"],
                "retrieval_slots_observed": summary["retrieval_slots_observed"],
                "telemetry": telemetry(rows),
            }

    pairwise: dict[str, Any] = {}
    for seed in (0, 1):
        pairwise[f"t3a_vs_t3b_seed_{seed}"] = paired_comparison(
            rows_by_condition[f"t3a_seed_{seed}"], rows_by_condition[f"t3b_seed_{seed}"]
        )
    for left in ("t3a", "t3b"):
        for right in ("shuffled", "random"):
            pairwise[f"{left}_vs_{right}_seed_0"] = paired_comparison(
                rows_by_condition[f"{left}_seed_0"], rows_by_condition[f"{right}_seed_0"]
            )

    replication = {}
    for arm in ("t3a", "t3b"):
        left = rows_by_condition[f"{arm}_seed_0"]
        right = rows_by_condition[f"{arm}_seed_1"]
        left_fixes, left_regressions = event_sets(left)
        right_fixes, right_regressions = event_sets(right)
        replication[arm] = {
            "fix_intersection": len(left_fixes & right_fixes),
            "fix_union": len(left_fixes | right_fixes),
            "fix_jaccard": jaccard(left_fixes, right_fixes),
            "regression_intersection": len(left_regressions & right_regressions),
            "regression_union": len(left_regressions | right_regressions),
            "regression_jaccard": jaccard(left_regressions, right_regressions),
            "prediction_changed_rows_between_seeds": sum(
                left[item_id]["prediction"] != right[item_id]["prediction"] for item_id in left
            ),
        }

    analysis = {
        "kind": "paper2_stage2a_t3_screen_analysis_v1",
        "registered_verdict": aggregate["verdict"],
        "registered_decision": {
            "t3a_delta_rows_by_seed": aggregate["t3a_delta_rows_by_seed"],
            "t3a_mean_delta_rows": aggregate["t3a_mean_delta_rows"],
            "t3a_threshold_rows": 8,
            "t3a_both_seeds_positive": aggregate["t3a_both_seeds_positive"],
            "t3a_threshold_pass": aggregate["t3a_minimum_net_row_gain_pass"],
            "controls_inside_equivalence_band": aggregate["controls_inside_equivalence_band"],
            "confirm_scored": aggregate["confirm_scored"],
            "eval_e_scored": aggregate["eval_e_scored"],
        },
        "descriptive_t3b": {
            "delta_rows_by_seed": aggregate["t3b_delta_rows_by_seed"],
            "mean_delta_rows": aggregate["t3b_mean_delta_rows"],
        },
        "conditions": conditions,
        "pairwise": pairwise,
        "replication": replication,
        "source_summary_sha256": sha256(args.receipt_root / "receipts" / "summary.json"),
        "interpretive_boundaries": [
            "DEV-only repeated panel; no confirmed capability claim",
            "T3b is descriptive and does not replace the registered T3a gate",
            "seed repeats on the same rows are not pooled as 2,048 independent observations",
            "no claim that teacher-derived concept memory beats literal n-gram memory",
            "CONFIRM and EVAL-E remained sealed",
        ],
    }
    output = args.output_dir / "analysis_summary.json"
    write_json(output, analysis)
    build_figure(analysis, args.figure_base)
    print(json.dumps({"analysis": str(output), "registered_verdict": aggregate["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
