"""Build the registered P3.5 DEV analysis and figures from durable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


LABELS = ("arm_s_seed_0", "arm_s_seed_1", "arm_r_seed_0")
PRIMARY_CONDITION = "step_4400_ema_ceiling_0.02"
ENDPOINT_CONDITIONS = (
    "step_4400_raw_ceiling_0.02",
    "step_4400_ema_ceiling_0.02",
    "step_4400_raw_ceiling_0.08",
    "step_4400_ema_ceiling_0.08",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def exact_paired_pvalue(left_only: int, right_only: int) -> float:
    """Two-sided exact sign/McNemar p-value over discordant paired rows."""

    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_binary(left: Iterable[bool], right: Iterable[bool]) -> dict[str, Any]:
    pairs = list(zip(left, right, strict=True))
    left_only = sum(bool(a) and not bool(b) for a, b in pairs)
    right_only = sum(not bool(a) and bool(b) for a, b in pairs)
    return {
        "rows": len(pairs),
        "left_only": left_only,
        "right_only": right_only,
        "net_right_minus_left": right_only - left_only,
        "two_sided_exact_p": exact_paired_pvalue(left_only, right_only),
    }


def matched_recall_precision(
    reference_rows: list[Mapping[str, Any]],
    candidate_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Read candidate precision at the closest attainable reference recall.

    Positive and negative audit rows are identified by their registered
    ``population`` label. The candidate threshold is selected only from its
    positive rows to match the reference's fixed-threshold recall; all negative
    rows then contribute false positives at that threshold.
    """

    reference = {str(row["record_id"]): row for row in reference_rows}
    candidate = {str(row["record_id"]): row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise RuntimeError("Arm R and Arm S audit row identities differ")
    ref_positive = [row for row in reference.values() if row["population"] == "positive"]
    cand_positive = [candidate[str(row["record_id"])] for row in ref_positive]
    cand_negative = [row for row in candidate.values() if row["population"] == "negative"]
    ref_true = sum(float(row["gate_unclamped"]) >= 0.5 for row in ref_positive)
    target_recall = ref_true / len(ref_positive)
    ranked = sorted((float(row["gate_unclamped"]) for row in cand_positive), reverse=True)
    selected = min(len(ranked), max(0, round(target_recall * len(ranked))))
    threshold = math.inf if selected == 0 else ranked[selected - 1]
    true_positive = sum(float(row["gate_unclamped"]) >= threshold for row in cand_positive)
    false_positive = sum(float(row["gate_unclamped"]) >= threshold for row in cand_negative)
    precision = true_positive / max(1, true_positive + false_positive)
    return {
        "reference_threshold": 0.5,
        "reference_recall": target_recall,
        "candidate_threshold": threshold,
        "candidate_recall": true_positive / len(cand_positive),
        "candidate_precision": precision,
        "candidate_true_positive": true_positive,
        "candidate_false_positive": false_positive,
        "positive_rows": len(cand_positive),
        "negative_rows": len(cand_negative),
    }


def _history_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    task = entry["task"]
    audit = entry["audit"]
    outcomes = task["score_preserving_telemetry"]["by_battery_and_outcome"]
    shares = entry.get("share_read_observe_only") or {}
    rows = int(task["rows"])
    base = round(float(task["base_accuracy"]) * rows)
    augmented = round(float(task["augmented_accuracy"]) * rows)
    return {
        "look": int(entry["look"]),
        "step": int(entry["step"]),
        "learning_rate": float(entry["learning_rate"]),
        "base_correct": base,
        "augmented_correct": augmented,
        "net_rows": augmented - base,
        "fixes": int(outcomes["outcome:fix"]["rows"]),
        "regressions": int(outcomes["outcome:regression"]["rows"]),
        "mean_answer_token_margin": float(
            task["answer_token_margins"]["pooled"]["mean_answer_token_margin"]
        ),
        "mean_row_minimum_margin": float(
            task["answer_token_margins"]["pooled"]["mean_row_minimum_margin"]
        ),
        "pi_dir": float(audit["pi_dir"]["point"]),
        "pi_dep": float(audit["pi_dep"]["point"]),
        "collateral_chi": float(audit["collateral_chi"]),
        "gate_precision_fixed_threshold": float(audit["gate"]["precision"]),
        "gate_recall_fixed_threshold": float(audit["gate"]["recall"]),
        "share_classification": shares.get("classification"),
        "failed_share_contracts": shares.get("failed_contracts", []),
        "guardrail": entry["guardrail"],
    }


def _locate(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def analyze(root: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for label in LABELS:
        run_path = _locate(root, f"receipts/train/{label}/summary.json")
        wave_path = _locate(root, f"receipts/train/{label}/wave_summary.json")
        score_path = _locate(root, f"private/score_bundle/{label}/summary.json")
        run = read_json(run_path)
        wave = read_json(wave_path)
        score = read_json(score_path)
        if run.get("status") != "complete" or score.get("status") != "complete_dev_only_no_training":
            raise RuntimeError(f"P3.5 arm incomplete: {label}")
        if run.get("confirm_scored") or run.get("eval_e_scored"):
            raise RuntimeError(f"sealed partition touched: {label}")
        if score.get("registered_primary") != PRIMARY_CONDITION:
            raise RuntimeError(f"registered primary changed: {label}")
        endpoint = score["conditions"][PRIMARY_CONDITION]["paired"]
        arms[label] = {
            "run_summary": {"path": str(run_path), "sha256": sha256_file(run_path)},
            "wave_summary": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
            "score_bundle": {"path": str(score_path), "sha256": sha256_file(score_path)},
            "history": [_history_row(entry) for entry in run["history"]],
            "endpoint": endpoint,
            "endpoint_conditions": {
                condition: score["conditions"][condition]["paired"]
                for condition in ENDPOINT_CONDITIONS
            },
            "adjacent_churn": score["ema_primary_adjacent_churn"],
            "raw_vs_ema": score["final_raw_vs_ema"],
            "confirm_scored": False,
            "eval_e_scored": False,
        }

    s0 = arms["arm_s_seed_0"]
    s1 = arms["arm_s_seed_1"]
    r0 = arms["arm_r_seed_0"]
    s0_net = int(s0["endpoint"]["net_rows"])
    s1_net = int(s1["endpoint"]["net_rows"])
    r0_net = int(r0["endpoint"]["net_rows"])
    mean_net = 0.5 * (s0_net + s1_net)
    if mean_net >= 10:
        branch = "A_p36_drafting"
    elif mean_net >= 8:
        branch = "B_margin_tiebreaker_and_lever_queue"
    else:
        branch = "C_effect_size_to_stage2a"

    primary_rows: dict[str, list[dict[str, Any]]] = {}
    audit_rows: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        primary_rows[label] = read_jsonl(
            _locate(root, f"private/score_bundle/{label}/{PRIMARY_CONDITION}.jsonl")
        )
        audit_rows[label] = read_jsonl(
            _locate(root, f"private/{label}/audit_rows_step_4400.jsonl")
        )
    s0_by_id = {str(row["item_id"]): row for row in primary_rows["arm_s_seed_0"]}
    r0_by_id = {str(row["item_id"]): row for row in primary_rows["arm_r_seed_0"]}
    if set(s0_by_id) != set(r0_by_id):
        raise RuntimeError("Arm R and Arm S task row identities differ")
    ordered = sorted(s0_by_id)
    paired_task = paired_binary(
        (bool(s0_by_id[key]["augmented_correct"]) for key in ordered),
        (bool(r0_by_id[key]["augmented_correct"]) for key in ordered),
    )
    matched_gate = matched_recall_precision(
        audit_rows["arm_s_seed_0"], audit_rows["arm_r_seed_0"]
    )
    s0_endpoint_history = s0["history"][-1]
    r0_endpoint_history = r0["history"][-1]
    arm_r = {
        "task_noninferior_on_registered_dev_net": r0_net >= s0_net,
        "task_net_difference_rows": r0_net - s0_net,
        "paired_task": paired_task,
        "gate_precision_at_s0_matched_recall": matched_gate,
        "fixed_threshold_gate": {
            "arm_s_seed_0_precision": s0_endpoint_history["gate_precision_fixed_threshold"],
            "arm_s_seed_0_recall": s0_endpoint_history["gate_recall_fixed_threshold"],
            "arm_r_seed_0_precision": r0_endpoint_history["gate_precision_fixed_threshold"],
            "arm_r_seed_0_recall": r0_endpoint_history["gate_recall_fixed_threshold"],
        },
        "pi_dep_difference": r0_endpoint_history["pi_dep"] - s0_endpoint_history["pi_dep"],
        "row_minimum_margin_difference": (
            float(r0["endpoint"]["margins"]["pooled"]["mean_row_minimum_margin"])
            - float(s0["endpoint"]["margins"]["pooled"]["mean_row_minimum_margin"])
        ),
        "promotion_requires_componentwise_strategy_read": True,
    }
    return {
        "kind": "paper2_phase3_p35_results_analysis_v1",
        "status": "complete_dev_only",
        "registered_primary": "EMA step 4400 at gate ceiling 0.02",
        "arms": arms,
        "stabilized_spine": {
            "seed_0_endpoint_net_rows": s0_net,
            "seed_1_endpoint_net_rows": s1_net,
            "mean_endpoint_net_rows": mean_net,
            "seed_0_benchmark_rows": 8,
            "seed_1_benchmark_rows": 9,
            "seed_0_benchmark_met": s0_net >= 8,
            "seed_1_benchmark_met": s1_net >= 9,
            "registered_branch": branch,
        },
        "arm_r_comparison": arm_r,
        "sealed_partitions": {"confirm_scored": False, "eval_e_scored": False},
        "interpretation_boundary": (
            "All task reads are exploratory DEV. Arm R gate components are reported "
            "separately because the lock did not define an arbitrary composite score."
        ),
    }


def plot(summary: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "arm_s_seed_0": "#146B8C",
        "arm_s_seed_1": "#D36B32",
        "arm_r_seed_0": "#5C7A29",
    }
    names = {
        "arm_s_seed_0": "Arm S seed 0",
        "arm_s_seed_1": "Arm S seed 1",
        "arm_r_seed_0": "Arm R seed 0",
    }
    markers = {
        "arm_s_seed_0": "o",
        "arm_s_seed_1": "s",
        "arm_r_seed_0": "^",
    }
    line_styles = {
        "arm_s_seed_0": "-",
        "arm_s_seed_1": "-",
        "arm_r_seed_0": "--",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for label in LABELS:
        history = summary["arms"][label]["history"]
        steps = [row["step"] for row in history]
        style = {
            "marker": markers[label],
            "linestyle": line_styles[label],
            "color": colors[label],
            "label": names[label],
            "linewidth": 1.8,
            "markersize": 5.5,
        }
        axes[0].plot(steps, [row["net_rows"] for row in history], **style)
        axes[1].plot(steps, [100 * row["pi_dir"] for row in history], **style)
        axes[2].plot(steps, [row["mean_row_minimum_margin"] for row in history], **style)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set(title="DEV task effect", ylabel="Net correct rows (of 1,024)", xlabel="Training step")
    axes[1].set(title="Causal direction capture", ylabel="pi_dir (%)", xlabel="Training step")
    axes[2].set(title="Continuous margin", ylabel="Mean row-minimum margin", xlabel="Training step")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("P3.5 stabilized landing and reader comparison", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"paper2_p35_results_20260815.{suffix}", dpi=220)
    plt.close(fig)

    condition_names = {
        "step_4400_raw_ceiling_0.02": "Raw / 0.02",
        "step_4400_ema_ceiling_0.02": "EMA / 0.02\n(primary)",
        "step_4400_raw_ceiling_0.08": "Raw / 0.08",
        "step_4400_ema_ceiling_0.08": "EMA / 0.08",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    width = 0.24
    x = list(range(len(ENDPOINT_CONDITIONS)))
    for index, label in enumerate(LABELS):
        endpoint_conditions = summary["arms"][label]["endpoint_conditions"]
        axes[0].bar(
            [value + (index - 1) * width for value in x],
            [endpoint_conditions[condition]["net_rows"] for condition in ENDPOINT_CONDITIONS],
            width=width,
            color=colors[label],
            label=names[label],
        )
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_xticks(x, [condition_names[condition] for condition in ENDPOINT_CONDITIONS])
    axes[0].set(title="Endpoint sensitivity matrix", ylabel="Net correct rows (of 1,024)")
    axes[0].legend(frameon=False, fontsize=8)

    primary = [summary["arms"][label]["endpoint"] for label in LABELS]
    y = list(range(len(LABELS)))
    axes[1].barh(
        [value + width / 2 for value in y],
        [entry["fixes"] for entry in primary],
        height=width,
        color="#2F7D62",
        label="Fixes",
    )
    axes[1].barh(
        [value - width / 2 for value in y],
        [entry["regressions"] for entry in primary],
        height=width,
        color="#B85248",
        label="Regressions",
    )
    axes[1].set_yticks(y, [names[label] for label in LABELS])
    axes[1].set(title="Registered primary paired changes", xlabel="Rows")
    axes[1].legend(
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(labelsize=8)
    fig.suptitle("P3.5 endpoint diagnostics", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"paper2_p35_endpoint_matrix_20260815.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.artifact_root)
    output = args.output_dir / "analysis_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(summary, args.output_dir)
    print(json.dumps(summary["stabilized_spine"], indent=2, sort_keys=True))
    print(json.dumps(summary["arm_r_comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
