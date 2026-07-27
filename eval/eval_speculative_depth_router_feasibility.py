"""Read-only D0 router-feasibility analysis on the locked calibration partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROUTER_AUDIT_SEED = 20260727
INTERIOR_BUDGET_FRACTIONS = (0.25, 0.50, 0.75)


def classify_depth_pair(loop1_correct: bool, loop2_correct: bool) -> str:
    if loop2_correct and not loop1_correct:
        return "recovered_at_2"
    if loop1_correct and not loop2_correct:
        return "harmed_at_2"
    if loop1_correct:
        return "both_correct"
    return "both_wrong"


def oracle_depth_profile(matches: Sequence[bool]) -> dict[str, Any]:
    if not matches:
        raise ValueError("oracle depth profile requires at least one loop")
    correct_depths = [index + 1 for index, matched in enumerate(matches) if bool(matched)]
    return {
        "any_correct": bool(correct_depths),
        "first_correct_depth": correct_depths[0] if correct_depths else None,
        "correct_depths": correct_depths,
        "selected_depth": correct_depths[0] if correct_depths else 1,
    }


def _teacher_demand_population(
    rows: Sequence[dict[str, Any]], *, teacher_key: str
) -> dict[str, Any]:
    if not rows:
        return {
            "positions": 0,
            "recoverable": 0,
            "never_correct": 0,
            "first_correct_depth_counts": {},
            "median_first_correct_depth_recoverable": None,
            "agreement_curve": [],
            "aggregate_peak_depth": None,
        }
    depths = {len(row["predictions"]) for row in rows}
    if len(depths) != 1 or next(iter(depths)) < 1:
        raise ValueError("teacher-demand rows require a common nonempty depth axis")
    max_depth = next(iter(depths))
    first_depths: list[int] = []
    agreement_counts = [0] * max_depth
    for row in rows:
        target = int(row[teacher_key])
        matches = [int(token) == target for token in row["predictions"]]
        for index, matched in enumerate(matches):
            agreement_counts[index] += int(matched)
        first = next((index + 1 for index, matched in enumerate(matches) if matched), None)
        if first is not None:
            first_depths.append(first)
    counts = {str(depth): first_depths.count(depth) for depth in range(1, max_depth + 1)}
    peak = max(range(max_depth), key=lambda index: agreement_counts[index]) + 1
    return {
        "positions": len(rows),
        "recoverable": len(first_depths),
        "recoverable_rate": len(first_depths) / len(rows),
        "never_correct": len(rows) - len(first_depths),
        "first_correct_depth_counts": counts,
        "median_first_correct_depth_recoverable": (
            float(statistics.median(first_depths)) if first_depths else None
        ),
        "agreement_curve": [count / len(rows) for count in agreement_counts],
        "aggregate_peak_depth": peak,
    }


def summarize_teacher_demand(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare teachers on clean, teacher-specific loop-1 rejection populations."""

    prepared = list(rows)
    if not prepared:
        raise ValueError("teacher-demand analysis requires rows")
    seven_rejected = [
        row for row in prepared if int(row["predictions"][0]) != int(row["teacher_7b"])
    ]
    fourteen_rejected = [
        row for row in prepared if int(row["predictions"][0]) != int(row["teacher_14b"])
    ]
    fourteen_endorses = sum(
        int(row["predictions"][0]) == int(row["teacher_14b"]) for row in seven_rejected
    )
    return {
        "teacher_7b_own_rejections": _teacher_demand_population(
            seven_rejected, teacher_key="teacher_7b"
        ),
        "teacher_14b_own_rejections": _teacher_demand_population(
            fourteen_rejected, teacher_key="teacher_14b"
        ),
        "teacher_overlap_on_7b_rejections": {
            "seven_rejections": len(seven_rejected),
            "fourteen_endorses_loop1": fourteen_endorses,
            "share": fourteen_endorses / len(seven_rejected) if seven_rejected else None,
            "scope": "teacher disagreement within the agreement target; not semantic correctness",
        },
        "teacher_14b_crossover_on_7b_rejections": _teacher_demand_population(
            seven_rejected, teacher_key="teacher_14b"
        ),
        "population_rule": (
            "each primary demand curve uses that teacher's own loop-1 rejection set; "
            "14B on 7B rejections is descriptive overlap only"
        ),
    }


def deterministic_group_split(
    rows: Iterable[dict[str, Any]], *, seed: int = ROUTER_AUDIT_SEED
) -> dict[tuple[int, str], str]:
    """Create deterministic 70/15/15 splits within each stratum by source row."""
    groups: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        groups[str(row["stratum"])].add(int(row["row_index"]))
    mapping: dict[tuple[int, str], str] = {}
    for stratum, row_indices in groups.items():
        ranked = sorted(
            row_indices,
            key=lambda value: hashlib.sha256(
                f"{seed}:{stratum}:{value}".encode("utf-8")
            ).digest(),
        )
        count = len(ranked)
        train_end = max(1, int(math.floor(0.70 * count)))
        validation_end = max(train_end + 1, int(math.floor(0.85 * count)))
        validation_end = min(validation_end, max(train_end + 1, count - 1))
        for index, row_index in enumerate(ranked):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            mapping[(row_index, stratum)] = split
    return mapping


def budget_policy_curve(
    rows: Sequence[dict[str, Any]],
    *,
    score_field: str,
    fractions: Sequence[float] = INTERIOR_BUDGET_FRACTIONS,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("budget policy curve requires rows")
    count = len(rows)
    base_correct = sum(int(bool(row["loop1_correct"])) for row in rows)
    deltas = [
        int(bool(row["loop2_correct"])) - int(bool(row["loop1_correct"])) for row in rows
    ]
    ranked_model = sorted(range(count), key=lambda index: float(rows[index][score_field]), reverse=True)
    ranked_oracle = sorted(range(count), key=lambda index: deltas[index], reverse=True)
    curve: list[dict[str, Any]] = []
    for fraction in fractions:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("budget fractions must be in [0, 1]")
        selected = min(count, max(0, int(round(float(fraction) * count))))
        model_correct = base_correct + sum(deltas[index] for index in ranked_model[:selected])
        oracle_correct = base_correct + sum(deltas[index] for index in ranked_oracle[:selected])
        random_expected = base_correct + (selected / count) * sum(deltas)
        curve.append(
            {
                "fraction": float(fraction),
                "selected_for_loop2": selected,
                "mean_loops": 1.0 + selected / count,
                "model_correct": model_correct,
                "model_accuracy": model_correct / count,
                "random_expected_correct": random_expected,
                "random_expected_accuracy": random_expected / count,
                "oracle_correct": oracle_correct,
                "oracle_accuracy": oracle_correct / count,
                "uplift_vs_random": (model_correct - random_expected) / count,
            }
        )
    return curve


def router_verdict(*, auroc: float, budget_points: Sequence[dict[str, Any]]) -> str:
    passing = [
        point
        for point in budget_points
        if float(point.get("uplift_vs_random", 0.0)) >= 0.01
        and float(point.get("bootstrap_low", float("-inf"))) > 0.0
    ]
    strong = [
        point
        for point in budget_points
        if float(point.get("uplift_vs_random", 0.0)) >= 0.02
        and float(point.get("bootstrap_low", float("-inf"))) > 0.0
    ]
    if float(auroc) >= 0.70 and len(strong) >= 2:
        return "strong_deployable_signal"
    if float(auroc) >= 0.60 and len(passing) >= 2:
        return "viable_deployable_signal"
    return "no_deployable_signal"


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Tie-aware AUROC via average ranks."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("AUROC requires aligned nonempty inputs")
    positives = sum(bool(value) for value in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(range(len(scores)), key=lambda index: float(scores[index]))
    positive_rank_sum = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        value = float(scores[ordered[cursor]])
        while end < len(ordered) and float(scores[ordered[end]]) == value:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(bool(labels[index]) for index in ordered[cursor:end])
        cursor = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def oracle_compute_frontier(
    matches: Sequence[Sequence[bool]], *, budgets: Sequence[float]
) -> list[dict[str, Any]]:
    if not matches:
        raise ValueError("oracle frontier requires rows")
    total = len(matches)
    base_correct = sum(bool(row[0]) for row in matches)
    upgrade_costs = sorted(
        profile["first_correct_depth"] - 1
        for row in matches
        if not bool(row[0])
        for profile in [oracle_depth_profile(row)]
        if profile["first_correct_depth"] is not None
    )
    points: list[dict[str, Any]] = []
    for budget in budgets:
        if not 1.0 <= float(budget) <= len(matches[0]):
            raise ValueError("oracle budget must be within the measured loop range")
        available = int(math.floor((float(budget) - 1.0) * total + 1e-12))
        used = 0
        selected = 0
        for cost in upgrade_costs:
            if used + int(cost) > available:
                break
            used += int(cost)
            selected += 1
        correct = base_correct + selected
        points.append(
            {
                "budget_mean_loops": float(budget),
                "realized_mean_loops": 1.0 + used / total,
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "upgraded_positions": selected,
            }
        )
    return points


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _signal_auc(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    labels = [bool(row[label]) for row in rows]
    fields = {
        "teacher_to_plain_drafter_kl": "kl",
        "drafter_token_rank_under_teacher": "rank",
        "rejection_run_length": "run_length",
        "teacher_entropy": "teacher_entropy",
        "negative_drafter_logprob_under_teacher": "negative_drafter_logprob_under_teacher",
    }
    result: dict[str, Any] = {}
    for name, field in fields.items():
        raw = binary_auroc([float(row[field]) for row in rows], labels)
        result[name] = {
            "raw_auroc": raw,
            "oriented_auroc": max(raw, 1.0 - raw) if raw is not None else None,
            "direction": "higher" if raw is not None and raw >= 0.5 else "lower",
        }
    return result


def summarize_oracle_router(rows: Sequence[dict[str, Any]], *, teacher: str) -> dict[str, Any]:
    match_key = f"matches_teacher_{teacher}"
    prepared: list[dict[str, Any]] = []
    for source in rows:
        matches = [bool(value) for value in source[match_key]]
        profile = oracle_depth_profile(matches)
        predictions = [int(value) for value in source["predictions"]]
        prepared.append(
            {
                **source,
                "matches": matches,
                "loop2_benefit": not matches[0] and matches[1],
                "any_extra_benefit": not matches[0] and any(matches[1:]),
                "oracle_profile": profile,
                "predictions": predictions,
            }
        )
    matches = [row["matches"] for row in prepared]
    total = len(prepared)
    fixed = [sum(int(row[depth]) for row in matches) for depth in range(6)]
    first_depth_counts = {str(depth): 0 for depth in range(1, 7)}
    never = 0
    for row in prepared:
        depth = row["oracle_profile"]["first_correct_depth"]
        if depth is None:
            never += 1
        else:
            first_depth_counts[str(depth)] += 1
    transitions: dict[str, Any] = {}
    for depth in range(2, 7):
        recovered = sum(not row[0] and row[depth - 1] for row in matches)
        harmed = sum(row[0] and not row[depth - 1] for row in matches)
        transitions[f"1_to_{depth}"] = {
            "recovered": recovered,
            "harmed": harmed,
            "net": recovered - harmed,
            "recovery_rate": recovered / total,
            "harm_rate": harmed / total,
        }
    post_loop_observables: dict[str, Any] = {}
    for loop in range(1, 6):
        eligible = [row for row in prepared if not row["matches"][loop - 1]]
        benefit = [any(row["matches"][loop:]) for row in eligible]
        if loop == 1:
            post_loop_observables[str(loop)] = {
                "currently_wrong": len(eligible),
                "later_recoverable": sum(benefit),
                "later_recoverable_rate": _safe_rate(sum(benefit), len(eligible)),
            }
            continue
        changed = [
            row["predictions"][loop - 1] != row["predictions"][loop - 2] for row in eligible
        ]
        changed_total = sum(changed)
        stable_total = len(changed) - changed_total
        changed_benefit = sum(flag and target for flag, target in zip(changed, benefit, strict=True))
        stable_benefit = sum((not flag) and target for flag, target in zip(changed, benefit, strict=True))
        post_loop_observables[str(loop)] = {
            "currently_wrong": len(eligible),
            "later_recoverable": sum(benefit),
            "prediction_changed": {
                "total": changed_total,
                "later_recoverable": changed_benefit,
                "rate": _safe_rate(changed_benefit, changed_total),
            },
            "prediction_stable": {
                "total": stable_total,
                "later_recoverable": stable_benefit,
                "rate": _safe_rate(stable_benefit, stable_total),
            },
        }
    any_correct = total - never
    selected_loop_sum = sum(
        int(row["oracle_profile"]["selected_depth"]) for row in prepared
    )
    best_fixed_depth = max(range(6), key=lambda index: fixed[index]) + 1
    return {
        "teacher": teacher,
        "positions": total,
        "fixed_depth": {
            str(depth): {"correct": fixed[depth - 1], "accuracy": fixed[depth - 1] / total}
            for depth in range(1, 7)
        },
        "best_fixed_depth": best_fixed_depth,
        "best_fixed_accuracy": fixed[best_fixed_depth - 1] / total,
        "oracle_any_depth": {
            "correct": any_correct,
            "accuracy": any_correct / total,
            "mean_loops_first_correct_else_one": selected_loop_sum / total,
            "uplift_over_best_fixed": (any_correct - max(fixed)) / total,
        },
        "first_correct_depth_counts": first_depth_counts,
        "never_correct_by_depth6": never,
        "transitions_from_loop1": transitions,
        "oracle_compute_frontier": oracle_compute_frontier(
            matches, budgets=(1.0, 1.25, 1.50, 1.75, 2.0, 3.0, 4.0, 5.0, 6.0)
        ),
        "teacher_signal_predictability": {
            "loop2_benefit": _signal_auc(prepared, "loop2_benefit"),
            "any_extra_benefit": _signal_auc(prepared, "any_extra_benefit"),
            "teacher_dependent_not_deployable": True,
        },
        "post_loop_token_observables": post_loop_observables,
    }


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor_private_rows", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--expected_private_rows_sha256", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.floor_private_rows).read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    summary = {
        "kind": "paper2_d0_oracle_router_audit",
        "status": "complete",
        "scope": "locked_calibration_7b_rejected_positions",
        "floor_private_rows_sha256": args.expected_private_rows_sha256,
        "no_training": True,
        "evaluation_partition_touched": False,
        "teacher_7b": summarize_oracle_router(rows, teacher="7b"),
        "teacher_14b_sensitivity": summarize_oracle_router(rows, teacher="14b"),
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
