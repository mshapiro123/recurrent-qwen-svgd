"""Locked scoring policy for the final N-24 synthetic-depth rung."""

from __future__ import annotations

from typing import Any


N24_SYMBOLS = 24
N24_MAX_EVAL_DEPTH = 22
N24_SUPPORT_DEPTH = 12
N24_ROWS_PER_DEPTH = 128
N24_TRAIN_ROWS_PER_DEPTH = 256
N24_TOTAL_STEPS = 6000
N24_CHECKPOINTS = (2000, 4000, 6000)

N24_STRONG_SCALING_MIN_CORRECT = 91
N24_ASYMPTOTE_BREAK_DEPTH15_MIN_CORRECT = 91
N24_CHANCE_REJECTION_MIN_CORRECT = 10
N24_ADJACENT_EXTENSION_MIN_CORRECT = 10
N24_NONREGRESSION_FLOORS = {
    **{str(depth): 0.93 for depth in range(1, 9)},
    **{str(depth): 0.85 for depth in range(9, 13)},
}


def diagonal_counts(active_summary: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    matrix = active_summary.get("active_matrix", {})
    out: dict[str, dict[str, float | int]] = {}
    for depth, by_loop in matrix.items():
        cell = by_loop.get(depth, {}) if isinstance(by_loop, dict) else {}
        out[str(depth)] = {
            "correct": int(cell.get("correct", 0)),
            "total": int(cell.get("total", 0)),
            "accuracy": float(cell.get("accuracy", 0.0)),
        }
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def locked_gate_summary(*, rows_per_depth: int = N24_ROWS_PER_DEPTH) -> dict[str, Any]:
    return {
        "n_symbols": N24_SYMBOLS,
        "rows_per_depth": int(rows_per_depth),
        "support_depth": N24_SUPPORT_DEPTH,
        "max_eval_depth": N24_MAX_EVAL_DEPTH,
        "strong_depths": [16, 17],
        "strong_scaling_min_correct": N24_STRONG_SCALING_MIN_CORRECT,
        "scaling_depth16_min_correct": N24_STRONG_SCALING_MIN_CORRECT,
        "law_consistent_depth17_chance_min_correct": N24_CHANCE_REJECTION_MIN_CORRECT,
        "asymptote_break_depth15_min_correct": N24_ASYMPTOTE_BREAK_DEPTH15_MIN_CORRECT,
        "adjacent_extension_depth13_min_correct": N24_ADJACENT_EXTENSION_MIN_CORRECT,
        "chance_floor": 1.0 / N24_SYMBOLS,
        "chance_rejection_min_correct": N24_CHANCE_REJECTION_MIN_CORRECT,
        "nonregression_floors": dict(N24_NONREGRESSION_FLOORS),
        "planned_steps": N24_TOTAL_STEPS,
        "planned_checkpoints": list(N24_CHECKPOINTS),
    }


def score_n24_rung(active_summary: dict[str, Any], *, rows_per_depth: int = N24_ROWS_PER_DEPTH) -> dict[str, Any]:
    counts = diagonal_counts(active_summary)
    nonregression = {
        depth: {
            "accuracy": float(counts.get(depth, {}).get("accuracy", 0.0)),
            "floor": floor,
            "pass": float(counts.get(depth, {}).get("accuracy", 0.0)) >= floor,
        }
        for depth, floor in N24_NONREGRESSION_FLOORS.items()
    }
    selected_correct = {
        str(depth): int(counts.get(str(depth), {}).get("correct", 0))
        for depth in range(13, N24_MAX_EVAL_DEPTH + 1)
    }
    strong = (
        selected_correct["16"] >= N24_STRONG_SCALING_MIN_CORRECT
        and selected_correct["17"] >= N24_STRONG_SCALING_MIN_CORRECT
    )
    scaling = selected_correct["16"] >= N24_STRONG_SCALING_MIN_CORRECT
    law_floor = selected_correct["17"] >= N24_CHANCE_REJECTION_MIN_CORRECT
    asymptote_broken = selected_correct["15"] < N24_ASYMPTOTE_BREAK_DEPTH15_MIN_CORRECT
    adjacent = selected_correct["13"] >= N24_ADJACENT_EXTENSION_MIN_CORRECT
    long_tail_above_chance = all(
        selected_correct[str(depth)] >= N24_CHANCE_REJECTION_MIN_CORRECT
        for depth in range(17, N24_MAX_EVAL_DEPTH + 1)
    )
    if strong:
        verdict = "strong_four_point_law"
    elif scaling:
        verdict = "scaling_depth16_only"
    elif asymptote_broken:
        verdict = "law_broken_or_ceiling_at_14"
    elif adjacent:
        verdict = "adjacent_extension_only"
    else:
        verdict = "no_extension"
    return {
        "diagonal_counts": counts,
        "locked_thresholds": locked_gate_summary(rows_per_depth=rows_per_depth),
        "nonregression": nonregression,
        "nonregression_pass": all(item["pass"] for item in nonregression.values()),
        "selected_correct": selected_correct,
        "strong_scaling_pass": strong,
        "scaling_pass": scaling,
        "law_consistent_floor_pass": law_floor,
        "asymptote_broken": asymptote_broken,
        "adjacent_extension_pass": adjacent,
        "long_tail_above_chance": long_tail_above_chance,
        "overall_pass": all(item["pass"] for item in nonregression.values()) and scaling,
        "verdict": verdict,
    }


def tier1_canary_verdict(*, accuracy_delta: float | None, ppl_relative_delta: float | None) -> dict[str, Any]:
    """Return the standing canary policy verdict for a checkpoint interval."""

    accuracy_red = accuracy_delta is not None and accuracy_delta < -0.03
    ppl_red = ppl_relative_delta is not None and ppl_relative_delta > 0.05
    if accuracy_red or ppl_red:
        status = "red_hard_stop"
    elif (accuracy_delta is not None and accuracy_delta < -0.015) or (
        ppl_relative_delta is not None and ppl_relative_delta > 0.025
    ):
        status = "yellow_review"
    else:
        status = "green_continue"
    return {
        "status": status,
        "accuracy_delta": accuracy_delta,
        "ppl_relative_delta": ppl_relative_delta,
        "accuracy_hard_stop_margin": -0.03,
        "ppl_hard_stop_relative": 0.05,
    }
