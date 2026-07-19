"""Locked scoring contracts for the Arm E adapter-parity battery."""

from __future__ import annotations

from typing import Any, Iterable


ARM_E_FINAL_SHA256 = "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839"
ARM_E_PRETRAINED_BASE_SHA256 = (
    "960f8bf265ba2850c9cdd60a388a00f8f366464babe0507521f010cb7f34971f"
)
ARM_E_RANK = 16
ARM_E_ALPHA = 32
ARM_E_FORWARD_ACTIVE_PARAMETERS = 6_007_425

E3A_STRONG_FLOOR = 0.70
E3A_PARTIAL_FLOOR = 0.40

E2_DIAGONAL_FLOOR = 0.93
E2_STRONG_CONTINUE_FLOOR = 0.85
E2_PARTIAL_CONTINUE_FLOOR = 0.50

E4_INVERSE_REQUIRED_CORRECT = 46
E4_INVERSE_TOTAL = 64
E4_SYNTHETIC_FLOOR = 0.93
E4_NATURAL_MAX_DROP = 0.03
E4_TIER1_BASELINE_CORRECT = 60
E4_TIER1_TOTAL = 64


def _accuracy(correct: int, total: int) -> float:
    if int(total) <= 0:
        raise ValueError("total must be positive")
    if not 0 <= int(correct) <= int(total):
        raise ValueError("correct must be between zero and total")
    return int(correct) / int(total)


def score_e3a_transfer(*, correct: int, total: int) -> dict[str, Any]:
    accuracy = _accuracy(correct, total)
    if accuracy >= E3A_STRONG_FLOOR:
        band = "strong"
    elif accuracy >= E3A_PARTIAL_FLOOR:
        band = "partial"
    else:
        band = "minimal"
    return {
        "accuracy": accuracy,
        "correct": int(correct),
        "total": int(total),
        "band": band,
        "thresholds": {
            "strong_floor": E3A_STRONG_FLOOR,
            "partial_floor": E3A_PARTIAL_FLOOR,
        },
    }


def score_e2_persistence(
    *,
    diagonal_correct: int,
    diagonal_total: int,
    continue_count: int,
    hold_count: int,
    above_total: int,
) -> dict[str, Any]:
    diagonal = _accuracy(diagonal_correct, diagonal_total)
    continuation = _accuracy(continue_count, above_total)
    hold = _accuracy(hold_count, above_total)
    if diagonal < E2_DIAGONAL_FLOOR:
        verdict = "failed"
    elif continuation >= E2_STRONG_CONTINUE_FLOOR:
        verdict = "strong"
    elif continuation >= E2_PARTIAL_CONTINUE_FLOOR:
        verdict = "partial"
    else:
        verdict = "failed"
    return {
        "verdict": verdict,
        "e4_authorized": diagonal >= E2_DIAGONAL_FLOOR,
        "active_diagonal": {
            "correct": int(diagonal_correct),
            "total": int(diagonal_total),
            "accuracy": diagonal,
            "floor": E2_DIAGONAL_FLOOR,
        },
        "above_diagonal": {
            "continue": int(continue_count),
            "hold": int(hold_count),
            "total": int(above_total),
            "continue_accuracy": continuation,
            "hold_accuracy": hold,
            "strong_continue_floor": E2_STRONG_CONTINUE_FLOOR,
            "partial_continue_floor": E2_PARTIAL_CONTINUE_FLOOR,
        },
    }


def validate_e4_source(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("kind") != "stage5_adapter_parity_e2":
        raise RuntimeError("E4 source is not an Arm E persistence receipt")
    if payload.get("source_checkpoint_sha256") != ARM_E_FINAL_SHA256:
        raise RuntimeError("E4 source does not descend from the locked Arm E checkpoint")
    if not bool((payload.get("decision") or {}).get("e4_authorized")):
        raise RuntimeError("E4 is not authorized because E2 did not preserve the diagonal")
    checkpoint_sha = str(payload.get("final_checkpoint_sha256") or "")
    if len(checkpoint_sha) != 64:
        raise RuntimeError("E4 source is missing the post-E2 checkpoint SHA")
    return {
        "status": "authorized",
        "arm_e_source_sha256": ARM_E_FINAL_SHA256,
        "post_e2_checkpoint_sha256": checkpoint_sha,
    }


def score_e4_retention(
    *,
    inverse_correct: int,
    inverse_total: int,
    synthetic_min_by_checkpoint: Iterable[float],
    natural_baseline_accuracy: float,
    natural_min_accuracy: float,
) -> dict[str, Any]:
    inverse_pass = (
        int(inverse_total) == E4_INVERSE_TOTAL
        and int(inverse_correct) >= E4_INVERSE_REQUIRED_CORRECT
    )
    synthetic_values = [float(value) for value in synthetic_min_by_checkpoint]
    if not synthetic_values:
        raise ValueError("at least one synthetic checkpoint readout is required")
    synthetic_pass = all(value >= E4_SYNTHETIC_FLOOR for value in synthetic_values)
    natural_floor = float(natural_baseline_accuracy) - E4_NATURAL_MAX_DROP
    natural_pass = float(natural_min_accuracy) >= natural_floor
    if inverse_pass and synthetic_pass and natural_pass:
        verdict = "wall_vanishes"
    elif inverse_pass:
        verdict = "wall_moves"
    else:
        verdict = "wall_holds"
    return {
        "verdict": verdict,
        "joint_pass": inverse_pass and synthetic_pass and natural_pass,
        "inverse": {
            "correct": int(inverse_correct),
            "total": int(inverse_total),
            "required_correct": E4_INVERSE_REQUIRED_CORRECT,
            "passed": inverse_pass,
        },
        "synthetic": {
            "minimums": synthetic_values,
            "floor": E4_SYNTHETIC_FLOOR,
            "passed_every_checkpoint": synthetic_pass,
        },
        "natural": {
            "baseline_accuracy": float(natural_baseline_accuracy),
            "minimum_accuracy": float(natural_min_accuracy),
            "floor": natural_floor,
            "max_drop": E4_NATURAL_MAX_DROP,
            "passed": natural_pass,
        },
    }


def score_e4_checkpoint_series(checkpoints: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in checkpoints]
    if not rows:
        raise ValueError("E4 requires at least one checkpoint readout")
    rows.sort(key=lambda row: int(row["step"]))
    for row in rows:
        row["inverse_pass"] = (
            int(row["inverse_total"]) == E4_INVERSE_TOTAL
            and int(row["inverse_correct"]) >= E4_INVERSE_REQUIRED_CORRECT
        )
        row["synthetic_pass"] = float(row["synthetic_min"]) >= E4_SYNTHETIC_FLOOR
        row["natural_floor"] = float(row["natural_baseline"]) - E4_NATURAL_MAX_DROP
        row["natural_pass"] = float(row["natural_accuracy"]) >= row["natural_floor"]
        row["tier1_accuracy"] = _accuracy(
            int(row["tier1_correct"]),
            int(row["tier1_total"]),
        )
        row["tier1_floor"] = (
            E4_TIER1_BASELINE_CORRECT / E4_TIER1_TOTAL
        ) - E4_NATURAL_MAX_DROP
        row["tier1_pass"] = (
            int(row["tier1_total"]) == E4_TIER1_TOTAL
            and row["tier1_accuracy"] >= row["tier1_floor"]
        )
        row["joint_pass"] = (
            row["inverse_pass"]
            and row["synthetic_pass"]
            and row["natural_pass"]
            and row["tier1_pass"]
        )
    final = rows[-1]
    all_retention_green = all(
        row["synthetic_pass"] and row["natural_pass"] and row["tier1_pass"]
        for row in rows
    )
    if final["joint_pass"] and all_retention_green:
        verdict = "wall_vanishes"
    elif any(row["joint_pass"] for row in rows):
        verdict = "wall_moves"
    else:
        verdict = "wall_holds"
    return {
        "verdict": verdict,
        "joint_pass_any_checkpoint": any(row["joint_pass"] for row in rows),
        "all_retention_checkpoints_green": all_retention_green,
        "final_joint_pass": bool(final["joint_pass"]),
        "checkpoints": rows,
    }


def derive_frontier(
    *,
    last_above_depth: int,
    last_above_accuracy: float,
    first_below_depth: int,
    first_below_accuracy: float,
    threshold: float,
    supported_depth: int = 8,
) -> dict[str, Any]:
    if int(first_below_depth) != int(last_above_depth) + 1:
        raise ValueError("frontier interpolation requires adjacent depths")
    if not float(last_above_accuracy) >= float(threshold) > float(first_below_accuracy):
        raise ValueError("accuracies do not bracket the threshold")
    span = float(last_above_accuracy) - float(first_below_accuracy)
    fraction = (float(last_above_accuracy) - float(threshold)) / span
    frontier = float(last_above_depth) + fraction
    return {
        "method": "linear_interpolation_between_adjacent_depths",
        "threshold": float(threshold),
        "last_above": {
            "depth": int(last_above_depth),
            "accuracy": float(last_above_accuracy),
        },
        "first_below": {
            "depth": int(first_below_depth),
            "accuracy": float(first_below_accuracy),
        },
        "supported_depth": int(supported_depth),
        "frontier": frontier,
        "frontier_to_support_ratio": frontier / int(supported_depth),
    }
