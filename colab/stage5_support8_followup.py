"""Shared gates for support-8 synthetic-depth follow-up experiments."""

from __future__ import annotations

from typing import Any


DEFAULT_SUPPORT8_SOURCE_SUMMARY = (
    "outputs/stage5/stage5_depth_support_ladder8_20260705_204923/summary.json"
)
DEFAULT_FROZEN_DEPTH14_EVAL_ID = "stage5_synthetic_depth_frozen_eval_v2_depth14"
SUPPORT8_RUN_ID = "stage5_depth_support_ladder8_20260705_204923"
SUPPORT8_TRAIN_SEED = "20260705"
SUPPORT8_TRAIN_MAX_DEPTH = 8
SUPPORT8_ROWS_PER_DEPTH = 256
SUPPORT8_FROZEN_ROWS_PER_DEPTH = 128

EXPECTED_SUPPORT8_SELECTED_CORRECT = {
    "9": 109,
    "10": 85,
    "11": 64,
    "12": 48,
    "13": 26,
    "14": 20,
}

STRONG_SCALING_MIN_CORRECT = 91
SOFT_DEPTH10_REVIVAL_MIN_CORRECT = 95
SOFT_DEPTH11_REVIVAL_MIN_CORRECT = 74


def selected_correct_as_strings(score: dict[str, Any]) -> dict[str, int]:
    return {str(depth): int(value) for depth, value in (score.get("selected_correct") or {}).items()}


def validate_support8_source_summary(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Fail fast if the dose/probe source is not the locked support-8 ladder run."""

    errors: list[str] = []
    if payload.get("run_id") != SUPPORT8_RUN_ID:
        errors.append(f"run_id={payload.get('run_id')!r}")
    if payload.get("kind") != "stage5_depth_support_ladder":
        errors.append(f"kind={payload.get('kind')!r}")
    if payload.get("status") != "finished_with_frozen_eval":
        errors.append(f"status={payload.get('status')!r}")
    if int(payload.get("train_max_depth", -1)) != SUPPORT8_TRAIN_MAX_DEPTH:
        errors.append(f"train_max_depth={payload.get('train_max_depth')!r}")
    if int(payload.get("rows_per_depth", -1)) != SUPPORT8_ROWS_PER_DEPTH:
        errors.append(f"rows_per_depth={payload.get('rows_per_depth')!r}")
    if not payload.get("final_checkpoint"):
        errors.append("missing final_checkpoint")
    if not payload.get("final_checkpoint_drive_backup"):
        errors.append("missing final_checkpoint_drive_backup")

    frozen = payload.get("frozen_eval_set") or {}
    if frozen.get("run_id") != DEFAULT_FROZEN_DEPTH14_EVAL_ID:
        errors.append(f"frozen_eval_set.run_id={frozen.get('run_id')!r}")
    if not frozen.get("base_route_identity_check"):
        errors.append("missing frozen_eval_set.base_route_identity_check")
    if not frozen.get("test_chain_mcq"):
        errors.append("missing frozen_eval_set.test_chain_mcq")

    score = payload.get("ladder_score") or {}
    selected = selected_correct_as_strings(score)
    if selected != EXPECTED_SUPPORT8_SELECTED_CORRECT:
        errors.append(f"selected_correct={selected!r}")

    if errors:
        raise RuntimeError(
            "Support-8 follow-up source is not the locked ladder run "
            f"{SUPPORT8_RUN_ID}: {path}; " + "; ".join(errors)
        )

    return {
        "source_summary": path,
        "source_run_id": payload.get("run_id"),
        "selected_correct": selected,
        "final_checkpoint": payload.get("final_checkpoint"),
        "final_checkpoint_drive_backup": payload.get("final_checkpoint_drive_backup"),
        "frozen_eval_set": frozen.get("run_id"),
        "frozen_test_chain_mcq": frozen.get("test_chain_mcq"),
    }


def active_diagonal(payload: dict[str, Any]) -> dict[str, float]:
    frozen = payload.get("frozen_active_eval") or {}
    return {str(depth): float(value) for depth, value in (frozen.get("active_diagonal") or {}).items()}


def first_depth_below(diagonal: dict[str, float], threshold: float) -> int | None:
    for depth in sorted((int(key) for key in diagonal), key=int):
        if float(diagonal[str(depth)]) < float(threshold):
            return int(depth)
    return None


def decay_alignment(active_diag: dict[str, float]) -> dict[str, Any]:
    strong_bar_accuracy = STRONG_SCALING_MIN_CORRECT / SUPPORT8_FROZEN_ROWS_PER_DEPTH
    return {
        "registered_prediction": "If drift binds, envelope exit should align with decay onset near depths 9/10.",
        "active_diagonal": dict(active_diag),
        "first_depth_below_0_90": first_depth_below(active_diag, 0.90),
        "first_depth_below_strong_scaling_bar": first_depth_below(active_diag, strong_bar_accuracy),
        "strong_scaling_bar_accuracy": strong_bar_accuracy,
    }


def score_dose_arm(ladder_score: dict[str, Any]) -> dict[str, Any]:
    selected = selected_correct_as_strings(ladder_score)
    depth10 = int(selected.get("10", 0))
    depth11 = int(selected.get("11", 0))
    depth10_delta = depth10 - EXPECTED_SUPPORT8_SELECTED_CORRECT["10"]
    depth11_delta = depth11 - EXPECTED_SUPPORT8_SELECTED_CORRECT["11"]
    locked_bar_revived = depth10 >= STRONG_SCALING_MIN_CORRECT
    soft_joint_revived = (
        depth10 >= SOFT_DEPTH10_REVIVAL_MIN_CORRECT
        and depth11 >= SOFT_DEPTH11_REVIVAL_MIN_CORRECT
    )
    scaling_revived = locked_bar_revived or soft_joint_revived
    deceleration_confirmed = depth10 < SOFT_DEPTH10_REVIVAL_MIN_CORRECT
    if soft_joint_revived:
        verdict = "soft_scaling_revived"
    elif locked_bar_revived:
        verdict = "locked_scaling_revived_but_soft_deceleration"
    elif deceleration_confirmed:
        verdict = "deceleration_confirmed"
    else:
        verdict = "ambiguous"
    return {
        "baseline_selected_correct": dict(EXPECTED_SUPPORT8_SELECTED_CORRECT),
        "selected_correct": selected,
        "depth10_delta_vs_support8": depth10_delta,
        "depth11_delta_vs_support8": depth11_delta,
        "locked_depth10_bar_revived": locked_bar_revived,
        "soft_joint_depth10_11_revived": soft_joint_revived,
        "scaling_revived": scaling_revived,
        "deceleration_confirmed_below_95": deceleration_confirmed,
        "locked_depth10_min_correct": STRONG_SCALING_MIN_CORRECT,
        "soft_depth10_min_correct": SOFT_DEPTH10_REVIVAL_MIN_CORRECT,
        "soft_depth11_min_correct": SOFT_DEPTH11_REVIVAL_MIN_CORRECT,
        "verdict": verdict,
    }
