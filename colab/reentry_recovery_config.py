"""Pure configuration helpers for Stage 4 re-entry recovery runs."""

from __future__ import annotations

from typing import Any


def int_dict_max_key(payload: Any, default: int) -> int:
    """Return the largest positive integer-like key from a count dict."""
    values: list[int] = []
    if isinstance(payload, dict):
        for key in payload:
            try:
                value = int(key)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
    return max(values) if values else default


def finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def repair_assessment_recovery_block_reason(assessment: dict[str, Any]) -> str | None:
    """Return why a Stage 3 repair assessment should not unlock Stage 4.

    Stage 4 is a more expensive recovery SFT run. It should not trust only the
    human-readable recommendation string from an older Stage 3 artifact. The
    repair smoke must also carry the current evidence fields: finite train
    metrics, depth-supervision metrics, loop-1 preservation, and live/moved
    bridge/re-entry components.
    """

    recommendation = str(assessment.get("recommendation") or "")
    if recommendation != "run_bounded_recovery_training_with_reentry_repair":
        status = str(assessment.get("status") or "")
        return f"Stage 3 repair smoke did not clear recovery training: status={status!r} recommendation={recommendation!r}."

    metrics = assessment.get("metrics")
    if not isinstance(metrics, dict):
        return "Stage 3 repair assessment is missing metrics; rerun reentry_repair_smoke with the current metric-hardened cell."

    if metrics.get("train_metrics_available") is not True:
        return "Stage 3 repair assessment lacks final training metrics; rerun reentry_repair_smoke before Stage 4."

    if finite_number(metrics.get("train_loss")) is None:
        return "Stage 3 repair assessment has missing or nonfinite final train_loss; do not start Stage 4."

    if metrics.get("depth_supervision_metrics_present") is not True:
        return "Stage 3 repair assessment lacks supervised depth metrics; rerun or fix reentry_repair_smoke before Stage 4."

    if metrics.get("loop1_preservation_available") is not True:
        return "Stage 3 repair assessment lacks comparable loop-1 preservation evidence; do not start Stage 4."

    if metrics.get("loop1_regressed") is True:
        return "Stage 3 repair smoke regressed loop-1 preservation; do not start Stage 4."

    if metrics.get("bridge_live") is not True or metrics.get("bridge_moved") is not True:
        return "Stage 3 repair assessment does not prove the bridge is both gradient-live and moved."

    if metrics.get("use_reentry_adapter") is True:
        if metrics.get("adapter_live") is not True:
            return "Stage 3 repair assessment does not prove the re-entry adapter is gradient-live."
        if metrics.get("adapter_moved") is not True:
            return "Stage 3 repair assessment does not prove the re-entry adapter moved."

    return None


def mode_rows_from_counts(mode_counts: Any) -> str:
    """Convert mode counts into the gate format: direct=12,deep_narrow=8."""
    if not isinstance(mode_counts, dict):
        return ""
    parts: list[str] = []
    for mode, count in sorted(mode_counts.items()):
        try:
            n = int(count)
        except (TypeError, ValueError):
            continue
        if n > 0:
            parts.append(f"{mode}={n}")
    return ",".join(parts)


def target_loop_rows_from_counts(target_loop_counts: Any) -> str:
    """Convert target-loop counts into strict SFT gate requirements.

    The Stage 4 depth curriculum depends on preserving the actual count per
    target loop. Collapsing every observed loop to a minimum of one row can let a
    fake ladder through the preflight gate and erase the intended depth signal.
    """
    if not isinstance(target_loop_counts, dict):
        return ""
    sortable: list[tuple[int, int]] = []
    for loop, count in target_loop_counts.items():
        try:
            loop_value = int(loop)
            count_value = int(count)
        except (TypeError, ValueError):
            continue
        if loop_value > 0 and count_value > 0:
            sortable.append((loop_value, count_value))
    return ",".join(f"{loop}={count}" for loop, count in sorted(sortable))
