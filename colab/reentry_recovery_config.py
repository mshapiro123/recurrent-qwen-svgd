"""Pure configuration helpers for Stage 4 re-entry recovery runs."""

from __future__ import annotations

from typing import Any


DEFAULT_STAGE4_MIN_POSITIVE_ROWS = 16


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


def positive_int_dict(payload: Any) -> dict[str, int]:
    """Return positive integer-like entries from a count dict."""

    out: dict[str, int] = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n > 0:
            out[str(key)] = n
    return out


def trace_curriculum_counts(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract Stage 4-relevant curriculum counts from a trace summary.

    Stage 4 accepts either the trace-collection wrapper shape, where counts
    live under ``curriculum.counts``, or the raw curriculum pipeline shape,
    where counts live at the top level.
    """

    curriculum = summary.get("curriculum") if isinstance(summary.get("curriculum"), dict) else {}
    counts = curriculum.get("counts") if isinstance(curriculum.get("counts"), dict) else summary.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    collection = summary.get("collection") if isinstance(summary.get("collection"), dict) else {}
    target_loop_counts = collection.get("target_loop_counts")
    if not isinstance(target_loop_counts, dict):
        target_loop_counts = counts.get("target_loop_counts")
    return {
        "positive_rows": int(counts.get("positive_sft_rows") or counts.get("typed_records") or 0),
        "typed_records": int(counts.get("typed_records") or 0),
        "mode_counts": positive_int_dict(counts.get("mode_counts")),
        "target_loop_counts": positive_int_dict(target_loop_counts),
        "tier_counts": positive_int_dict(counts.get("tier_counts")),
    }


def assess_trace_curriculum_for_reentry_recovery(
    summary: dict[str, Any],
    *,
    min_positive_rows: int = DEFAULT_STAGE4_MIN_POSITIVE_ROWS,
) -> dict[str, Any]:
    """Assess whether a trace curriculum is usable for bounded Stage 4 SFT.

    This is a readiness report, not a scientific pass/fail claim. A small
    curriculum can be acceptable for the bounded recovery smoke while still
    being too thin for a performance claim.
    """

    counts = trace_curriculum_counts(summary)
    positive_rows = int(counts["positive_rows"])
    mode_counts = counts["mode_counts"]
    target_loop_counts = counts["target_loop_counts"]
    issues: list[str] = []
    warnings: list[str] = []

    if summary.get("status") not in {"trace_curriculum_gate_ready", "complete"}:
        issues.append(f"unexpected_status:{summary.get('status')}")
    gate = summary.get("gate") if isinstance(summary.get("gate"), dict) else {}
    if gate and gate.get("go") is not True:
        issues.append("curriculum_gate_not_go")
    if positive_rows < min_positive_rows:
        issues.append(f"positive_rows_below_min:{positive_rows}<{min_positive_rows}")
    if mode_counts.get("direct", 0) <= 0:
        issues.append("missing_direct_rows")
    if mode_counts.get("deep_narrow", 0) <= 0 and mode_counts.get("deep", 0) <= 0:
        issues.append("missing_deep_rows")
    if target_loop_counts.get("1", 0) <= 0:
        issues.append("missing_target_loop_1")
    if not any(int(loop) > 1 for loop in target_loop_counts):
        issues.append("missing_deeper_target_loops")

    max_loop = max((int(loop) for loop in target_loop_counts), default=1)
    if positive_rows < 200:
        warnings.append("small_recovery_curriculum_not_claim_sized")
    if max_loop < 3:
        warnings.append("no_target_loop_3_or_higher")
    elif target_loop_counts.get(str(max_loop), 0) < 16:
        warnings.append(f"sparse_highest_loop_bucket:{max_loop}={target_loop_counts.get(str(max_loop), 0)}")

    status = "stage4_curriculum_ready" if not issues else "stage4_curriculum_blocked"
    return {
        "status": status,
        "go": not issues,
        "issues": issues,
        "warnings": warnings,
        "counts": counts,
        "strict_target_loop_gate": target_loop_rows_from_counts(target_loop_counts),
        "strict_mode_gate": mode_rows_from_counts(mode_counts),
        "max_target_loop": max_loop,
        "next_step": (
            "Use for bounded Stage 4 recovery smoke after Stage 3 passes; do not treat as claim-sized."
            if not issues
            else "Fix or regenerate the trace curriculum before Stage 4 recovery training."
        ),
    }
