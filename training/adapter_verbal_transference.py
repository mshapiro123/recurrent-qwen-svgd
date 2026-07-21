"""Locked scoring helpers for the E3b adapter verbal-transference experiment."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


TRANSFER_THRESHOLD = 0.71
PAIRED_ALPHA = 0.05
SYNTHETIC_RETAINED_FLOOR = 0.93
SYNTHETIC_COLLAPSE_CEILING = 0.10


def guardrail_near_miss_context(
    *,
    baseline_hits: Iterable[bool],
    observed_hits: Iterable[bool],
    hard_stop_delta: float,
) -> dict[str, Any]:
    """Describe a finite-row hard stop without relaxing the locked decision."""

    baseline = list(map(bool, baseline_hits))
    observed = list(map(bool, observed_hits))
    if len(baseline) != len(observed):
        raise ValueError("Guardrail comparisons require identical rows")
    if not baseline:
        raise ValueError("Guardrail comparisons require at least one row")

    baseline_correct = sum(baseline)
    observed_correct = sum(observed)
    total = len(baseline)
    baseline_accuracy = baseline_correct / total
    observed_accuracy = observed_correct / total
    accuracy_delta = observed_accuracy - baseline_accuracy
    triggered = accuracy_delta < float(hard_stop_delta)
    paired = paired_binary_test(baseline, observed)

    return {
        "baseline": {
            "correct": baseline_correct,
            "total": total,
            "accuracy": baseline_accuracy,
        },
        "observed": {
            "correct": observed_correct,
            "total": total,
            "accuracy": observed_accuracy,
        },
        "accuracy_delta": accuracy_delta,
        "accuracy_delta_points": round(100.0 * accuracy_delta, 10),
        "hard_stop_delta": float(hard_stop_delta),
        "hard_stop_delta_points": 100.0 * float(hard_stop_delta),
        "hard_stop_triggered": triggered,
        "item_resolution_points": 100.0 / total,
        "boundary_excess_points": (
            round(100.0 * (float(hard_stop_delta) - accuracy_delta), 10) if triggered else 0.0
        ),
        "paired": {
            "baseline_only": paired["t_only"],
            "observed_only": paired["s_only"],
            "ties": paired["ties"],
            "discordant": paired["discordant"],
            "two_sided_p": paired["two_sided_p"],
            "test": paired["test"],
        },
        "interpretation": (
            "near_boundary_discrete_hard_stop"
            if triggered and float(hard_stop_delta) - accuracy_delta < 1.0 / total
            else "hard_stop" if triggered else "guardrail_green"
        ),
        "decision_note": "The preregistered stop is preserved; this context is descriptive only.",
    }


def summarize_archived_active_diagonal(
    diagonal: Mapping[str, float], *, rows_per_depth: int
) -> dict[str, int | float]:
    correct = sum(round(float(value) * rows_per_depth) for value in diagonal.values())
    total = len(diagonal) * rows_per_depth
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def paired_binary_test(t_hits: Iterable[bool], s_hits: Iterable[bool]) -> dict[str, Any]:
    pairs = [(bool(t), bool(s)) for t, s in zip(t_hits, s_hits, strict=True)]
    t_only = sum(t and not s for t, s in pairs)
    s_only = sum(s and not t for t, s in pairs)
    discordant = t_only + s_only
    smaller = min(t_only, s_only)
    lower = (
        sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
        if discordant
        else 0.5
    )
    return {
        "t_only": t_only,
        "s_only": s_only,
        "ties": len(pairs) - discordant,
        "discordant": discordant,
        "net_correct": t_only - s_only,
        "two_sided_p": 1.0 if not discordant else min(1.0, 2.0 * lower),
        "test": "exact_paired_sign_mcnemar",
    }


def score_transference(
    *, t_hits: Iterable[bool], s_hits: Iterable[bool], alpha: float = PAIRED_ALPHA
) -> dict[str, Any]:
    t = list(map(bool, t_hits))
    s = list(map(bool, s_hits))
    if len(t) != len(s):
        raise ValueError("Arm T and Arm S must be scored on identical rows")
    paired = paired_binary_test(t, s)
    t_accuracy = sum(t) / len(t) if t else 0.0
    s_accuracy = sum(s) / len(s) if s else 0.0
    supported = t_accuracy > s_accuracy and paired["two_sided_p"] < alpha
    return {
        "verdict": "positive" if supported else "null",
        "arm_t": {"correct": sum(t), "total": len(t), "accuracy": t_accuracy},
        "arm_s": {"correct": sum(s), "total": len(s), "accuracy": s_accuracy},
        "delta_accuracy": t_accuracy - s_accuracy,
        "paired": paired,
        "alpha": alpha,
        "positive_requires": "T>S and exact paired two-sided p<alpha at matched endpoint",
    }


def first_threshold_crossing(
    curve: Mapping[str, float], *, threshold: float = TRANSFER_THRESHOLD
) -> int | None:
    for step, accuracy in sorted(((int(key), float(value)) for key, value in curve.items())):
        if accuracy >= threshold:
            return step
    return None


def classify_regression(
    by_step_depth: Mapping[str, Mapping[str, float]],
    *,
    retained_floor: float = SYNTHETIC_RETAINED_FLOOR,
    collapse_ceiling: float = SYNTHETIC_COLLAPSE_CEILING,
) -> dict[str, Any]:
    if not by_step_depth:
        raise ValueError("Regression classification requires at least one checkpoint")
    minima = {
        str(step): min(map(float, depths.values())) if depths else 0.0
        for step, depths in by_step_depth.items()
    }
    all_retained = all(value >= retained_floor for value in minima.values())
    final_step = max(minima, key=lambda key: int(key))
    final_min = minima[final_step]
    if all_retained:
        verdict = "retained"
    elif final_min < collapse_ceiling:
        verdict = "collapsed"
    else:
        verdict = "partial"
    return {
        "verdict": verdict,
        "min_accuracy_by_step": minima,
        "minimum_observed": min(minima.values()),
        "final_step": int(final_step),
        "final_min_accuracy": final_min,
        "retained_floor": retained_floor,
        "collapse_ceiling": collapse_ceiling,
        "collapse_reference": "E4 near-chance minimum stratum was 0.09375",
    }
