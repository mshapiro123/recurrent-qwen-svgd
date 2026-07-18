"""Locked decision logic for the Phase G forced-injection causal probe."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable


LOCKED_INJECTION_FACTORS = (1.0, 3.0, 10.0, 30.0, 100.0)
LOCKED_CONTROL_ROWS = 106
LOCKED_CONTROL_GROUPS = 32
CHANNEL_SWITCHING_MIN = 16
NO_CHANNEL_SWITCHING_MAX_EXCLUSIVE = 8
VALIDITY_FLOOR_EXCLUSIVE = 0.50


def summarize_factor_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != LOCKED_CONTROL_ROWS:
        raise AssertionError(
            f"Forced-injection probe requires {LOCKED_CONTROL_ROWS} rows, got {len(rows)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_problem_id"])].append(row)
    if len(grouped) != LOCKED_CONTROL_GROUPS:
        raise AssertionError(
            f"Forced-injection probe requires {LOCKED_CONTROL_GROUPS} groups, got {len(grouped)}"
        )
    if any(len({str(row["target"]) for row in members}) < 2 for members in grouped.values()):
        raise AssertionError("Every forced-injection control group must contain multiple targets")

    switching_groups = sum(
        len({str(row["prediction"]) for row in members}) >= 2
        for members in grouped.values()
    )
    selected_target_correct = sum(
        str(row["prediction"]) == str(row["target"]) for row in rows
    )
    valid = sum(
        str(row["prediction"]) in {str(value) for value in row["reachable_symbols"]}
        for row in rows
    )
    changed_from_factor_1 = sum(
        str(row["prediction"]) != str(row["factor_1_prediction"]) for row in rows
    )
    scalar_metrics: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in dict(row.get("phase_g_metrics") or {}).items():
            scalar_metrics[str(name)].append(float(value))
    return {
        "rows": len(rows),
        "groups": len(grouped),
        "switching_groups": int(switching_groups),
        "selected_target_correct": int(selected_target_correct),
        "selected_target_fidelity": selected_target_correct / len(rows),
        "valid": int(valid),
        "K1_validity": valid / len(rows),
        "changed_from_factor_1": int(changed_from_factor_1),
        "changed_from_factor_1_rate": changed_from_factor_1 / len(rows),
        "mean_phase_g_metrics": {
            name: fmean(values) for name, values in sorted(scalar_metrics.items())
        },
    }


def _factor_points(arms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for arm in arms:
        label = str(arm["label"])
        raw_factors = dict(arm["factors"])
        factors = {float(value): metrics for value, metrics in raw_factors.items()}
        if set(factors) != set(LOCKED_INJECTION_FACTORS):
            raise AssertionError("Every arm must contain exactly the locked factors")
        for factor in LOCKED_INJECTION_FACTORS:
            metrics = factors[factor]
            points.append(
                {
                    "arm": label,
                    "factor": factor,
                    "switching_groups": int(metrics["switching_groups"]),
                    "K1_validity": float(metrics["K1_validity"]),
                    "selected_target_fidelity": float(
                        metrics["selected_target_fidelity"]
                    ),
                }
            )
    return points


def score_forced_injection_probe(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the strategy-locked channel/no-channel decision rule."""

    if len(arms) != 2:
        raise AssertionError("Forced-injection probe requires exactly both KL arms")
    points = _factor_points(arms)
    channel_points = [
        point
        for point in points
        if point["switching_groups"] >= CHANNEL_SWITCHING_MIN
        and point["K1_validity"] > VALIDITY_FLOOR_EXCLUSIVE
    ]
    high_switching_points = [
        point for point in points if point["switching_groups"] >= CHANNEL_SWITCHING_MIN
    ]
    all_below_no_channel_switching = all(
        point["switching_groups"] < NO_CHANNEL_SWITCHING_MAX_EXCLUSIVE
        for point in points
    )
    high_switching_only_with_invalidity = bool(high_switching_points) and all(
        point["K1_validity"] < VALIDITY_FLOOR_EXCLUSIVE
        for point in high_switching_points
    )

    if channel_points:
        measured_verdict = "CHANNEL-EXISTS"
        authorization = "successor_spec_authorized"
        next_step = (
            "new_preregistered_same_route_larger_trained_scale_with_preservation_gates"
        )
    elif all_below_no_channel_switching or high_switching_only_with_invalidity:
        measured_verdict = "NO-CHANNEL"
        authorization = "closed"
        next_step = (
            "additive_reentry_injection_closed; any_successor_must_change_conditioning_route"
        )
    else:
        measured_verdict = "AMBIGUOUS"
        authorization = "closed_by_default"
        next_step = (
            "report_ambiguous_and_default_to_no_channel_for_authorization"
        )

    return {
        "kind": "phase_g_forced_injection_causal_probe_gate",
        "status": (
            "channel_exists"
            if measured_verdict == "CHANNEL-EXISTS"
            else "blocked_no_authorized_successor"
        ),
        "measured_verdict": measured_verdict,
        "authorization": authorization,
        "next_step": next_step,
        "locked_rule": {
            "factors": list(LOCKED_INJECTION_FACTORS),
            "channel_exists": (
                f"switching_groups >= {CHANNEL_SWITCHING_MIN} at any factor and "
                f"K1_validity > {VALIDITY_FLOOR_EXCLUSIVE:.2f}"
            ),
            "no_channel": (
                f"switching_groups < {NO_CHANNEL_SWITCHING_MAX_EXCLUSIVE} at every "
                f"factor, or switching reaches {CHANNEL_SWITCHING_MIN} only while "
                f"K1_validity < {VALIDITY_FLOOR_EXCLUSIVE:.2f}"
            ),
            "ambiguous": "all intermediate outcomes; closed by default",
        },
        "channel_qualifying_points": channel_points,
        "high_switching_points": high_switching_points,
        "points": points,
    }
