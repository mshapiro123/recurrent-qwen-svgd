"""Locked contracts and scoring for the terminal oracle interface probe."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable


LOCKED_ROUTES = ("additive", "film")
LOCKED_CONTROL_ROWS = 106
LOCKED_CONTROL_GROUPS = 32
LOCKED_CONTROL_TRANSITIONS = 305
NONDEFAULT_CONTROL_FLOOR = 0.85
OVERALL_CONTROL_FLOOR = 0.90
TRANSITION_LEGALITY_FLOOR = 0.95
TERMINAL_VALIDITY_FLOOR = 0.71
ALLOWED_TRAINABLE_PREFIX = "oracle_reentry_conditioner."


def is_oracle_interface_trainable(name: str) -> bool:
    return str(name).startswith(ALLOWED_TRAINABLE_PREFIX)


def assert_oracle_frozen_parameter_contract(
    named_parameters: Iterable[tuple[str, Any]],
) -> dict[str, list[str]]:
    allowed: list[str] = []
    unexpected: list[str] = []
    for name, parameter in named_parameters:
        if not bool(parameter.requires_grad):
            continue
        if is_oracle_interface_trainable(name):
            allowed.append(str(name))
        else:
            unexpected.append(str(name))
    if unexpected:
        raise AssertionError(
            "Oracle interface probe freezes the keeper; unexpected trainable "
            "parameters: " + ", ".join(unexpected[:20])
        )
    if not allowed:
        raise AssertionError("Oracle interface probe has no trainable conditioner tensors")
    return {"allowed_trainable": sorted(allowed), "unexpected_trainable": unexpected}


def assert_oracle_frozen_gradients_zero(
    named_parameters: Iterable[tuple[str, Any]],
) -> None:
    violations: list[str] = []
    for name, parameter in named_parameters:
        if is_oracle_interface_trainable(name) or parameter.grad is None:
            continue
        if int(parameter.grad.detach().count_nonzero().item()):
            violations.append(str(name))
    if violations:
        raise AssertionError(
            "Frozen keeper received oracle-probe gradients: "
            + ", ".join(violations[:20])
        )


def _rate(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _transition_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["controlled"]) for row in rows)
    legal = sum(bool(row["legal"]) for row in rows)
    margins = [float(row["target_margin"]) for row in rows]
    return {
        "correct": correct,
        "total": len(rows),
        "control_rate": _rate(correct, len(rows)),
        "legal": legal,
        "legality_rate": _rate(legal, len(rows)),
        "mean_target_margin": fmean(margins) if margins else 0.0,
    }


def summarize_oracle_arm(
    transition_rows: list[dict[str, Any]],
    terminal_rows: list[dict[str, Any]],
    *,
    route: str,
    identity_exact: bool,
    frozen_lineage_unchanged: bool,
    expected_rows: int = LOCKED_CONTROL_ROWS,
    expected_groups: int = LOCKED_CONTROL_GROUPS,
    expected_transitions: int = LOCKED_CONTROL_TRANSITIONS,
) -> dict[str, Any]:
    if route not in (*LOCKED_ROUTES, "layerwise_film"):
        raise AssertionError(f"Unknown oracle route: {route}")
    if len(transition_rows) != int(expected_transitions):
        raise AssertionError(
            f"Oracle interface evaluation requires {expected_transitions} transitions, "
            f"got {len(transition_rows)}"
        )
    if len(terminal_rows) != int(expected_rows):
        raise AssertionError(
            f"Oracle interface evaluation requires {expected_rows} terminal rows, "
            f"got {len(terminal_rows)}"
        )
    groups = {str(row["base_problem_id"]) for row in terminal_rows}
    if len(groups) != int(expected_groups):
        raise AssertionError(
            f"Oracle interface evaluation requires {expected_groups} groups, got {len(groups)}"
        )

    default_rows = [row for row in transition_rows if bool(row["command_is_default"])]
    nondefault_rows = [
        row for row in transition_rows if not bool(row["command_is_default"])
    ]
    if not default_rows or not nondefault_rows:
        raise AssertionError("Oracle gate requires both default and non-default transitions")
    terminal_valid = sum(bool(row["valid"]) for row in terminal_rows)

    by_depth: dict[str, Any] = {}
    for depth in sorted({int(row["depth"]) for row in transition_rows}):
        selected = [row for row in transition_rows if int(row["depth"]) == depth]
        by_depth[str(depth)] = _transition_slice(selected)
    by_loop: dict[str, Any] = {}
    for loop_index in sorted({int(row["loop_index"]) for row in transition_rows}):
        selected = [
            row for row in transition_rows if int(row["loop_index"]) == loop_index
        ]
        by_loop[str(loop_index)] = _transition_slice(selected)

    overall = _transition_slice(transition_rows)
    default = _transition_slice(default_rows)
    nondefault = _transition_slice(nondefault_rows)
    checks = {
        "nondefault_branch_control": {
            "observed": nondefault["control_rate"],
            "minimum": NONDEFAULT_CONTROL_FLOOR,
            "correct": nondefault["correct"],
            "total": nondefault["total"],
            "passed": nondefault["control_rate"] >= NONDEFAULT_CONTROL_FLOOR,
        },
        "overall_transition_control": {
            "observed": overall["control_rate"],
            "minimum": OVERALL_CONTROL_FLOOR,
            "correct": overall["correct"],
            "total": overall["total"],
            "passed": overall["control_rate"] >= OVERALL_CONTROL_FLOOR,
        },
        "transition_legality": {
            "observed": overall["legality_rate"],
            "minimum": TRANSITION_LEGALITY_FLOOR,
            "legal": overall["legal"],
            "total": overall["total"],
            "passed": overall["legality_rate"] >= TRANSITION_LEGALITY_FLOOR,
        },
        "terminal_validity": {
            "observed": _rate(terminal_valid, len(terminal_rows)),
            "minimum": TERMINAL_VALIDITY_FLOOR,
            "valid": terminal_valid,
            "total": len(terminal_rows),
            "passed": _rate(terminal_valid, len(terminal_rows))
            >= TERMINAL_VALIDITY_FLOOR,
        },
        "zeroed_conditioning_identity": {
            "observed": bool(identity_exact),
            "required": True,
            "passed": bool(identity_exact),
        },
        "frozen_keeper_lineage": {
            "observed": bool(frozen_lineage_unchanged),
            "required": True,
            "passed": bool(frozen_lineage_unchanged),
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "kind": "phase_g_oracle_interface_arm",
        "route": route,
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "checks": checks,
        "transition_control": {
            "overall": overall,
            "default": default,
            "nondefault": nondefault,
            "by_depth": by_depth,
            "by_loop_index": by_loop,
        },
    }


def score_oracle_interface_probe(arms: list[dict[str, Any]]) -> dict[str, Any]:
    if len(arms) != 2:
        raise AssertionError("Oracle interface probe requires exactly two arms")
    by_route = {str(arm["route"]): arm for arm in arms}
    if set(by_route) != set(LOCKED_ROUTES):
        raise AssertionError("Oracle interface probe requires additive and FiLM arms")
    additive_pass = bool(by_route["additive"]["passed"])
    film_pass = bool(by_route["film"]["passed"])

    if film_pass and not additive_pass:
        reading = "FILM_CONTROLS_ADDITIVE_DOES_NOT"
        interpretation = "interface_localized_to_featurewise_modulation"
    elif not film_pass and not additive_pass:
        reading = "BOTH_FAIL"
        interpretation = "reentry_conditioning_closed_on_frozen_substrate"
    elif film_pass and additive_pass:
        reading = "BOTH_PASS"
        interpretation = "A0_failure_localized_to_variational_objective_or_amortization"
    else:
        reading = "ADDITIVE_ONLY_UNEXPECTED"
        interpretation = (
            "A0_failure_localized_away_from_additive_plumbing; "
            "FiLM_not_preferred; strategy_review_required"
        )
    return {
        "kind": "phase_g_oracle_interface_probe_gate",
        "status": "finished_terminal_probe",
        "measured_reading": reading,
        "interpretation": interpretation,
        "automatic_successor_authorized": False,
        "arms": {route: by_route[route] for route in LOCKED_ROUTES},
        "locked_gate_order": [
            "nondefault_branch_control",
            "overall_transition_control",
            "transition_legality",
            "terminal_validity",
            "zeroed_conditioning_identity",
            "frozen_keeper_lineage",
        ],
    }


def preregistration_payload() -> dict[str, Any]:
    return {
        "kind": "phase_g_oracle_interface_probe_preregistration",
        "status": "locked_before_training",
        "terminal_probe": True,
        "source_verdict": "NO-CHANNEL additive A0 ratified",
        "keeper": {
            "sha256": "0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f",
            "frozen": True,
        },
        "arms": list(LOCKED_ROUTES),
        "matched_arm_contract": {
            "same_conditioner_parameter_count": True,
            "same_training_rows": 1899,
            "same_training_seed": 20260718,
            "same_steps": 1500,
            "same_optimizer": "AdamW",
            "only_variable": "combination_rule_additive_vs_featurewise_affine",
        },
        "conditioning": (
            "frozen true-next-symbol embedding plus pooled current loop-input state"
        ),
        "objective": "per-loop commanded-chain cross entropy; no KL or stochastic latent",
        "heldout": {
            "rows": LOCKED_CONTROL_ROWS,
            "groups": LOCKED_CONTROL_GROUPS,
            "transitions": LOCKED_CONTROL_TRANSITIONS,
        },
        "gates": {
            "nondefault_branch_control": NONDEFAULT_CONTROL_FLOOR,
            "overall_transition_control": OVERALL_CONTROL_FLOOR,
            "transition_legality": TRANSITION_LEGALITY_FLOOR,
            "terminal_validity": TERMINAL_VALIDITY_FLOOR,
            "zeroed_conditioning_identity": "exact",
            "frozen_keeper_lineage": "exact",
        },
        "deferred": [
            "variational_successor",
            "coverage",
            "selection",
            "per_trajectory_halting",
            "particles",
            "SVGD",
        ],
    }
