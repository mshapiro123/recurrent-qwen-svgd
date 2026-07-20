"""Locked contract for the distributed oracle-control localization probe."""

from __future__ import annotations

from typing import Any

from training.oracle_interface_probe_spec import (
    NONDEFAULT_CONTROL_FLOOR,
    OVERALL_CONTROL_FLOOR,
    TERMINAL_VALIDITY_FLOOR,
    TRANSITION_LEGALITY_FLOOR,
)


LOCKED_ROUTE = "layerwise_film"
ALLOWED_TRAINABLE_PREFIX = "oracle_intrablock_conditioner."


def is_oracle_intrablock_trainable(name: str) -> bool:
    return str(name).startswith(ALLOWED_TRAINABLE_PREFIX)


def assert_oracle_intrablock_frozen_parameter_contract(
    named_parameters: Any,
) -> dict[str, list[str]]:
    allowed: list[str] = []
    unexpected: list[str] = []
    for name, parameter in named_parameters:
        if not bool(parameter.requires_grad):
            continue
        if is_oracle_intrablock_trainable(name):
            allowed.append(str(name))
        else:
            unexpected.append(str(name))
    if unexpected:
        raise AssertionError(
            "Layerwise oracle probe freezes the keeper; unexpected trainable "
            "parameters: " + ", ".join(unexpected[:20])
        )
    if not allowed:
        raise AssertionError(
            "Layerwise oracle probe has no trainable conditioner tensors"
        )
    return {
        "allowed_trainable": sorted(allowed),
        "unexpected_trainable": unexpected,
    }


def assert_oracle_intrablock_frozen_gradients_zero(named_parameters: Any) -> None:
    violations: list[str] = []
    for name, parameter in named_parameters:
        if is_oracle_intrablock_trainable(name) or parameter.grad is None:
            continue
        if int(parameter.grad.detach().count_nonzero().item()):
            violations.append(str(name))
    if violations:
        raise AssertionError(
            "Frozen keeper received layerwise-oracle gradients: "
            + ", ".join(violations[:20])
        )


def score_oracle_intrablock_control(arm: dict[str, Any]) -> dict[str, Any]:
    if str(arm.get("route")) != LOCKED_ROUTE:
        raise AssertionError(
            f"Distributed oracle probe requires route={LOCKED_ROUTE!r}"
        )
    passed = bool(arm.get("passed"))
    if passed:
        reading = "DISTRIBUTED_INTERFACE_CONTROLS"
        interpretation = "single_entry_access_was_the_binding_constraint"
    else:
        reading = "DISTRIBUTED_INTERFACE_FAILS"
        interpretation = (
            "frozen_substrate_not_oracle_controllable_under_tested_small_interfaces"
        )
    return {
        "kind": "phase_g_oracle_intrablock_control_gate",
        "status": "finished_terminal_localization",
        "measured_reading": reading,
        "interpretation": interpretation,
        "automatic_successor_authorized": False,
        "arm": arm,
    }


def preregistration_payload() -> dict[str, Any]:
    return {
        "kind": "phase_g_oracle_intrablock_control_preregistration",
        "status": "locked_before_training",
        "terminal_probe": True,
        "source_verdict": "single-entry additive and FiLM oracle routes both failed",
        "keeper": {
            "sha256": (
                "0f657b653078ba403cbc666410e7598ca"
                "20c836d5bd6e19a0e85a186a82c5d2f"
            ),
            "frozen": True,
        },
        "arms": ["single_entry_film_control", LOCKED_ROUTE],
        "only_variable": "command_access_location",
        "conditioner": {
            "combination_rule": "film",
            "shared_across_recurrent_layers": True,
            "parameter_matched_to_single_entry_film": True,
            "zero_initialized_identity": "exact",
        },
        "objective": "per-loop commanded-chain cross entropy; no KL or stochastic latent",
        "heldout": {
            "rows": 106,
            "groups": 32,
            "transitions": 305,
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
