"""Locked architecture policy for the staged horizontal/vertical composite."""

from __future__ import annotations

from typing import Any, Iterable

import torch


COMPOSITE_TRAINING_POLICY: dict[str, Any] = {
    "kind": "composite_training_design_policy",
    "as_of": "2026-07-29",
    "horizontal_append_cap": 3,
    "vertical_depth_by_stage": {"A": 1, "B": 1, "C": 1, "D": 2},
    "per_position_vertical_routing": False,
    "stage_c_joint_finetune": False,
    "stage_c_interface_frozen": True,
    "control_signal_injected": False,
    "control_readout": (
        "continue/stop logits at the real decision position and after every transient slot"
    ),
    "control_visible_generation": False,
    "stage_d_shape_test": {
        "horizontal": {"append_steps": 2, "vertical_loops": 1},
        "vertical": {"append_steps": 1, "vertical_loops": 2},
        "registered_layer_applications_each": 72,
    },
    "teacher_policy": {
        "training_teacher_stages_a_to_c": "7B",
        "teacher_14b_roles": [
            "descriptive_crossover",
            "optional_stage_c_label_referee",
            "post_stage_c_distillation",
        ],
        "exact_match_signal_mixing": False,
    },
    "open_for_markup": [
        "stage_b_trainable_set",
        "stage_b_lambda_grid",
        "stage_c_14b_referee_exclusion",
        "stage_b_14b_crossover_cache_policy",
    ],
}


def registered_layer_applications(
    *,
    append_steps: int,
    vertical_loops: int,
    prelude_layers: int = 12,
    recurrent_layers: int = 12,
) -> int:
    """Return the design's registered compute count, excluding attention overhead."""

    return (int(append_steps) + 1) * (
        int(prelude_layers) + int(recurrent_layers) * int(vertical_loops)
    )


def assert_composite_stage_contract(
    *,
    stage: str,
    append_steps: int,
    vertical_loops: int,
) -> dict[str, Any]:
    """Fail before execution if a staged run drifts from a locked architecture constant."""

    normalized = str(stage).upper()
    expected = COMPOSITE_TRAINING_POLICY["vertical_depth_by_stage"].get(normalized)
    if expected is None:
        raise AssertionError(f"unknown composite stage {stage!r}")
    if not 0 <= int(append_steps) <= int(COMPOSITE_TRAINING_POLICY["horizontal_append_cap"]):
        raise AssertionError("composite execution requires 0 <= k <= 3")
    if int(vertical_loops) != int(expected):
        raise AssertionError(
            f"stage {normalized} requires global vertical depth L={expected}; "
            f"observed L={vertical_loops}"
        )
    return {
        "stage": normalized,
        "append_steps": int(append_steps),
        "vertical_loops": int(vertical_loops),
        "registered_layer_applications": registered_layer_applications(
            append_steps=int(append_steps),
            vertical_loops=int(vertical_loops),
        ),
        "passed": True,
    }


def extract_horizontal_control_logits(
    logits: torch.Tensor,
    control_token_ids: Iterable[int],
) -> torch.Tensor:
    """Read continue/stop rows without sampling, masking, or changing execution."""

    token_ids = tuple(int(value) for value in control_token_ids)
    if len(token_ids) != 2:
        raise ValueError("horizontal control readout requires continue and stop token ids")
    if token_ids[0] == token_ids[1]:
        raise ValueError("horizontal control token ids must be distinct")
    if min(token_ids) < 0 or max(token_ids) >= int(logits.shape[-1]):
        raise ValueError(
            f"horizontal control token id outside logits vocabulary {logits.shape[-1]}"
        )
    return logits[..., list(token_ids)]

