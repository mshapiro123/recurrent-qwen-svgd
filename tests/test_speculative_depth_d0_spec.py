from __future__ import annotations

import pytest

from training.speculative_depth_d0_spec import (
    D0ExecutionPolicy,
    build_only_contract,
    calibrated_depth_targets,
    d0_draft,
    depth_recoverable_fraction,
    dynamic_depth_target,
    mask_teacher_added_token_probabilities,
    unresolved_paths,
    validate_locked_d0,
)

import torch


def test_d0_draft_is_fail_closed_and_has_no_launcher() -> None:
    spec = d0_draft()
    assert spec["status"] == "draft_not_locked"
    assert spec["training_authorized"] is False
    assert spec["launch_target_exists"] is False
    assert spec["substrate_family"] == "Qwen"
    assert spec["dependency"]["t1_lite_verdict_required"] is True
    assert spec["dependency"]["automatic_launch_from_t1"] is False
    assert len(unresolved_paths(spec)) >= 10


def test_d0_draft_cannot_validate_as_locked() -> None:
    with pytest.raises(AssertionError, match="not locked"):
        validate_locked_d0(d0_draft())


def test_d0_build_only_contract_forbids_labeling_and_training() -> None:
    contract = build_only_contract()
    assert contract["status"] == "build_only_no_labeling_no_training"
    assert contract["labeling_gpu_authorized"] is False
    assert contract["training_authorized"] is False
    D0ExecutionPolicy().assert_allowed(labeling=False, training=False)
    with pytest.raises(RuntimeError, match="labeling"):
        D0ExecutionPolicy().assert_allowed(labeling=True, training=False)
    with pytest.raises(RuntimeError, match="training"):
        D0ExecutionPolicy().assert_allowed(labeling=False, training=True)


def test_d0_dynamic_target_uses_first_teacher_match_and_caps_at_four() -> None:
    assert dynamic_depth_target([False, True, True, True], max_depth=4) == 2
    assert dynamic_depth_target([False, False, False, False], max_depth=4) == 4
    assert dynamic_depth_target([True], max_depth=4) == 1


def test_d0_graded_mapping_selects_smallest_depth_near_depth4_plateau() -> None:
    curves = {
        "q1": [0.40, 0.48, 0.50, 0.50],
        "q2": [0.30, 0.31, 0.31, 0.31],
        "q3": [0.20, 0.25, 0.29, 0.30],
        "q4": [0.10, 0.10, 0.10, 0.10],
    }
    result = calibrated_depth_targets(curves)
    assert result["branch"] == "graded_floor_curve"
    assert result["targets"] == {"q1": 3, "q2": 1, "q3": 3, "q4": 1}


def test_d0_teacher_added_token_mask_renormalizes() -> None:
    probabilities = torch.tensor([0.2, 0.3, 0.1, 0.4])
    masked = mask_teacher_added_token_probabilities(probabilities, added_token_ids=[2, 3])
    assert torch.allclose(masked, torch.tensor([0.4, 0.6, 0.0, 0.0]))
    assert float(masked.sum()) == pytest.approx(1.0)


def test_d0_depth_recoverable_fraction_is_incremental_match_rate() -> None:
    receipt = depth_recoverable_fraction(loop1_matches=20, self_halted_matches=35, rejected_positions=100)
    assert receipt["depth_recoverable_fraction"] == pytest.approx(0.15)
