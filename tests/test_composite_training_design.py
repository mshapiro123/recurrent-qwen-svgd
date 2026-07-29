from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from models.coconut_composite import MAX_HORIZONTAL_APPEND_STEPS
from training.composite_training_design import (
    COMPOSITE_TRAINING_POLICY,
    assert_composite_stage_contract,
    extract_horizontal_control_logits,
)


ROOT = Path(__file__).resolve().parents[1]


def test_governing_drive_artifacts_are_byte_verified() -> None:
    expected = {
        "docs/COMPOSITE_TRAINING_DESIGN_20260729.md": (
            "0ae848f560dda18abc89deb7716b53b24f40b49f5a7d44a6d5f2e514c9d5ed7b"
        ),
        "docs/STRATEGY_ADDENDUM_DC1_ROADMAP_20260729.md": (
            "67a38f52529fadf79a9b229e8a88d045a645a1f36cdfc2be89a1effec953a78b"
        ),
        "docs/figures/composite_architecture_20260729.svg": (
            "444aa15ae4210096a7082d23ec9ec88380f25b1c96624808fc3107ee7907cf9f"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_roadmap_locks_horizontal_cap_and_global_vertical_depth() -> None:
    assert MAX_HORIZONTAL_APPEND_STEPS == 3
    assert COMPOSITE_TRAINING_POLICY["horizontal_append_cap"] == 3
    assert COMPOSITE_TRAINING_POLICY["vertical_depth_by_stage"] == {
        "A": 1,
        "B": 1,
        "C": 1,
        "D": 2,
    }
    assert COMPOSITE_TRAINING_POLICY["per_position_vertical_routing"] is False


@pytest.mark.parametrize("stage", ["A", "B", "C"])
def test_stages_a_through_c_reject_vertical_depth_above_one(stage: str) -> None:
    with pytest.raises(AssertionError, match="global vertical depth"):
        assert_composite_stage_contract(stage=stage, append_steps=1, vertical_loops=2)


def test_stage_d_accepts_only_the_locked_shape_test_depth() -> None:
    assert_composite_stage_contract(stage="D", append_steps=1, vertical_loops=2)
    with pytest.raises(AssertionError, match="global vertical depth"):
        assert_composite_stage_contract(stage="D", append_steps=1, vertical_loops=1)


def test_every_stage_rejects_k_above_three() -> None:
    with pytest.raises(AssertionError, match="k <= 3"):
        assert_composite_stage_contract(stage="A", append_steps=4, vertical_loops=1)


def test_control_readout_extracts_only_continue_stop_rows() -> None:
    logits = torch.arange(2 * 3 * 11, dtype=torch.float32).reshape(2, 3, 11)
    control = extract_horizontal_control_logits(logits, (7, 9))
    assert control.shape == (2, 3, 2)
    torch.testing.assert_close(control[..., 0], logits[..., 7])
    torch.testing.assert_close(control[..., 1], logits[..., 9])


def test_control_readout_rejects_aliasing_or_out_of_range_rows() -> None:
    logits = torch.zeros(1, 2, 8)
    with pytest.raises(ValueError, match="distinct"):
        extract_horizontal_control_logits(logits, (4, 4))
    with pytest.raises(ValueError, match="outside"):
        extract_horizontal_control_logits(logits, (4, 8))
