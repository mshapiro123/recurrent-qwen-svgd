from __future__ import annotations

import pytest

from training.internal_think_token_t1_spec import (
    phase_t1_draft,
    validate_locked_phase_t1,
)


def test_t1_draft_has_two_explicit_references_and_all_four_gates() -> None:
    spec = phase_t1_draft()

    assert spec["status"] == "draft_not_locked"
    assert spec["training_authorized"] is False
    assert set(spec["fresh_base_lineages"]) == {
        "full_block",
        "r16_adapter_bridge",
    }
    assert spec["fresh_base_lineages"]["full_block"]["nonhalting_reference"][
        "trained_depths_correct"
    ] == 1005
    assert spec["fresh_base_lineages"]["full_block"]["nonhalting_reference"][
        "checkpoint_sha256"
    ].startswith("dc00f7b6")
    assert spec["fresh_base_lineages"]["r16_adapter_bridge"][
        "nonhalting_reference"
    ]["trained_depths_correct"] == 1021
    assert spec["gates"]["control_selection"]["minimum_each_depth"] == 0.90
    assert spec["gates"]["causal_override"]["required"] is True
    assert spec["gates"]["all_four_required_for_positive"] is True
    assert spec["data"]["rehearsal_fraction"] == 0.30
    assert spec["evaluation"]["frozen_row_id_sha256"].startswith("14482ca4")


def test_draft_cannot_authorize_training() -> None:
    with pytest.raises(AssertionError, match="not locked"):
        validate_locked_phase_t1(phase_t1_draft())
