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
    selection = spec["gates"]["control_selection"]
    assert selection["metric"] == "row_level_exact_selected_depth"
    assert selection["minimum_correct_each_depth"] == 115
    assert selection["rows_each_depth"] == 128
    assert selection["minimum_correct_pooled"] == 922
    assert selection["rows_pooled"] == 1024
    assert selection["transition_micro_accuracy_is_gate"] is False
    assert spec["gates"]["causal_override"]["required"] is True
    assert spec["gates"]["causal_override"]["forced_stop_executions"] == 4608
    assert spec["gates"]["causal_override"]["forced_continue_executions"] == 1024
    assert spec["gates"]["all_four_required_for_positive"] is True
    assert spec["data"]["rehearsal_fraction"] == 0.30
    assert spec["evaluation"]["gated"]["rows"] == 1024
    assert spec["evaluation"]["calibration"]["rows"] == 512
    assert spec["evaluation"]["extrapolation"]["depths"] == list(range(9, 15))
    assert spec["evaluation"]["self_halt_max_loops"]["gated"] == 12
    assert spec["evaluation"]["self_halt_max_loops"]["extrapolation"] == 16


def test_t1_draft_encodes_p0_without_authorizing_registered_training() -> None:
    spec = phase_t1_draft()
    pilot = spec["pilot_p0"]

    assert pilot["authorized_before_lock"] is True
    assert pilot["registered_t1_training"] is False
    assert pilot["lineage"] == "r16_adapter_bridge"
    assert pilot["seed"] == 9999
    assert pilot["steps_per_cell"] == 1500
    assert len(pilot["cells"]) == 10
    assert pilot["evaluation_steps"] == [500, 1000, 1500]
    assert pilot["selection"]["minimum_stop_recall"] == 0.60
    assert pilot["selection"]["minimum_continue_recall"] == 0.60
    assert spec["training_authorized"] is False


def test_draft_cannot_authorize_training() -> None:
    with pytest.raises(AssertionError, match="not locked"):
        validate_locked_phase_t1(phase_t1_draft())
