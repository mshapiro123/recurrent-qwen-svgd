from __future__ import annotations

import pytest

from training.internal_think_token_t1_spec import (
    phase_t1_draft,
    phase_t1_locked,
    validate_locked_phase_t1,
)


def test_t1_lite_draft_has_one_full_block_reference_and_all_four_gates() -> None:
    spec = phase_t1_draft()

    assert spec["status"] == "draft_not_locked"
    assert spec["training_authorized"] is False
    assert spec["program_mode"] == "t1_lite_full_block_actuator_qualification"
    assert set(spec["fresh_base_lineages"]) == {"full_block"}
    assert spec["fresh_base_lineages"]["full_block"]["nonhalting_reference"][
        "trained_depths_correct"
    ] == 1005
    assert spec["fresh_base_lineages"]["full_block"]["nonhalting_reference"][
        "checkpoint_sha256"
    ].startswith("dc00f7b6")
    assert spec["descoped_lineage"]["lineage"] == "r16_adapter_bridge"
    assert spec["descoped_lineage"]["capacity_contrast_forfeited"] is True
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
    assert pilot["registered_t1_lineage"] == "full_block"
    assert pilot["matched_lineage_evidence"] is False
    assert pilot["seed"] == 9999
    assert pilot["steps_per_cell"] == 1500
    assert len(pilot["cells"]) == 10
    assert pilot["evaluation_steps"] == [500, 1000, 1500]
    assert pilot["selection"]["minimum_stop_recall"] == 0.60
    assert pilot["selection"]["minimum_continue_recall"] == 0.60
    assert pilot["selection"]["tie_break"] == "toward_lambda_1_then_ratio_3p5"
    assert spec["training_authorized"] is False
    assert spec["proposed_training_budget"]["lineages"] == ["full_block"]
    assert spec["d0_status"] == "preregistration_drafting_only"


def test_draft_cannot_authorize_training() -> None:
    with pytest.raises(AssertionError, match="not locked"):
        validate_locked_phase_t1(phase_t1_draft())


def test_t1_lite_locked_spec_records_p0_selection_and_stage_guardrails() -> None:
    spec = phase_t1_locked()

    validate_locked_phase_t1(spec)
    assert spec["status"] == "locked_before_training"
    assert spec["training_authorized"] is True
    assert spec["pilot_p0"]["status"] == "complete_uncitable_prelock_pilot"
    assert spec["pilot_p0"]["selected_cell_id"] == "lambda0p5_ratio1"
    assert spec["loss"]["control_loss_lambda"] == 0.5
    assert spec["loss"]["stop_to_continue_ratio"] == 1.0
    assert spec["loss"]["normalized_class_weights"] == {
        "continue": 1.0,
        "stop": 1.0,
    }
    guardrails = spec["stage_boundary_liveness"]
    assert [row["step"] for row in guardrails["boundaries"]] == [500, 2500, 6500, 8500]
    assert guardrails["abort_rule"]["all_conditions_required"] is True
    assert guardrails["may_change_registered_constants"] is False


def test_t1_lite_locked_spec_freezes_all_three_evaluation_manifests() -> None:
    spec = phase_t1_locked()
    evaluation = spec["evaluation"]

    assert evaluation["gated"]["rows"] == 1024
    assert evaluation["gated"]["row_id_sha256"] == (
        "7aa673d046803c691226dd0a9950972ca141b4aaa89fcc118cc049b7e71fdcbe"
    )
    assert evaluation["extrapolation"]["rows"] == 768
    assert evaluation["extrapolation"]["row_id_sha256"] == (
        "74c56235a033cc783963bc71584e2203b0b6936ba3996cf174616da3d1414b48"
    )
    assert evaluation["calibration"]["rows"] == 512
    assert evaluation["calibration"]["seed"] == 2026072401
    assert evaluation["calibration"]["row_id_sha256"] == (
        "ebc17c1012db868fe5788241e632c463e304bbadef0117a8b6af32a4fff6d6b2"
    )
