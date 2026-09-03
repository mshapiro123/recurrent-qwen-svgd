from __future__ import annotations

from dataclasses import replace

import pytest

from analysis.weft1_preflight_cjac1 import (
    CJAC1_AUTHORITY_BYTES,
    CJAC1_AUTHORITY_SHA256,
    CJAC1_LEGACY_DIAGNOSTIC_STATUS,
    CJAC1_SCRATCH_COMPONENT_MAPPING,
    CurrentGraphCJac1Receipt,
    run_current_graph_cjac1,
)


@pytest.fixture(scope="module")
def cjac1_receipt() -> CurrentGraphCJac1Receipt:
    return run_current_graph_cjac1()


def test_pf3_cjac1_current_graph_emits_converged_two_number_receipt(
    cjac1_receipt: CurrentGraphCJac1Receipt,
) -> None:
    receipt = cjac1_receipt
    core = receipt.loop_receipt.joint_core_estimate.core_estimate

    assert receipt.authority_byte_verified
    assert receipt.authority_bytes == CJAC1_AUTHORITY_BYTES == 14_632
    assert receipt.authority_sha256 == CJAC1_AUTHORITY_SHA256 == (
        "7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef"
    )
    assert receipt.current_graph_components == ("h", "scratch")
    assert receipt.absent_ratified_components == ("lanes", "carrier")
    assert receipt.scratch_component_mapping == CJAC1_SCRATCH_COMPONENT_MAPPING
    assert "private_API_name_lanes" in receipt.scratch_component_mapping
    assert "absent_full_width_bicameral_lanes_component" in (
        receipt.scratch_component_mapping
    )
    assert receipt.loop_receipt.joint_core_estimate.layout.metric == (
        "plain_euclidean_concatenation_no_per_block_reweighting"
    )
    assert receipt.terminal_visit == 8
    assert receipt.visits_materialized_before_primal == 7
    assert receipt.lambda_adapters == pytest.approx(1.0)
    assert receipt.lambda_hat_core == pytest.approx(1.041439431237759, abs=1e-12)
    assert receipt.loop_receipt.two_number_line == (
        ("Lambda_adapters", receipt.lambda_adapters),
        ("Lambda_hat_core", receipt.lambda_hat_core),
    )
    assert receipt.convergence_passed
    assert core.converged
    assert core.iterations == 47
    assert core.minimum_iterations == 3
    assert core.last_relative_change == pytest.approx(
        0.0009611520575406325,
        abs=1e-15,
    )
    assert core.last_relative_change < core.convergence_tolerance == 1e-3
    assert receipt.transition_output_matches_direct_visit
    assert receipt.require_converged_measurement() is None


def test_pf3_cjac1_enumerates_current_factors_without_claiming_missing_topology(
    cjac1_receipt: CurrentGraphCJac1Receipt,
) -> None:
    receipt = cjac1_receipt
    adapter = receipt.loop_receipt.adapter_certificate

    assert tuple(factor.name for factor in adapter.factors) == (
        "anchored_reentry_bridge",
        "two_lane_birkhoff_scratch_carrier",
        "loop_embedding",
    )
    assert all(factor.bound == 1.0 for factor in adapter.factors)
    assert {placeholder.name for placeholder in adapter.placeholders} == {
        "integrated_rotor_carrier",
        "per_band_callosum",
        "sidecar",
    }
    assert tuple(item.name for item in receipt.factor_treatments) == (
        "anchored_reentry_bridge",
        "position_aligned_scratch_update_and_injection",
        "two_lane_birkhoff_scratch_carrier",
        "loop_embedding",
        "two_shared_core_blocks",
    )
    assert receipt.production_certificate_authorized is False
    assert receipt.production_alarm_authorized is False
    assert receipt.loop_receipt.production_claim == (
        "current_graph_measurement_only_topology_incomplete"
    )
    assert receipt.training_performed is False
    assert receipt.checkpoint_used is False
    assert receipt.sealed_data_touched is False
    assert receipt.a100_hours == 0.0


def test_pf3_cjac1_legacy_dimension_reweighted_probe_cannot_be_promoted(
    cjac1_receipt: CurrentGraphCJac1Receipt,
) -> None:
    receipt = cjac1_receipt

    assert receipt.legacy_model_diagnostic_status == CJAC1_LEGACY_DIAGNOSTIC_STATUS
    assert receipt.loop_receipt.joint_core_estimate.legacy_model_diagnostic_status == (
        CJAC1_LEGACY_DIAGNOSTIC_STATUS
    )
    with pytest.raises(ValueError, match="legacy diagnostic"):
        replace(receipt, legacy_model_diagnostic_status="governing")


def test_pf3_cjac1_receipt_rejects_forged_measurement_and_production_claims(
    cjac1_receipt: CurrentGraphCJac1Receipt,
) -> None:
    receipt = cjac1_receipt
    core = receipt.loop_receipt.joint_core_estimate.core_estimate

    receipt_mutations = (
        {"lambda_adapters": 2.0},
        {"lambda_hat_core": 0.0},
        {"measurement_status": "passed"},
        {"scratch_component_mapping": "lanes_and_scratch_are_the_same"},
        {"convergence_passed": False},
        {"transition_output_matches_direct_visit": False},
        {"production_certificate_authorized": True},
        {"production_alarm_authorized": True},
        {"sealed_data_touched": True},
        {"authority_byte_verified": False},
        {"state_definition": "z=[h]"},
        {"root_seed": -1},
        {"cpu_runtime": "forged"},
        {"torch_version": "forged"},
        {"training_performed": True},
        {"checkpoint_used": True},
        {"a100_hours": 1.0},
    )
    for updates in receipt_mutations:
        with pytest.raises(ValueError, match="C-JAC-1"):
            replace(receipt, **updates)

    with pytest.raises(ValueError, match="lambda_hat_core"):
        replace(core, lambda_hat_core=0.0)
    with pytest.raises(ValueError, match="convergence"):
        replace(core, converged=False)
    with pytest.raises(ValueError, match="Rayleigh"):
        replace(
            core,
            rayleigh_quotient_sequence=(
                *core.rayleigh_quotient_sequence[:-1],
                -1.0,
            ),
        )


def test_pf3_cjac1_replay_is_bit_identical(
    cjac1_receipt: CurrentGraphCJac1Receipt,
) -> None:
    replay = run_current_graph_cjac1()

    assert replay.to_dict() == cjac1_receipt.to_dict()
