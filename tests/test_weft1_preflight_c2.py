from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from analysis.weft1_preflight_c2 import (
    C2_AUTHORITY_SHA256,
    C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD,
    C2_PF2_AUTHORITY_BYTES,
    C2_PF2_AUTHORITY_SHA256,
    C2_PF3_AUTHORITY_BYTES,
    C2_PF3_AUTHORITY_SHA256,
    C2_REPRESENTATIVE_MISSING_FULL_TOY_INTEGRATIONS,
    C2_RATIFIED_VISITS,
    C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD,
    C2PreflightReceipt,
    _GradientTraceTensor,
    _config_identity_sha256,
    _gradient_drift,
    c2_current_toy_config,
    run_preflight_c2,
)


@pytest.fixture(scope="module")
def c2_receipt() -> C2PreflightReceipt:
    return run_preflight_c2()


def test_c2_current_toy_configuration_is_exactly_4_2_4_d64_k8() -> None:
    config = c2_current_toy_config()

    assert C2_AUTHORITY_SHA256 == (
        "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
    )
    assert C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD == 1e-2
    assert C2_LANE_GRADIENT_RELATIVE_L2_THRESHOLD == 5e-2
    assert C2_PF2_AUTHORITY_BYTES == 13_097
    assert C2_PF2_AUTHORITY_SHA256 == (
        "be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05"
    )
    assert C2_PF3_AUTHORITY_BYTES == 14_632
    assert C2_PF3_AUTHORITY_SHA256 == (
        "7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef"
    )
    assert C2_RATIFIED_VISITS == 8
    assert (config.n_prelude_layers, config.n_core_blocks, config.n_coda_layers) == (
        4,
        2,
        4,
    )
    assert config.d_model == 64
    assert config.scratch_lanes == 2 and config.scratch_width == 8
    assert config.recurrent_steps == config.max_recurrent_steps == 8
    assert config.use_recurrence and config.use_static_kv_core
    assert config.use_bicameral_core is False
    assert config.kv_policy == "live"
    assert config.use_scratch and config.use_lane_carrier


def test_historical_c2_identity_rejects_active_or_nondefault_step2_fields() -> None:
    config = c2_current_toy_config()
    assert len(_config_identity_sha256(config)) == 64
    with pytest.raises(ValueError, match="forbids the bicameral"):
        _config_identity_sha256(
            replace(
                config,
                use_bicameral_core=True,
                use_static_kv_core=False,
                static_kv_midpoint_refresh=False,
            )
        )
    with pytest.raises(ValueError, match="inactive default K/V-policy"):
        _config_identity_sha256(replace(config, kv_policy="static"))


def test_c2_pf3_gate_excludes_valid_ineligible_zeros_and_passes(
    c2_receipt: C2PreflightReceipt,
) -> None:
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    receipt = c2_receipt

    assert torch.are_deterministic_algorithms_enabled() is previous_determinism
    assert receipt.trace_matches_main_forward_fp32
    assert receipt.authority_byte_verified
    assert receipt.authority_bytes == 15_575
    assert receipt.ratification_bytes == 2_233
    assert receipt.pf2_authority.endswith("#2-PF-2.2")
    assert receipt.pf2_authority_bytes == C2_PF2_AUTHORITY_BYTES
    assert receipt.pf2_authority_sha256 == C2_PF2_AUTHORITY_SHA256
    assert receipt.pf3_authority.endswith("#2-PF-3.2")
    assert receipt.pf3_authority_bytes == C2_PF3_AUTHORITY_BYTES
    assert receipt.pf3_authority_sha256 == C2_PF3_AUTHORITY_SHA256
    assert receipt.visits == 8
    assert len(receipt.per_visit) == 8
    assert tuple(item.visit for item in receipt.per_visit) == tuple(range(1, 9))
    assert receipt.block_split == (4, 2, 4)
    assert receipt.d_model == 64
    assert receipt.scratch_shape == (2, 8)
    assert "all_requires_grad_named_parameters" in receipt.gradient_population
    assert "eligible_zero_or_precision_mismatch_fails" in (
        receipt.relative_l2_denominator_policy
    )
    assert receipt.training_performed is False
    assert receipt.checkpoint_used is False
    assert "not_a_learned_checkpoint" in receipt.weight_state
    assert receipt.root_seed == 20_260_902
    assert receipt.config_identity_sha256 == (
        "57ef01bbaa4aa89ea18b098cbb51ece021fc0849f9b958435f7f534f2f6d8e28"
    )
    assert receipt.input_panel_sha256 == (
        "1bce335742f00f14c7e9abd88396776dc4d473fb955f60a923868618bf1d15f6"
    )
    assert receipt.initial_model_state_sha256 == (
        "8420c11aa7746c2191206cb061bd9bcbecf217fe264e0fc24cdb1761ee2dc2fb"
    )
    assert receipt.state_logit_relative_l2_threshold == 1e-2
    assert receipt.lane_gradient_relative_l2_threshold == 5e-2
    assert receipt.threshold_source_authority == receipt.pf2_authority
    assert receipt.thresholds_bound_after_data is True
    assert receipt.thresholds_preregistered is False
    assert "after_observing" in receipt.threshold_binding_disclosure
    assert receipt.threshold_applied is True
    assert receipt.threshold_passed is True
    assert receipt.catch_number is None
    assert receipt.catch_reason is None
    assert receipt.measurement_status.startswith(
        "cpu_current_integrated_composition_pf3_2_complete_gate_passed_"
    )
    assert "terminal_k8_passed" in receipt.measurement_status
    assert "PF-3.2_zero_reference_eligibility" in (
        receipt.threshold_metric_binding_status
    )
    assert receipt.full_weft1_toy_step_claim is False
    assert (
        receipt.representative_missing_full_toy_integrations
        == C2_REPRESENTATIVE_MISSING_FULL_TOY_INTEGRATIONS
    )
    assert (
        "integrated_learned_rotor_carrier"
        in receipt.representative_missing_full_toy_integrations
    )
    assert "Birkhoff".lower() in receipt.carrier_accumulation_decision.lower()
    assert all(cell.status == "deferred" for cell in receipt.deferred_gpu_cells)
    assert receipt.a100_hours == 0.0
    assert receipt.require_passed() is None
    assert receipt.summary.max_hidden_relative_l2 >= 0.0
    assert receipt.summary.max_scratch_lane_relative_l2 >= 0.0
    assert receipt.summary.max_logit_relative_l2 >= 0.0
    assert receipt.summary.max_full_gradient_relative_l2 >= 0.0
    assert receipt.summary.max_worst_module_gradient_relative_l2 is not None
    assert receipt.summary.max_hidden_relative_l2 == pytest.approx(
        0.002343670477777934, abs=5e-10
    )
    assert receipt.summary.max_scratch_lane_relative_l2 == pytest.approx(
        0.016916588480255675, abs=5e-10
    )
    assert receipt.summary.max_logit_relative_l2 == pytest.approx(
        0.0038046872811950652, abs=5e-10
    )
    assert receipt.summary.max_full_gradient_relative_l2 == pytest.approx(
        0.0071831022367519065, abs=5e-10
    )
    assert receipt.summary.max_worst_module_gradient_relative_l2 == pytest.approx(
        0.026736695789816682, abs=5e-10
    )
    assert (
        receipt.summary.max_hidden_relative_l2_visit,
        receipt.summary.max_scratch_lane_relative_l2_visit,
        receipt.summary.max_logit_relative_l2_visit,
        receipt.summary.max_full_gradient_relative_l2_visit,
        receipt.summary.max_worst_module_gradient_relative_l2_visit,
    ) == (8, 8, 5, 7, 2)
    assert (
        receipt.summary.max_worst_module_gradient_module
        == "core_blocks.0.attention.key_norm"
    )
    assert (
        receipt.summary.max_worst_module_gradient_parameter
        == "core_blocks.0.attention.key_norm.weight"
    )
    assert receipt.summary.max_relative_loss_drift == pytest.approx(
        7.33063034306664e-05, abs=5e-10
    )
    assert receipt.summary.max_relative_loss_drift_visit == 1

    assert receipt.summary.gradient_maxima_complete is True
    assert receipt.summary.zero_reference_failures == ()
    excluded = receipt.summary.zero_reference_cells
    assert tuple(
        (cell.visit, cell.module_name, cell.parameter_name)
        for cell in excluded
    ) == (
        (1, "reentry_bridge", "reentry_bridge.layer_scale"),
        (
            1,
            "reentry_bridge.prelude_norm",
            "reentry_bridge.prelude_norm.weight",
        ),
        (
            1,
            "reentry_bridge.projection",
            "reentry_bridge.projection.weight",
        ),
    )
    assert all(
        not cell.fp32_autograd_connected
        and not cell.bf16_autograd_connected
        and cell.fp32_l2 == 0.0
        and cell.bf16_compute_l2 == 0.0
        and cell.disposition == "ineligible (zero reference)"
        and cell.excluded_from_relative_error_population
        and not cell.structurally_eligible
        and cell.passed
        for cell in excluded
    )
    module_maxima = {
        item.module_name: item for item in receipt.summary.per_module_gradient_maxima
    }
    assert len(module_maxima) == 142
    for module_name in (
        "reentry_bridge",
        "reentry_bridge.prelude_norm",
        "reentry_bridge.projection",
    ):
        assert module_maxima[module_name].complete is True
        assert module_maxima[module_name].excluded_zero_reference_visits == (1,)
        assert module_maxima[module_name].failed_zero_reference_visits == ()
        assert module_maxima[module_name].max_relative_l2 is not None

    final_visit = receipt.per_visit[-1]
    assert final_visit.hidden.relative_l2_error == pytest.approx(
        0.002343670477777934, abs=5e-10
    )
    assert final_visit.scratch_lanes.relative_l2_error == pytest.approx(
        0.016916588480255675, abs=5e-10
    )
    assert final_visit.logits.relative_l2_error == pytest.approx(
        0.00366168723457278, abs=5e-10
    )
    assert final_visit.gradient.full_parameter_vector.relative_l2_error == pytest.approx(
        0.007065333916557898, abs=5e-10
    )
    assert final_visit.relative_loss_drift == pytest.approx(
        4.892405245362929e-05, abs=5e-10
    )

    assert final_visit.gradient.trainable_parameter_tensors == 144
    assert final_visit.gradient.trainable_parameter_elements == 484_795
    assert final_visit.gradient.complete is True
    assert final_visit.gradient.zero_reference_cells == ()
    assert final_visit.gradient.worst_module_name == "core_blocks.0.attention.key_norm"
    assert (
        final_visit.gradient.worst_parameter_name
        == "core_blocks.0.attention.key_norm.weight"
    )
    assert final_visit.gradient.worst_tensor is not None
    assert final_visit.gradient.worst_tensor.relative_l2_error == pytest.approx(
        0.023292915877290394, abs=5e-10
    )

    terminal = receipt.terminal_gate
    assert terminal.visit == 8
    assert terminal.metric == "vector_relative_l2_per_tensor"
    assert terminal.hidden_passed
    assert terminal.scratch_lanes_passed
    assert terminal.logits_passed
    assert terminal.full_gradient_passed
    assert terminal.gradient_population_complete
    assert terminal.every_module_worst_gradient_passed
    assert terminal.passed
    assert terminal.worst_module_gradient_relative_l2 is not None
    assert terminal.worst_module_gradient_relative_l2 <= 5e-2

    for visit in receipt.per_visit:
        assert visit.fp32_loss > 0.0 and visit.bf16_compute_loss > 0.0
        assert math.isfinite(visit.relative_loss_drift)
        for measurement in (
            visit.hidden,
            visit.scratch_lanes,
            visit.logits,
            visit.gradient.full_parameter_vector,
        ):
            assert measurement.reference_l2 > 0.0
            assert measurement.bf16_compute_l2 > 0.0
            assert math.isfinite(measurement.relative_l2_error)
            assert math.isfinite(measurement.relative_norm_drift)
            assert measurement.cosine_similarity is not None
            assert math.isfinite(measurement.cosine_similarity)


def test_c2_receipt_never_infers_an_integrated_rotor_from_the_scratch_carrier(
    c2_receipt: C2PreflightReceipt,
) -> None:
    receipt = c2_receipt

    assert "two_lane_birkhoff_scratch_carrier" in receipt.current_integrated_modules
    assert "integrated_learned_rotor_carrier" not in receipt.current_integrated_modules
    assert (
        "integrated_learned_rotor_carrier"
        in receipt.representative_missing_full_toy_integrations
    )
    assert receipt.carrier_accumulation_decision.startswith("deferred_")


def test_c2_receipt_rejects_forged_promotion_and_derived_state(
    c2_receipt: C2PreflightReceipt,
) -> None:
    receipt = c2_receipt
    mutations = (
        {"threshold_passed": False},
        {"catch_number": 34},
        {"catch_reason": "forged"},
        {
            "measurement_status": (
                "catch_34_pf2_2_zero_reference_population_incomplete_"
                "terminal_k8_passed_full_weft1_and_carrier_deferred"
            )
        },
        {
            "summary": replace(
                receipt.summary,
                gradient_maxima_complete=False,
            )
        },
        {"terminal_gate": replace(receipt.terminal_gate, passed=False)},
        {"authority_byte_verified": False},
        {"pf3_authority_sha256": "0" * 64},
        {"full_weft1_toy_step_claim": True},
        {"training_performed": True},
        {"checkpoint_used": True},
        {"trace_matches_main_forward_fp32": False},
        {"deferred_gpu_cells": ()},
        {"a100_hours": 999.0},
        {"current_composition": "full WEFT-1 integrated"},
        {"current_integrated_modules": ()},
        {"representative_missing_full_toy_integrations": ()},
        {"root_seed": -1},
        {"config_identity_sha256": "0" * 64},
        {"input_panel_sha256": "0" * 64},
        {"initial_model_state_sha256": "0" * 64},
        {"block_split": (9, 4, 9)},
        {"cpu_runtime": "forged"},
        {"torch_version": "forged"},
    )

    for updates in mutations:
        with pytest.raises(ValueError, match="C2"):
            replace(receipt, **updates)


def test_c2_pf3_zero_reference_cell_invariants_reject_forgery(
    c2_receipt: C2PreflightReceipt,
) -> None:
    cell = c2_receipt.summary.zero_reference_cells[0]

    for updates in (
        {"passed": False},
        {"disposition": "fail (eligible zero reference)"},
        {"excluded_from_relative_error_population": False},
        {"structurally_eligible": True},
        {"bf16_compute_l2": 1.0},
    ):
        with pytest.raises(ValueError, match="C2 zero-reference"):
            replace(cell, **updates)


def test_c2_pf3_eligible_zero_and_ineligible_precision_mismatch_fail() -> None:
    nonzero_reference = _GradientTraceTensor(
        parameter_name="core_blocks.0.weight",
        value=torch.ones(1),
        autograd_connected=True,
    )
    nonzero_observed = replace(nonzero_reference, value=torch.ones(1))
    eligible_zero_reference = _GradientTraceTensor(
        parameter_name="core_blocks.0.bias",
        value=torch.zeros(1),
        autograd_connected=True,
    )
    eligible_zero_observed = replace(
        eligible_zero_reference,
        value=torch.zeros(1),
    )
    eligible_failure = _gradient_drift(
        (nonzero_reference, eligible_zero_reference),
        (nonzero_observed, eligible_zero_observed),
        visit=1,
    )

    assert eligible_failure.complete is False
    assert eligible_failure.zero_reference_cells[0].disposition == (
        "fail (eligible zero reference)"
    )
    assert not eligible_failure.zero_reference_cells[0].passed

    ineligible_zero_reference = _GradientTraceTensor(
        parameter_name="reentry_bridge.layer_scale",
        value=torch.zeros(1),
        autograd_connected=False,
    )
    ineligible_nonzero_observed = replace(
        ineligible_zero_reference,
        value=torch.ones(1),
    )
    mismatch_failure = _gradient_drift(
        (nonzero_reference, ineligible_zero_reference),
        (nonzero_observed, ineligible_nonzero_observed),
        visit=1,
    )

    assert mismatch_failure.complete is False
    assert mismatch_failure.zero_reference_cells[0].disposition == (
        "fail (ineligible precision mismatch)"
    )
