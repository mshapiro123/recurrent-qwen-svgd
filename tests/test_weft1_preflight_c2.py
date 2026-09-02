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
    C2_REPRESENTATIVE_MISSING_FULL_TOY_INTEGRATIONS,
    C2_RATIFIED_VISITS,
    C2_STATE_LOGIT_RELATIVE_L2_THRESHOLD,
    C2GateCatch,
    C2PreflightReceipt,
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
    assert config.use_scratch and config.use_lane_carrier


def test_c2_pf2_gate_reports_defined_metrics_and_returns_zero_denominator_catch(
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
    assert receipt.visits == 8
    assert len(receipt.per_visit) == 8
    assert tuple(item.visit for item in receipt.per_visit) == tuple(range(1, 9))
    assert receipt.block_split == (4, 2, 4)
    assert receipt.d_model == 64
    assert receipt.scratch_shape == (2, 8)
    assert "all_requires_grad_named_parameters" in receipt.gradient_population
    assert "fail_closed_if_zero" in receipt.relative_l2_denominator_policy
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
        "52b993089afe982dd4027eb16500f761b0ed8cfc49a938de7f00e2e102831aac"
    )
    assert receipt.state_logit_relative_l2_threshold == 1e-2
    assert receipt.lane_gradient_relative_l2_threshold == 5e-2
    assert receipt.threshold_source_authority == receipt.pf2_authority
    assert receipt.thresholds_bound_after_data is True
    assert receipt.thresholds_preregistered is False
    assert "after_observing" in receipt.threshold_binding_disclosure
    assert receipt.threshold_applied is True
    assert receipt.threshold_passed is False
    assert receipt.catch_number == 34
    assert receipt.measurement_status.startswith("catch_34_")
    assert "terminal_k8_passed" in receipt.measurement_status
    assert receipt.catch_reason is not None
    assert "zero-reference" in receipt.catch_reason
    assert "zero_reference_eligibility_unbound" in receipt.threshold_metric_binding_status
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
    with pytest.raises(C2GateCatch, match=r"CATCH #34"):
        receipt.require_passed()
    assert receipt.summary.max_hidden_relative_l2 >= 0.0
    assert receipt.summary.max_scratch_lane_relative_l2 >= 0.0
    assert receipt.summary.max_logit_relative_l2 >= 0.0
    assert receipt.summary.max_full_gradient_relative_l2 >= 0.0
    assert receipt.summary.max_worst_module_gradient_relative_l2 is not None
    assert receipt.summary.max_hidden_relative_l2 == pytest.approx(
        0.002378677322798898, abs=5e-10
    )
    assert receipt.summary.max_scratch_lane_relative_l2 == pytest.approx(
        0.016478415427698328, abs=5e-10
    )
    assert receipt.summary.max_logit_relative_l2 == pytest.approx(
        0.0037359662511258457, abs=5e-10
    )
    assert receipt.summary.max_full_gradient_relative_l2 == pytest.approx(
        0.0071749515521762446, abs=5e-10
    )
    assert receipt.summary.max_worst_module_gradient_relative_l2 == pytest.approx(
        0.06243657691629686, abs=5e-10
    )
    assert (
        receipt.summary.max_hidden_relative_l2_visit,
        receipt.summary.max_scratch_lane_relative_l2_visit,
        receipt.summary.max_logit_relative_l2_visit,
        receipt.summary.max_full_gradient_relative_l2_visit,
        receipt.summary.max_worst_module_gradient_relative_l2_visit,
    ) == (8, 8, 6, 8, 4)
    assert receipt.summary.max_worst_module_gradient_module == "engram"
    assert (
        receipt.summary.max_worst_module_gradient_parameter
        == "engram.gate_bias"
    )
    assert receipt.summary.max_relative_loss_drift == pytest.approx(
        6.992828237628056e-05, abs=5e-10
    )
    assert receipt.summary.max_relative_loss_drift_visit == 4

    assert receipt.summary.gradient_maxima_complete is False
    undefined = receipt.summary.undefined_gradient_cells
    assert tuple(
        (cell.visit, cell.module_name, cell.parameter_name)
        for cell in undefined
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
        for cell in undefined
    )
    module_maxima = {
        item.module_name: item for item in receipt.summary.per_module_gradient_maxima
    }
    assert len(module_maxima) == 141
    for module_name in (
        "reentry_bridge",
        "reentry_bridge.prelude_norm",
        "reentry_bridge.projection",
    ):
        assert module_maxima[module_name].complete is False
        assert module_maxima[module_name].undefined_visits == (1,)
        assert module_maxima[module_name].max_relative_l2 is not None

    final_visit = receipt.per_visit[-1]
    assert final_visit.hidden.relative_l2_error == pytest.approx(
        0.002378677322798898, abs=5e-10
    )
    assert final_visit.scratch_lanes.relative_l2_error == pytest.approx(
        0.016478415427698328, abs=5e-10
    )
    assert final_visit.logits.relative_l2_error == pytest.approx(
        0.003659068615870767, abs=5e-10
    )
    assert final_visit.gradient.full_parameter_vector.relative_l2_error == pytest.approx(
        0.0071749515521762446, abs=5e-10
    )
    assert final_visit.relative_loss_drift == pytest.approx(
        1.4189866282529046e-05, abs=5e-10
    )

    assert final_visit.gradient.trainable_parameter_tensors == 143
    assert final_visit.gradient.trainable_parameter_elements == 488_859
    assert final_visit.gradient.complete is True
    assert final_visit.gradient.undefined_relative_l2_cells == ()
    assert final_visit.gradient.worst_module_name == "engram"
    assert (
        final_visit.gradient.worst_parameter_name
        == "engram.raw_residual_scale"
    )
    assert final_visit.gradient.worst_tensor is not None
    assert final_visit.gradient.worst_tensor.relative_l2_error == pytest.approx(
        0.02583730846608379, abs=5e-10
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
    assert receipt.summary.max_worst_module_gradient_relative_l2 > 5e-2
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
        {"threshold_passed": True},
        {"catch_number": None},
        {"catch_reason": None},
        {
            "measurement_status": (
                "cpu_current_integrated_composition_pf2_2_complete_gate_passed_"
                "terminal_k8_passed_full_weft1_and_carrier_deferred"
            )
        },
        {
            "summary": replace(
                receipt.summary,
                gradient_maxima_complete=True,
            )
        },
        {"terminal_gate": replace(receipt.terminal_gate, passed=False)},
        {"authority_byte_verified": False},
    )

    for updates in mutations:
        with pytest.raises(ValueError, match="C2"):
            replace(receipt, **updates)
