from __future__ import annotations

import math

import pytest
import torch

from analysis.weft1_preflight_c2 import (
    C2_AUTHORITY_SHA256,
    C2_GRADIENT_PARAMETER,
    C2_REPRESENTATIVE_MISSING_FULL_TOY_INTEGRATIONS,
    C2_RATIFIED_VISITS,
    C2_REGISTERED_RELATIVE_THRESHOLD,
    c2_current_toy_config,
    run_preflight_c2,
)


def test_c2_current_toy_configuration_is_exactly_4_2_4_d64_k8() -> None:
    config = c2_current_toy_config()

    assert C2_AUTHORITY_SHA256 == (
        "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
    )
    assert C2_REGISTERED_RELATIVE_THRESHOLD == 1e-2
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


def test_c2_cpu_check_measures_all_visits_and_fails_closed_on_missing_bindings() -> None:
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    receipt = run_preflight_c2()

    assert torch.are_deterministic_algorithms_enabled() is previous_determinism
    assert receipt.trace_matches_main_forward_fp32
    assert receipt.authority_byte_verified
    assert receipt.authority_bytes == 15_575
    assert receipt.ratification_bytes == 2_233
    assert receipt.visits == 8
    assert len(receipt.per_visit) == 8
    assert tuple(item.visit for item in receipt.per_visit) == tuple(range(1, 9))
    assert receipt.block_split == (4, 2, 4)
    assert receipt.d_model == 64
    assert receipt.scratch_shape == (2, 8)
    assert receipt.gradient_parameter == C2_GRADIENT_PARAMETER
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
    assert receipt.registered_relative_threshold == 1e-2
    assert receipt.threshold_source_ratified is True
    assert receipt.threshold_applied is False
    assert receipt.threshold_passed is None
    assert "underspecified" in receipt.threshold_metric_binding_status
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
    assert receipt.summary.max_hidden_relative_l2 >= 0.0
    assert receipt.summary.max_scratch_lane_relative_l2 >= 0.0
    assert receipt.summary.max_logit_relative_l2 >= 0.0
    assert receipt.summary.max_gradient_relative_l2 >= 0.0
    assert "descriptive_only" in receipt.summary.crossing_semantics
    assert receipt.summary.max_hidden_relative_l2 == pytest.approx(
        0.002378677322798898, abs=5e-10
    )
    assert receipt.summary.max_scratch_lane_relative_l2 == pytest.approx(
        0.016478415427698328, abs=5e-10
    )
    assert receipt.summary.max_logit_relative_l2 == pytest.approx(
        0.0037359662511258457, abs=5e-10
    )
    assert receipt.summary.max_gradient_relative_l2 == pytest.approx(
        0.01145752471878963, abs=5e-10
    )
    assert (
        receipt.summary.max_hidden_relative_l2_visit,
        receipt.summary.max_scratch_lane_relative_l2_visit,
        receipt.summary.max_logit_relative_l2_visit,
        receipt.summary.max_gradient_relative_l2_visit,
    ) == (8, 8, 6, 7)
    assert set(
        receipt.summary.candidate_relative_l2_crossings_of_registered_literal
    ) == {
        *(("gradient", visit) for visit in range(1, 9)),
        *(("scratch_lanes", visit) for visit in range(5, 9)),
    }
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
    assert final_visit.gradient.relative_l2_error == pytest.approx(
        0.010893679494000596, abs=5e-10
    )
    assert final_visit.relative_loss_drift == pytest.approx(
        1.4189866282529046e-05, abs=5e-10
    )
    assert final_visit.hidden.relative_norm_drift == pytest.approx(
        0.00010911398835916895, abs=5e-10
    )
    assert final_visit.scratch_lanes.relative_norm_drift == pytest.approx(
        0.007337845706321525, abs=5e-10
    )
    assert final_visit.logits.relative_norm_drift == pytest.approx(
        0.00035368685698889015, abs=5e-10
    )
    assert final_visit.gradient.relative_norm_drift == pytest.approx(
        0.0026168309043040064, abs=5e-10
    )

    for visit in receipt.per_visit:
        assert visit.fp32_loss > 0.0 and visit.bf16_compute_loss > 0.0
        assert math.isfinite(visit.relative_loss_drift)
        for measurement in (
            visit.hidden,
            visit.scratch_lanes,
            visit.logits,
            visit.gradient,
        ):
            assert measurement.reference_l2 > 0.0
            assert measurement.bf16_compute_l2 > 0.0
            assert math.isfinite(measurement.relative_l2_error)
            assert math.isfinite(measurement.relative_norm_drift)
            assert math.isfinite(measurement.cosine_similarity)


def test_c2_receipt_never_infers_an_integrated_rotor_from_the_scratch_carrier() -> None:
    receipt = run_preflight_c2()

    assert "two_lane_birkhoff_scratch_carrier" in receipt.current_integrated_modules
    assert "integrated_learned_rotor_carrier" not in receipt.current_integrated_modules
    assert (
        "integrated_learned_rotor_carrier"
        in receipt.representative_missing_full_toy_integrations
    )
    assert receipt.carrier_accumulation_decision.startswith("deferred_")
