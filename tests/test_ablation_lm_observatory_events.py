from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest
import torch

from models.ablation_lm.config import RATIFIED_TARGET_AUTHORITY_SHA256
from models.ablation_lm.observatory import (
    ObservatoryBlocked,
    ObservatoryGuard,
    T14bReceipt,
    measure_t14b_future_gradients,
)
from models.ablation_lm.observatory_events import (
    EVENT_KIND,
    INSTRUMENT_REGISTRY,
    InstrumentTier,
    ObservatoryEvent,
    PrecisionRecord,
    emit_observatory_event,
)


GRAPH_SHA = hashlib.sha256(b"weft1-event-graph").hexdigest()
CONFIG_SHA = hashlib.sha256(b"weft1-event-config").hexdigest()
REGISTRATION_SHA = hashlib.sha256(b"weft1-event-registration").hexdigest()
K_VALUES = (1, 2, 4, 8)
STAGES = ("core",)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _receipt() -> T14bReceipt:
    evidence = []
    for k_value in K_VALUES:
        for mode in ("packed", "padded"):
            source = torch.linspace(-1.0, 1.0, 8).reshape(1, 4, 2).requires_grad_(True)
            segments = (
                torch.tensor([[0, 0, 1, 1]])
                if mode == "packed"
                else torch.tensor([[0, 0, -1, -1]])
            )
            output = torch.cat(
                (source[:, :2].cumsum(1), source[:, 2:].cumsum(1)),
                dim=1,
            )
            if mode == "padded":
                output = torch.cat((output[:, :2], output[:, 2:] * 0.0), dim=1)
            evidence.append(
                measure_t14b_future_gradients(
                    stage="core",
                    k_value=k_value,
                    coverage_mode=mode,
                    hook_identity="model.core",
                    input_manifest_sha256=_hash(f"{mode}-manifest"),
                    causal_segment_ids=segments,
                    stage_output=output,
                    sequence_source=source,
                )
            )
    return T14bReceipt.from_gradient_evidence(
        authority_sha256=RATIFIED_TARGET_AUTHORITY_SHA256,
        graph_fingerprint=GRAPH_SHA,
        config_fingerprint=CONFIG_SHA,
        tested_k_values=K_VALUES,
        sequence_axis_stages=STAGES,
        evidence=tuple(evidence),
    )


def _guard(receipt: T14bReceipt) -> ObservatoryGuard:
    return ObservatoryGuard(
        expected_graph_fingerprint=GRAPH_SHA,
        expected_config_fingerprint=CONFIG_SHA,
        expected_receipt_sha256=receipt.receipt_sha256,
        required_k_values=K_VALUES,
        required_sequence_axis_stages=STAGES,
    )


def _precision() -> PrecisionRecord:
    return PrecisionRecord(
        effective_n=128.0,
        null_id="matched_smooth_noise",
        interval_low=0.7,
        interval_high=0.9,
    )


def _delta_measurements() -> dict[str, object]:
    return {
        "band_index": 0,
        "visit_index": 3,
        "rho": 0.1,
        "disagreement_eigenvalue": 0.8,
        "steps": 3,
        "amplitude_law_exponent": 3,
        "energy_law_exponent": 6,
        "expected_amplitude_retention": 0.512,
        "observed_amplitude_retention": 0.511,
        "expected_energy_retention": 0.262144,
        "observed_energy_retention": 0.261121,
        "amplitude_absolute_error": 0.001,
        "energy_absolute_error": 0.001023,
        "scope": "callosum_only_excludes_intervening_core_dynamics",
    }


def test_ratified_tier_registry_has_exact_eight_and_latest_binding() -> None:
    tier_one = {
        name
        for name, spec in INSTRUMENT_REGISTRY.items()
        if spec.tier is InstrumentTier.DECISION
    }

    assert tier_one == {
        "t14_causality",
        "obs_inv",
        "carrier_retention",
        "mu_r_exponent",
        "loop_marginal_gain",
        "resp_leak",
        "g_inv",
        "t5_rotor_signature",
    }
    assert INSTRUMENT_REGISTRY["a_state"].tier is InstrumentTier.DIAGNOSTIC
    with pytest.raises(TypeError):
        INSTRUMENT_REGISTRY["resp_leak"] = INSTRUMENT_REGISTRY["a_state"]  # type: ignore[index]


def test_event_cannot_bypass_guarded_internal_factory() -> None:
    with pytest.raises(TypeError, match="guarded internal factory"):
        ObservatoryEvent(
            instrument_id="a_state",
            tier=InstrumentTier.DIAGNOSTIC,
            run_id="weft1.seed0",
            model_rung="proxy",
            checkpoint_step=0,
            seed=0,
            cohort_id="bringup",
            graph_fingerprint=GRAPH_SHA,
            config_fingerprint=CONFIG_SHA,
            t14b_receipt_sha256=_hash("fake-receipt"),
            authority_sha256=RATIFIED_TARGET_AUTHORITY_SHA256,
            measurements_frozen=(),  # type: ignore[arg-type]
            precision=_precision(),
        )


def test_tier_one_event_is_derived_hash_bound_and_immutable() -> None:
    receipt = _receipt()
    event = emit_observatory_event(
        _guard(receipt),
        receipt,
        instrument_id="resp_leak",
        run_id="weft1.seed0",
        model_rung="proxy",
        checkpoint_step=400,
        seed=0,
        cohort_id="joint_tuning_dev",
        measurements={
            "sidecar_only_delta_q_disabled": 0.8,
            "checkpoint_delta_q_disabled": 0.6,
            "retained_fraction": 0.75,
        },
        registration_sha256=REGISTRATION_SHA,
        branch_outcome="separation_retained",
    )

    payload = event.as_dict()
    assert payload["kind"] == EVENT_KIND
    assert payload["instrument_tier"] == "tier_1_decision"
    assert payload["t14b_receipt_sha256"] == receipt.receipt_sha256
    assert payload["event_sha256"] == hashlib.sha256(event.canonical_payload()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        event.tier = InstrumentTier.DIAGNOSTIC  # type: ignore[misc]


def test_tier_two_requires_precision_and_cannot_carry_decision_fields() -> None:
    receipt = _receipt()
    kwargs = dict(
        guard=_guard(receipt),
        receipt=receipt,
        instrument_id="a_state",
        run_id="weft1.seed0",
        model_rung="proxy",
        checkpoint_step=400,
        seed=0,
        cohort_id="diagnostic_dev",
        measurements={
            "intervention": "cross_example",
            "numerator": 0.4,
            "denominator": 0.8,
            "ratio_unclipped": 0.5,
        },
    )

    with pytest.raises(ValueError, match="effective n, null, and interval"):
        emit_observatory_event(**kwargs)
    with pytest.raises(ValueError, match="cannot carry decision"):
        emit_observatory_event(
            **kwargs,
            precision=_precision(),
            registration_sha256=REGISTRATION_SHA,
            branch_outcome="promoted",
        )
    event = emit_observatory_event(**kwargs, precision=_precision())
    assert event.tier is InstrumentTier.DIAGNOSTIC
    assert event.precision is not None


def test_missing_or_stale_t14b_blocks_before_measurement_validation() -> None:
    receipt = _receipt()
    guard = _guard(receipt)

    with pytest.raises(ObservatoryBlocked, match="required before observation"):
        emit_observatory_event(
            guard,
            None,
            instrument_id="not_registered",
            run_id="invalid id",
            model_rung="proxy",
            checkpoint_step=0,
            seed=0,
            cohort_id="dev",
            measurements={},
        )


def test_required_fields_finiteness_and_t14_receipt_ownership_fail_closed() -> None:
    receipt = _receipt()
    guard = _guard(receipt)
    common = dict(
        guard=guard,
        receipt=receipt,
        instrument_id="delta_mode_energy",
        run_id="weft1.seed0",
        model_rung="proxy",
        checkpoint_step=0,
        seed=0,
        cohort_id="bringup",
        precision=_precision(),
    )

    with pytest.raises(ValueError, match="missing fields"):
        emit_observatory_event(**common, measurements={"rho": 0.1})
    with pytest.raises(ValueError, match="finite"):
        invalid = _delta_measurements()
        invalid["observed_energy_retention"] = float("nan")
        emit_observatory_event(
            **common,
            measurements=invalid,
        )
    with pytest.raises(ValueError, match="recorded by T14bReceipt"):
        emit_observatory_event(
            guard,
            receipt,
            instrument_id="t14_causality",
            run_id="weft1.seed0",
            model_rung="proxy",
            checkpoint_step=0,
            seed=0,
            cohort_id="bringup",
            measurements={
                "max_future_gradient": 0.0,
                "minimum_allowed_gradient_max": 1.0,
            },
            registration_sha256=REGISTRATION_SHA,
            branch_outcome="passed",
        )


def test_measurement_payload_is_defensively_frozen() -> None:
    receipt = _receipt()
    measurements = {
        "invocation_rate_by_step": [0.0, 0.0, 0.25, 0.5],
        "kappa_alignment": 0.7,
        "empty_array": [],
        "empty_object": {},
        "string_pairs": [["a", 1], ["b", 2]],
        "nested": {"values": [1, 2]},
    }
    event = emit_observatory_event(
        _guard(receipt),
        receipt,
        instrument_id="g_inv",
        run_id="weft1.seed0",
        model_rung="proxy",
        checkpoint_step=0,
        seed=0,
        cohort_id="bringup",
        measurements=measurements,
        registration_sha256=REGISTRATION_SHA,
        branch_outcome="conditionality_retained",
    )
    original_hash = event.event_sha256

    measurements["invocation_rate_by_step"][2] = 1.0
    measurements["nested"]["values"][0] = 99

    assert event.measurements["invocation_rate_by_step"] == [0.0, 0.0, 0.25, 0.5]
    assert event.measurements["empty_array"] == []
    assert event.measurements["empty_object"] == {}
    assert event.measurements["string_pairs"] == [["a", 1], ["b", 2]]
    assert event.measurements["nested"] == {"values": [1, 2]}
    thawed = event.measurements
    thawed["nested"]["values"][0] = -1
    assert event.measurements["nested"] == {"values": [1, 2]}
    assert event.event_sha256 == original_hash
    assert event.event_sha256 == hashlib.sha256(event.canonical_payload()).hexdigest()


def test_delta_mode_schema_records_distinct_amplitude_and_energy_laws() -> None:
    receipt = _receipt()
    event = emit_observatory_event(
        _guard(receipt),
        receipt,
        instrument_id="delta_mode_energy",
        run_id="weft1.seed0",
        model_rung="proxy",
        checkpoint_step=0,
        seed=0,
        cohort_id="bringup",
        measurements=_delta_measurements(),
        precision=_precision(),
    )

    assert event.measurements["amplitude_law_exponent"] == 3
    assert event.measurements["energy_law_exponent"] == 6
    assert "(1-2rho)^K" in INSTRUMENT_REGISTRY["delta_mode_energy"].branch_rule
    assert "(1-2rho)^(2K)" in INSTRUMENT_REGISTRY["delta_mode_energy"].branch_rule

    invalid = _delta_measurements()
    invalid["energy_law_exponent"] = 3
    with pytest.raises(ValueError, match="energy law exponent must equal 2K"):
        emit_observatory_event(
            _guard(receipt),
            receipt,
            instrument_id="delta_mode_energy",
            run_id="weft1.seed0",
            model_rung="proxy",
            checkpoint_step=0,
            seed=0,
            cohort_id="bringup",
            measurements=invalid,
            precision=_precision(),
        )
