from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest
import torch

from models.ablation_lm.config import RATIFIED_TARGET_AUTHORITY_SHA256
from models.ablation_lm.observatory import (
    ObservatoryBlocked,
    ObservatoryGuard,
    T14B_MEASUREMENT_ALGORITHM_SHA256,
    T14bGradientEvidence,
    T14bReceipt,
    measure_t14b_future_gradients,
)


GRAPH_SHA = hashlib.sha256(b"weft1-test-graph").hexdigest()
CONFIG_SHA = hashlib.sha256(b"weft1-test-config").hexdigest()
STALE_GRAPH_SHA = hashlib.sha256(b"weft1-stale-graph").hexdigest()
K_VALUES = (1, 2, 4, 8)
# Generic test stages avoid claiming the unintegrated callosum or sidecar passed.
STAGES = ("stage_a", "stage_b")


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _evidence(
    *,
    k_values: tuple[int, ...] = K_VALUES,
    stages: tuple[str, ...] = STAGES,
    modes: tuple[str, ...] = ("packed", "padded"),
    nonzero_coordinate: tuple[int, str, str] | None = None,
    vacuous_coordinate: tuple[int, str, str] | None = None,
    alternate_manifest_coordinate: tuple[int, str, str] | None = None,
    alternate_hook_coordinate: tuple[int, str, str] | None = None,
) -> tuple[T14bGradientEvidence, ...]:
    rows = []
    for k_value in k_values:
        for stage in stages:
            for mode in modes:
                source = torch.linspace(-1.0, 1.0, 12).reshape(1, 4, 3)
                source = source.clone().requires_grad_(True)
                causal_segment_ids = (
                    torch.tensor([[0, 0, 1, 1]])
                    if mode == "packed"
                    else torch.tensor([[0, 0, -1, -1]])
                )
                if mode == "packed":
                    stage_output = torch.cat(
                        (source[:, :2].cumsum(dim=1), source[:, 2:].cumsum(dim=1)),
                        dim=1,
                    )
                else:
                    stage_output = torch.cat(
                        (source[:, :2].cumsum(dim=1), source[:, 2:] * 0.0),
                        dim=1,
                    )
                coordinate = (k_value, stage, mode)
                if coordinate == vacuous_coordinate:
                    stage_output = source * 0.0
                elif coordinate == nonzero_coordinate:
                    leak_mask = torch.zeros(1, 4, 1)
                    leak_mask[:, 1] = 1.0
                    stage_output = stage_output + leak_mask * source[:, -1:, :]
                rows.append(
                    measure_t14b_future_gradients(
                        stage=stage,
                        k_value=k_value,
                        coverage_mode=mode,
                        hook_identity=(
                            "model.alternate_hook"
                            if coordinate == alternate_hook_coordinate
                            else f"model.{stage}"
                        ),
                        input_manifest_sha256=_hash(
                            f"{mode}-alternate-input-manifest"
                            if coordinate == alternate_manifest_coordinate
                            else f"{mode}-input-manifest"
                        ),
                        causal_segment_ids=causal_segment_ids,
                        stage_output=stage_output,
                        sequence_source=source,
                    )
                )
    return tuple(rows)


def _receipt(
    *,
    graph_fingerprint: str = GRAPH_SHA,
    config_fingerprint: str = CONFIG_SHA,
    k_values: tuple[int, ...] = K_VALUES,
    stages: tuple[str, ...] = STAGES,
    modes: tuple[str, ...] = ("packed", "padded"),
    nonzero_coordinate: tuple[int, str, str] | None = None,
    vacuous_coordinate: tuple[int, str, str] | None = None,
) -> T14bReceipt:
    return T14bReceipt.from_gradient_evidence(
        authority_sha256=RATIFIED_TARGET_AUTHORITY_SHA256,
        graph_fingerprint=graph_fingerprint,
        config_fingerprint=config_fingerprint,
        tested_k_values=k_values,
        sequence_axis_stages=stages,
        evidence=_evidence(
            k_values=k_values,
            stages=stages,
            modes=modes,
            nonzero_coordinate=nonzero_coordinate,
            vacuous_coordinate=vacuous_coordinate,
        ),
    )


def _guard(receipt: T14bReceipt) -> ObservatoryGuard:
    return ObservatoryGuard(
        expected_graph_fingerprint=GRAPH_SHA,
        expected_config_fingerprint=CONFIG_SHA,
        expected_receipt_sha256=receipt.receipt_sha256,
        required_k_values=K_VALUES,
        required_sequence_axis_stages=STAGES,
    )


def _counting_callback(calls: list[str]) -> str:
    calls.append("called")
    return "statistic-result"


def test_t14b_receipt_is_immutable_and_hashes_canonical_tensor_evidence() -> None:
    first = _receipt()
    second = _receipt()

    assert first.passed is True
    assert first.liveness_passed is True
    assert first.max_future_gradient == 0.0
    assert first.minimum_allowed_gradient_max > 0.0
    assert all(
        row.measurement_algorithm_sha256 == T14B_MEASUREMENT_ALGORITHM_SHA256
        for row in _evidence()
    )
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.evidence_sha256 == second.evidence_sha256
    assert first.receipt_sha256 == hashlib.sha256(first.canonical_payload()).hexdigest()
    assert first.canonical_payload() == second.canonical_payload()
    with pytest.raises(FrozenInstanceError):
        first.passed = False  # type: ignore[misc]


def test_receipt_cannot_be_minted_by_asserting_passed_or_max_gradient() -> None:
    with pytest.raises(TypeError, match="minted from gradient evidence"):
        T14bReceipt()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="minted from gradient evidence"):
        T14bReceipt(passed=True, max_future_gradient=0.0)  # type: ignore[call-arg]


def test_missing_and_measured_failed_receipts_block_before_callback() -> None:
    passing = _receipt()
    guard = _guard(passing)
    calls: list[str] = []

    with pytest.raises(ObservatoryBlocked, match="required before observation"):
        guard.invoke(None, _counting_callback, calls)
    failed = _receipt(nonzero_coordinate=(4, "stage_b", "packed"))
    assert failed.passed is False
    assert failed.max_future_gradient == pytest.approx(1.0)
    with pytest.raises(ObservatoryBlocked, match="did not pass"):
        guard.invoke(failed, _counting_callback, calls)

    assert calls == []


def test_zero_connected_graph_fails_the_liveness_control() -> None:
    vacuous = _receipt(vacuous_coordinate=(2, "stage_a", "padded"))

    assert vacuous.max_future_gradient == 0.0
    assert vacuous.minimum_allowed_gradient_max == 0.0
    assert vacuous.liveness_passed is False
    assert vacuous.passed is False


def test_full_channel_basis_catches_projection_cancellation_and_cross_batch_leak() -> None:
    source = torch.randn(2, 4, 2, requires_grad=True)
    stage_output = source.cumsum(dim=1)
    leak_mask = torch.zeros(2, 4, 1)
    leak_mask[0, 1] = 1.0
    # Equal two-channel leakage is orthogonal to the old [-1, +1] projection.
    cancellation_leak = source[0, -1, 0].expand(2).view(1, 1, 2)
    # Also make batch 0 depend on a different batch item.
    cross_batch_leak = source[1, 0].view(1, 1, 2)
    stage_output = stage_output + leak_mask * (cancellation_leak + cross_batch_leak)

    evidence = measure_t14b_future_gradients(
        stage="stage_a",
        k_value=1,
        coverage_mode="packed",
        hook_identity="model.stage_a",
        input_manifest_sha256=_hash("adversarial-input-manifest"),
        causal_segment_ids=torch.tensor(
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            dtype=torch.long,
        ),
        stage_output=stage_output,
        sequence_source=source,
    )

    assert evidence.future_gradients.abs().max().item() == pytest.approx(1.0)
    assert evidence.allowed_gradients.abs().max().item() > 0.0


def test_probe_panel_is_derived_from_every_valid_position_including_terminal() -> None:
    packed = _evidence(k_values=(1,), stages=("stage_a",), modes=("packed",))[0]
    padded = _evidence(k_values=(1,), stages=("stage_a",), modes=("padded",))[0]

    assert packed.probe_positions == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert padded.probe_positions == ((0, 0), (0, 1))


@pytest.mark.parametrize(
    ("segment_ids", "coverage_mode", "message"),
    [
        (torch.tensor([[0, 1, 0]]), "packed", "exactly one contiguous run"),
        (torch.tensor([[-1, 0, -1]]), "padded", "contiguous edge run"),
    ],
)
def test_malformed_causal_segment_layout_is_rejected(
    segment_ids: torch.Tensor,
    coverage_mode: str,
    message: str,
) -> None:
    source = torch.randn(1, 3, 2, requires_grad=True)

    with pytest.raises(ValueError, match=message):
        measure_t14b_future_gradients(
            stage="stage_a",
            k_value=1,
            coverage_mode=coverage_mode,
            hook_identity="model.stage_a",
            input_manifest_sha256=_hash("malformed-segment-input-manifest"),
            causal_segment_ids=segment_ids,
            stage_output=source,
            sequence_source=source,
        )


def test_left_edge_padding_is_a_valid_contiguous_layout() -> None:
    source = torch.randn(1, 4, 2, requires_grad=True)

    evidence = measure_t14b_future_gradients(
        stage="stage_a",
        k_value=1,
        coverage_mode="padded",
        hook_identity="model.stage_a",
        input_manifest_sha256=_hash("left-padding-input-manifest"),
        causal_segment_ids=torch.tensor([[-1, -1, 0, 0]]),
        stage_output=source,
        sequence_source=source,
    )

    assert evidence.probe_positions == ((0, 2), (0, 3))
    assert evidence.future_gradients.abs().max().item() == 0.0


def test_packed_mask_treats_an_earlier_document_as_forbidden() -> None:
    source = torch.randn(1, 4, 2, requires_grad=True)
    # Deliberately leaky across the boundary: cumsum carries segment 0 into 1.
    stage_output = source.cumsum(dim=1)

    evidence = measure_t14b_future_gradients(
        stage="stage_a",
        k_value=2,
        coverage_mode="packed",
        hook_identity="model.stage_a",
        input_manifest_sha256=_hash("packed-boundary-input-manifest"),
        causal_segment_ids=torch.tensor([[0, 0, 1, 1]]),
        stage_output=stage_output,
        sequence_source=source,
    )

    assert evidence.future_gradients.abs().max().item() == pytest.approx(1.0)
    assert evidence.allowed_gradients.abs().max().item() == pytest.approx(1.0)


def test_stale_graph_receipt_blocks_before_callback() -> None:
    passing = _receipt()
    stale = _receipt(graph_fingerprint=STALE_GRAPH_SHA)
    guard = _guard(passing)
    calls: list[str] = []

    with pytest.raises(ObservatoryBlocked, match="graph fingerprint is stale"):
        guard.invoke(stale, _counting_callback, calls)

    assert calls == []


def test_missing_stage_coverage_blocks_before_callback() -> None:
    passing = _receipt()
    incomplete = _receipt(stages=STAGES[:-1])
    guard = _guard(passing)
    calls: list[str] = []

    with pytest.raises(ObservatoryBlocked, match="stage coverage"):
        guard.invoke(incomplete, _counting_callback, calls)

    assert calls == []


def test_registered_k_and_packing_coverage_cannot_be_waived() -> None:
    with pytest.raises(ValueError, match="registered K"):
        _receipt(k_values=(1, 2, 4))
    with pytest.raises(ValueError, match="both packed and padded"):
        _receipt(modes=("packed",))
    passing = _receipt()
    with pytest.raises(ValueError, match="requires K"):
        ObservatoryGuard(
            expected_graph_fingerprint=GRAPH_SHA,
            expected_config_fingerprint=CONFIG_SHA,
            expected_receipt_sha256=passing.receipt_sha256,
            required_k_values=(1, 2, 4),
            required_sequence_axis_stages=STAGES,
        )
    with pytest.raises(ValueError, match="cannot waive"):
        ObservatoryGuard(
            expected_graph_fingerprint=GRAPH_SHA,
            expected_config_fingerprint=CONFIG_SHA,
            expected_receipt_sha256=passing.receipt_sha256,
            required_k_values=K_VALUES,
            required_sequence_axis_stages=STAGES,
            require_padded=False,
        )


def test_valid_exact_receipt_allows_one_callback() -> None:
    passing = _receipt()
    guard = _guard(passing)
    calls: list[str] = []

    result = guard.invoke(passing, _counting_callback, calls)

    assert result == "statistic-result"
    assert calls == ["called"]


def test_evidence_factory_rejects_inexact_or_incomplete_panels() -> None:
    with pytest.raises(TypeError, match="exact tuple"):
        T14bReceipt.from_gradient_evidence(
            authority_sha256=RATIFIED_TARGET_AUTHORITY_SHA256,
            graph_fingerprint=GRAPH_SHA,
            config_fingerprint=CONFIG_SHA,
            tested_k_values=[1, 2],  # type: ignore[arg-type]
            sequence_axis_stages=STAGES,
            evidence=_evidence(k_values=(1, 2)),
        )
    with pytest.raises(ValueError, match="coverage cross-product"):
        T14bReceipt.from_gradient_evidence(
            authority_sha256=RATIFIED_TARGET_AUTHORITY_SHA256,
            graph_fingerprint=GRAPH_SHA,
            config_fingerprint=CONFIG_SHA,
            tested_k_values=K_VALUES,
            sequence_axis_stages=STAGES,
            evidence=_evidence()[:-1],
        )
    with pytest.raises(ValueError, match="share one input and probe panel"):
        T14bReceipt.from_gradient_evidence(
            authority_sha256=RATIFIED_TARGET_AUTHORITY_SHA256,
            graph_fingerprint=GRAPH_SHA,
            config_fingerprint=CONFIG_SHA,
            tested_k_values=K_VALUES,
            sequence_axis_stages=STAGES,
            evidence=_evidence(
                alternate_manifest_coordinate=(1, "stage_a", "packed")
            ),
        )
    with pytest.raises(ValueError, match="must use one hook identity"):
        T14bReceipt.from_gradient_evidence(
            authority_sha256=RATIFIED_TARGET_AUTHORITY_SHA256,
            graph_fingerprint=GRAPH_SHA,
            config_fingerprint=CONFIG_SHA,
            tested_k_values=K_VALUES,
            sequence_axis_stages=STAGES,
            evidence=_evidence(alternate_hook_coordinate=(1, "stage_a", "packed")),
        )
    with pytest.raises(TypeError, match="autograd measurer"):
        T14bGradientEvidence(
            stage="stage_a",
            k_value=1,
            coverage_mode="packed",
            future_gradients=torch.zeros(2, dtype=torch.int64),
        )  # type: ignore[call-arg]
    floating_stage = torch.zeros(1, 3, 2, requires_grad=True)
    integer_source = torch.zeros(1, 3, 2, dtype=torch.int64)
    with pytest.raises(TypeError, match="floating dtypes"):
        measure_t14b_future_gradients(
            stage="stage_a",
            k_value=1,
            coverage_mode="packed",
            hook_identity="model.stage_a",
            input_manifest_sha256=_hash("packed-input-manifest"),
            causal_segment_ids=torch.tensor([[0, 0, 1]]),
            stage_output=floating_stage,
            sequence_source=integer_source,
        )
