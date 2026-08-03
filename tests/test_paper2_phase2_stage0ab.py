from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from eval.eval_paper2_phase2_exp0a import (
    _canonical_statistics,
    _fit_probe,
    _fit_projector,
    _probe_metrics,
    _transform,
)
from training.paper2_phase2_stage0ab import (
    WHITEN_EPS_ABS,
    CanonicalizerTransform,
    SharedResidualFlowPilot,
    affine_interpolate,
    build_anchor_targets,
    document_split,
    effective_eigenvalues,
    finite_quantiles,
    probability_scale_coherence,
    safe_coarse_lattice_metrics,
)


def test_future_whitening_default_and_health_assertion() -> None:
    assert WHITEN_EPS_ABS == 1e-6
    with pytest.raises(ValueError, match="fit-health"):
        effective_eigenvalues(torch.tensor([1e-4, 1e-8]))


ROOT = Path(__file__).resolve().parents[1]


def test_finite_quantiles_counts_nonfinite_values_without_poisoning_summary() -> None:
    result = finite_quantiles([0.0, 1.0, float("inf"), float("-inf"), float("nan")])
    assert result["count"] == 5
    assert result["finite_count"] == 2
    assert result["positive_infinity_count"] == 1
    assert result["negative_infinity_count"] == 1
    assert result["nan_count"] == 1
    assert result["finite_mean"] == pytest.approx(0.5)
    assert result["finite_median"] == pytest.approx(0.5)


def test_safe_coarse_metrics_report_support_miss_and_remain_finite() -> None:
    student = torch.tensor([math.log(0.8), float("-inf"), math.log(0.2)])
    teacher = torch.tensor([math.log(0.2), math.log(0.6), math.log(0.2)])
    result = safe_coarse_lattice_metrics(
        student_log_probs=student,
        teacher_log_probs=[teacher],
        student_topk_mask=torch.tensor([True, False, False]),
    )
    assert math.isfinite(result["student_gap_coarse_kl_clipped"])
    assert result["student_support_miss_mass"] == pytest.approx(0.6)
    assert result["teachability_student_topk"] == pytest.approx(0.2)


def test_probability_scale_coherence_ignores_log_zero_cells() -> None:
    seven = torch.tensor([math.log(0.8), math.log(0.2), float("-inf")])
    fourteen = torch.tensor([math.log(0.6), math.log(0.3), math.log(0.1)])
    thirty_two = torch.tensor([math.log(0.4), math.log(0.4), math.log(0.2)])
    value = probability_scale_coherence([seven, fourteen, thirty_two])
    assert value is not None
    assert math.isfinite(value)
    assert -1.0 <= value <= 1.0


def test_document_split_is_deterministic_and_document_disjoint() -> None:
    documents = ["a", "a", "b", "c", "d", "d", "e"]
    first = document_split(documents, calibration_fraction=0.8, seed=17)
    second = document_split(documents, calibration_fraction=0.8, seed=17)
    assert torch.equal(first, second)
    for document in set(documents):
        indices = [index for index, value in enumerate(documents) if value == document]
        assert len({bool(first[index]) for index in indices}) == 1
    assert first.any() and (~first).any()


def test_effective_eigenvalue_floor_is_applied_once() -> None:
    raw = torch.tensor([10.0, 1.0, 1e-8])
    effective = effective_eigenvalues(raw, tau=1e-4, eps_abs=1e-8)
    assert torch.allclose(effective, torch.tensor([10.0, 1.0, 1e-3]))
    transform = CanonicalizerTransform(
        projector_weight=torch.eye(3),
        teacher_mean=torch.zeros(3),
        canonical_mean=torch.zeros(1, 3),
        whiten_basis=torch.eye(3),
        whiten_eigenvalues=effective,
        whiten_alpha=1.0,
        layer_weights=torch.tensor([1.0]),
        n_slots=1,
        latent_dim=3,
    )
    result = transform(torch.tensor([[[1.0, 1.0, 1.0]]]))
    expected = torch.tensor([[[1 / math.sqrt(10), 1.0, 1 / math.sqrt(1e-3)]]])
    assert torch.allclose(result, expected)


def test_alpha_zero_keeps_shared_basis_without_equalization() -> None:
    basis = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    transform = CanonicalizerTransform(
        projector_weight=torch.eye(2),
        teacher_mean=torch.zeros(2),
        canonical_mean=torch.zeros(1, 2),
        whiten_basis=basis,
        whiten_eigenvalues=torch.tensor([4.0, 0.25]),
        whiten_alpha=0.0,
        layer_weights=torch.tensor([1.0]),
        n_slots=1,
        latent_dim=2,
    )
    normalized = torch.tensor([2.0, 3.0]) / torch.sqrt(torch.tensor(6.5 + 1e-6))
    expected = normalized.flip(0).view(1, 1, 2)
    assert torch.allclose(transform(torch.tensor([[[2.0, 3.0]]])), expected)


def test_anchor_targets_fill_four_future_slots_and_leave_span_slots_masked() -> None:
    topk_ids = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long
    )
    topk_log_probs = torch.log(torch.tensor([[0.6, 0.4]]).repeat(4, 1))
    middle = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
    targets, mask = build_anchor_targets(
        topk_ids=topk_ids,
        topk_log_probs=topk_log_probs,
        middle_states=middle,
        horizons=torch.tensor([1, 2, 3, 4]),
        anchor_indices=torch.zeros(4, dtype=torch.long),
        anchor_count=1,
        latent_dim=4,
        n_slots=8,
        seed=11,
    )
    assert targets.shape == (1, 8, 4)
    assert mask.tolist() == [[True, True, True, True, False, False, False, False]]
    assert torch.count_nonzero(targets[:, 4:]) == 0


def test_anchor_target_construction_is_gradient_isolated() -> None:
    topk_ids = torch.arange(16).view(4, 4)
    topk_log_probs = torch.randn(4, 4, requires_grad=True)
    middle_states = torch.randn(4, 6, requires_grad=True)
    targets, _mask = build_anchor_targets(
        topk_ids=topk_ids,
        topk_log_probs=topk_log_probs,
        middle_states=middle_states,
        horizons=torch.tensor([1, 2, 3, 4]),
        anchor_indices=torch.zeros(4, dtype=torch.long),
        anchor_count=1,
        latent_dim=4,
        n_slots=8,
        seed=9,
    )
    assert not targets.requires_grad


def test_affine_interpolation_does_not_renormalize_the_path() -> None:
    start = torch.tensor([[2.0, 0.0]])
    stop = torch.tensor([[0.0, 4.0]])
    midpoint = affine_interpolate(start, stop, 0.5)
    assert torch.equal(midpoint, torch.tensor([[1.0, 2.0]]))
    assert midpoint.norm() != pytest.approx(start.norm())


def test_flow_pilot_enforces_loop_cap_and_uses_residual_update() -> None:
    module = SharedResidualFlowPilot(latent_dim=4, context_dim=4, max_steps=4)
    state = torch.randn(2, 3, 4)
    context = torch.randn(2, 4)
    output = module(state, context, steps=4)
    assert output.shape == state.shape
    with pytest.raises(ValueError, match="loop cap"):
        module(state, context, steps=5)


def test_exp0a_linear_fit_probe_and_alpha_path_smoke() -> None:
    torch.manual_seed(23)
    x = torch.randn(40, 16)
    y = torch.randn(40, 8 * 128)
    teacher_mean, projector, _receipt = _fit_projector(
        x[:32], y[:32], method="predictive_rrr", seed=23
    )
    canonical_mean, basis, eigenvalues, condition = _canonical_statistics(
        x[:32], teacher_mean, projector
    )
    assert condition["effective"] <= 10_000.01
    z = _transform(
        x,
        teacher_mean=teacher_mean,
        projector=projector,
        canonical_mean=canonical_mean,
        basis=basis,
        eigenvalues=eigenvalues,
        alpha=0.5,
    )
    horizons = torch.tensor(([1, 2, 3, 4] * 10), dtype=torch.long)
    hidden = torch.randn(40, 32)
    decoder, bias = _fit_probe(z[:32], hidden[:32], horizons[:32])
    topk_ids = torch.arange(8).repeat(8, 1)
    topk_log_probs = torch.log_softmax(torch.randn(8, 8), dim=1)
    metrics = _probe_metrics(
        z=z[32:],
        decoder=decoder,
        decoder_bias=bias,
        hidden=hidden[32:],
        topk_ids=topk_ids,
        topk_log_probs=topk_log_probs,
        horizons=horizons[32:],
        observed_token_ids=topk_ids[:, 0],
        lm_head=torch.randn(8, 32),
        batch_size=4,
    )
    assert metrics["samples"] == 8
    assert math.isfinite(metrics["future_topk_kl_mean"])


def test_stage0ab_launchers_are_separate_and_preserve_dev_only_boundaries() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    expected = {
        "paper2_phase2_stage0a_repair": (
            "STAGE5_PAPER2_PHASE2_STAGE0A_REPAIR_CELL.py",
            "CPU-only cached lattice repair no model inference no training",
        ),
        "paper2_phase2_exp0a": (
            "STAGE5_PAPER2_PHASE2_EXP0A_CELL.py",
            "DEV-only canonicalizer and whitening screening no backbone training",
        ),
        "paper2_phase2_exp0b": (
            "STAGE5_PAPER2_PHASE2_EXP0B_CELL.py",
            "DEV-only interpolation and serial-flow geometry screening",
        ),
    }
    for target, markers in expected.items():
        assert target in bootstrap
        for marker in markers:
            assert marker in bootstrap
    for filename in (
        "colab/run_stage5_paper2_phase2_stage0a_repair.py",
        "colab/run_stage5_paper2_phase2_exp0a.py",
        "colab/run_stage5_paper2_phase2_exp0b.py",
    ):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "frozen_evaluation_partitions_touched" in text
        assert "DEV-C" in text
