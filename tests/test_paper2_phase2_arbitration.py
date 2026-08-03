from __future__ import annotations

import pytest
import torch

from eval.eval_paper2_phase2_canonicalizer_arbitration import (
    arbitration_decision,
    eigenvalue_floor_support,
    paired_bootstrap_ci,
)
from eval.eval_paper2_phase2_exp0a import _pool


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    delta = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    first = paired_bootstrap_ci(delta, seed=17, replicates=1000)
    second = paired_bootstrap_ci(delta, seed=17, replicates=1000)

    assert first == second
    assert first["mean"] == pytest.approx(0.25)
    assert first["ci95_low"] > 0


def test_arbitration_requires_nonzero_ci_and_stratum_sign_consistency() -> None:
    assert arbitration_decision(
        agreement_ci=(0.001, 0.01), stratum_deltas={"code": 0.01, "text": 0.02}
    )["primary"] == "learned_mixture_rrr"
    assert arbitration_decision(
        agreement_ci=(-0.001, 0.01), stratum_deltas={"code": 0.01, "text": 0.02}
    )["primary"] == "uniform_mixture_rrr"
    assert arbitration_decision(
        agreement_ci=(0.001, 0.01), stratum_deltas={"code": 0.01, "text": -0.02}
    )["primary"] == "uniform_mixture_rrr"


def test_fit_noise_comparable_to_prior_edge_forces_parsimony() -> None:
    decision = arbitration_decision(
        agreement_ci=(0.001, 0.01),
        stratum_deltas={"code": 0.01, "text": 0.02},
        fit_noise_comparable_to_previous_edge=True,
    )
    assert decision["primary"] == "uniform_mixture_rrr"
    assert "fit_noise" in decision["reason"]


def test_floor_support_counts_clamped_eigenvalues() -> None:
    effective = torch.tensor([4.0, 1.0, 4e-4, 4e-4])
    report = eigenvalue_floor_support(effective, tau=1e-4)
    assert report["at_floor_count"] == 2
    assert report["floored_fraction"] == pytest.approx(0.5)
    assert report["raw_fraction_recoverable"] is False


def test_chunked_layer_pool_matches_direct_formula() -> None:
    torch.manual_seed(23)
    states = torch.randn(11, 3, 7).to(torch.bfloat16)
    weights = torch.tensor([0.584, 0.332, 0.084])
    observed = _pool(states, weights, chunk_size=3)
    values = states.float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
    expected = (normalized * (weights / weights.sum()).view(1, 3, 1)).sum(dim=1)
    assert torch.equal(observed, expected)
