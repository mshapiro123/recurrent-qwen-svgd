from __future__ import annotations

import pytest
import torch

from models.phase_g_guidance import (
    PhaseGGuidance,
    balanced_diagonal_gaussian_kl,
    extract_trajectory_candidates,
)


def test_guidance_uses_only_prior_at_inference_and_distinct_seed_streams() -> None:
    guidance = PhaseGGuidance(hidden_dim=8, latent_dim=3, projection_seed=19).eval()
    states = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.long)

    first = guidance(states, mask, trajectory_seeds=[101, 202])
    repeated = guidance(states, mask, trajectory_seeds=[101, 202])

    assert first.source == "prior"
    assert first.seed_manifest == [101, 202]
    assert torch.equal(first.injected_states, repeated.injected_states)
    assert not torch.equal(first.latent_samples[0], first.latent_samples[1])


def test_guidance_factor_one_is_identity_and_multiplier_scales_only_residual() -> None:
    guidance = PhaseGGuidance(
        hidden_dim=8,
        latent_dim=3,
        projection_seed=19,
        injection_scale_init=0.1,
    ).eval()
    states = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.long)

    default = guidance(states, mask, trajectory_seeds=[101, 202])
    factor_one = guidance(
        states,
        mask,
        trajectory_seeds=[101, 202],
        injection_multiplier=1.0,
    )
    factor_three = guidance(
        states,
        mask,
        trajectory_seeds=[101, 202],
        injection_multiplier=3.0,
    )

    assert torch.equal(default.injected_states, factor_one.injected_states)
    assert torch.equal(default.latent_samples, factor_three.latent_samples)
    assert torch.allclose(factor_three.injection_scale, 3.0 * default.injection_scale)
    assert torch.allclose(
        factor_three.injected_states - states,
        3.0 * (default.injected_states - states),
        atol=1e-6,
    )


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("inf"), float("nan")])
def test_guidance_refuses_invalid_injection_multiplier(multiplier: float) -> None:
    guidance = PhaseGGuidance(hidden_dim=8, latent_dim=3).eval()
    with pytest.raises(ValueError, match="finite and positive"):
        guidance(
            torch.randn(1, 2, 8),
            None,
            injection_multiplier=multiplier,
        )


def test_posterior_conditioning_is_structurally_unreachable_at_inference() -> None:
    guidance = PhaseGGuidance(hidden_dim=8, latent_dim=3).eval()
    states = torch.randn(1, 4, 8)
    targets = torch.randn(1, 8)

    with pytest.raises(RuntimeError, match="training-only"):
        guidance(states, None, posterior_targets=targets, use_posterior=True)


def test_posterior_training_path_and_balanced_kl_are_finite() -> None:
    guidance = PhaseGGuidance(hidden_dim=8, latent_dim=3).train()
    states = torch.randn(2, 4, 8)
    targets = torch.randn(2, 8)
    output = guidance(
        states,
        None,
        posterior_targets=targets,
        use_posterior=True,
        trajectory_seeds=[11],
    )

    assert output.source == "posterior"
    assert output.posterior is not None
    kl = balanced_diagonal_gaussian_kl(
        output.posterior,
        output.prior,
        balance=0.8,
    )
    assert kl.shape == (2,)
    assert torch.isfinite(kl).all()


def test_candidate_extraction_refuses_missing_unpooled_logits() -> None:
    class Output:
        logits = torch.randn(1, 3, 7)
        trajectory_logits = None

    with pytest.raises(RuntimeError, match="unpooled"):
        extract_trajectory_candidates(Output())


def test_candidate_extraction_preserves_distinct_trajectory_sets() -> None:
    class Output:
        logits = torch.randn(1, 1, 5)
        trajectory_logits = torch.tensor(
            [[[[0.0, 9.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 9.0, 0.0]]]]
        )

    candidates = extract_trajectory_candidates(Output())

    assert candidates.tolist() == [[1, 3]]
