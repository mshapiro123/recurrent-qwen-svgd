"""Guided stochastic re-entry components for Phase G-alpha."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .halting import masked_mean


@dataclass
class DiagonalGaussian:
    mean: torch.Tensor
    logvar: torch.Tensor


@dataclass
class PhaseGGuidanceOutput:
    injected_states: torch.Tensor
    latent_samples: torch.Tensor
    prior: DiagonalGaussian
    posterior: DiagonalGaussian | None
    source: str
    seed_manifest: list[int]
    injection_scale: torch.Tensor


class ConditionalGaussianHead(nn.Module):
    """A small normalized head producing a diagonal Gaussian."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden = max(latent_dim * 2, input_dim // 2)
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, values: torch.Tensor) -> DiagonalGaussian:
        work = values.to(dtype=self.norm.weight.dtype)
        hidden = self.net(self.norm(work))
        return DiagonalGaussian(
            mean=self.mean(hidden),
            logvar=self.logvar(hidden).clamp(-12.0, 8.0),
        )


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(float(value)))


def fixed_orthonormal_projection(
    hidden_dim: int,
    latent_dim: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if latent_dim < 1 or latent_dim > hidden_dim:
        raise ValueError("latent_dim must be in [1, hidden_dim]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(hidden_dim, latent_dim, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.to(dtype=dtype).contiguous()


def seeded_reparameterize(
    distribution: DiagonalGaussian,
    trajectory_seeds: Sequence[int],
) -> tuple[torch.Tensor, list[int]]:
    seeds = [int(seed) for seed in trajectory_seeds]
    if not seeds:
        raise ValueError("trajectory_seeds cannot be empty")
    samples = []
    for row_idx in range(distribution.mean.shape[0]):
        seed = seeds[row_idx % len(seeds)]
        generator = torch.Generator(device=distribution.mean.device)
        generator.manual_seed(seed)
        epsilon = torch.randn(
            distribution.mean[row_idx].shape,
            generator=generator,
            device=distribution.mean.device,
            dtype=distribution.mean.dtype,
        )
        samples.append(
            distribution.mean[row_idx]
            + torch.exp(0.5 * distribution.logvar[row_idx]) * epsilon
        )
    return torch.stack(samples, dim=0), seeds


def diagonal_gaussian_kl(
    posterior: DiagonalGaussian,
    prior: DiagonalGaussian,
) -> torch.Tensor:
    variance_ratio = torch.exp(posterior.logvar - prior.logvar)
    mean_term = (posterior.mean - prior.mean).pow(2) * torch.exp(-prior.logvar)
    return 0.5 * (
        prior.logvar
        - posterior.logvar
        + variance_ratio
        + mean_term
        - 1.0
    ).sum(dim=-1)


def balanced_diagonal_gaussian_kl(
    posterior: DiagonalGaussian,
    prior: DiagonalGaussian,
    *,
    balance: float,
) -> torch.Tensor:
    """KL balancing: prior and posterior receive separately weighted gradients."""

    if not 0.0 <= balance <= 1.0:
        raise ValueError("balance must be in [0, 1]")
    posterior_detached = DiagonalGaussian(
        posterior.mean.detach(),
        posterior.logvar.detach(),
    )
    prior_detached = DiagonalGaussian(
        prior.mean.detach(),
        prior.logvar.detach(),
    )
    prior_learning = diagonal_gaussian_kl(posterior_detached, prior)
    posterior_learning = diagonal_gaussian_kl(posterior, prior_detached)
    return float(balance) * prior_learning + (1.0 - float(balance)) * posterior_learning


def extract_trajectory_candidates(output: Any) -> torch.Tensor:
    """Return one token candidate per trajectory without consulting pooled logits."""

    logits = getattr(output, "trajectory_logits", None)
    if logits is None:
        raise RuntimeError(
            "Phase G candidate extraction requires unpooled trajectory_logits; "
            "pooled output.logits is not an admissible fallback."
        )
    if logits.dim() != 4:
        raise ValueError(
            "trajectory_logits must be [batch, trajectories, sequence, vocabulary]"
        )
    return logits[:, :, -1, :].argmax(dim=-1)


class PhaseGGuidance(nn.Module):
    """Standalone guided transition used by the wrapper and CPU contract tests."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 64,
        *,
        projection_seed: int = 20260717,
        injection_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.phase_g_prior_head = ConditionalGaussianHead(hidden_dim, latent_dim)
        self.phase_g_posterior_head = ConditionalGaussianHead(2 * hidden_dim, latent_dim)
        self.phase_g_injection_scale = nn.Parameter(
            torch.tensor(inverse_softplus(injection_scale_init), dtype=torch.float32)
        )
        self.register_buffer(
            "phase_g_projection",
            fixed_orthonormal_projection(hidden_dim, latent_dim, seed=projection_seed),
            persistent=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        posterior_targets: torch.Tensor | None = None,
        use_posterior: bool = False,
        trajectory_seeds: Sequence[int] = (0,),
        injection_multiplier: float = 1.0,
    ) -> PhaseGGuidanceOutput:
        if use_posterior and not self.training:
            raise RuntimeError("The target-conditioned posterior is training-only")
        if posterior_targets is not None and not self.training:
            raise RuntimeError("Posterior targets are training-only and forbidden at inference")
        if not math.isfinite(float(injection_multiplier)) or float(injection_multiplier) <= 0.0:
            raise ValueError("injection_multiplier must be finite and positive")

        pooled = masked_mean(hidden_states, attention_mask)
        prior = self.phase_g_prior_head(pooled)
        posterior = None
        source = prior
        source_name = "prior"
        if use_posterior:
            if posterior_targets is None:
                raise ValueError("use_posterior=True requires posterior_targets")
            if posterior_targets.shape != pooled.shape:
                raise ValueError(
                    "posterior_targets must match pooled hidden shape; "
                    f"got {tuple(posterior_targets.shape)} versus {tuple(pooled.shape)}"
                )
            posterior = self.phase_g_posterior_head(
                torch.cat([pooled, posterior_targets.to(dtype=pooled.dtype)], dim=-1)
            )
            source = posterior
            source_name = "posterior"

        samples, seeds = seeded_reparameterize(source, trajectory_seeds)
        projection = self.phase_g_projection.to(device=samples.device, dtype=samples.dtype)
        scale = F.softplus(self.phase_g_injection_scale).to(
            device=samples.device,
            dtype=samples.dtype,
        ) * float(injection_multiplier)
        delta = (samples @ projection.transpose(0, 1)).unsqueeze(1)
        return PhaseGGuidanceOutput(
            injected_states=hidden_states + (scale * delta).to(dtype=hidden_states.dtype),
            latent_samples=samples,
            prior=prior,
            posterior=posterior,
            source=source_name,
            seed_manifest=seeds,
            injection_scale=scale,
        )
