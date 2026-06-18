"""SVGD-style particle updates for recurrent latent trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .halting import masked_mean


@dataclass(frozen=True)
class SVGDStats:
    bandwidth: torch.Tensor
    mean_pairwise_distance: torch.Tensor
    drift_rms: torch.Tensor
    repulsion_rms_pre_clip: torch.Tensor
    repulsion_rms: torch.Tensor
    repulsion_clip_fraction: torch.Tensor
    velocity_rms: torch.Tensor


def svgd_particle_update(
    previous_state: torch.Tensor,
    standard_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_particles: int,
    eps: float = 1.0,
    repulsion_scale: float = 1.0,
    bandwidth: str = "median",
    bandwidth_floor: float = 1e-6,
    repulsion_max_norm: Optional[float] = None,
    kernel_projection_dim: Optional[int] = None,
    projection_seed: int = 0,
) -> tuple[torch.Tensor, SVGDStats]:
    """Apply one learned-drift SVGD update to flattened particle states.

    Args:
        previous_state: Tensor shaped ``[B*K, T, D]`` before recurrent block.
        standard_state: Tensor shaped ``[B*K, T, D]`` after the standard recurrent block.
        attention_mask: Optional flattened padding mask shaped ``[B*K, T]``.
        num_particles: Particle count K.

    With ``K=1`` and ``eps=1`` this returns ``standard_state`` exactly.
    """

    if num_particles < 1:
        raise ValueError("num_particles must be >= 1")
    if previous_state.shape != standard_state.shape:
        raise ValueError("previous_state and standard_state must have the same shape")

    if num_particles == 1:
        zero = standard_state.new_zeros(())
        stats = SVGDStats(
            bandwidth=zero,
            mean_pairwise_distance=zero,
            drift_rms=_rms((standard_state - previous_state).float()).to(device=standard_state.device),
            repulsion_rms_pre_clip=zero,
            repulsion_rms=zero,
            repulsion_clip_fraction=zero,
            velocity_rms=_rms((standard_state - previous_state).float()).to(device=standard_state.device),
        )
        if float(eps) == 1.0:
            return standard_state, stats
        updated = previous_state.float() + float(eps) * (standard_state - previous_state).float()
        return updated.to(dtype=standard_state.dtype), stats

    flat_batch, seq_len, hidden_dim = previous_state.shape
    if flat_batch % num_particles != 0:
        raise ValueError(
            "Flattened batch must be divisible by num_particles. "
            f"Got batch={flat_batch}, num_particles={num_particles}."
        )

    batch_size = flat_batch // num_particles
    work_prev = previous_state.float()
    work_std = standard_state.float()
    drift = work_std - work_prev

    pooled = masked_mean(work_std, attention_mask).view(batch_size, num_particles, hidden_dim)
    drift_particles = drift.view(batch_size, num_particles, seq_len, hidden_dim)
    projection = _projection_matrix(
        hidden_dim=hidden_dim,
        projection_dim=kernel_projection_dim,
        seed=projection_seed,
        device=work_std.device,
    )
    kernel_pooled = pooled @ projection if projection is not None else pooled

    diff = kernel_pooled.unsqueeze(2) - kernel_pooled.unsqueeze(1)
    sq_dist = diff.pow(2).sum(dim=-1)
    h = _resolve_bandwidth(sq_dist, num_particles, bandwidth, bandwidth_floor)
    kernel = torch.exp(-sq_dist / h)

    attraction = torch.einsum("bij,bjtd->bitd", kernel, drift_particles) / float(num_particles)
    repulsion_pre_clip = (2.0 / h) * (kernel.unsqueeze(-1) * diff).sum(dim=2) / float(num_particles)
    if projection is not None:
        repulsion_pre_clip = repulsion_pre_clip @ projection.t()
    repulsion, clip_fraction = _clamp_repulsion(repulsion_pre_clip, repulsion_max_norm)
    repulsion_tokens = repulsion.unsqueeze(2).expand(batch_size, num_particles, seq_len, hidden_dim)
    repulsion_tokens_pre_clip = repulsion_pre_clip.unsqueeze(2).expand(
        batch_size,
        num_particles,
        seq_len,
        hidden_dim,
    )

    velocity = attraction + float(repulsion_scale) * repulsion_tokens
    updated = work_prev.view(batch_size, num_particles, seq_len, hidden_dim) + float(eps) * velocity

    stats = SVGDStats(
        bandwidth=h.detach().reshape(()),
        mean_pairwise_distance=_mean_pairwise_distance(sq_dist, num_particles).detach(),
        drift_rms=_rms(drift_particles).detach(),
        repulsion_rms_pre_clip=_rms(repulsion_tokens_pre_clip).detach(),
        repulsion_rms=_rms(repulsion_tokens).detach(),
        repulsion_clip_fraction=clip_fraction.detach(),
        velocity_rms=_rms(velocity).detach(),
    )
    return updated.view(flat_batch, seq_len, hidden_dim).to(dtype=standard_state.dtype), stats


def _resolve_bandwidth(
    sq_dist: torch.Tensor,
    num_particles: int,
    bandwidth: str,
    floor: float,
) -> torch.Tensor:
    if bandwidth != "median":
        raise ValueError("Only median SVGD bandwidth is implemented")
    off_diag = _off_diagonal_values(sq_dist, num_particles)
    h = off_diag.detach().median() / max(math.log(float(num_particles)), 1e-9)
    return h.clamp_min(float(floor))


def _off_diagonal_values(matrix: torch.Tensor, num_particles: int) -> torch.Tensor:
    mask = ~torch.eye(num_particles, dtype=torch.bool, device=matrix.device)
    return matrix[:, mask]


def _mean_pairwise_distance(sq_dist: torch.Tensor, num_particles: int) -> torch.Tensor:
    off_diag = _off_diagonal_values(sq_dist, num_particles)
    return off_diag.clamp_min(0.0).sqrt().mean()


_PROJECTION_CACHE: dict[tuple[str, int, int, int], torch.Tensor] = {}


def _projection_matrix(
    hidden_dim: int,
    projection_dim: Optional[int],
    seed: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if projection_dim is None or projection_dim <= 0 or projection_dim >= hidden_dim:
        return None
    key = (str(device), hidden_dim, int(projection_dim), int(seed))
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    projection = torch.randn(
        hidden_dim,
        int(projection_dim),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    projection = projection / math.sqrt(float(projection_dim))
    _PROJECTION_CACHE[key] = projection
    return projection


def _clamp_repulsion(repulsion: torch.Tensor, max_norm: Optional[float]) -> tuple[torch.Tensor, torch.Tensor]:
    if max_norm is None or max_norm <= 0:
        return repulsion, repulsion.new_zeros(())
    norm = repulsion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (float(max_norm) / norm).clamp_max(1.0)
    clip_fraction = (scale < 1.0).float().mean()
    return repulsion * scale, clip_fraction


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pow(2).mean().sqrt()
