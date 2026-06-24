"""Similarity-sensitive diversity diagnostics for recurrent pathway states."""

from __future__ import annotations

import math
from typing import Iterable

import torch


DEFAULT_QS = (0.0, 1.0, 2.0, math.inf)


def pairwise_squared_distances(states: torch.Tensor) -> torch.Tensor:
    """Return pairwise squared Euclidean distances for ``[N, D]`` states."""

    if states.dim() != 2:
        raise ValueError(f"states must be rank-2 [N, D], got shape={tuple(states.shape)}")
    diff = states.float().unsqueeze(1) - states.float().unsqueeze(0)
    return diff.pow(2).sum(dim=-1)


def median_nearest_neighbor_sigma(
    states: torch.Tensor,
    *,
    bw_factor: float = 2.0,
    sigma_floor: float = 1e-6,
) -> torch.Tensor:
    """Bandwidth from the median nearest-neighbor scale.

    This is the local scale requested by the effective-pathway diagnostic. It
    intentionally ignores global pairwise spread, which is often dominated by
    task or prompt identity rather than within-prompt particle structure.
    """

    if states.shape[0] <= 1:
        return states.new_tensor(float(sigma_floor), dtype=torch.float32)
    d2 = pairwise_squared_distances(states)
    eye = torch.eye(d2.shape[0], dtype=torch.bool, device=d2.device)
    masked = d2.masked_fill(eye, float("inf"))
    nn = masked.min(dim=1).values.clamp_min(0.0).sqrt()
    finite = nn[torch.isfinite(nn)]
    if finite.numel() == 0:
        return states.new_tensor(float(sigma_floor), dtype=torch.float32)
    sigma = float(bw_factor) * finite.median()
    return sigma.clamp_min(float(sigma_floor)).to(dtype=torch.float32)


def gaussian_similarity(
    states: torch.Tensor,
    *,
    sigma: torch.Tensor | float | None = None,
    bw_factor: float = 2.0,
    sigma_floor: float = 1e-6,
) -> torch.Tensor:
    """Gaussian similarity matrix with bandwidth from local particle scale."""

    work = states.float()
    if work.shape[0] == 0:
        raise ValueError("states must contain at least one pathway")
    resolved_sigma = (
        median_nearest_neighbor_sigma(work, bw_factor=bw_factor, sigma_floor=sigma_floor)
        if sigma is None
        else torch.as_tensor(sigma, dtype=torch.float32, device=work.device).clamp_min(float(sigma_floor))
    )
    d2 = pairwise_squared_distances(work)
    return torch.exp(-d2 / (2.0 * resolved_sigma.pow(2)))


def effective_pathways_from_similarity(
    similarity: torch.Tensor,
    *,
    qs: Iterable[float] = DEFAULT_QS,
    probabilities: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Leinster-Cobbold similarity-sensitive diversity for a similarity matrix.

    The returned keys are strings so JSON summaries preserve the special
    ``inf`` order unambiguously.
    """

    if similarity.dim() != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"similarity must be square, got shape={tuple(similarity.shape)}")
    n = similarity.shape[0]
    if n == 0:
        raise ValueError("similarity must be non-empty")
    work = similarity.float().clamp_min(0.0)
    if probabilities is None:
        p = torch.full((n,), 1.0 / float(n), dtype=torch.float32, device=work.device)
    else:
        if probabilities.shape != (n,):
            raise ValueError(f"probabilities must have shape {(n,)}, got {tuple(probabilities.shape)}")
        p = probabilities.float().to(device=work.device)
        p = p / p.sum().clamp_min(float(eps))
    zp = (work @ p).clamp_min(float(eps))
    out: dict[str, float] = {}
    for q_value in qs:
        q = float(q_value)
        if math.isinf(q):
            value = 1.0 / zp.max()
            key = "inf"
        elif abs(q - 1.0) < 1e-9:
            value = torch.exp(-(p * torch.log(zp)).sum())
            key = "1"
        else:
            value = (p * zp.pow(q - 1.0)).sum().pow(1.0 / (1.0 - q))
            key = str(int(q)) if q.is_integer() else f"{q:g}"
        out[key] = float(value.detach().cpu())
    return out


def effective_pathways(
    states: torch.Tensor,
    *,
    qs: Iterable[float] = DEFAULT_QS,
    bw_factor: float = 2.0,
    sigma_floor: float = 1e-6,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute effective pathway counts and bandwidth diagnostics for states."""

    sigma = median_nearest_neighbor_sigma(states, bw_factor=bw_factor, sigma_floor=sigma_floor)
    similarity = gaussian_similarity(states, sigma=sigma, sigma_floor=sigma_floor)
    diversity = effective_pathways_from_similarity(similarity, qs=qs)
    diagnostics = {
        "sigma": float(sigma.detach().cpu()),
        "mean_similarity": float(similarity.float().mean().detach().cpu()),
        "median_nearest_neighbor": float((sigma / float(bw_factor)).detach().cpu()),
    }
    if states.shape[0] > 1:
        d2 = pairwise_squared_distances(states)
        mask = ~torch.eye(states.shape[0], dtype=torch.bool, device=states.device)
        diagnostics["mean_pairwise_distance"] = float(d2[mask].clamp_min(0.0).sqrt().mean().detach().cpu())
    else:
        diagnostics["mean_pairwise_distance"] = 0.0
    return diversity, diagnostics

