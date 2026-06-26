"""Eval-only low-rank tail damping for recurrent re-entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def strength_scaled_damper(damper_scale: torch.Tensor, strength: float, *, eps: float = 1e-8) -> torch.Tensor:
    """Interpolate from identity to the calibrated damper in log scale."""

    clamped = damper_scale.float().clamp(min=eps, max=1.0)
    strength_value = max(0.0, min(float(strength), 1.0))
    return torch.exp(torch.log(clamped) * strength_value)


def apply_tail_damper(
    hidden_states: torch.Tensor,
    *,
    mean: torch.Tensor,
    basis: torch.Tensor,
    damper_scale: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Attenuate calibrated tail coordinates while preserving the orthogonal complement."""

    if strength <= 0:
        return hidden_states
    input_dtype = hidden_states.dtype
    device = hidden_states.device
    work = hidden_states.float()
    mean = mean.to(device=device, dtype=work.dtype).view(1, 1, -1)
    basis = basis.to(device=device, dtype=work.dtype)
    scale = strength_scaled_damper(damper_scale.to(device=device), strength).to(device=device, dtype=work.dtype)
    centered = work - mean
    coeff = centered @ basis
    delta = (coeff * (scale - 1.0).view(1, 1, -1)) @ basis.t()
    return (work + delta).to(dtype=input_dtype)


def load_tail_damper(path: str | Path) -> dict[str, torch.Tensor | Any]:
    """Load and validate a saved re-entry tail-damper artifact."""

    payload = torch.load(Path(path).expanduser(), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Tail damper artifact must be a dictionary")
    required = ("mean", "basis", "damper_scale")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Tail damper artifact missing required keys: {missing}")
    mean = payload["mean"].detach().float().contiguous()
    basis = payload["basis"].detach().float().contiguous()
    damper_scale = payload["damper_scale"].detach().float().contiguous()
    if mean.dim() != 1:
        raise ValueError(f"Tail damper mean must be 1D, got {tuple(mean.shape)}")
    if basis.dim() != 2:
        raise ValueError(f"Tail damper basis must be 2D, got {tuple(basis.shape)}")
    if damper_scale.dim() != 1:
        raise ValueError(f"Tail damper scale must be 1D, got {tuple(damper_scale.shape)}")
    if basis.shape[0] != mean.shape[0] or basis.shape[1] != damper_scale.shape[0]:
        raise ValueError(
            "Tail damper artifact shape mismatch: "
            f"mean={tuple(mean.shape)}, basis={tuple(basis.shape)}, scale={tuple(damper_scale.shape)}"
        )
    return {
        **payload,
        "mean": mean,
        "basis": basis,
        "damper_scale": damper_scale,
    }

