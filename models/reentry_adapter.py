"""Identity-preserving adapters for recurrent loop re-entry."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralLowRankCorrection(nn.Module):
    """Non-expansive low-rank directional correction for loop re-entry.

    The effective map is a convex interpolation between identity and a
    spectrally-normalized low-rank map. That keeps the correction
    non-expansive under composition while making the initial perturbation tiny
    and gradient-live.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 8,
        max_depth: int = 8,
        theta_init: float = -8.0,
        init_std: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.max_depth = int(max_depth)
        self.eps = float(eps)
        self.U = nn.Parameter(torch.empty(hidden_size, rank))
        self.V = nn.Parameter(torch.empty(hidden_size, rank))
        self.theta = nn.Parameter(torch.full((max_depth,), float(theta_init)))
        self.register_buffer("power_v", torch.empty(hidden_size), persistent=True)
        self.reset_parameters(init_std=init_std, theta_init=theta_init)

    def reset_parameters(self, *, init_std: float = 1e-4, theta_init: float = -8.0) -> None:
        with torch.no_grad():
            self.U.normal_(mean=0.0, std=float(init_std))
            self.V.normal_(mean=0.0, std=float(init_std))
            self.theta.fill_(float(theta_init))
            self.power_v.normal_()
            self.power_v.div_(self.power_v.norm().clamp_min(self.eps))

    def _matrix_vector(self, vector: torch.Tensor) -> torch.Tensor:
        return self.U @ (self.V.t() @ vector)

    def _transpose_matrix_vector(self, vector: torch.Tensor) -> torch.Tensor:
        return self.V @ (self.U.t() @ vector)

    def _apply_low_rank(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (hidden_states @ self.V) @ self.U.t()

    def _depth_index(self, loop_idx: int | None) -> int:
        if loop_idx is None:
            return 0
        return min(max(int(loop_idx), 0), self.max_depth - 1)

    def strength(self, loop_idx: int | None = None) -> torch.Tensor:
        return torch.sigmoid(self.theta[self._depth_index(loop_idx)])

    def spectral_norm(self, *, num_iters: int = 1, update: bool = False) -> torch.Tensor:
        """Estimate the top singular value without forming the dense matrix."""

        v = self.power_v.detach().clone().to(device=self.U.device, dtype=self.U.dtype)
        for _ in range(max(int(num_iters), 1)):
            u = F.normalize(self._matrix_vector(v), dim=0, eps=self.eps)
            v = F.normalize(self._transpose_matrix_vector(u), dim=0, eps=self.eps)
        if update:
            with torch.no_grad():
                self.power_v.copy_(v.detach().to(device=self.power_v.device, dtype=self.power_v.dtype))
        sigma = self._matrix_vector(v).norm()
        return sigma

    def forward(self, hidden_states: torch.Tensor, loop_idx: int | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        work = hidden_states.to(dtype=self.U.dtype)
        sigma = self.spectral_norm(num_iters=1, update=self.training)
        normalized = self._apply_low_rank(work) / sigma.clamp_min(self.eps)
        gamma = self.strength(loop_idx).to(device=work.device, dtype=work.dtype)
        # Convex interpolation keeps the whole effective map non-expansive when
        # the normalized low-rank map has spectral norm <= 1.
        out = (1.0 - gamma) * work + gamma * normalized
        return out.to(dtype=input_dtype)

    def stats(self, loop_idx: int | None = None) -> dict[str, float]:
        with torch.no_grad():
            sigma = self.spectral_norm(num_iters=8, update=False)
            gamma = self.strength(loop_idx)
        return {
            "rank": float(self.rank),
            "sigma": float(sigma.detach().float().cpu()),
            "strength": float(gamma.detach().float().cpu()),
            "effective_spectral_bound": 1.0,
        }


class ReentryAffineAdapter(nn.Module):
    """A tiny loop-only affine correction.

    The adapter is initialized to exact identity, so enabling it does not change
    the initial recurrent computation. Unlike a zero-gated residual, both scale
    and bias receive gradients immediately when the adapter is used.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        spectral_rank: int = 8,
        spectral_max_depth: int = 8,
        spectral_theta_init: float = -8.0,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.spectral_correction = SpectralLowRankCorrection(
            hidden_size,
            rank=spectral_rank,
            max_depth=spectral_max_depth,
            theta_init=spectral_theta_init,
        )

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.scale.fill_(1.0)
            self.bias.zero_()
        self.spectral_correction.reset_parameters()

    def affine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        work_dtype = self.scale.dtype
        work = hidden_states.to(dtype=work_dtype)
        scale = self.scale.to(device=hidden_states.device, dtype=work_dtype)
        bias = self.bias.to(device=hidden_states.device, dtype=work_dtype)
        return (work * scale.view(1, 1, -1) + bias.view(1, 1, -1)).to(dtype=hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        loop_idx: int | None = None,
        mode: str = "affine",
    ) -> torch.Tensor:
        if mode == "affine":
            return self.affine(hidden_states)
        if mode == "spectral":
            return self.spectral_correction(hidden_states, loop_idx=loop_idx)
        if mode == "affine_spectral":
            return self.spectral_correction(self.affine(hidden_states), loop_idx=loop_idx)
        raise ValueError("mode must be one of: affine, spectral, affine_spectral")
