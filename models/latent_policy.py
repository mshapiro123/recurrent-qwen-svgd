"""Stochastic latent trajectory modules for recurrent-depth Qwen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .halting import masked_mean


@dataclass
class LatentStats:
    mu: torch.Tensor
    logvar: torch.Tensor
    sample: torch.Tensor
    kl: torch.Tensor


class LatentPolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first_linear = self.net[0]
        x = self.net(pooled.to(dtype=first_linear.weight.dtype))
        return self.mu(x), self.logvar(x)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
    if not sample:
        return mu
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


def standard_normal_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(u|x) || N(0, I)) per example."""

    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


class LatentAdapter(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        latent_scale_init: float = 0.01,
        adapter_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_dim)
        self.latent_scale = nn.Parameter(torch.tensor(float(latent_scale_init)))
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=adapter_std)
            nn.init.zeros_(self.proj.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(dtype=self.proj.weight.dtype)
        scale = self.latent_scale.to(device=latent.device, dtype=latent.dtype)
        return scale * self.proj(latent)


class LatentTrajectoryModule(nn.Module):
    """Sample and inject one latent variable per sequence per recurrent pass."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 256,
        latent_scale_init: float = 0.01,
        adapter_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.policy = LatentPolicyHead(hidden_dim, latent_dim)
        self.adapter = LatentAdapter(latent_dim, hidden_dim, latent_scale_init, adapter_std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        sample: bool = True,
    ) -> tuple[torch.Tensor, LatentStats]:
        pooled = masked_mean(hidden_states, attention_mask)
        mu, logvar = self.policy(pooled)
        latent = reparameterize(mu, logvar, sample=sample)
        latent_delta = self.adapter(latent).unsqueeze(1)
        stats = LatentStats(
            mu=mu,
            logvar=logvar,
            sample=latent,
            kl=standard_normal_kl(mu, logvar),
        )
        return hidden_states + latent_delta.to(dtype=hidden_states.dtype), stats
