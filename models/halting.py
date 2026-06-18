"""Sequence-level PonderNet halting utilities."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def masked_mean(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean-pool sequence states over non-padding tokens.

    Args:
        hidden_states: Tensor shaped ``[batch, seq_len, hidden]``.
        attention_mask: Usually shaped ``[batch, seq_len]`` with 1 for real
            tokens and 0 for padding. If ``None``, all tokens are averaged.
    """

    if attention_mask is None:
        return hidden_states.mean(dim=1)

    if attention_mask.dim() != 2:
        raise ValueError(
            "masked_mean expects a 2D padding mask shaped [batch, seq_len]. "
            f"Got shape {tuple(attention_mask.shape)}."
        )

    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(eps)
    return (hidden_states * mask).sum(dim=1) / denom


class SequenceHaltingPredictor(nn.Module):
    """Predicts a single halt probability per sequence and recurrent pass."""

    def __init__(self, hidden_size: int, initial_halt_prob: float = 0.25) -> None:
        super().__init__()
        if not 0.0 < initial_halt_prob < 1.0:
            raise ValueError("initial_halt_prob must be in (0, 1)")

        self.proj = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            self.proj.bias.fill_(math.log(initial_halt_prob / (1.0 - initial_halt_prob)))

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        pooled = pooled_hidden.to(dtype=self.proj.weight.dtype)
        return torch.sigmoid(self.proj(pooled))


def pondernet_halting_probabilities(
    halt_probs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert per-pass halt probabilities into PonderNet lambda weights.

    The final loop absorbs all remaining survival probability, so the returned
    probabilities sum to one even if the final predicted halt probability is low.

    Args:
        halt_probs: Tensor shaped ``[..., num_loops]``.
    """

    if halt_probs.shape[-1] < 1:
        raise ValueError("halt_probs must contain at least one loop")

    probs = halt_probs.clamp(min=eps, max=1.0 - eps)
    num_loops = probs.shape[-1]
    if num_loops == 1:
        return torch.ones_like(probs)

    survival = torch.ones_like(probs[..., 0])
    weights = []
    for idx in range(num_loops):
        if idx == num_loops - 1:
            weights.append(survival)
        else:
            weight = survival * probs[..., idx]
            weights.append(weight)
            survival = survival * (1.0 - probs[..., idx])

    lambdas = torch.stack(weights, dim=-1)
    return lambdas / lambdas.sum(dim=-1, keepdim=True).clamp_min(eps)


def expected_loop_count(halting_weights: torch.Tensor) -> torch.Tensor:
    loop_ids = torch.arange(
        1,
        halting_weights.shape[-1] + 1,
        device=halting_weights.device,
        dtype=halting_weights.dtype,
    )
    return (halting_weights * loop_ids).sum(dim=-1)


def target_loop_counts(
    input_tokens: torch.Tensor | int | float,
    cot_tokens: torch.Tensor | int | float,
    max_train_loops: int,
) -> torch.Tensor:
    """Compute the staged target loop heuristic from the handoff."""

    input_t = torch.as_tensor(input_tokens, dtype=torch.float32).clamp_min(1.0)
    cot_t = torch.as_tensor(cot_tokens, dtype=torch.float32).clamp_min(1.0)
    target = 2.0 + 0.75 * torch.log2(input_t / 512.0) + 0.75 * torch.log2(cot_t / 128.0)
    return torch.round(target).clamp(1, max_train_loops).to(torch.long)


def centered_geometric_prior(
    target_loops: torch.Tensor,
    max_loops: int,
    decay: float = 0.55,
) -> torch.Tensor:
    """A normalized geometric-shaped prior centered around target loops."""

    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1)")

    target = target_loops.to(dtype=torch.float32)
    loop_ids = torch.arange(1, max_loops + 1, device=target.device, dtype=torch.float32)
    weights = decay ** torch.abs(loop_ids.view(*([1] * target.dim()), -1) - target.unsqueeze(-1))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def categorical_kl(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(q || p) over the last dimension."""

    q_safe = q.clamp_min(eps)
    p_safe = p.clamp_min(eps)
    return (q_safe * (q_safe.log() - p_safe.log())).sum(dim=-1)
