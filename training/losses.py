"""Loss functions for deterministic and stochastic recurrent-depth training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from models.halting import categorical_kl
from models.trajectory_utils import average_pairwise_cosine_distance


def sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Autoregressive CE per example, shifted like Hugging Face causal LMs."""

    if logits.dim() != 3:
        raise ValueError(f"logits must be [batch, seq_len, vocab], got {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be [batch, seq_len], got {tuple(labels.shape)}")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape[0], -1)
    valid = shift_labels.ne(ignore_index).to(dtype=flat_loss.dtype)
    return (flat_loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)


def pondernet_weighted_ce(
    per_loop_ce: torch.Tensor,
    halting_weights: torch.Tensor,
) -> torch.Tensor:
    """Expected CE under PonderNet loop probabilities."""

    if per_loop_ce.shape != halting_weights.shape:
        raise ValueError(
            "per_loop_ce and halting_weights must have the same shape; "
            f"got {tuple(per_loop_ce.shape)} and {tuple(halting_weights.shape)}"
        )
    return (per_loop_ce * halting_weights).sum(dim=-1).mean()


def halting_kl_loss(
    halting_weights: torch.Tensor,
    target_prior: Optional[torch.Tensor],
) -> torch.Tensor:
    if target_prior is None:
        return halting_weights.new_zeros(())
    return categorical_kl(halting_weights, target_prior.to(halting_weights.device)).mean()


def trajectory_diversity_reward(final_pooled: torch.Tensor) -> torch.Tensor:
    return average_pairwise_cosine_distance(final_pooled)
