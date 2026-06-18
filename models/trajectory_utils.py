"""Helpers for batched multi-trajectory recurrent inference."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def repeat_for_trajectories(tensor: Optional[torch.Tensor], num_trajectories: int) -> Optional[torch.Tensor]:
    if tensor is None or num_trajectories == 1:
        return tensor
    return tensor.repeat_interleave(num_trajectories, dim=0)


def unflatten_trajectories(tensor: torch.Tensor, batch_size: int, num_trajectories: int) -> torch.Tensor:
    return tensor.view(batch_size, num_trajectories, *tensor.shape[1:])


def flatten_trajectories(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 3:
        raise ValueError("Expected a tensor with batch and trajectory dimensions")
    return tensor.view(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def average_pairwise_cosine_distance(pooled_by_trajectory: torch.Tensor) -> torch.Tensor:
    """Average pairwise cosine distance for ``[batch, K, hidden]`` states."""

    if pooled_by_trajectory.dim() != 3:
        raise ValueError("Expected [batch, num_trajectories, hidden]")

    num_trajectories = pooled_by_trajectory.shape[1]
    if num_trajectories < 2:
        return pooled_by_trajectory.new_zeros(())

    # Compute the diagnostic/reward in fp32 even when the base model runs in
    # bf16/fp16. Small trajectory differences otherwise round to zero.
    work = pooled_by_trajectory.float()
    normalized = F.normalize(work, dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    distance = 1.0 - similarity
    mask = torch.triu(
        torch.ones(
            num_trajectories,
            num_trajectories,
            device=pooled_by_trajectory.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    return distance[:, mask].mean()
