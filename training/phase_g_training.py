"""Training utilities for the frozen-substrate Phase G-alpha experiment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


def first_active_loop_token_ids(loop_labels: torch.Tensor) -> torch.Tensor:
    """Return the first supervised token for every batch/loop cell."""

    if loop_labels.dim() != 3:
        raise ValueError("loop_labels must be [batch, loops, sequence]")
    active = loop_labels.ne(-100)
    missing = ~active.any(dim=-1)
    if bool(missing.any()):
        locations = missing.nonzero(as_tuple=False).tolist()
        raise ValueError(f"Every requested Phase G loop needs a gold symbol; missing={locations}")
    first = active.to(dtype=torch.int64).argmax(dim=-1)
    return loop_labels.gather(dim=-1, index=first.unsqueeze(-1)).squeeze(-1)


def posterior_target_embeddings(
    base_model: torch.nn.Module,
    loop_labels: torch.Tensor,
) -> torch.Tensor:
    """Embed gold next symbols without opening a gradient path into the keeper."""

    token_ids = first_active_loop_token_ids(loop_labels)
    embedding = base_model.get_input_embeddings()
    with torch.no_grad():
        targets = embedding(token_ids)
    return targets.detach()


@dataclass(frozen=True)
class PhaseGBackwardResult:
    loss: float
    metrics: dict[str, float]
    microbatch_count: int


def _is_additive_metric(name: str) -> bool:
    return (
        name == "per_loop_label_active"
        or name == "per_loop_label_weighted_active"
        or name.startswith("per_loop_label_active_")
    )


def backward_phase_g_trajectories(
    module: torch.nn.Module,
    *,
    forward_kwargs: dict[str, Any],
    trajectory_seeds: list[int],
    microbatch_size: int,
) -> PhaseGBackwardResult:
    """Backpropagate the exact mean objective over independent trajectories.

    Phase G-alpha has no training-time cross-trajectory interaction. Splitting
    the seeded trajectories therefore changes only peak activation memory, not
    the objective or optimizer-step cadence.
    """

    if not trajectory_seeds:
        raise ValueError("trajectory_seeds cannot be empty")
    if "num_trajectories" in forward_kwargs or "phase_g_trajectory_seeds" in forward_kwargs:
        raise ValueError(
            "forward_kwargs must not override num_trajectories or phase_g_trajectory_seeds"
        )
    total_trajectories = len(trajectory_seeds)
    chunk_size = (
        total_trajectories
        if int(microbatch_size) <= 0
        else min(int(microbatch_size), total_trajectories)
    )
    weighted_loss = 0.0
    metrics: dict[str, float] = {}
    microbatch_count = 0
    for start in range(0, total_trajectories, chunk_size):
        seeds = trajectory_seeds[start : start + chunk_size]
        weight = len(seeds) / total_trajectories
        output = module(
            **forward_kwargs,
            num_trajectories=len(seeds),
            phase_g_trajectory_seeds=seeds,
        )
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise FloatingPointError(
                f"Nonfinite Phase G loss for trajectory seeds {seeds}"
            )
        (output.loss * weight).backward()
        weighted_loss += float(output.loss.detach().float().cpu().item()) * weight
        for name, value in output.metrics.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                continue
            scalar = float(value.detach().float().cpu().item())
            metrics[name] = metrics.get(name, 0.0) + (
                scalar if _is_additive_metric(name) else scalar * weight
            )
        microbatch_count += 1
    return PhaseGBackwardResult(
        loss=weighted_loss,
        metrics=metrics,
        microbatch_count=microbatch_count,
    )


class PhaseGEMA:
    """EMA over exactly the three registered Phase G parameter groups."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        decay: float,
    ) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in named_parameters
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires at least one trainable parameter")

    @torch.no_grad()
    def update(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        current = dict(named_parameters)
        if set(current).issuperset(self.shadow) is False:
            raise KeyError("EMA parameter set changed")
        for name, shadow in self.shadow.items():
            value = current[name].detach().float().cpu()
            shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> dict[str, torch.Tensor]:
        current = dict(named_parameters)
        backup: dict[str, torch.Tensor] = {}
        for name, shadow in self.shadow.items():
            parameter = current[name]
            backup[name] = parameter.detach().cpu().clone()
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
        return backup

    @staticmethod
    @torch.no_grad()
    def restore(
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        backup: dict[str, torch.Tensor],
    ) -> None:
        current = dict(named_parameters)
        for name, value in backup.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }
