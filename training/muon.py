"""Small Muon optimizer utilities for recurrent-block unfreeze experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(matrix: torch.Tensor, *, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Return an approximate orthogonalized update using Newton-Schulz steps."""

    if matrix.ndim != 2:
        raise ValueError("zeropower_via_newtonschulz5 expects a 2D tensor")
    x = matrix.float()
    norm = x.norm()
    if not torch.isfinite(norm) or float(norm) == 0.0:
        return torch.zeros_like(matrix)
    x = x / norm.clamp_min(eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    # Coefficients from the public Muon reference implementation. The iteration
    # converges toward the polar factor for well-scaled update matrices.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.T
    return x.to(dtype=matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Minimal Muon optimizer for matrix-like parameters.

    Vector and scalar parameters should usually be trained by AdamW in a
    separate parameter group. This class intentionally rejects non-matrix
    parameters so experiments do not silently apply the wrong update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim < 2:
                    raise ValueError("Muon only supports parameters with ndim >= 2")
                grad = param.grad.detach()
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad.add(buffer, alpha=momentum) if nesterov else buffer
                original_shape = update.shape
                if update.ndim > 2:
                    update = update.reshape(update.shape[0], -1)
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                if len(original_shape) > 2:
                    update = update.reshape(original_shape)
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                scale = math.sqrt(max(1.0, update.numel() / max(1, update.shape[0])))
                param.add_(update.to(dtype=param.dtype), alpha=-lr * scale)
        return loss


class OptimizerBundle:
    """Tiny wrapper that steps multiple optimizers as one training object."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("OptimizerBundle requires at least one optimizer")
        self.optimizers = optimizers

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)


def split_muon_and_adamw_params(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainables into matrix-like Muon params and AdamW fallback params."""

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for _name, param in named_parameters:
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params
