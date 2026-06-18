"""Numerical stability guards for lightweight training loops."""

from __future__ import annotations

import torch


def assert_finite_tensor(name: str, value: torch.Tensor, step: int) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name} at step {step}: {value.detach().float().cpu()}")


def assert_finite_metrics(metrics: dict[str, torch.Tensor], step: int) -> None:
    for name, value in metrics.items():
        if torch.is_tensor(value):
            assert_finite_tensor(name, value, step)


def assert_finite_trainable_parameters(module: torch.nn.Module, step: int) -> None:
    bad_names = []
    for name, param in module.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            bad_names.append(name)
    if bad_names:
        joined = ", ".join(bad_names[:20])
        raise FloatingPointError(f"Nonfinite trainable parameters at step {step}: {joined}")


def assert_finite_trainable_gradients(module: torch.nn.Module, step: int) -> None:
    bad_names = []
    for name, param in module.named_parameters():
        if param.requires_grad and param.grad is not None and not torch.isfinite(param.grad).all():
            bad_names.append(name)
    if bad_names:
        joined = ", ".join(bad_names[:20])
        raise FloatingPointError(f"Nonfinite trainable gradients at step {step}: {joined}")


def assert_finite_training_state(
    module: torch.nn.Module,
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    step: int,
) -> None:
    assert_finite_tensor("loss", loss, step)
    assert_finite_metrics(metrics, step)
    assert_finite_trainable_parameters(module, step)
