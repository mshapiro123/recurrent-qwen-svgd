"""Checkpoint helpers for lightweight recurrent-depth experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .stability import assert_finite_trainable_parameters


def trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def save_trainable_checkpoint(
    wrapper: torch.nn.Module,
    output_dir: str | Path,
    phase: str,
    step: int,
    config: dict[str, Any],
) -> Path:
    assert_finite_trainable_parameters(wrapper, step)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path / f"{phase}_step_{step}.pt"
    torch.save(
        {
            "phase": phase,
            "step": step,
            "config": config,
            "trainable_state_dict": trainable_state_dict(wrapper),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_trainable_checkpoint(
    wrapper: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["trainable_state_dict"]
    current = wrapper.state_dict()
    compatible = {}
    skipped = {}
    for name, tensor in state.items():
        if name not in current:
            skipped[name] = "missing"
            continue
        if current[name].shape != tensor.shape:
            skipped[name] = f"shape {tuple(tensor.shape)} != {tuple(current[name].shape)}"
            continue
        compatible[name] = tensor.to(device=current[name].device, dtype=current[name].dtype)

    missing, unexpected = wrapper.load_state_dict(compatible, strict=False)
    if strict and (skipped or unexpected):
        raise RuntimeError(
            f"Checkpoint load was not strict-clean. skipped={skipped}, unexpected={unexpected}"
        )

    return {
        "checkpoint": checkpoint,
        "loaded_keys": sorted(compatible),
        "skipped": skipped,
        "missing": missing,
        "unexpected": unexpected,
    }
