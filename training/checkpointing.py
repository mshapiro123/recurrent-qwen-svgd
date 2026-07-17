"""Checkpoint helpers for lightweight recurrent-depth experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .stability import assert_finite_trainable_parameters


def trainable_state_dict(
    module: torch.nn.Module,
    *,
    include_frozen_prefixes: tuple[str, ...] = (),
    include_frozen_lora: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if (
            param.requires_grad
            or any(name.startswith(prefix) for prefix in include_frozen_prefixes)
            or (
                include_frozen_lora
                and (".lora_a." in name or ".lora_b." in name)
            )
        )
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
            "trainable_state_dict": trainable_state_dict(
                wrapper,
                include_frozen_prefixes=tuple(config.get("checkpoint_include_frozen_prefixes") or ()),
                include_frozen_lora=bool(config.get("checkpoint_include_frozen_lora", False)),
            ),
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
    upgraded = {}
    for name, tensor in state.items():
        if name == "bridge.proj.weight" and "bridge.prelude_proj.weight" in current and "bridge.state_proj.weight" in current:
            hidden = current["bridge.state_proj.weight"].shape[0]
            if tensor.dim() == 2 and tensor.shape == current["bridge.proj.weight"].shape:
                compatible[name] = tensor.to(device=current[name].device, dtype=current[name].dtype)
                compatible["bridge.prelude_proj.weight"] = tensor[:, :hidden].to(
                    device=current["bridge.prelude_proj.weight"].device,
                    dtype=current["bridge.prelude_proj.weight"].dtype,
                )
                compatible["bridge.state_proj.weight"] = tensor[:, hidden:].to(
                    device=current["bridge.state_proj.weight"].device,
                    dtype=current["bridge.state_proj.weight"].dtype,
                )
                upgraded[name] = "split_concat_projection_into_prelude_state"
                continue
            if tensor.dim() == 2 and tensor.shape == current["bridge.state_proj.weight"].shape:
                upgraded_tensor = current[name].detach().clone()
                upgraded_tensor[:, :hidden].zero_()
                upgraded_tensor[:, hidden:] = tensor.to(
                    device=upgraded_tensor.device,
                    dtype=upgraded_tensor.dtype,
                )
                compatible[name] = upgraded_tensor.to(device=current[name].device, dtype=current[name].dtype)
                compatible["bridge.prelude_proj.weight"] = torch.zeros_like(current["bridge.prelude_proj.weight"])
                compatible["bridge.state_proj.weight"] = tensor.to(
                    device=current["bridge.state_proj.weight"].device,
                    dtype=current["bridge.state_proj.weight"].dtype,
                )
                upgraded[name] = (
                    f"warm_start_split_state_half {tuple(tensor.shape)} -> "
                    f"{tuple(current[name].shape)}"
                )
                continue
        if (
            name == "bridge.proj.bias"
            and "bridge.state_proj.bias" in current
            and tensor.shape == current["bridge.state_proj.bias"].shape
        ):
            compatible[name] = tensor.to(device=current[name].device, dtype=current[name].dtype)
            compatible["bridge.state_proj.bias"] = tensor.to(
                device=current["bridge.state_proj.bias"].device,
                dtype=current["bridge.state_proj.bias"].dtype,
            )
            upgraded[name] = "copy_concat_bias_to_split_state_bias"
            continue
        if name not in current:
            skipped[name] = "missing"
            continue
        if current[name].shape != tensor.shape:
            if (
                name == "bridge.proj.weight"
                and tensor.dim() == 2
                and current[name].dim() == 2
                and current[name].shape[0] == tensor.shape[0]
                and current[name].shape[1] == 2 * tensor.shape[1]
            ):
                upgraded_tensor = current[name].detach().clone()
                upgraded_tensor[:, : tensor.shape[1]].zero_()
                upgraded_tensor[:, tensor.shape[1] :] = tensor.to(
                    device=upgraded_tensor.device,
                    dtype=upgraded_tensor.dtype,
                )
                compatible[name] = upgraded_tensor.to(device=current[name].device, dtype=current[name].dtype)
                upgraded[name] = f"warm_start_state_half {tuple(tensor.shape)} -> {tuple(current[name].shape)}"
                continue
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
        "upgraded": upgraded,
        "skipped": skipped,
        "missing": missing,
        "unexpected": unexpected,
    }
