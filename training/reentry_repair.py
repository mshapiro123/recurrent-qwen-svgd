"""Utilities for explicit recurrent re-entry repair controls."""

from __future__ import annotations

from typing import Any

import torch


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _bridge_identity_max_abs_diff(wrapper: torch.nn.Module) -> float:
    bridge = wrapper.bridge
    weight = bridge.proj.weight.detach().float().cpu()
    eye = torch.eye(weight.shape[0], dtype=weight.dtype)
    return float((weight - eye).abs().max().item())


def apply_reentry_repair_controls(wrapper: torch.nn.Module, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit bridge/re-entry repair controls after checkpoint load.

    The normal checkpoint path must preserve learned bridge parameters exactly.
    This helper only mutates the bridge when a config opts in through
    ``bridge_reset_identity`` and/or ``bridge_gate_override``.  The useful smoke
    setting for a dead identity bridge is ``bridge_reset_identity: true`` and
    ``bridge_gate_override: 1.0``: output remains identity, but the bridge
    projection receives gradients immediately.
    """

    bridge = wrapper.bridge
    before_gate = float(bridge.bridge_gate.detach().float().cpu().item())
    before_identity_diff = _bridge_identity_max_abs_diff(wrapper)
    reset_identity = _as_bool(cfg.get("bridge_reset_identity"), default=False)
    gate_override = cfg.get("bridge_gate_override")
    applied = False

    with torch.no_grad():
        if reset_identity:
            bridge.reset_parameters()
            applied = True
        if gate_override is not None:
            bridge.bridge_gate.fill_(float(gate_override))
            applied = True

    after_gate = float(bridge.bridge_gate.detach().float().cpu().item())
    after_identity_diff = _bridge_identity_max_abs_diff(wrapper)
    return {
        "applied": applied,
        "bridge_reset_identity": reset_identity,
        "bridge_gate_override": None if gate_override is None else float(gate_override),
        "bridge_gate_before": before_gate,
        "bridge_gate_after": after_gate,
        "bridge_identity_max_abs_diff_before": before_identity_diff,
        "bridge_identity_max_abs_diff_after": after_identity_diff,
    }
