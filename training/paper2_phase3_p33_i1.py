"""Locked contracts for the P3.3 i1 aim-focused continuation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase2_matched_alpha import clip_module_groups
from training.paper2_phase3_p33 import (
    P33_PRESERVATION_WEIGHT,
    p33_forward_losses,
)


P33_I1_TOTAL_STEPS = 1_000
P33_I1_LOOKS = 20
P33_I1_LOOK_INTERVAL = 50
P33_I1_LEARNING_RATE = 3e-4
P33_I1_WARMUP_STEPS = 100
P33_I1_AIM_SHARE_FLOOR = 0.70
P33_I1_PRESERVATION_SHARE_CEILING = 0.25
P33_I1_CALIBRATION_TARGET = 0.75
P33_I1_CALIBRATION_BATCHES = 4
P33_I1_CALIBRATION_WEIGHT_MIN = 1.0
P33_I1_CALIBRATION_WEIGHT_MAX = 1_000.0
P33_I1_BASELINE_PI_DIR = 0.14901016586409846
P33_I1_P34_AUTOMATIC_THRESHOLD = 0.25
P33_I1_BOUNDARY_THRESHOLD = 0.05

P33_I1_GATE_NAMES = frozenset(
    {
        "bridge.gate_logits",
        "bridge.gate_hidden.weight",
        "bridge.gate_scratch.weight",
        "bridge.gate_control.weight",
    }
)
P33_I1_DIRECTION_NAMES = frozenset(
    {
        "bridge.output_projection.weight",
    }
)


def set_p33_i1_trainable(module: Phase3StudentModules) -> dict[str, nn.Parameter]:
    """Expose the post-selector direction projection and keep the selector immutable."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for name, parameter in module.named_parameters():
        if name in P33_I1_DIRECTION_NAMES:
            parameter.requires_grad_(True)
    named = dict(module.named_parameters())
    missing = sorted((P33_I1_GATE_NAMES | P33_I1_DIRECTION_NAMES) - set(named))
    if missing:
        raise RuntimeError(f"P3.3 i1 expected parameters are absent: {missing}")
    if any(named[name].requires_grad for name in P33_I1_GATE_NAMES):
        raise RuntimeError("P3.3 i1 gate freeze failed")
    if any(
        parameter.requires_grad
        for name, parameter in named.items()
        if name.startswith("control.")
    ):
        raise RuntimeError("P3.3 i1 upstream selector freeze failed")
    trainable = {name: value for name, value in named.items() if value.requires_grad}
    if set(name for name in trainable if name.startswith("bridge.")) != set(
        P33_I1_DIRECTION_NAMES
    ):
        raise RuntimeError("P3.3 i1 bridge trainable set changed")
    return trainable


def p33_i1_forward_losses(**kwargs: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    losses, metrics = p33_forward_losses(**kwargs)
    return {"aim": losses["aim"], "preserve": losses["preserve"]}, metrics


def p33_i1_total(losses: Mapping[str, torch.Tensor], *, aim_weight: float) -> torch.Tensor:
    if set(losses) != {"aim", "preserve"}:
        raise ValueError("P3.3 i1 loss set changed")
    if not math.isfinite(float(aim_weight)) or float(aim_weight) <= 0.0:
        raise ValueError("P3.3 i1 aim weight must be positive and finite")
    return float(aim_weight) * losses["aim"] + P33_PRESERVATION_WEIGHT * losses["preserve"]


def build_p33_i1_adamw_groups(
    trainable: Mapping[str, nn.Parameter], *, weight_decay: float
) -> list[dict[str, object]]:
    if set(trainable) != {"bridge.output_projection.weight"}:
        raise RuntimeError("P3.3 i1 optimizer surface changed")
    return [
        {
            "params": list(trainable.values()),
            "weight_decay": float(weight_decay),
            "group_name": "direction_projection_decay",
        }
    ]


def postclip_gradient_shares(
    *,
    losses: Mapping[str, torch.Tensor],
    module: nn.Module,
    parameters: Sequence[nn.Parameter],
    aim_weight: float,
) -> dict[str, Any]:
    """Attribute each loss after applying the exact combined group clip factors."""

    if set(losses) != {"aim", "preserve"}:
        raise ValueError("P3.3 i1 share audit requires aim and preserve")
    active = [parameter for parameter in parameters if parameter.requires_grad]
    if not active:
        raise ValueError("P3.3 i1 share audit has no trainable parameters")
    weighted = {
        "aim": float(aim_weight) * losses["aim"],
        "preserve": P33_PRESERVATION_WEIGHT * losses["preserve"],
    }
    gradients: dict[str, dict[int, torch.Tensor]] = {}
    for index, name in enumerate(("aim", "preserve")):
        values = torch.autograd.grad(
            weighted[name],
            active,
            retain_graph=index == 0,
            allow_unused=True,
        )
        gradients[name] = {
            id(parameter): gradient.detach()
            for parameter, gradient in zip(active, values)
            if gradient is not None
        }

    parameter_group: dict[int, tuple[str, float]] = {}
    for group_name, (group_parameters, ceiling) in clip_module_groups(module).items():
        for parameter in group_parameters:
            if parameter.requires_grad:
                parameter_group[id(parameter)] = (group_name, float(ceiling))
    uncovered = [parameter for parameter in active if id(parameter) not in parameter_group]
    if uncovered:
        raise RuntimeError("P3.3 i1 clip attribution does not cover every trainable parameter")

    group_combined_sq: dict[str, float] = {}
    group_ceiling: dict[str, float] = {}
    for parameter in active:
        group_name, ceiling = parameter_group[id(parameter)]
        group_ceiling[group_name] = ceiling
        combined = sum(
            (
                gradients[name].get(id(parameter), torch.zeros_like(parameter))
                for name in ("aim", "preserve")
            ),
            torch.zeros_like(parameter),
        )
        group_combined_sq[group_name] = group_combined_sq.get(group_name, 0.0) + float(
            combined.double().square().sum()
        )
    group_scales = {
        name: min(1.0, group_ceiling[name] / max(math.sqrt(value), 1e-30))
        for name, value in group_combined_sq.items()
    }
    postclip_norms: dict[str, float] = {}
    for name in ("aim", "preserve"):
        squared = 0.0
        for parameter in active:
            gradient = gradients[name].get(id(parameter))
            if gradient is None:
                continue
            group_name, _ = parameter_group[id(parameter)]
            squared += float((gradient.double() * group_scales[group_name]).square().sum())
        postclip_norms[name] = math.sqrt(squared)
    denominator = sum(postclip_norms.values())
    shares = {name: value / max(denominator, 1e-30) for name, value in postclip_norms.items()}
    return {
        "aim_weight": float(aim_weight),
        "postclip_gradient_norms": postclip_norms,
        "shares": shares,
        "aim_share": shares["aim"],
        "preservation_share": shares["preserve"],
        "group_combined_norms": {
            name: math.sqrt(value) for name, value in group_combined_sq.items()
        },
        "group_clip_scales": group_scales,
        "classification": (
            "gross"
            if shares["aim"] < 0.60 or shares["preserve"] > 0.35
            else "marginal"
            if shares["aim"] < P33_I1_AIM_SHARE_FLOOR
            or shares["preserve"] > P33_I1_PRESERVATION_SHARE_CEILING
            else "pass"
        ),
    }


def i1_result_band(pi_dir: float) -> str:
    value = float(pi_dir)
    if value >= P33_I1_P34_AUTOMATIC_THRESHOLD:
        return "p34_charter_funded"
    if value < P33_I1_BOUNDARY_THRESHOLD:
        return "boundary_diagnostic"
    return "middle_band_strategy_review"
