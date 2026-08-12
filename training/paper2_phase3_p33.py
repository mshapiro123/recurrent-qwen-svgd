"""Locked pure contracts for the Phase 3.3 aimed-writeback pilot."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase2_matched_alpha import (
    masked_sparse_kl,
    normalize_sparse_with_tail,
    reconstruct_sparse_residual_seed,
)
from training.paper2_phase3_p32 import GateLabel
from training.paper2_phase3_p33_prep import P33_GATE_CEILING


P33_TOTAL_STEPS = 1_000
P33_LOOKS = 20
P33_LOOK_INTERVAL = 50
P33_BATCH_SIZE = 128
P33_LEARNING_RATE = 3e-4
P33_WARMUP_STEPS = 100
P33_WEIGHT_DECAY = 0.01
P33_ADAM_BETAS = (0.9, 0.999)
P33_PRESERVATION_WEIGHT = 1.0
P33_GATE_CLASSIFICATION_THRESHOLD = 0.5


def look_steps(*, total_steps: int = P33_TOTAL_STEPS, looks: int = P33_LOOKS) -> list[int]:
    if total_steps <= 0 or looks <= 0 or total_steps % looks:
        raise ValueError("P3.3 requires evenly spaced integer look steps")
    steps = list(range(total_steps // looks, total_steps + 1, total_steps // looks))
    if len(steps) != looks or steps[-1] != total_steps:
        raise RuntimeError("P3.3 look construction changed")
    return steps


def learning_rate_at_step(step: int) -> float:
    if step <= 0:
        return 0.0
    return P33_LEARNING_RATE * min(1.0, int(step) / P33_WARMUP_STEPS)


def set_p33_trainable(module: Phase3StudentModules) -> dict[str, nn.Parameter]:
    """Freeze the installed substrate and expose only bridge plus control."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for name, parameter in module.named_parameters():
        if name.startswith(("bridge.", "control.")):
            parameter.requires_grad_(True)
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def p33_forward_losses(
    *,
    module: Phase3StudentModules,
    hidden4: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    base_candidates: torch.Tensor,
    base_tail: torch.Tensor,
    gate_labels: torch.Tensor,
    oracle_directions: torch.Tensor,
    position_bucket: torch.Tensor,
    tied_embedding: nn.Embedding,
    steps: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute the three locked P3.3 losses on one anchor batch."""

    if hidden4.ndim != 3 or hidden4.shape[1:] != (4, 896):
        raise ValueError("P3.3 hidden cache must be [batch,4,896]")
    if gate_labels.shape != hidden4.shape[:2]:
        raise ValueError("P3.3 gate labels must be [batch,4]")
    if oracle_directions.shape != hidden4.shape:
        raise ValueError("P3.3 oracle directions must match hidden horizons")
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    candidate_embeddings = tied_embedding(candidate_ids.clamp_min(0)).float()
    raw_base = torch.einsum("bhd,bhcd->bhc", hidden4.float(), candidate_embeddings)
    residual_seed = reconstruct_sparse_residual_seed(
        base_candidates, raw_base, candidate_mask
    ).detach()
    output = module(
        hidden=hidden,
        previous_logits=residual_seed,
        steps=steps,
        attention_mask=attention,
        position_bucket=position_bucket,
        candidate_ids=candidate_ids,
    )
    pre_gate_direction = output.bridge.delta[:, 1:].float()
    positive = gate_labels.eq(int(GateLabel.POSITIVE))
    negative = gate_labels.eq(int(GateLabel.NEGATIVE))
    active = positive | negative
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("P3.3 batches must contain both gate classes")
    aim_rows = 1.0 - F.cosine_similarity(
        pre_gate_direction[positive], oracle_directions.float()[positive], dim=-1
    )
    aim = aim_rows.mean()

    gate_probability = output.bridge.position_gate_unclamped[:, 1:, 0].float()
    target = positive.float()
    positive_count = positive.sum().float()
    negative_count = negative.sum().float()
    active_count = positive_count + negative_count
    class_weight = torch.zeros_like(gate_probability)
    class_weight[positive] = active_count / (2.0 * positive_count)
    class_weight[negative] = active_count / (2.0 * negative_count)
    gate_rows = F.binary_cross_entropy(
        gate_probability.clamp(1e-7, 1.0 - 1e-7), target, reduction="none"
    )
    gate = (gate_rows[active] * class_weight[active]).mean()

    bridge_delta = torch.einsum(
        "bhd,bhcd->bhc",
        output.hidden[:, 1:].float() - hidden4.float(),
        candidate_embeddings,
    )
    bridge_sparse = residual_seed + bridge_delta
    base_log = normalize_sparse_with_tail(
        base_candidates, base_tail, candidate_mask
    ).detach()
    bridge_log = normalize_sparse_with_tail(
        bridge_sparse, base_tail, candidate_mask
    )
    preserve_rows = masked_sparse_kl(base_log, bridge_log, candidate_mask)
    preserve = preserve_rows[negative].mean()
    losses = {"aim": aim, "gate": gate, "preserve": preserve}
    metrics = {
        "pre_gate_direction": pre_gate_direction,
        "gate_probability_unclamped": gate_probability,
        "gate_probability_deployed": output.bridge.position_gate[:, 1:, 0].float(),
        "bridge_log": bridge_log,
        "base_log": base_log,
        "positive_mask": positive,
        "negative_mask": negative,
        "writeback": output.hidden[:, 1:].float() - hidden4.float(),
        "flow_states": output.flow.states,
    }
    return losses, metrics


def weighted_total(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if set(losses) != {"aim", "gate", "preserve"}:
        raise ValueError("P3.3 loss set changed")
    return losses["aim"] + losses["gate"] + P33_PRESERVATION_WEIGHT * losses["preserve"]


def independent_gradient_shares(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[nn.Parameter],
) -> dict[str, Any]:
    norms = {}
    names = ("aim", "gate", "preserve")
    for index, name in enumerate(names):
        gradients = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=index + 1 < len(names),
            allow_unused=True,
        )
        squared = sum(
            float(gradient.detach().double().square().sum())
            for gradient in gradients
            if gradient is not None
        )
        norms[name] = math.sqrt(squared)
    denominator = sum(norms.values())
    shares = {name: norms[name] / max(denominator, 1e-30) for name in names}
    primary = shares["aim"] + shares["gate"]
    return {
        "raw_gradient_norms": norms,
        "shares": shares,
        "primary_share": primary,
        "preservation_share": shares["preserve"],
        "classification": (
            "gross"
            if primary < 0.40 or shares["preserve"] > 0.35
            else "marginal"
            if primary < 0.50 or shares["preserve"] > 0.25
            else "pass"
        ),
    }


def gate_classification(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = P33_GATE_CLASSIFICATION_THRESHOLD,
) -> dict[str, float | int]:
    positive = labels.eq(int(GateLabel.POSITIVE))
    negative = labels.eq(int(GateLabel.NEGATIVE))
    predicted = probabilities.float() >= float(threshold)
    true_positive = int((predicted & positive).sum())
    false_positive = int((predicted & negative).sum())
    false_negative = int((~predicted & positive).sum())
    true_negative = int((~predicted & negative).sum())
    return {
        "threshold_on_unclamped_probability": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "recall": true_positive / max(1, true_positive + false_negative),
        "precision": true_positive / max(1, true_positive + false_positive),
        "false_positive_rate": false_positive / max(1, false_positive + true_negative),
    }


def activate_operating_clamp(module: Phase3StudentModules) -> dict[str, Any]:
    before = torch.sigmoid(module.bridge.gate_logits.detach().float())
    module.bridge.set_gate_ceiling(P33_GATE_CEILING)
    return {
        "ceiling": P33_GATE_CEILING,
        "unclamped_migrated_gate_by_loop": before.tolist(),
        "clamp_binds_at_init_by_loop": (before > P33_GATE_CEILING).tolist(),
    }
