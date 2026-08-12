"""Pure contracts for the build-only P3.4 prerequisite cycle.

This module deliberately contains no optimizer or campaign runner.  The
executed lock must bind the remaining calibration values before training code
is allowed to import these contracts into a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase2_matched_alpha import (
    masked_sparse_kl,
    normalize_sparse_with_tail,
)
from training.paper2_phase3_p33 import p33_forward_losses


P34_FLOW_LOOPS = 4
P34_GATE_CEILINGS = (0.02, 0.08, 0.20, 0.50)
P34_PRESERVATION_WEIGHTS = (1.0, 0.5, 0.2, 0.05)
P34_ADVANCE_PI_DEP = (0.10, 0.25, 0.40)
P34_INITIAL_RUNG = 1
P34_LOSS_NAMES = ("kl", "aim", "ce", "gate", "preserve")


@dataclass(frozen=True)
class TaskInferenceContract:
    scratch_lifetime: str = "fresh_per_emitted_token"
    flow_loops: int = P34_FLOW_LOOPS
    draft_head_scoring: bool = False
    cross_token_state_persistence: bool = False
    decoding: str = "greedy"
    position_zero_closed: bool = True
    serving_reader: str = "bf16_serving_matmul_v1"

    def validate(self) -> None:
        expected = {
            "scratch_lifetime": "fresh_per_emitted_token",
            "flow_loops": 4,
            "draft_head_scoring": False,
            "cross_token_state_persistence": False,
            "decoding": "greedy",
            "position_zero_closed": True,
            "serving_reader": "bf16_serving_matmul_v1",
        }
        observed = asdict(self)
        if observed != expected:
            raise RuntimeError(
                f"P3.4 task-inference contract changed: {observed} != {expected}"
            )


@dataclass(frozen=True)
class LossShareBounds:
    kl_floor: float = 0.35
    aim_floor: float = 0.15
    ce_floor: float = 0.10
    gate_floor: float = 0.03
    preserve_ceiling: float = 0.25

    def as_mapping(self) -> dict[str, float]:
        return {
            "kl": self.kl_floor,
            "aim": self.aim_floor,
            "ce": self.ce_floor,
            "gate": self.gate_floor,
            "preserve": self.preserve_ceiling,
        }


@dataclass(frozen=True)
class AnnealingState:
    rung: int
    chi_max: float
    window: int = 0

    @property
    def gate_ceiling(self) -> float:
        return P34_GATE_CEILINGS[self.rung]

    @property
    def preservation_weight(self) -> float:
        return P34_PRESERVATION_WEIGHTS[self.rung]


def initial_annealing_state(*, chi_max: float) -> AnnealingState:
    if not 0.0 <= float(chi_max) <= 1.0:
        raise ValueError("chi_max must be a probability")
    return AnnealingState(rung=P34_INITIAL_RUNG, chi_max=float(chi_max))


def controller_transition(
    state: AnnealingState,
    *,
    pi_dep: float,
    chi: float,
    tier_w_event: bool,
) -> tuple[AnnealingState, dict[str, Any]]:
    """Apply at most one registered controller transition per window."""

    if state.rung not in range(len(P34_GATE_CEILINGS)):
        raise ValueError("controller rung is outside the registered ladder")
    if not 0.0 <= float(pi_dep) <= 1.0 or not 0.0 <= float(chi) <= 1.0:
        raise ValueError("pi_dep and chi must be probabilities")
    before = state.rung
    reason = "hold"
    if tier_w_event and before > 0:
        after = before - 1
        reason = "tier_w_demote"
    elif before < len(P34_GATE_CEILINGS) - 1:
        required = P34_ADVANCE_PI_DEP[before]
        if float(pi_dep) >= required and float(chi) <= state.chi_max:
            after = before + 1
            reason = "advance"
        else:
            after = before
    else:
        after = before
    updated = AnnealingState(rung=after, chi_max=state.chi_max, window=state.window + 1)
    return updated, {
        "window": updated.window,
        "rung_before": before,
        "rung_after": after,
        "reason": reason,
        "pi_dep": float(pi_dep),
        "chi": float(chi),
        "chi_max": state.chi_max,
        "gate_ceiling": updated.gate_ceiling,
        "preservation_weight": updated.preservation_weight,
        "at_most_one_rung": abs(after - before) <= 1,
    }


def set_p34_trainable(module: Phase3StudentModules) -> dict[str, nn.Parameter]:
    """Expose the chartered bridge, gate, and control surface only."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for name, parameter in module.named_parameters():
        if name.startswith(("bridge.", "control.")):
            parameter.requires_grad_(True)
    trainable = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(
        name.startswith(("flow.", "draft.", "initializer.")) for name in trainable
    ):
        raise RuntimeError("P3.4 trainable surface crossed the charter boundary")
    return trainable


def classify_loss_shares(
    shares: Mapping[str, float],
    *,
    prior_consecutive_misses: int = 0,
    bounds: LossShareBounds = LossShareBounds(),
) -> dict[str, Any]:
    if set(shares) != set(P34_LOSS_NAMES):
        raise ValueError(f"P3.4 share names changed: {sorted(shares)}")
    if any(not 0.0 <= float(value) <= 1.0 for value in shares.values()):
        raise ValueError("P3.4 gradient shares must be probabilities")
    misses = {
        "kl": float(shares["kl"]) < bounds.kl_floor,
        "aim": float(shares["aim"]) < bounds.aim_floor,
        "ce": float(shares["ce"]) < bounds.ce_floor,
        "gate": float(shares["gate"]) < bounds.gate_floor,
        "preserve": float(shares["preserve"]) > bounds.preserve_ceiling,
    }
    failed = sorted(name for name, missed in misses.items() if missed)
    consecutive = prior_consecutive_misses + 1 if failed else 0
    return {
        "shares": {name: float(shares[name]) for name in P34_LOSS_NAMES},
        "bounds": bounds.as_mapping(),
        "failed_contracts": failed,
        "classification": "stop" if consecutive >= 2 else "warn" if failed else "pass",
        "consecutive_misses": consecutive,
        "rule": "first miss warns; two consecutive trailing-window misses stop",
    }


def p34_forward_losses(
    *,
    module: Phase3StudentModules,
    tied_embedding: nn.Embedding,
    teacher_candidates: torch.Tensor,
    teacher_tail: torch.Tensor,
    teacher_token_index: torch.Tensor,
    kl_mask: torch.Tensor,
    ce_mask: torch.Tensor,
    **p33_inputs: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute the five P3.4 losses without choosing their scalar weights."""

    inherited, metrics = p33_forward_losses(
        module=module,
        tied_embedding=tied_embedding,
        steps=P34_FLOW_LOOPS,
        **p33_inputs,
    )
    candidate_mask = p33_inputs["candidate_mask"].bool()
    target = normalize_sparse_with_tail(
        teacher_candidates, teacher_tail, candidate_mask
    )
    kl_rows = masked_sparse_kl(target, metrics["bridge_log"], candidate_mask)
    if kl_mask.shape != kl_rows.shape or ce_mask.shape != kl_rows.shape:
        raise ValueError("P3.4 KL and CE masks must align with batch and horizon")
    if not bool(kl_mask.any()) or not bool(ce_mask.any()):
        raise ValueError("P3.4 batches require active KL and verified CE rows")
    if teacher_token_index.shape != kl_rows.shape:
        raise ValueError("teacher_token_index must align with batch and horizon")
    if bool((teacher_token_index[ce_mask] < 0).any()):
        raise ValueError("verified CE rows must locate the teacher token in the union")
    selected = metrics["bridge_log"][..., :-1].gather(
        -1, teacher_token_index.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    losses = {
        "kl": kl_rows[kl_mask].mean(),
        "aim": inherited["aim"],
        "ce": -selected[ce_mask].mean(),
        "gate": inherited["gate"],
        "preserve": inherited["preserve"],
    }
    return losses, {**metrics, "teacher_log": target, "kl_rows": kl_rows}


def weighted_p34_total(
    losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]
) -> torch.Tensor:
    if set(losses) != set(P34_LOSS_NAMES) or set(weights) != set(P34_LOSS_NAMES):
        raise ValueError("P3.4 weighted total requires all five named losses")
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError("P3.4 loss weights must be non-negative")
    return sum(float(weights[name]) * losses[name] for name in P34_LOSS_NAMES)


def gap_closed(*, augmented: float, base: float, teacher: float) -> dict[str, Any]:
    values = (float(augmented), float(base), float(teacher))
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("accuracy inputs must be probabilities")
    denominator = float(teacher) - float(base)
    raw_delta = float(augmented) - float(base)
    return {
        "augmented_accuracy": float(augmented),
        "base_accuracy": float(base),
        "teacher_accuracy": float(teacher),
        "raw_delta": raw_delta,
        "teacher_gap": denominator,
        "gap_closed": raw_delta / denominator if denominator > 0.0 else None,
        "defined": denominator > 0.0,
    }
