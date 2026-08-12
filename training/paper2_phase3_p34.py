"""Pure contracts for the build-only P3.4 prerequisite cycle.

This module deliberately contains no optimizer or campaign runner.  The
executed lock must bind the remaining calibration values before training code
is allowed to import these contracts into a run.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

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
P34_SLOT_LOSS_NAMES = ("kl", "aim", "ce", "gate", "slot", "preserve")
P34_CHI_MAX_BY_RUNG = (0.0005, 0.0005, 0.0005, 0.0010)


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
    slot_floor: float = 0.10
    preserve_ceiling: float = 0.25

    def as_mapping(self, *, slot_arm: bool = False) -> dict[str, float]:
        result = {
            "kl": self.kl_floor,
            "aim": self.aim_floor,
            "ce": self.ce_floor,
            "gate": self.gate_floor,
            "preserve": self.preserve_ceiling,
        }
        if slot_arm:
            result["slot"] = self.slot_floor
        return result


@dataclass(frozen=True)
class AnnealingState:
    rung: int
    chi_max_by_rung: tuple[float, ...] = P34_CHI_MAX_BY_RUNG
    window: int = 0

    @property
    def gate_ceiling(self) -> float:
        return P34_GATE_CEILINGS[self.rung]

    @property
    def preservation_weight(self) -> float:
        return P34_PRESERVATION_WEIGHTS[self.rung]

    @property
    def chi_max(self) -> float:
        return self.chi_max_by_rung[self.rung]


def initial_annealing_state(
    *, chi_max_by_rung: tuple[float, ...] = P34_CHI_MAX_BY_RUNG
) -> AnnealingState:
    if len(chi_max_by_rung) != len(P34_GATE_CEILINGS) or any(
        not 0.0 <= float(value) <= 1.0 for value in chi_max_by_rung
    ):
        raise ValueError("chi_max_by_rung must bind one probability per controller rung")
    return AnnealingState(
        rung=P34_INITIAL_RUNG,
        chi_max_by_rung=tuple(float(value) for value in chi_max_by_rung),
    )


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
    updated = AnnealingState(
        rung=after,
        chi_max_by_rung=state.chi_max_by_rung,
        window=state.window + 1,
    )
    return updated, {
        "window": updated.window,
        "rung_before": before,
        "rung_after": after,
        "reason": reason,
        "pi_dep": float(pi_dep),
        "chi": float(chi),
        "chi_max_before": state.chi_max,
        "chi_max_after": updated.chi_max,
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
    slot_arm = "slot" in shares
    names = P34_SLOT_LOSS_NAMES if slot_arm else P34_LOSS_NAMES
    if set(shares) != set(names):
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
    if slot_arm:
        misses["slot"] = float(shares["slot"]) < bounds.slot_floor
    failed = sorted(name for name, missed in misses.items() if missed)
    consecutive = prior_consecutive_misses + 1 if failed else 0
    if consecutive >= 4:
        classification = "stop"
    elif consecutive >= 2:
        classification = "warn"
    elif failed:
        classification = "breach_observed"
    else:
        classification = "pass"
    return {
        "shares": {name: float(shares[name]) for name in names},
        "bounds": bounds.as_mapping(slot_arm=slot_arm),
        "failed_contracts": failed,
        "classification": classification,
        "consecutive_misses": consecutive,
        "rule": (
            "first breach is observed; two consecutive trailing-window breaches warn; "
            "four consecutive breaches stop"
        ),
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
    names = P34_SLOT_LOSS_NAMES if "slot" in losses else P34_LOSS_NAMES
    if set(losses) != set(names) or set(weights) != set(names):
        raise ValueError("P3.4 weighted total requires the arm's complete named loss set")
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError("P3.4 loss weights must be non-negative")
    return sum(float(weights[name]) * losses[name] for name in names)


class SlotSupervisionLift(nn.Module):
    """LOTUS-style token decoder for the four populated future slots."""

    def __init__(self, *, latent_dim: int = 128, hidden_size: int = 896) -> None:
        super().__init__()
        self.lift = nn.Linear(latent_dim, hidden_size, bias=False)
        nn.init.zeros_(self.lift.weight)

    def forward(self, slots: torch.Tensor, tied_weight: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[1] < 4:
            raise ValueError("P3.4 slot supervision requires four future slots")
        hidden = self.lift(slots[:, :4].float())
        return hidden @ tied_weight.detach().float().T


def slot_supervision_loss(
    *,
    lift: SlotSupervisionLift,
    flow_states: tuple[torch.Tensor, ...],
    tied_weight: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Deep-supervise slots 0..3 against horizons 1..4 through the frozen head."""

    if len(flow_states) != P34_FLOW_LOOPS + 1:
        raise ValueError("P3.4 slot loss requires initial plus four flow states")
    if teacher_tokens.shape != teacher_mask.shape or teacher_tokens.ndim != 2:
        raise ValueError("P3.4 slot token targets and mask must share [batch,horizon]")
    if teacher_tokens.shape[1] != 4 or not bool(teacher_mask.any()):
        raise ValueError("P3.4 slot supervision requires active horizons one through four")
    losses = []
    accuracies = []
    decoded = []
    for loop_index, state in enumerate(flow_states[1:], start=1):
        logits = lift(state, tied_weight)
        selected_logits = logits[teacher_mask]
        selected_tokens = teacher_tokens[teacher_mask]
        losses.append(F.cross_entropy(selected_logits, selected_tokens.long()))
        predictions = logits.argmax(dim=-1)
        accuracies.append(float(predictions[teacher_mask].eq(selected_tokens).float().mean().detach()))
        decoded.append(predictions.detach())
    weights = torch.arange(
        1,
        P34_FLOW_LOOPS + 1,
        device=losses[0].device,
        dtype=losses[0].dtype,
    ) / P34_FLOW_LOOPS
    stacked = torch.stack(losses)
    loss = (stacked * weights).sum() / weights.sum()
    return loss, {
        "loop_losses": [float(value.detach()) for value in losses],
        "loop_accuracies": accuracies,
        "final_slot_decodes": decoded[-1],
        "deep_supervision_weights": weights.detach().cpu().tolist(),
        "future_slot_indices": [0, 1, 2, 3],
        "target_horizons": [1, 2, 3, 4],
        "tied_head_frozen": True,
    }


def share_targets(*, slot_arm: bool) -> dict[str, float]:
    """Resolve floor ratios onto the 75% primary budget with 25% preservation."""

    raw = {"kl": 0.35, "aim": 0.15, "ce": 0.10, "gate": 0.03}
    if slot_arm:
        raw["slot"] = 0.10
    scale = 0.75 / sum(raw.values())
    targets = {name: value * scale for name, value in raw.items()}
    targets["preserve"] = 0.25
    return targets


def solve_static_loss_weights(
    raw_gradient_norms: Mapping[str, float], *, slot_arm: bool
) -> dict[str, Any]:
    names = P34_SLOT_LOSS_NAMES if slot_arm else P34_LOSS_NAMES
    if set(raw_gradient_norms) != set(names):
        raise ValueError("P3.4 loss-share calibration names changed")
    if any(float(raw_gradient_norms[name]) <= 0.0 for name in names):
        raise ValueError("P3.4 loss-share calibration requires positive raw norms")
    targets = share_targets(slot_arm=slot_arm)
    weights = {
        name: targets[name] / float(raw_gradient_norms[name]) for name in names
    }
    anchor = weights["kl"]
    weights = {name: value / anchor for name, value in weights.items()}
    realized_mass = {
        name: weights[name] * float(raw_gradient_norms[name]) for name in names
    }
    denominator = sum(realized_mass.values())
    realized = {name: value / denominator for name, value in realized_mass.items()}
    return {
        "slot_arm": slot_arm,
        "raw_gradient_norms": {
            name: float(raw_gradient_norms[name]) for name in names
        },
        "target_shares": targets,
        "static_weights_kl_normalized": weights,
        "solved_shares": realized,
        "preservation_at_or_below_25_percent": realized["preserve"] <= 0.25 + 1e-12,
    }


def postclip_loss_gradient_norms(
    *,
    losses: Mapping[str, torch.Tensor],
    module: Phase3StudentModules,
    parameters: Sequence[nn.Parameter],
    slot_lift: SlotSupervisionLift | None = None,
) -> dict[str, Any]:
    """Attribute unit-weight losses after the registered group clipping operation."""

    bundle = loss_gradient_bundle(
        losses=losses,
        module=module,
        parameters=parameters,
        slot_lift=slot_lift,
    )
    return postclip_gradient_norms_from_bundle(bundle)


def loss_gradient_bundle(
    *,
    losses: Mapping[str, torch.Tensor],
    module: Phase3StudentModules,
    parameters: Sequence[nn.Parameter],
    slot_lift: SlotSupervisionLift | None = None,
) -> dict[str, Any]:
    """Detach one batch's per-loss gradients for exact static-weight solving."""

    names = P34_SLOT_LOSS_NAMES if "slot" in losses else P34_LOSS_NAMES
    if set(losses) != set(names):
        raise ValueError("P3.4 post-clip attribution requires the complete arm loss set")
    active = [parameter for parameter in parameters if parameter.requires_grad]
    if not active:
        raise ValueError("P3.4 post-clip attribution has no trainable parameters")

    gradients: dict[str, dict[int, torch.Tensor]] = {}
    for index, name in enumerate(names):
        values = torch.autograd.grad(
            losses[name],
            active,
            retain_graph=index + 1 < len(names),
            allow_unused=True,
        )
        gradients[name] = {
            id(parameter): gradient.detach()
            for parameter, gradient in zip(active, values)
            if gradient is not None
        }

    parameter_group: dict[int, tuple[str, float]] = {}
    for parameter in module.bridge.parameters():
        if parameter.requires_grad:
            parameter_group[id(parameter)] = ("bridge", 0.5)
    for parameter in module.control.parameters():
        if parameter.requires_grad:
            parameter_group[id(parameter)] = ("heads", 1.0)
    if slot_lift is not None:
        for parameter in slot_lift.parameters():
            if parameter.requires_grad:
                parameter_group[id(parameter)] = ("heads", 1.0)
    uncovered = [parameter for parameter in active if id(parameter) not in parameter_group]
    if uncovered:
        raise RuntimeError("P3.4 clip attribution does not cover every trainable parameter")

    return {
        "names": names,
        "parameter_ids": [id(parameter) for parameter in active],
        "gradients": gradients,
        "parameter_groups": parameter_group,
    }


def postclip_gradient_norms_from_bundle(
    bundle: Mapping[str, Any], *, weights: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Apply exact combined group clips to one detached gradient bundle."""

    names = tuple(bundle["names"])
    weights = {name: 1.0 for name in names} if weights is None else {
        name: float(weights[name]) for name in names
    }
    if set(weights) != set(names) or any(value < 0.0 for value in weights.values()):
        raise ValueError("P3.4 post-clip weights must cover every non-negative loss")
    gradients = bundle["gradients"]
    parameter_ids = list(bundle["parameter_ids"])
    parameter_group = bundle["parameter_groups"]
    group_combined_sq: dict[str, float] = {}
    group_ceiling: dict[str, float] = {}
    for parameter_id in parameter_ids:
        group_name, ceiling = parameter_group[parameter_id]
        group_ceiling[group_name] = float(ceiling)
        available = [
            weights[name] * gradients[name][parameter_id]
            for name in names
            if parameter_id in gradients[name]
        ]
        if not available:
            continue
        combined = sum(available[1:], available[0])
        group_combined_sq[group_name] = group_combined_sq.get(group_name, 0.0) + float(
            combined.double().square().sum()
        )
    group_scales = {
        name: min(1.0, group_ceiling[name] / max(math.sqrt(value), 1e-30))
        for name, value in group_combined_sq.items()
    }
    postclip_norms: dict[str, float] = {}
    for name in names:
        squared = 0.0
        for parameter_id in parameter_ids:
            gradient = gradients[name].get(parameter_id)
            if gradient is None:
                continue
            group_name, _ = parameter_group[parameter_id]
            squared += float(
                (weights[name] * gradient.double() * group_scales[group_name]).square().sum()
            )
        postclip_norms[name] = math.sqrt(squared)
    if any(value <= 0.0 for value in postclip_norms.values()):
        raise RuntimeError("P3.4 calibration found a loss with zero post-clip gradient")
    denominator = sum(postclip_norms.values())
    return {
        "estimator": "unit-weight independent loss gradients under combined registered group clips",
        "weights": weights,
        "postclip_gradient_norms": postclip_norms,
        "unit_weight_shares": {
            name: value / denominator for name, value in postclip_norms.items()
        },
        "group_combined_norms": {
            name: math.sqrt(value) for name, value in group_combined_sq.items()
        },
        "group_clip_scales": group_scales,
        "group_clip_ceilings": group_ceiling,
    }


def solve_static_loss_weights_from_bundles(
    bundles: Sequence[Mapping[str, Any]],
    *,
    slot_arm: bool,
    tolerance: float = 1e-8,
    maximum_iterations: int = 200,
) -> dict[str, Any]:
    """Solve weights against mean post-clip norms across fixed strata."""

    if not bundles:
        raise ValueError("P3.4 static-weight solve needs at least one stratum bundle")
    names = P34_SLOT_LOSS_NAMES if slot_arm else P34_LOSS_NAMES
    if any(tuple(bundle["names"]) != names for bundle in bundles):
        raise ValueError("P3.4 gradient bundle loss names changed")
    targets = share_targets(slot_arm=slot_arm)
    unit_reads = [postclip_gradient_norms_from_bundle(bundle) for bundle in bundles]
    unit_mean = {
        name: sum(read["postclip_gradient_norms"][name] for read in unit_reads)
        / len(unit_reads)
        for name in names
    }
    initial = solve_static_loss_weights(unit_mean, slot_arm=slot_arm)
    weights = dict(initial["static_weights_kl_normalized"])
    converged = False
    final_reads: list[dict[str, Any]] = []
    shares: dict[str, float] = {}
    for iteration in range(1, maximum_iterations + 1):
        final_reads = [
            postclip_gradient_norms_from_bundle(bundle, weights=weights)
            for bundle in bundles
        ]
        mean_norms = {
            name: sum(read["postclip_gradient_norms"][name] for read in final_reads)
            / len(final_reads)
            for name in names
        }
        denominator = sum(mean_norms.values())
        shares = {name: mean_norms[name] / denominator for name in names}
        maximum_error = max(abs(shares[name] - targets[name]) for name in names)
        if maximum_error <= tolerance:
            converged = True
            break
        weights = {
            name: weights[name] * targets[name] / max(shares[name], 1e-30)
            for name in names
        }
        anchor = weights["kl"]
        weights = {name: value / anchor for name, value in weights.items()}
    if not converged:
        raise RuntimeError("P3.4 exact post-clip static-weight solve did not converge")
    return {
        "slot_arm": slot_arm,
        "strata": len(bundles),
        "unit_weight_mean_postclip_gradient_norms": unit_mean,
        "target_shares": targets,
        "static_weights_kl_normalized": weights,
        "solved_mean_postclip_shares": shares,
        "iterations": iteration,
        "maximum_share_error": max(abs(shares[name] - targets[name]) for name in names),
        "tolerance": tolerance,
        "converged": converged,
        "per_stratum_solved_reads": final_reads,
        "preservation_at_or_below_25_percent": shares["preserve"] <= 0.25 + tolerance,
    }


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
