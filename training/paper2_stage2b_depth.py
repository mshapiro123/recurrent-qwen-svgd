"""Registered contracts and statistics for the Stage 2B-D depth campaign."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from models.lora import LoopScopedLoRALinear


STAGE2B_TOTAL_STEPS = 24_000
STAGE2B_KILL_GATE_STEP = 5_000
STAGE2B_LOOK_INTERVAL = 1_000
STAGE2B_BATCH_SIZE = 128
STAGE2B_FLOW_LOOPS = 4
STAGE2B_EMA_DECAY = 0.999
STAGE2B_WARMUP_STEPS = 500
STAGE2B_LANDING_START = 21_601
STAGE2B_AMPLITUDE_LOW = 0.02
STAGE2B_AMPLITUDE_HIGH = 0.11
STAGE2B_READ_AMPLITUDE = 0.05

STAGE_ALLOCATIONS = {
    "M0": [0, 0],
    "M1": [0, 0],
    "M2": [1, 2_500],
    "M3": [2_501, 5_000],
    "M4": [5_001, 24_000],
}


@dataclass(frozen=True)
class DepthObjectiveWeights:
    ce: float
    kl: float
    monotonicity: float
    verified_depth: float = 0.0

    def validate(self) -> None:
        values = (self.ce, self.kl, self.monotonicity, self.verified_depth)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("objective weights must be finite and nonnegative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("objective weights must sum to one")
        if min(self.ce, self.kl, self.monotonicity) <= 0.0:
            raise ValueError("CE, KL, and monotonicity must each remain active")


def configure_stage2b_trainable_groups(wrapper: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Freeze the substrate and expose the three registered optimizer groups."""

    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    attachment = getattr(wrapper, "stage2b_depth_attachment", None)
    if attachment is None:
        raise ValueError("Stage 2B attachment must be installed before grouping parameters")
    groups: dict[str, list[torch.nn.Parameter]] = {
        "new_modules": [],
        "gates": [],
        "loop_lora": [],
    }
    for name, parameter in attachment.named_parameters():
        group = "gates" if name.startswith(("control.", "bridge.")) else "new_modules"
        parameter.requires_grad_(True)
        groups[group].append(parameter)
    for module in wrapper.modules():
        if isinstance(module, LoopScopedLoRALinear):
            for parameter in module.lora_parameters():
                parameter.requires_grad_(True)
                groups["loop_lora"].append(parameter)
    identities = [id(parameter) for values in groups.values() for parameter in values]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Stage 2B optimizer groups overlap")
    if any(not values for values in groups.values()):
        raise RuntimeError("Stage 2B optimizer group is unexpectedly empty")
    return groups


def calibrated_gradient_share_weights(
    raw_gradient_norms: Mapping[str, float],
    *,
    targets: Mapping[str, float] | None = None,
) -> DepthObjectiveWeights:
    targets = targets or {"kl": 0.5, "ce": 0.3, "monotonicity": 0.2}
    if set(raw_gradient_norms) != {"ce", "kl", "monotonicity"}:
        raise ValueError("gradient calibration requires CE, KL, and monotonicity norms")
    if set(targets) != set(raw_gradient_norms) or abs(sum(targets.values()) - 1.0) > 1e-9:
        raise ValueError("gradient-share targets must name the three active losses and sum to one")
    unnormalized = {}
    for name, norm in raw_gradient_norms.items():
        if not math.isfinite(float(norm)) or float(norm) <= 0.0:
            raise ValueError(f"nonpositive gradient norm for {name}")
        unnormalized[name] = float(targets[name]) / float(norm)
    scale = sum(unnormalized.values())
    result = DepthObjectiveWeights(
        ce=unnormalized["ce"] / scale,
        kl=unnormalized["kl"] / scale,
        monotonicity=unnormalized["monotonicity"] / scale,
    )
    result.validate()
    return result


def sparse_forward_kl(
    student_logits: torch.Tensor,
    teacher_topk_token_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if student_logits.ndim != 3:
        raise ValueError("student logits must have shape [batch, positions, vocabulary]")
    batch, positions, vocabulary = student_logits.shape
    if teacher_topk_token_ids.shape != (batch, positions, 128):
        raise ValueError("teacher token lattice must have shape [batch, positions, 128]")
    if teacher_topk_logits.shape != teacher_topk_token_ids.shape:
        raise ValueError("teacher top-k logits must align with teacher token IDs")
    if loss_mask.shape != student_logits.shape[:-1] or loss_mask.dtype != torch.bool:
        raise ValueError("loss mask must align with token logits")
    if bool(((teacher_topk_token_ids < 0) | (teacher_topk_token_ids >= vocabulary)).any()):
        raise ValueError("teacher lattice token is outside the student vocabulary")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    counts = loss_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every example requires at least one supervised position")

    selected_student = student_logits.gather(-1, teacher_topk_token_ids.long()).float()
    teacher_log = F.log_softmax(teacher_topk_logits.float() / temperature, dim=-1)
    student_log = F.log_softmax(selected_student / temperature, dim=-1)
    token_kl = (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1)
    mask = loss_mask.to(token_kl.dtype)
    per_example = (token_kl * mask).sum(dim=1) / counts.to(token_kl.dtype)
    return per_example.mean() * temperature**2


def monotonicity_hinge(loop_kls: Sequence[torch.Tensor], *, delta: float) -> torch.Tensor:
    if len(loop_kls) != STAGE2B_FLOW_LOOPS:
        raise ValueError("Stage 2B-D requires KL at exactly four loop depths")
    if delta < 0 or not math.isfinite(delta):
        raise ValueError("hinge margin must be finite and nonnegative")
    return torch.stack(
        [F.relu(loop_kls[index] - loop_kls[index - 1] + delta) for index in range(1, 4)]
    ).sum()


def depth_objective(
    *,
    loop_logits: Sequence[torch.Tensor],
    teacher_topk_token_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    weights: DepthObjectiveWeights,
    hinge_delta: float,
    verified_depth_losses: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights.validate()
    if len(loop_logits) != STAGE2B_FLOW_LOOPS:
        raise ValueError("deep supervision requires four loop outputs")
    if not loop_logits or loop_logits[0].ndim != 3:
        raise ValueError("loop logits must have shape [batch, positions, vocabulary]")
    batch, positions, vocabulary = loop_logits[0].shape
    if any(logits.shape != loop_logits[0].shape for logits in loop_logits):
        raise ValueError("all loop logits must share one shape")
    if teacher_tokens.shape != (batch, positions):
        raise ValueError("teacher targets must have shape [batch, positions]")
    if loss_mask.shape != (batch, positions) or loss_mask.dtype != torch.bool:
        raise ValueError("loss mask must be boolean [batch, positions]")
    if bool(((teacher_tokens < 0) | (teacher_tokens >= vocabulary)).any()):
        raise ValueError("teacher target token is outside the student vocabulary")
    counts = loss_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every example requires at least one supervised position")
    kls = [
        sparse_forward_kl(
            logits,
            teacher_topk_token_ids,
            teacher_topk_logits,
            loss_mask,
        )
        for logits in loop_logits
    ]
    ce_tokens = teacher_tokens.masked_fill(~loss_mask, -100)
    ce = torch.stack(
        [
            (
                F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                ce_tokens.reshape(-1),
                ignore_index=-100,
                reduction="none",
                )
                .reshape(batch, positions)
                .sum(dim=1)
                / counts.to(torch.float32)
            ).mean()
            for logits in loop_logits
        ]
    ).mean()
    kl = torch.stack(kls).mean()
    mono = monotonicity_hinge(kls, delta=hinge_delta)
    if weights.verified_depth > 0:
        if not verified_depth_losses:
            raise ValueError("verified-depth weight is active but no verified targets were supplied")
        verified = torch.stack(list(verified_depth_losses)).mean()
    else:
        verified = ce.new_zeros(())
    components = {"ce": ce, "kl": kl, "monotonicity": mono, "verified_depth": verified}
    total = sum(getattr(weights, name) * loss for name, loss in components.items())
    return total, components


def stage2b_learning_rate(step: int, *, peak: float) -> float:
    if not 1 <= int(step) <= STAGE2B_TOTAL_STEPS:
        raise ValueError("Stage 2B-D step must be in 1..24000")
    if step <= STAGE2B_WARMUP_STEPS:
        return peak * step / STAGE2B_WARMUP_STEPS
    if step < STAGE2B_LANDING_START:
        return peak
    progress = (step - STAGE2B_LANDING_START + 1) / (
        STAGE2B_TOTAL_STEPS - STAGE2B_LANDING_START + 1
    )
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def stage_for_step(step: int) -> str:
    if step == 0:
        return "M1"
    for stage in ("M2", "M3", "M4"):
        low, high = STAGE_ALLOCATIONS[stage]
        if low <= step <= high:
            return stage
    raise ValueError("step is outside the Stage 2B-D schedule")


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 20_260_818,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() < 2:
        raise ValueError("bootstrap interval requires at least two rows")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indexes = torch.randint(
        tensor.numel(), (draws, tensor.numel()), generator=generator
    )
    means = tensor[indexes].mean(dim=1).sort().values
    low_index = max(0, int(math.floor((alpha / 2.0) * draws)))
    high_index = min(draws - 1, int(math.ceil((1.0 - alpha / 2.0) * draws)) - 1)
    return {
        "rows": tensor.numel(),
        "draws": draws,
        "seed": seed,
        "mean": float(tensor.mean().item()),
        "ci95_low": float(means[low_index].item()),
        "ci95_high": float(means[high_index].item()),
    }


def kill_gate_seed_read(
    per_loop_margins: Sequence[Sequence[float]],
    *,
    draws: int = 10_000,
    seed: int = 20_260_818,
) -> dict[str, Any]:
    if len(per_loop_margins) != STAGE2B_FLOW_LOOPS:
        raise ValueError("kill gate requires margins from loops one through four")
    lengths = {len(values) for values in per_loop_margins}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("kill-gate loop margins must be row-aligned")
    transitions = {}
    for index in range(1, STAGE2B_FLOW_LOOPS):
        differences = [
            float(current) - float(previous)
            for current, previous in zip(per_loop_margins[index], per_loop_margins[index - 1])
        ]
        transitions[f"k{index}_to_k{index + 1}"] = bootstrap_mean_interval(
            differences, draws=draws, seed=seed + index
        )
    required = (transitions["k2_to_k3"], transitions["k3_to_k4"])
    separating = all(float(result["mean"]) > 0 and float(result["ci95_low"]) > 0 for result in required)
    return {"transitions": transitions, "separating": separating}


def kill_gate_verdict(seed_reads: Mapping[int, Mapping[str, Any]]) -> str:
    if set(seed_reads) != {0, 1}:
        raise ValueError("registered kill gate requires both seeds")
    separating = [bool(seed_reads[seed]["separating"]) for seed in (0, 1)]
    return "continue_m4" if any(separating) else "terminate_and_bank_boundary"


def paired_sign_test_power(
    *,
    rows: int,
    net_improvement: int,
    discordance_rate: float,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """One-sided exact paired-sign power under a fixed discordance scenario."""

    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError("discordance rate must be inside (0, 1]")
    discordant = max(abs(net_improvement), int(round(rows * discordance_rate)))
    probability = 0.5 * (1.0 + net_improvement / discordant)
    probability = min(max(probability, 0.0), 1.0)

    def tail(n: int, p: float, threshold: int) -> float:
        return sum(
            math.comb(n, value) * p**value * (1.0 - p) ** (n - value)
            for value in range(threshold, n + 1)
        )

    critical = discordant
    for candidate in range(discordant + 1):
        if tail(discordant, 0.5, candidate) <= alpha:
            critical = candidate
            break
    return {
        "rows": rows,
        "net_improvement": net_improvement,
        "discordance_rate": discordance_rate,
        "discordant_rows": discordant,
        "alternative_fix_probability": probability,
        "critical_fixes": critical,
        "one_sided_alpha": alpha,
        "power": tail(discordant, probability, critical),
    }


def validate_lock(lock: Mapping[str, Any], *, require_signature: bool) -> None:
    if lock.get("kind") != "paper2_stage2b_depth_executed_lock_v1":
        raise RuntimeError("Stage 2B-D lock kind changed")
    if lock.get("charter", {}).get("sha256") != (
        "48aed379110a4614b6091592890713af9fe40b444f0a918c8ddef3cf5845d3b0"
    ):
        raise RuntimeError("Stage 2B-D governing charter changed")
    if lock.get("sealed_partitions") != {
        "confirm_scored": False,
        "eval_e_scored": False,
        "remain_sealed": True,
    }:
        raise RuntimeError("sealed-partition contract changed")
    if lock.get("training", {}).get("steps") != STAGE2B_TOTAL_STEPS:
        raise RuntimeError("Stage 2B-D dose changed")
    if lock.get("training", {}).get("stage_allocations") != STAGE_ALLOCATIONS:
        raise RuntimeError("Stage 2B-D curriculum changed")
    if lock.get("kill_gate", {}).get("step") != STAGE2B_KILL_GATE_STEP:
        raise RuntimeError("Stage 2B-D kill-gate position changed")
    if require_signature:
        if not (
            lock.get("status") == "approved_for_training"
            and lock.get("locked_before_training") is True
            and lock.get("training_authorized") is True
            and lock.get("mark_signed") is True
            and lock.get("unresolved_lock_fields") == []
        ):
            raise RuntimeError("Stage 2B-D training remains disabled pending Mark's signature")
    else:
        if lock.get("training_authorized") is not False:
            raise RuntimeError("draft Stage 2B-D lock unexpectedly authorizes training")


def load_and_validate_lock(path: str | Path, *, require_signature: bool) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_lock(lock, require_signature=require_signature)
    return lock


def assert_optimizer_construction_authorized(lock_path: str | Path) -> None:
    """Mandatory call immediately before any optimizer is constructed."""

    load_and_validate_lock(lock_path, require_signature=True)
