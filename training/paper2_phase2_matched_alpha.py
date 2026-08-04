"""Pure contracts and metrics for the locked Phase-2 matched-alpha pilot."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable

import torch
from torch import nn


PROTOCOL_LOCK_COMMIT = "cf6747264e48e2de657eb2a1646f1e7c4f152ea5"


def stable_fraction(value: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def document_partition(
    document_ids: Iterable[str], *, evaluation_fraction: float, seed: int
) -> torch.Tensor:
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must lie strictly between zero and one")
    values = [stable_fraction(str(value), seed=seed) < evaluation_fraction for value in document_ids]
    return torch.tensor(values, dtype=torch.bool)


def alpha_transform(
    raw: torch.Tensor, basis: torch.Tensor, eigenvalues: torch.Tensor, alpha: float
) -> torch.Tensor:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if raw.shape[-1] != basis.shape[0] or basis.shape[1] != eigenvalues.numel():
        raise ValueError("canonical transform shapes do not align")
    return (raw.float() @ basis.float()) * eigenvalues.float().pow(-0.5 * float(alpha))


def normalize_sparse_with_tail(
    candidate_logits: torch.Tensor, tail_logit: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if candidate_logits.shape != mask.shape:
        raise ValueError("candidate logits and mask must share shape")
    if tail_logit.shape != candidate_logits.shape[:-1]:
        raise ValueError("tail logit must omit only the candidate dimension")
    masked = candidate_logits.float().masked_fill(~mask.bool(), float("-inf"))
    joined = torch.cat([masked, tail_logit.float().unsqueeze(-1)], dim=-1)
    return torch.log_softmax(joined, dim=-1)


def masked_sparse_kl(
    target_log_probs: torch.Tensor,
    predicted_log_probs: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if target_log_probs.shape != predicted_log_probs.shape:
        raise ValueError("target and predicted sparse distributions must share shape")
    if target_log_probs.shape[:-1] != candidate_mask.shape[:-1]:
        raise ValueError("candidate mask leading dimensions do not align")
    if target_log_probs.shape[-1] != candidate_mask.shape[-1] + 1:
        raise ValueError("sparse distribution must contain candidates plus one tail")
    support = torch.cat(
        [
            candidate_mask.bool(),
            torch.ones((*candidate_mask.shape[:-1], 1), dtype=torch.bool, device=candidate_mask.device),
        ],
        dim=-1,
    )
    positive_target = support & torch.isfinite(target_log_probs)
    safe_target = torch.where(
        positive_target, target_log_probs, torch.zeros_like(target_log_probs)
    )
    safe_predicted = torch.where(
        positive_target, predicted_log_probs, torch.zeros_like(predicted_log_probs)
    )
    probability = torch.where(
        positive_target, safe_target.exp(), torch.zeros_like(safe_target)
    )
    return (probability * (safe_target - safe_predicted)).sum(dim=-1)


def distribution_overlap(target_log_probs: torch.Tensor, draft_log_probs: torch.Tensor) -> torch.Tensor:
    if target_log_probs.shape != draft_log_probs.shape:
        raise ValueError("target and draft distributions must share shape")
    return torch.minimum(target_log_probs.exp(), draft_log_probs.exp()).sum(dim=-1)


def expected_accepted_length(overlaps: torch.Tensor) -> torch.Tensor:
    if overlaps.ndim != 2:
        raise ValueError("overlaps must have shape [batch, horizons]")
    return overlaps.clamp(0.0, 1.0).cumprod(dim=1).sum(dim=1)


def wilson_lower(correct: int, total: int, *, z: float = 1.959963984540054) -> float:
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("invalid binomial counts")
    p = correct / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (center - radius) / denominator


def quality_noninferior(
    retained: int,
    baseline_correct: int,
    *,
    point_floor: float = 0.997,
    wilson_floor: float = 0.990,
) -> bool:
    if baseline_correct <= 0:
        return False
    return retained / baseline_correct >= point_floor and wilson_lower(
        retained, baseline_correct
    ) >= wilson_floor


def build_adamw_groups(module: nn.Module, *, weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    norm_parameter_ids = {
        id(parameter)
        for child in module.modules()
        if isinstance(child, nn.RMSNorm)
        for parameter in child.parameters(recurse=False)
    }
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        is_scalar = parameter.ndim == 0 or parameter.numel() == 1
        is_scalar_bank = name.endswith(("gate_logits", "rho_logits"))
        if name.endswith("bias") or id(parameter) in norm_parameter_ids or is_scalar or is_scalar_bank:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay or not no_decay:
        raise ValueError("registered optimizer requires both decay and no-decay groups")
    return [
        {"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]


def clip_module_groups(module: nn.Module) -> dict[str, tuple[list[nn.Parameter], float]]:
    heads = list(module.initializer.parameters()) + list(module.control.parameters()) + list(
        module.draft.parameters()
    )
    return {
        "refiner": (list(module.flow.parameters()), 1.0),
        "bridge": (list(module.bridge.parameters()), 0.5),
        "heads": (heads, 1.0),
    }


def trust_saturated(history: Iterable[bool], *, window: int = 100, maximum_nonzero: int = 50) -> bool:
    values = list(bool(value) for value in history)
    return len(values) >= window and sum(values[-window:]) > maximum_nonzero


def summarize_clip_fractions(events: Iterable[tuple[str, bool]]) -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    clipped: dict[str, int] = defaultdict(int)
    for name, was_clipped in events:
        totals[name] += 1
        clipped[name] += int(bool(was_clipped))
    return {name: clipped[name] / totals[name] for name in sorted(totals)}


def paired_bootstrap_interval(
    differences: torch.Tensor, *, seed: int, draws: int = 10_000
) -> tuple[float, float, float]:
    values = differences.detach().float().cpu().reshape(-1)
    if values.numel() < 2:
        raise ValueError("paired bootstrap requires at least two rows")
    generator = torch.Generator().manual_seed(int(seed))
    means = []
    for _ in range(int(draws)):
        indices = torch.randint(values.numel(), (values.numel(),), generator=generator)
        means.append(values.index_select(0, indices).mean())
    samples = torch.stack(means)
    return float(values.mean()), float(torch.quantile(samples, 0.025)), float(
        torch.quantile(samples, 0.975)
    )


def practical_equivalence(
    *, difference_ci: tuple[float, float], reference_mean: float, relative_band: float
) -> bool:
    width = abs(float(reference_mean)) * float(relative_band)
    return difference_ci[0] >= -width and difference_ci[1] <= width
