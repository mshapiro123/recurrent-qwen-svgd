"""Shared numerical and geometry contracts for Phase-2 Stage 0A/0B."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


WHITEN_TAU = 1e-4
WHITEN_EPS_ABS = 1e-8
WHITEN_ALPHAS = (0.0, 0.5, 1.0)


def finite_quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize finite values while retaining explicit non-finite counts."""

    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {
            "count": 0,
            "finite_count": 0,
            "positive_infinity_count": 0,
            "negative_infinity_count": 0,
            "nan_count": 0,
            "finite_min": None,
            "finite_p10": None,
            "finite_median": None,
            "finite_p90": None,
            "finite_p95": None,
            "finite_max": None,
            "finite_mean": None,
        }
    finite = tensor[torch.isfinite(tensor)]
    result: dict[str, float | int | None] = {
        "count": int(tensor.numel()),
        "finite_count": int(finite.numel()),
        "positive_infinity_count": int(torch.isposinf(tensor).sum()),
        "negative_infinity_count": int(torch.isneginf(tensor).sum()),
        "nan_count": int(torch.isnan(tensor).sum()),
    }
    if finite.numel() == 0:
        result.update(
            {
                "finite_min": None,
                "finite_p10": None,
                "finite_median": None,
                "finite_p90": None,
                "finite_p95": None,
                "finite_max": None,
                "finite_mean": None,
            }
        )
        return result
    result.update(
        {
            "finite_min": float(finite.min()),
            "finite_p10": float(torch.quantile(finite, 0.10)),
            "finite_median": float(torch.quantile(finite, 0.50)),
            "finite_p90": float(torch.quantile(finite, 0.90)),
            "finite_p95": float(torch.quantile(finite, 0.95)),
            "finite_max": float(finite.max()),
            "finite_mean": float(finite.mean()),
        }
    )
    return result


def _normalized_probabilities(log_probs: torch.Tensor) -> torch.Tensor:
    values = log_probs.float()
    if values.ndim != 1:
        raise ValueError("coarse distributions must be rank one")
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        raise ValueError("coarse distribution has no finite support")
    safe = torch.full_like(values, float("-inf"))
    safe[finite] = values[finite]
    return torch.softmax(safe, dim=0)


def safe_coarse_lattice_metrics(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: Sequence[torch.Tensor],
    student_topk_mask: torch.Tensor,
    probability_floor: float = 1e-30,
) -> dict[str, Any]:
    """Compute finite diagnostics without hiding teacher mass outside student support."""

    if not teacher_log_probs:
        raise ValueError("at least one teacher distribution is required")
    student = _normalized_probabilities(student_log_probs)
    teachers = [_normalized_probabilities(value) for value in teacher_log_probs]
    if any(value.shape != student.shape for value in teachers):
        raise ValueError("student and teacher coarse distributions must align")
    mask = student_topk_mask.bool()
    if mask.shape != student.shape:
        raise ValueError("student_topk_mask must match the coarse distribution")

    teacher_stack = torch.stack(teachers)
    mixture = teacher_stack.mean(dim=0)
    mixture_log = mixture.clamp_min(probability_floor).log()
    teacher_logs = teacher_stack.clamp_min(probability_floor).log()
    js = torch.stack(
        [torch.sum(prob * (log_prob - mixture_log)) for prob, log_prob in zip(teacher_stack, teacher_logs)]
    ).mean()
    teacher_count = len(teachers)
    agreement = 1.0 if teacher_count == 1 else 1.0 - float(js) / math.log(teacher_count)
    support_miss = mixture[(student <= 0) & (mixture > 0)].sum()
    clipped_student = student.clamp_min(probability_floor)
    clipped_student = clipped_student / clipped_student.sum()
    clipped_kl = torch.sum(mixture * (mixture_log - clipped_student.log()))
    return {
        "teacher_count": teacher_count,
        "normalized_teacher_agreement": max(0.0, min(1.0, agreement)),
        "student_gap_coarse_kl_clipped": float(clipped_kl),
        "student_support_miss_mass": float(support_miss),
        "teachability_student_topk": float(mixture[mask].sum()),
        "teacher_tail_mass": float(mixture[-1]),
        "probability_floor": float(probability_floor),
    }


def probability_scale_coherence(vectors: Sequence[torch.Tensor]) -> float | None:
    """Cosine of successive teacher-scale probability increments."""

    if len(vectors) != 3:
        return None
    seven, fourteen, thirty_two = [_normalized_probabilities(value) for value in vectors]
    first = fourteen - seven
    second = thirty_two - fourteen
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-12:
        return None
    result = torch.dot(first, second) / denominator
    if not bool(torch.isfinite(result)):
        return None
    return float(result.clamp(-1.0, 1.0))


def document_split(
    document_ids: Sequence[str], *, calibration_fraction: float, seed: int
) -> torch.Tensor:
    """Return a deterministic document-level calibration mask."""

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between zero and one")
    if len(set(document_ids)) < 2:
        raise ValueError("document-disjoint split requires at least two documents")
    scores: dict[str, float] = {}
    for document_id in set(document_ids):
        digest = hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).digest()
        scores[document_id] = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    mask = torch.tensor(
        [scores[document_id] < calibration_fraction for document_id in document_ids],
        dtype=torch.bool,
    )
    if not bool(mask.any()) or not bool((~mask).any()):
        ordered = sorted(scores, key=scores.get)
        cutoff = max(1, min(len(ordered) - 1, round(len(ordered) * calibration_fraction)))
        calibration = set(ordered[:cutoff])
        mask = torch.tensor(
            [document_id in calibration for document_id in document_ids], dtype=torch.bool
        )
    return mask


def effective_eigenvalues(
    raw: torch.Tensor, *, tau: float = WHITEN_TAU, eps_abs: float = WHITEN_EPS_ABS
) -> torch.Tensor:
    """Apply the single registered relative/absolute eigenvalue floor."""

    values = raw.float()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("eigenvalues must be a nonempty rank-one tensor")
    if bool((values < 0).any()):
        raise ValueError("covariance eigenvalues cannot be negative")
    floor = max(float(values.max()) * float(tau), float(eps_abs))
    return values.clamp_min(floor)


def _rms_unit(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)


class CanonicalizerTransform(nn.Module):
    """Frozen affine canonicalizer with one shared PCA basis across alpha arms."""

    def __init__(
        self,
        *,
        projector_weight: torch.Tensor,
        teacher_mean: torch.Tensor,
        canonical_mean: torch.Tensor,
        whiten_basis: torch.Tensor,
        whiten_eigenvalues: torch.Tensor,
        whiten_alpha: float,
        layer_weights: torch.Tensor,
        n_slots: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        if projector_weight.shape[1] != n_slots * latent_dim:
            raise ValueError("projector output does not match slot geometry")
        if canonical_mean.shape != (n_slots, latent_dim):
            raise ValueError("canonical mean does not match slot geometry")
        if whiten_basis.shape != (latent_dim, latent_dim):
            raise ValueError("whitening basis must act within each slot")
        if whiten_eigenvalues.shape != (latent_dim,):
            raise ValueError("whitening eigenvalues must match latent width")
        if layer_weights.ndim != 1:
            raise ValueError("layer weights must be rank one")
        self.n_slots = int(n_slots)
        self.latent_dim = int(latent_dim)
        self.register_buffer("projector_weight", projector_weight.float())
        self.register_buffer("teacher_mean", teacher_mean.float())
        self.register_buffer("canonical_mean", canonical_mean.float())
        self.register_buffer("whiten_basis", whiten_basis.float())
        self.register_buffer("whiten_eigenvalues", whiten_eigenvalues.float())
        self.register_buffer("whiten_alpha", torch.tensor(float(whiten_alpha)))
        self.register_buffer("layer_weights", layer_weights.float())

    def pooled_teacher(self, selected_states: torch.Tensor) -> torch.Tensor:
        if selected_states.ndim != 3:
            raise ValueError("selected teacher states must have shape [batch, layers, width]")
        if selected_states.shape[1] != self.layer_weights.numel():
            raise ValueError("teacher layer count differs from frozen layer weights")
        weights = self.layer_weights / self.layer_weights.sum()
        return (_rms_unit(selected_states.float()) * weights.view(1, -1, 1)).sum(dim=1)

    def forward(self, selected_states: torch.Tensor) -> torch.Tensor:
        pooled = self.pooled_teacher(selected_states)
        z = (pooled - self.teacher_mean) @ self.projector_weight
        z = z.view(-1, self.n_slots, self.latent_dim) - self.canonical_mean
        scale = self.whiten_eigenvalues.pow(-0.5 * self.whiten_alpha)
        return (z @ self.whiten_basis) * scale


def _token_sketch(
    ids: torch.Tensor, log_probs: torch.Tensor, *, width: int, seed: int
) -> torch.Tensor:
    probabilities = torch.softmax(log_probs.float(), dim=-1)
    token_ids = ids.long()
    buckets = torch.remainder(token_ids * 1_103_515_245 + int(seed), width)
    signs = torch.where(
        torch.remainder(token_ids * 2_654_435_761 + int(seed) * 17, 2) == 0,
        torch.ones_like(probabilities),
        -torch.ones_like(probabilities),
    )
    result = torch.zeros(
        (ids.shape[0], width), dtype=torch.float32, device=ids.device
    )
    result.scatter_add_(1, buckets, probabilities * signs)
    return result


def build_anchor_targets(
    *,
    topk_ids: torch.Tensor,
    topk_log_probs: torch.Tensor,
    middle_states: torch.Tensor,
    horizons: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_count: int,
    latent_dim: int,
    n_slots: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build four future slots; unavailable span slots remain zero and masked."""

    if latent_dim % 2:
        raise ValueError("latent_dim must be even for the two-part target")
    if n_slots < 4:
        raise ValueError("at least four canonical slots are required")
    if not (
        topk_ids.shape == topk_log_probs.shape
        and topk_ids.shape[0] == middle_states.shape[0] == horizons.numel() == anchor_indices.numel()
    ):
        raise ValueError("target inputs are not sample aligned")
    half = latent_dim // 2
    logit_features = _token_sketch(topk_ids, topk_log_probs, width=half, seed=seed)
    generator = torch.Generator(device=middle_states.device).manual_seed(int(seed) + 1)
    state_projection = torch.randn(
        middle_states.shape[-1],
        half,
        generator=generator,
        dtype=torch.float32,
        device=middle_states.device,
    ) / math.sqrt(middle_states.shape[-1])
    state_features = middle_states.float() @ state_projection
    anchor_indices = anchor_indices.to(device=middle_states.device, dtype=torch.long)
    horizons = horizons.to(device=middle_states.device, dtype=torch.long)
    final_features = torch.zeros(
        (anchor_count, half), dtype=torch.float32, device=middle_states.device
    )
    final_seen = torch.zeros(anchor_count, dtype=torch.bool, device=middle_states.device)
    final_mask = horizons.eq(4)
    final_features[anchor_indices[final_mask]] = state_features[final_mask]
    final_seen[anchor_indices[final_mask]] = True
    if not bool(final_seen.all()):
        raise ValueError("every anchor must contain a horizon-four state")

    if bool(((horizons < 1) | (horizons > 4)).any()):
        raise ValueError("Stage 0A targets require horizons one through four")
    targets = torch.zeros(
        (anchor_count, n_slots, latent_dim), dtype=torch.float32, device=middle_states.device
    )
    mask = torch.zeros(
        (anchor_count, n_slots), dtype=torch.bool, device=middle_states.device
    )
    state_delta = final_features.index_select(0, anchor_indices) - state_features
    sample_targets = torch.cat([logit_features.to(middle_states.device), state_delta], dim=1)
    targets[anchor_indices, horizons - 1] = sample_targets
    mask[anchor_indices, horizons - 1] = True
    return targets, mask


def affine_interpolate(start: torch.Tensor, stop: torch.Tensor, tau: float) -> torch.Tensor:
    if not 0.0 <= float(tau) <= 1.0:
        raise ValueError("interpolation fraction must be in [0, 1]")
    return (1.0 - float(tau)) * start + float(tau) * stop


class SharedResidualFlowPilot(nn.Module):
    """Small slotwise serial-flow pilot; persistent state is never renormalized."""

    def __init__(
        self, *, latent_dim: int, context_dim: int, max_steps: int = 4, expansion: int = 4
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        hidden = int(expansion) * int(latent_dim)
        self.context_proj = nn.Linear(context_dim, latent_dim, bias=False)
        self.step_embedding = nn.Embedding(max_steps, latent_dim)
        self.input = nn.Linear(3 * latent_dim, hidden)
        self.delta = nn.Linear(hidden, latent_dim)
        self.gate = nn.Linear(hidden, latent_dim)
        nn.init.normal_(self.delta.weight, std=1e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def step(self, state: torch.Tensor, context: torch.Tensor, index: int) -> torch.Tensor:
        normalized = _rms_unit(state)
        innovation = _rms_unit(state - state.mean(dim=1, keepdim=True))
        projected_context = self.context_proj(context).unsqueeze(1).expand_as(state)
        step_feature = self.step_embedding.weight[index].view(1, 1, -1).expand_as(state)
        hidden = torch.nn.functional.silu(
            self.input(torch.cat([normalized, innovation + projected_context, step_feature], dim=-1))
        )
        return state + torch.sigmoid(self.gate(hidden)) * self.delta(hidden)

    def forward(self, state: torch.Tensor, context: torch.Tensor, *, steps: int) -> torch.Tensor:
        if steps < 0 or steps > self.max_steps:
            raise ValueError(f"requested steps violate loop cap {self.max_steps}")
        result = state
        for index in range(steps):
            result = self.step(result, context, index)
        return result
