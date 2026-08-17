"""Registered answer-region objective for the Paper Two Stage 2A screen."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Stage2AObjectiveReadout:
    """Loss values and per-example support for an auditable training step."""

    loss: torch.Tensor
    cross_entropy: torch.Tensor
    forward_kl: torch.Tensor
    answer_positions_per_example: torch.Tensor


def stage2a_answer_region_objective(
    *,
    student_logits: torch.Tensor,
    teacher_topk_token_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
    answer_region_mask: torch.Tensor,
    ce_weight: float = 0.5,
    kl_weight: float = 0.5,
    temperature: float = 1.0,
) -> Stage2AObjectiveReadout:
    """Compute the registered 0.5 CE + 0.5 forward-KL objective.

    The KL is renormalized over the cached teacher top-128 lattice. Losses are
    averaged over answer-bearing positions within each example and then over
    the batch, so examples with longer normalized answers do not receive more
    weight. Position zero and prompt/formatting positions must be masked out by
    the caller's battery-specific KP-1R answer-region mask.
    """

    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, positions, vocab]")
    batch, positions, vocab = student_logits.shape
    expected_prefix = (batch, positions)
    if teacher_topk_token_ids.ndim != 3 or teacher_topk_token_ids.shape[:2] != expected_prefix:
        raise ValueError("teacher_topk_token_ids must have shape [batch, positions, K]")
    if teacher_topk_logits.shape != teacher_topk_token_ids.shape:
        raise ValueError("teacher_topk_logits must match teacher_topk_token_ids")
    if teacher_topk_token_ids.shape[-1] != 128:
        raise ValueError("the registered teacher lattice contains exactly 128 tokens")
    if teacher_token_ids.shape != expected_prefix:
        raise ValueError("teacher_token_ids must have shape [batch, positions]")
    if answer_region_mask.shape != expected_prefix or answer_region_mask.dtype != torch.bool:
        raise ValueError("answer_region_mask must be boolean [batch, positions]")
    if positions < 1 or bool(answer_region_mask[:, 0].any()):
        raise ValueError("position zero must carry no Stage 2A loss")
    counts = answer_region_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every example requires at least one answer-bearing position")
    if teacher_topk_token_ids.dtype == torch.bool or teacher_topk_token_ids.is_floating_point():
        raise TypeError("teacher_topk_token_ids must contain integer token IDs")
    if teacher_token_ids.dtype == torch.bool or teacher_token_ids.is_floating_point():
        raise TypeError("teacher_token_ids must contain integer token IDs")
    if bool(((teacher_topk_token_ids < 0) | (teacher_topk_token_ids >= vocab)).any()):
        raise ValueError("teacher lattice token ID is outside the student vocabulary")
    if bool(((teacher_token_ids < 0) | (teacher_token_ids >= vocab)).any()):
        raise ValueError("teacher target token ID is outside the student vocabulary")
    if not float(temperature) > 0.0:
        raise ValueError("temperature must be positive")
    if abs(float(ce_weight) - 0.5) > 1e-12 or abs(float(kl_weight) - 0.5) > 1e-12:
        raise ValueError("Stage 2A binds CE and KL weights to 0.5 each")

    lattice_student_logits = student_logits.gather(
        dim=-1, index=teacher_topk_token_ids.long()
    ).float()
    teacher_log_probs = F.log_softmax(teacher_topk_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(lattice_student_logits / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    per_position_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    per_position_ce = F.cross_entropy(
        student_logits.float().reshape(-1, vocab),
        teacher_token_ids.long().reshape(-1),
        reduction="none",
    ).reshape(batch, positions)

    mask = answer_region_mask.to(dtype=per_position_ce.dtype)
    denominator = counts.to(dtype=per_position_ce.dtype)
    ce = ((per_position_ce * mask).sum(dim=1) / denominator).mean()
    kl = ((per_position_kl * mask).sum(dim=1) / denominator).mean()
    loss = float(ce_weight) * ce + float(kl_weight) * kl
    return Stage2AObjectiveReadout(
        loss=loss,
        cross_entropy=ce,
        forward_kl=kl,
        answer_positions_per_example=counts,
    )
