"""Phase 3.2 cache schema, gate labels, and oracle-gradient batching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Mapping, Optional

import torch


class GateLabel(IntEnum):
    IGNORED = -1
    NEGATIVE = 0
    POSITIVE = 1


@dataclass(frozen=True)
class AgreementLabelInputs:
    student_top1: int
    teacher_14b_top1: int
    teacher_32b_top1: Optional[int]
    teachability: float
    confident_agreement_margin: float


@dataclass(frozen=True)
class VerifiedLabelInputs:
    student_right: bool
    teacher_right: bool
    confident_agreement: bool


def agreement_gate_label(
    inputs: AgreementLabelInputs,
    *,
    teachability_threshold: float,
    confident_agreement_margin_threshold: float,
    require_32b: bool = True,
) -> GateLabel:
    cross_scale = (
        inputs.teacher_32b_top1 is not None
        and inputs.teacher_14b_top1 == inputs.teacher_32b_top1
    )
    if require_32b and not cross_scale:
        return GateLabel.IGNORED
    disagreement = inputs.student_top1 != inputs.teacher_14b_top1
    if disagreement and cross_scale and inputs.teachability >= teachability_threshold:
        return GateLabel.POSITIVE
    if (
        inputs.student_top1 == inputs.teacher_14b_top1
        and inputs.confident_agreement_margin >= confident_agreement_margin_threshold
    ):
        return GateLabel.NEGATIVE
    return GateLabel.IGNORED


def verified_gate_label(inputs: VerifiedLabelInputs) -> GateLabel:
    if inputs.teacher_right and not inputs.student_right:
        return GateLabel.POSITIVE
    if inputs.student_right and inputs.teacher_right and inputs.confident_agreement:
        return GateLabel.NEGATIVE
    return GateLabel.IGNORED


def gate_loss_mask(labels: torch.Tensor) -> torch.Tensor:
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("gate labels must be an integer tensor")
    valid = (labels == int(GateLabel.POSITIVE)) | (labels == int(GateLabel.NEGATIVE))
    invalid = ~(valid | (labels == int(GateLabel.IGNORED)))
    if bool(invalid.any()):
        raise ValueError("gate labels must be positive, negative, or ignored")
    return valid


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_cache_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required_common = {
        "record_id",
        "source_stratum",
        "battery",
        "document_id",
        "item_id",
        "prediction_position",
        "loop_index",
        "student_top1",
        "teacher_14b_top1",
        "gate_label",
    }
    missing = sorted(required_common - set(record))
    if missing:
        raise ValueError(f"P3.2 cache record missing common fields: {missing}")
    stratum = str(record["source_stratum"])
    if stratum not in {"agreement", "verified"}:
        raise ValueError("P3.2 source stratum must be agreement or verified")
    label = GateLabel(int(record["gate_label"]))
    if stratum == "agreement":
        required = {
            "teacher_32b_top1",
            "cross_scale_consistent",
            "teachability",
            "confident_agreement_margin",
            "teacher_topk_ids",
            "teacher_topk_log_probs",
        }
        absent = sorted(required - set(record))
        if absent:
            raise ValueError(f"agreement record missing fields: {absent}")
        if record["teacher_32b_top1"] is None or not bool(record["cross_scale_consistent"]):
            raise ValueError("agreement record lacks required 14B/32B concurrence")
        if label == GateLabel.POSITIVE and record["student_top1"] == record["teacher_14b_top1"]:
            raise ValueError("agreement positive must be a teacher/student disagreement")
        if "teacher_right" in record or "student_right" in record:
            raise ValueError("agreement records cannot carry unverified correctness labels")
    else:
        required = {"verifier_kind", "student_right", "teacher_right", "verifier_receipt"}
        absent = sorted(required - set(record))
        if absent:
            raise ValueError(f"verified record missing fields: {absent}")
        if label == GateLabel.POSITIVE and not (
            bool(record["teacher_right"]) and not bool(record["student_right"])
        ):
            raise ValueError("verified positive must be teacher-right/student-wrong")
    return {
        "record_id": str(record["record_id"]),
        "source_stratum": stratum,
        "battery": str(record["battery"]),
        "document_id": str(record["document_id"]),
        "item_id": str(record["item_id"]),
        "gate_label": int(label),
        "record_sha256": canonical_sha256(dict(record)),
    }


def cache_manifest(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    validated = [validate_cache_record(record) for record in records]
    ids = [record["record_id"] for record in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("P3.2 cache record ids must be unique")
    counts = {
        "agreement": sum(record["source_stratum"] == "agreement" for record in validated),
        "verified": sum(record["source_stratum"] == "verified" for record in validated),
        "positive": sum(record["gate_label"] == int(GateLabel.POSITIVE) for record in validated),
        "negative": sum(record["gate_label"] == int(GateLabel.NEGATIVE) for record in validated),
        "ignored": sum(record["gate_label"] == int(GateLabel.IGNORED) for record in validated),
    }
    stable = sorted(validated, key=lambda record: record["record_id"])
    return {
        "kind": "paper2_phase3_p32_cache_manifest_v1",
        "status": "schema_validated_no_model_training",
        "counts": counts,
        "record_index_sha256": canonical_sha256(stable),
        "cross_scale_agreement_required": True,
        "agreement_correctness_labels_prohibited": True,
        "gate_label_semantics": {"positive": 1, "negative": 0, "ignored": -1},
    }


@dataclass
class OracleGradientBatch:
    directions: torch.Tensor
    gradient_norms: torch.Tensor
    margins_before: torch.Tensor
    source_tokens: torch.Tensor
    target_tokens: torch.Tensor


def batched_oracle_directions(
    *,
    insertion_states: torch.Tensor,
    forward_from_insertion: Callable[[torch.Tensor], torch.Tensor],
    prediction_positions: torch.Tensor,
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    insertion_positions: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> OracleGradientBatch:
    """Compute independent per-row agreement directions with one backward pass."""

    if insertion_states.ndim != 3:
        raise ValueError("insertion states must have [batch, sequence, hidden]")
    batch, sequence, _ = insertion_states.shape
    expected = (batch,)
    for name, value in (
        ("prediction_positions", prediction_positions),
        ("source_tokens", source_tokens),
        ("target_tokens", target_tokens),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape [batch]")
    if insertion_positions is None:
        insertion_positions = prediction_positions
    if tuple(insertion_positions.shape) != expected:
        raise ValueError("insertion_positions must have shape [batch]")
    if bool((prediction_positions < 0).any()) or bool((prediction_positions >= sequence).any()):
        raise ValueError("prediction position is outside the sequence")
    if bool((insertion_positions < 0).any()) or bool((insertion_positions >= sequence).any()):
        raise ValueError("insertion position is outside the sequence")

    states = insertion_states.detach().clone().requires_grad_(True)
    logits = forward_from_insertion(states)
    if logits.ndim != 3 or logits.shape[:2] != states.shape[:2]:
        raise ValueError("forward callback must return [batch, sequence, vocabulary] logits")
    rows = torch.arange(batch, device=states.device)
    selected = logits[rows, prediction_positions]
    source = selected.gather(1, source_tokens.unsqueeze(1)).squeeze(1)
    target = selected.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
    margins = source - target
    gradients = torch.autograd.grad(margins.sum(), states, create_graph=False)[0]
    selected_gradients = gradients[rows, insertion_positions]
    norms = selected_gradients.float().norm(dim=-1)
    directions = -selected_gradients / norms.clamp_min(eps).unsqueeze(-1)
    directions = torch.where((norms > eps).unsqueeze(-1), directions, torch.zeros_like(directions))
    return OracleGradientBatch(
        directions=directions.detach(),
        gradient_norms=norms.detach(),
        margins_before=margins.detach(),
        source_tokens=source_tokens.detach().clone(),
        target_tokens=target_tokens.detach().clone(),
    )


def oracle_batch_equivalence(
    *,
    insertion_states: torch.Tensor,
    forward_from_insertion: Callable[[torch.Tensor], torch.Tensor],
    prediction_positions: torch.Tensor,
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    insertion_positions: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    batched = batched_oracle_directions(
        insertion_states=insertion_states,
        forward_from_insertion=forward_from_insertion,
        prediction_positions=prediction_positions,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        insertion_positions=insertion_positions,
    )
    singles = []
    for index in range(insertion_states.shape[0]):
        single_insertion = None if insertion_positions is None else insertion_positions[index : index + 1]
        singles.append(
            batched_oracle_directions(
                insertion_states=insertion_states[index : index + 1],
                forward_from_insertion=forward_from_insertion,
                prediction_positions=prediction_positions[index : index + 1],
                source_tokens=source_tokens[index : index + 1],
                target_tokens=target_tokens[index : index + 1],
                insertion_positions=single_insertion,
            )
        )
    directions = torch.cat([item.directions for item in singles], dim=0)
    norms = torch.cat([item.gradient_norms for item in singles], dim=0)
    margins = torch.cat([item.margins_before for item in singles], dim=0)
    return {
        "kind": "paper2_phase3_p32_oracle_batch_equivalence_v1",
        "batch_size": insertion_states.shape[0],
        "maximum_direction_difference": float((batched.directions - directions).abs().max()),
        "maximum_norm_difference": float((batched.gradient_norms - norms).abs().max()),
        "maximum_margin_difference": float((batched.margins_before - margins).abs().max()),
        "all_finite": bool(
            torch.isfinite(batched.directions).all()
            and torch.isfinite(batched.gradient_norms).all()
            and torch.isfinite(batched.margins_before).all()
        ),
    }
