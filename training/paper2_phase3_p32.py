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
) -> GateLabel:
    cross_scale = (
        inputs.teacher_32b_top1 is not None
        and inputs.teacher_14b_top1 == inputs.teacher_32b_top1
    )
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
            "flip_candidate_14b",
            "teachability",
            "confident_agreement_margin",
            "teacher_topk_ids",
            "teacher_topk_log_probs",
        }
        absent = sorted(required - set(record))
        if absent:
            raise ValueError(f"agreement record missing fields: {absent}")
        cascade_covered = record["teacher_32b_top1"] is not None
        cross_scale_consistent = bool(
            cascade_covered
            and record["teacher_14b_top1"] == record["teacher_32b_top1"]
        )
        if bool(record["cross_scale_consistent"]) != cross_scale_consistent:
            raise ValueError("agreement cross-scale consistency field is inaccurate")
        disagreement = record["student_top1"] != record["teacher_14b_top1"]
        if bool(record["flip_candidate_14b"]) and not disagreement:
            raise ValueError("14B flip candidate must be a teacher/student disagreement")
        if label == GateLabel.POSITIVE and not (
            bool(record["flip_candidate_14b"]) and cross_scale_consistent
        ):
            raise ValueError("agreement positive requires a concurrent 14B/32B flip target")
        if label == GateLabel.NEGATIVE and disagreement:
            raise ValueError("agreement negative must be a confident 14B/student agreement")
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
    is_agreement = stratum == "agreement"
    cascade_covered = bool(is_agreement and record.get("teacher_32b_top1") is not None)
    cross_scale_consistent = bool(is_agreement and record.get("cross_scale_consistent"))
    flip_candidate_14b = bool(is_agreement and record.get("flip_candidate_14b"))
    loss_eligibility = {
        "l_kl": is_agreement,
        "aim_target": is_agreement and label == GateLabel.POSITIVE and cross_scale_consistent,
        "gate_positive": label == GateLabel.POSITIVE,
        "gate_negative": label == GateLabel.NEGATIVE,
        "preservation": label == GateLabel.NEGATIVE,
    }
    return {
        "record_id": str(record["record_id"]),
        "source_stratum": stratum,
        "battery": str(record["battery"]),
        "document_id": str(record["document_id"]),
        "item_id": str(record["item_id"]),
        "gate_label": int(label),
        "cascade_covered": cascade_covered,
        "cross_scale_consistent": cross_scale_consistent,
        "flip_candidate_14b": flip_candidate_14b,
        "targeted_32b_extension_candidate": bool(
            is_agreement and flip_candidate_14b and not cascade_covered
        ),
        "cross_scale_conflict": bool(
            is_agreement and cascade_covered and not cross_scale_consistent
        ),
        "loss_eligibility": loss_eligibility,
        "record_sha256": canonical_sha256(dict(record)),
    }


def cache_manifest(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    validated = [validate_cache_record(record) for record in records]
    ids = [record["record_id"] for record in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("P3.2 cache record ids must be unique")
    counts = {
        "total": len(validated),
        "agreement": sum(record["source_stratum"] == "agreement" for record in validated),
        "verified": sum(record["source_stratum"] == "verified" for record in validated),
        "positive": sum(record["gate_label"] == int(GateLabel.POSITIVE) for record in validated),
        "negative": sum(record["gate_label"] == int(GateLabel.NEGATIVE) for record in validated),
        "ignored": sum(record["gate_label"] == int(GateLabel.IGNORED) for record in validated),
    }
    agreement = [record for record in validated if record["source_stratum"] == "agreement"]
    cascade_covered = sum(record["cascade_covered"] for record in agreement)
    concurrent = sum(record["cross_scale_consistent"] for record in agreement)
    label_names = {
        GateLabel.POSITIVE: "positive",
        GateLabel.NEGATIVE: "negative",
        GateLabel.IGNORED: "ignored",
    }
    per_label_class = {}
    for label, name in label_names.items():
        selected = [record for record in agreement if record["gate_label"] == int(label)]
        covered = sum(record["cascade_covered"] for record in selected)
        per_label_class[name] = {
            "total": len(selected),
            "cascade_covered": covered,
            "cross_scale_concurrent": sum(
                record["cross_scale_consistent"] for record in selected
            ),
            "14b_only": len(selected) - covered,
        }
    per_loss_class = {
        loss: {
            "eligible": sum(record["loss_eligibility"][loss] for record in validated),
            "agreement_14b_only_eligible": sum(
                record["source_stratum"] == "agreement"
                and not record["cascade_covered"]
                and record["loss_eligibility"][loss]
                for record in validated
            ),
        }
        for loss in ("l_kl", "aim_target", "gate_positive", "gate_negative", "preservation")
    }
    extension_candidates = sum(
        record["targeted_32b_extension_candidate"] for record in agreement
    )
    stable = sorted(validated, key=lambda record: record["record_id"])
    return {
        "kind": "paper2_phase3_p32_cache_manifest_v2",
        "status": "schema_validated_no_model_training",
        "counts": counts,
        "coverage": {
            "total_anchors": len(validated),
            "agreement_anchors": len(agreement),
            "cascade_covered_agreement_anchors": cascade_covered,
            "cross_scale_concurrent_agreement_anchors": concurrent,
            "concurrence_rate_within_cascade_coverage": (
                concurrent / cascade_covered if cascade_covered else None
            ),
            "per_label_class": per_label_class,
            "per_loss_class": per_loss_class,
            "targeted_32b_extension_candidates": extension_candidates,
            "cross_scale_conflicts": sum(record["cross_scale_conflict"] for record in agreement),
            "write_stratum_thinness_threshold": None,
            "write_stratum_thinness_decision": "pending_p33_lock",
        },
        "record_index_sha256": canonical_sha256(stable),
        "admission_policy": {
            "aim_target": "14b_and_32b_required_and_concurrent",
            "gate_positive": "agreement_positive_requires_14b_32b_concurrence_verified_positive_requires_programmatic_correctness",
            "l_kl": "14b_only_admissible",
            "gate_negative": "confident_14b_student_agreement_14b_only_admissible",
            "preservation": "14b_only_admissible",
            "fallback": "targeted_32b_extension_over_uncovered_flip_candidates_never_dilution",
        },
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
