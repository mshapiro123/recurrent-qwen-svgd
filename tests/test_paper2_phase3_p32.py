from __future__ import annotations

import pytest
import torch
from torch import nn

from training.paper2_phase3_p32 import (
    AgreementLabelInputs,
    GateLabel,
    VerifiedLabelInputs,
    agreement_gate_label,
    batched_oracle_directions,
    cache_manifest,
    gate_loss_mask,
    oracle_batch_equivalence,
    verified_gate_label,
)


def test_agreement_gate_requires_cross_scale_concurrence() -> None:
    common = dict(
        student_top1=3,
        teacher_14b_top1=5,
        teachability=0.9,
        confident_agreement_margin=0.0,
    )
    assert agreement_gate_label(
        AgreementLabelInputs(teacher_32b_top1=5, **common),
        teachability_threshold=0.8,
        confident_agreement_margin_threshold=1.0,
    ) == GateLabel.POSITIVE
    assert agreement_gate_label(
        AgreementLabelInputs(teacher_32b_top1=7, **common),
        teachability_threshold=0.8,
        confident_agreement_margin_threshold=1.0,
    ) == GateLabel.IGNORED
    assert agreement_gate_label(
        AgreementLabelInputs(teacher_32b_top1=None, **common),
        teachability_threshold=0.8,
        confident_agreement_margin_threshold=1.0,
    ) == GateLabel.IGNORED


def test_gate_labels_are_tri_state_and_ignored_rows_leave_loss() -> None:
    labels = torch.tensor([1, 0, -1], dtype=torch.long)
    assert torch.equal(gate_loss_mask(labels), torch.tensor([True, True, False]))
    assert verified_gate_label(
        VerifiedLabelInputs(student_right=False, teacher_right=True, confident_agreement=False)
    ) == GateLabel.POSITIVE
    assert verified_gate_label(
        VerifiedLabelInputs(student_right=False, teacher_right=False, confident_agreement=False)
    ) == GateLabel.IGNORED
    with pytest.raises(ValueError, match="positive, negative, or ignored"):
        gate_loss_mask(torch.tensor([2]))


def test_agreement_cache_cannot_claim_unverified_correctness() -> None:
    record = {
        "record_id": "a",
        "source_stratum": "agreement",
        "battery": "lattice",
        "document_id": "doc",
        "item_id": "item",
        "prediction_position": 2,
        "loop_index": 1,
        "student_top1": 3,
        "teacher_14b_top1": 5,
        "teacher_32b_top1": 5,
        "cross_scale_consistent": True,
        "teachability": 0.9,
        "confident_agreement_margin": 0.0,
        "teacher_topk_ids": [5, 7],
        "teacher_topk_log_probs": [-0.1, -2.0],
        "gate_label": 1,
        "teacher_right": True,
    }
    with pytest.raises(ValueError, match="unverified correctness"):
        cache_manifest([record])


def test_cache_manifest_keeps_agreement_and_verified_semantics_separate() -> None:
    agreement = {
        "record_id": "a",
        "source_stratum": "agreement",
        "battery": "lattice",
        "document_id": "doc-a",
        "item_id": "item-a",
        "prediction_position": 2,
        "loop_index": 1,
        "student_top1": 3,
        "teacher_14b_top1": 5,
        "teacher_32b_top1": 5,
        "cross_scale_consistent": True,
        "teachability": 0.9,
        "confident_agreement_margin": 0.0,
        "teacher_topk_ids": [5, 7],
        "teacher_topk_log_probs": [-0.1, -2.0],
        "gate_label": 1,
    }
    verified = {
        "record_id": "v",
        "source_stratum": "verified",
        "battery": "gsm8k",
        "document_id": "doc-v",
        "item_id": "item-v",
        "prediction_position": 4,
        "loop_index": 1,
        "student_top1": 9,
        "teacher_14b_top1": 11,
        "gate_label": 1,
        "verifier_kind": "final_number",
        "student_right": False,
        "teacher_right": True,
        "verifier_receipt": "hash",
    }
    manifest = cache_manifest([agreement, verified])
    assert manifest["counts"] == {
        "agreement": 1,
        "verified": 1,
        "positive": 2,
        "negative": 0,
        "ignored": 0,
    }


def test_oracle_gradient_batch_matches_single_rows() -> None:
    torch.manual_seed(20260809)
    head = nn.Linear(8, 13, bias=False)

    def forward(states: torch.Tensor) -> torch.Tensor:
        return head(states)

    states = torch.randn(5, 4, 8)
    positions = torch.tensor([0, 1, 2, 3, 1])
    sources = torch.tensor([1, 2, 3, 4, 5])
    targets = torch.tensor([6, 7, 8, 9, 10])
    batch = batched_oracle_directions(
        insertion_states=states,
        forward_from_insertion=forward,
        prediction_positions=positions,
        source_tokens=sources,
        target_tokens=targets,
    )
    equivalence = oracle_batch_equivalence(
        insertion_states=states,
        forward_from_insertion=forward,
        prediction_positions=positions,
        source_tokens=sources,
        target_tokens=targets,
    )
    assert batch.directions.shape == (5, 8)
    assert torch.allclose(batch.directions.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert equivalence["all_finite"]
    assert equivalence["maximum_direction_difference"] == 0.0
    assert equivalence["maximum_norm_difference"] == 0.0
    assert equivalence["maximum_margin_difference"] < 1e-6
