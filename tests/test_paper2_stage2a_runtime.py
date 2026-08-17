from __future__ import annotations

import pytest
import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_stage2a_runtime import (
    Stage2AMemorySystem,
    canonical_fingerprint_query,
    exact_prefix_features,
    memory_augmented_logits,
    stage2a_learning_rate,
)
from training.run_paper2_stage2a import _manifest_core
from training.paper2_stage2a_objective import (
    stage2a_answer_region_objective,
    stage2a_flat_answer_objective,
)


def _memory(arm: str) -> Stage2AMemorySystem:
    return Stage2AMemorySystem(
        arm=arm,
        memory_slots=32,
        memory_keys=torch.randn(32, 128),
        teacher_values=torch.randn(32, 128),
        seed=0,
    )


def test_t3b_uses_the_same_aggregate_value_parameter_budget() -> None:
    memory = _memory("t3b")
    assert memory.aggregate_table_slots == 32
    assert sum(table.weight.numel() for table in memory.reader.tables) == 32 * 128


@pytest.mark.parametrize("arm", ["t3a", "t3b", "shuffled", "random"])
def test_stage2a_arm_trainable_surface_is_narrow(arm: str) -> None:
    memory = _memory(arm)
    trainable = memory.allowed_trainable()
    assert trainable
    assert all(
        name.startswith(("reader.values", "reader.compatibility_projection.", "reader.tables.", "injection."))
        for name in trainable
    )
    if arm in ("shuffled", "random"):
        assert "reader.values" not in trainable


def test_fingerprint_leave_one_out_removes_owned_slot() -> None:
    memory = _memory("t3a")
    query = memory.reader.keys[:2].float()
    readout = memory.read_fingerprint(
        query, excluded_slot_indices=torch.tensor([0, 1])
    )
    assert not bool((readout.slot_indices[0] == 0).any())
    assert not bool((readout.slot_indices[1] == 1).any())


def test_zero_gate_is_exact_and_open_gate_reaches_logits() -> None:
    embedding = nn.Embedding(43, 16)
    sidecar = Phase3StudentModules(
        tied_embedding=embedding,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        control_dim=4,
        draft_rank=4,
        stage2a_memory_dim=None,
    )
    for parameter in sidecar.parameters():
        parameter.requires_grad_(False)
    memory = Stage2AMemorySystem(
        arm="t3a",
        memory_slots=32,
        memory_keys=torch.randn(32, 128),
        teacher_values=torch.randn(32, 128),
        seed=0,
    )
    # Match the compact test sidecar's latent width.
    memory.injection = memory.injection.__class__(memory_dim=128, scratch_dim=8, n_slots=8)
    hidden = torch.randn(7, 16)
    positions = torch.tensor([2, 5])
    scratch, context, current = exact_prefix_features(sidecar, hidden, positions)
    values = torch.randn(2, 128)
    baseline, _ = memory_augmented_logits(
        sidecar=sidecar,
        memory_system=memory,
        scratch0=scratch,
        contexts=context,
        current_hidden=current,
        memory_value=torch.zeros_like(values),
        lm_head_weight=embedding.weight,
        amplitude=0.05,
    )
    attached, _ = memory_augmented_logits(
        sidecar=sidecar,
        memory_system=memory,
        scratch0=scratch,
        contexts=context,
        current_hidden=current,
        memory_value=values,
        lm_head_weight=embedding.weight,
        amplitude=0.05,
    )
    torch.testing.assert_close(attached, baseline, rtol=0.0, atol=0.0)
    with torch.no_grad():
        memory.injection.gate.fill_(0.5)
    opened, _ = memory_augmented_logits(
        sidecar=sidecar,
        memory_system=memory,
        scratch0=scratch,
        contexts=context,
        current_hidden=current,
        memory_value=values,
        lm_head_weight=embedding.weight,
        amplitude=0.05,
    )
    assert not torch.equal(opened, baseline)


def test_canonical_query_and_schedule_contract() -> None:
    hidden = torch.randn(3, 7)
    mean = torch.randn(7)
    basis = torch.randn(7, 4)
    torch.testing.assert_close(
        canonical_fingerprint_query(hidden, student_mean=mean, student_basis=basis),
        (hidden.float() - mean.float()) @ basis.float(),
    )
    torch.testing.assert_close(
        canonical_fingerprint_query(
            hidden, student_mean=mean[None, :], student_basis=basis
        ),
        (hidden.float() - mean.float()) @ basis.float(),
    )
    assert stage2a_learning_rate(1) < stage2a_learning_rate(50)
    assert stage2a_learning_rate(50) == pytest.approx(5e-4)
    assert stage2a_learning_rate(1080) == pytest.approx(5e-4)
    assert stage2a_learning_rate(1200) == pytest.approx(0.0, abs=1e-12)


def test_flat_objective_matches_padded_registered_reduction() -> None:
    generator = torch.Generator().manual_seed(9)
    student = torch.randn(2, 4, 257, generator=generator)
    top_ids = torch.randint(0, 257, (2, 4, 128), generator=generator)
    top_logits = torch.randn(2, 4, 128, generator=generator)
    targets = torch.randint(0, 257, (2, 4), generator=generator)
    mask = torch.tensor([[False, True, True, False], [False, True, True, True]])
    padded = stage2a_answer_region_objective(
        student_logits=student,
        teacher_topk_token_ids=top_ids,
        teacher_topk_logits=top_logits,
        teacher_token_ids=targets,
        answer_region_mask=mask,
    )
    positions = mask.nonzero(as_tuple=False)
    flat = stage2a_flat_answer_objective(
        student_logits=student[mask],
        teacher_topk_token_ids=top_ids[mask],
        teacher_topk_logits=top_logits[mask],
        teacher_token_ids=targets[mask],
        example_index=positions[:, 0],
        example_count=2,
    )
    torch.testing.assert_close(flat.loss, padded.loss)
    torch.testing.assert_close(flat.cross_entropy, padded.cross_entropy)
    torch.testing.assert_close(flat.forward_kl, padded.forward_kl)


def test_population_owner_comparison_is_field_scoped() -> None:
    population = {
        "battery": "gsm8k",
        "item_id": "row-1",
        "content_sha256": "abc",
        "owns_memory_slot": True,
        "owner_slot": 7,
    }
    owner = population | {"retrieval_contract": "leave_one_out"}
    assert _manifest_core([population]) == _manifest_core([owner])
    assert _manifest_core([population]) != _manifest_core(
        [owner | {"owner_slot": 8}]
    )
