from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_stage2a_lock import (
    assert_stage2a_training_authorized,
    load_stage2a_lock,
    validate_stage2a_lock,
)


def test_checked_in_stage2a_companion_is_valid_but_training_disabled() -> None:
    lock = load_stage2a_lock()
    assert validate_stage2a_lock(lock) == []
    assert lock["unresolved_before_ratification"]
    with pytest.raises(RuntimeError, match="structurally disabled"):
        assert_stage2a_training_authorized(lock)


def test_stage2a_rejects_reuse_of_dev_fitted_geometry() -> None:
    lock = load_stage2a_lock()
    lock["geometry_fit"]["fit_population"] = "diagnostic_dev_alignment_fit"
    lock["geometry_fit"]["dev_rows_used"] = 614
    lock["geometry_fit"]["diagnostic_dev_transform_reused"] = True
    errors = validate_stage2a_lock(lock)
    assert any("non-DEV" in error for error in errors)
    assert any("touched DEV" in error for error in errors)
    assert any("may not be reused" in error for error in errors)


def test_stage2a_rejects_trainable_query_transform() -> None:
    lock = load_stage2a_lock()
    lock["arms"]["t3a_fingerprint"]["query_transform_trainable"] = True
    lock["trainable_surface"].append("memory.query_projection")
    errors = validate_stage2a_lock(lock)
    assert any("query transform" in error for error in errors)
    assert any("ratified allowlist" in error for error in errors)


def test_stage2a_rejects_objective_or_leave_one_out_drift() -> None:
    lock = load_stage2a_lock()
    lock["training"]["objective"]["kl_direction"] = "student_to_teacher"
    lock["arms"]["t3a_fingerprint"]["training_leave_one_out"] = "disabled"
    errors = validate_stage2a_lock(lock)
    assert any("objective field changed: kl_direction" in error for error in errors)
    assert any("leave-one-out" in error for error in errors)


def test_ratification_flags_cannot_flip_with_open_fields() -> None:
    lock = copy.deepcopy(load_stage2a_lock())
    lock["status"] = "approved_for_training"
    lock["locked_before_training"] = True
    lock["training_authorized"] = True
    lock["mark_ratified"] = True
    errors = validate_stage2a_lock(lock)
    assert any("unresolved signature fields" in error for error in errors)


def test_stage2a_writer_is_post_initializer_and_zero_gate_exact() -> None:
    embedding = nn.Embedding(31, 16)
    module = Phase3StudentModules(
        tied_embedding=embedding,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        stage2a_memory_dim=8,
    )
    hidden = torch.randn(2, 4, 16)
    previous_logits = torch.randn(2, 4, 31)
    memory = torch.randn(2, 8)
    baseline = module(hidden=hidden, previous_logits=previous_logits, steps=1)
    attached = module(
        hidden=hidden,
        previous_logits=previous_logits,
        steps=1,
        stage2a_memory_value=memory,
        stage2a_amplitude_ceiling=0.05,
    )
    torch.testing.assert_close(attached.scratch, baseline.scratch, rtol=0.0, atol=0.0)
    assert module.stage2a_memory_injection is not None
    with torch.no_grad():
        module.stage2a_memory_injection.gate.fill_(0.5)
    opened = module(
        hidden=hidden,
        previous_logits=previous_logits,
        steps=1,
        stage2a_memory_value=memory,
        stage2a_amplitude_ceiling=0.05,
    )
    assert not torch.equal(opened.scratch, baseline.scratch)
