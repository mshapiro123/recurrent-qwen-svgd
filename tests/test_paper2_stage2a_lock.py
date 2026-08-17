from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_stage2a_lock import (
    assert_stage2a_training_authorized,
    load_stage2a_lock,
    materialize_stage2a_lock,
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


def test_stage2a_rejects_sizing_rule_or_realized_count_drift() -> None:
    lock = load_stage2a_lock()
    lock["data_separation"]["post_concurrence_admitted_rows"] = 4_095
    lock["data_separation"]["memory_slots"] = 4_096
    errors = validate_stage2a_lock(lock)
    assert any("exceed" in error for error in errors)

    lock = load_stage2a_lock()
    lock["data_separation"]["post_concurrence_admitted_rows"] = 4_095
    lock["data_separation"]["memory_slots"] = 2_048
    assert validate_stage2a_lock(lock) == []


def test_stage2a_rejects_validation_or_subselection_seed_drift() -> None:
    lock = load_stage2a_lock()
    lock["data_separation"]["validation_split_seed"] = 7
    lock["data_separation"]["selection_seed"] = 8
    errors = validate_stage2a_lock(lock)
    assert any("validation split seed" in error for error in errors)
    assert any("subselection seed" in error for error in errors)


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


def test_score_blind_receipts_materialize_but_do_not_authorize() -> None:
    lock = load_stage2a_lock()
    summary = {
        "kind": "paper2_stage2a_content_geometry_summary_v1",
        "status": "complete_score_blind_pre_signature",
        "validation": {"sha256": "1" * 64},
        "firm_knowledge": {"sha256": "2" * 64},
        "memory": {"slots": 2_048, "admitted_nonpanel_rows": 4_095},
        "geometry": {"fit_manifest_sha256": "3" * 64, "artifact_sha256": "4" * 64},
        "training_authorized": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    materialized = materialize_stage2a_lock(lock, summary, summary_sha256="5" * 64)
    assert validate_stage2a_lock(materialized) == []
    assert materialized["data_separation"]["memory_slots"] == 2_048
    assert materialized["geometry_fit"]["artifact_sha256"] == "4" * 64
    assert materialized["unresolved_before_ratification"] == [
        "Mark signs the fully materialized executed lock."
    ]
    with pytest.raises(RuntimeError, match="structurally disabled"):
        assert_stage2a_training_authorized(materialized)
