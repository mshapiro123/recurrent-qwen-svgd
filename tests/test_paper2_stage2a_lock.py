from __future__ import annotations

import copy

import pytest

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
    assert any("four-part allowlist" in error for error in errors)


def test_ratification_flags_cannot_flip_with_open_fields() -> None:
    lock = copy.deepcopy(load_stage2a_lock())
    lock["status"] = "approved_for_training"
    lock["locked_before_training"] = True
    lock["training_authorized"] = True
    lock["mark_ratified"] = True
    errors = validate_stage2a_lock(lock)
    assert any("unresolved signature fields" in error for error in errors)

