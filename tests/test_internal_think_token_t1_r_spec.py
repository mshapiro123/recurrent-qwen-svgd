from __future__ import annotations

import pytest

from training.internal_think_token_t1_r_spec import (
    ORIGINAL_T1_LOCK,
    ORIGINAL_T1_LOCK_CANONICAL_LF_SHA256,
    ORIGINAL_T1_LOCK_SHA256,
    phase_t1_lite_r_locked,
    validate_phase_t1_lite_r_locked,
)


def test_original_lock_canonical_hash_is_checkout_newline_independent(tmp_path) -> None:
    from colab.run_stage5_paper2_t1_lite_r import sha256_canonical_lf

    lf = b'{\n  "locked": true\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_path.write_bytes(lf)
    crlf_path.write_bytes(crlf)
    assert sha256_canonical_lf(lf_path) == sha256_canonical_lf(crlf_path)


def test_committed_original_lock_matches_canonical_hash() -> None:
    from colab.run_stage5_paper2_t1_lite_r import ROOT, sha256_canonical_lf

    path = ROOT / ORIGINAL_T1_LOCK
    assert sha256_canonical_lf(path) == ORIGINAL_T1_LOCK_CANONICAL_LF_SHA256


def test_t1_lite_r_changes_only_registered_replication_fields() -> None:
    spec = phase_t1_lite_r_locked()
    validate_phase_t1_lite_r_locked(spec)

    assert spec["status"] == "locked_before_training"
    assert spec["replication"]["primary_seed"] == 1
    assert spec["endpoint_policy"]["primary"] == "raw_final_step"
    assert spec["endpoint_policy"]["continuous_ema"]["role"] == "passive_descriptive_shadow"
    assert spec["endpoint_policy"]["stage_reset_ema"]["role"] == "passive_descriptive_shadow"
    assert spec["amendment"]["original_lock_sha256"] == ORIGINAL_T1_LOCK_SHA256
    assert spec["proposed_training_budget"]["total_steps"] == 10500
    assert spec["loss"]["control_loss_lambda"] == 0.5
    assert spec["gates"]["chain_accuracy"]["full_block_minimum_correct"] == 975
    assert spec["gates"]["control_selection"]["minimum_correct_each_depth"] == 115
    assert spec["gates"]["causal_override"]["required_exact_agreement"] == 5632


def test_t1_lite_r_requires_all_boundary_artifacts() -> None:
    spec = phase_t1_lite_r_locked()
    assert spec["stage_checkpoint_manifest"]["required_steps"] == [500, 2500, 6500, 8500, 10500]
    assert spec["stage_checkpoint_manifest"]["states"] == [
        "raw",
        "continuous_ema",
        "stage_reset_ema",
    ]
    assert spec["stage_checkpoint_manifest"]["atomic_write"] is True
    assert spec["stage_checkpoint_manifest"]["end_of_run_availability_required"] is True


def test_t1_lite_r_rejects_any_seed_or_gate_drift() -> None:
    bad_seed = phase_t1_lite_r_locked()
    bad_seed["replication"]["primary_seed"] = 0
    with pytest.raises(AssertionError, match="seed 1"):
        validate_phase_t1_lite_r_locked(bad_seed)

    bad_gate = phase_t1_lite_r_locked()
    bad_gate["gates"]["chain_accuracy"]["full_block_minimum_correct"] = 974
    with pytest.raises(AssertionError, match="base T1-lite contract"):
        validate_phase_t1_lite_r_locked(bad_gate)
