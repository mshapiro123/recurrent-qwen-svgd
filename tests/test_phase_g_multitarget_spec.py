from __future__ import annotations

import pytest

from training.branching_relations_task import BranchingRelationsConfig
from training.phase_g_multitarget_spec import (
    assert_posterior_control_gate_lock,
    assert_multitarget_curriculum,
    build_posterior_control_gate_lock,
    frozen_gradient_assertion_count,
    preregistration_payload,
    required_posterior_control_thresholds,
    resolve_posterior_control_gate_lock_path,
)
from training.phase_g_multitarget_task import build_multitarget_rows


def rows(*, cap: int | None) -> list[dict]:
    return build_multitarget_rows(
        BranchingRelationsConfig(rows_per_depth=2, max_depth=2, seed=22),
        split="multitarget_spec",
        rendering="symbolic",
        n_symbols=12,
        targets_per_prompt=cap,
    )


def test_multitarget_contract_requires_repeated_prompt_support() -> None:
    report = assert_multitarget_curriculum(
        rows(cap=None),
        require_all_reachable_targets=True,
    )

    assert report["all_reachable_targets_covered"] is True
    assert report["groups_with_multiple_targets"] == report["base_problem_groups"]


def test_multitarget_contract_can_reject_capped_support() -> None:
    capped = rows(cap=2)

    assert_multitarget_curriculum(capped, require_all_reachable_targets=False)
    with pytest.raises(AssertionError, match="every reachable terminal"):
        assert_multitarget_curriculum(capped, require_all_reachable_targets=True)


def test_multitarget_preregistration_defers_unearned_mechanisms() -> None:
    payload = preregistration_payload()

    assert payload["curriculum"]["sampling_policy"] == (
        "base_problem_uniform_then_target_variant_uniform"
    )
    assert payload["gate_order"][0] == "posterior_exact_target_control_on_repeated_prompt_holdout"
    assert "SVGD" in payload["deferred"]


def test_posterior_control_gate_lock_binds_thresholds_to_exact_rows() -> None:
    control_rows = rows(cap=None)
    thresholds = {
        "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS": 4,
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE": 0.6,
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT": 0.15,
        "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS": 3,
        "STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE": 0.05,
    }
    lock = build_posterior_control_gate_lock(control_rows, thresholds)

    restored = assert_posterior_control_gate_lock(lock, control_rows)

    assert restored == required_posterior_control_thresholds(
        {name: str(value) for name, value in thresholds.items()}
    )
    changed = [dict(row) for row in control_rows]
    changed[0]["target"] = changed[1]["target"]
    with pytest.raises(AssertionError):
        assert_posterior_control_gate_lock(lock, changed)


def test_posterior_control_gate_lock_path_fails_before_gpu_setup(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="GATE_LOCK"):
        resolve_posterior_control_gate_lock_path(tmp_path, None)
    with pytest.raises(FileNotFoundError, match="Missing Phase G"):
        resolve_posterior_control_gate_lock_path(tmp_path, "missing.json")

    lock = tmp_path / "gate.json"
    lock.write_text("{}", encoding="utf-8")
    assert resolve_posterior_control_gate_lock_path(tmp_path, "gate.json") == lock


def test_frozen_gradient_receipt_is_read_from_the_training_config() -> None:
    summary = {"config": {"frozen_gradient_assertions": 1000}}

    assert frozen_gradient_assertion_count(summary) == 1000
    with pytest.raises(AssertionError, match="config receipt"):
        frozen_gradient_assertion_count({"frozen_gradient_assertions": 1000})
