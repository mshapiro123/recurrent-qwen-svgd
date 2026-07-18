from __future__ import annotations

import pytest

from training.branching_relations_task import BranchingRelationsConfig
from training.phase_g_multitarget_spec import (
    assert_multitarget_curriculum,
    preregistration_payload,
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
