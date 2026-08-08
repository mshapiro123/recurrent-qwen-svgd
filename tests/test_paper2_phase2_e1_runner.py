from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eval.eval_paper2_phase2_e1_confirmation import (
    _claim_lease,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "training/paper2_phase2_e1_confirmation_preregistration.json"


def test_read_once_lease_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    first = _claim_lease(path, "a" * 64)
    assert first["read_once_scoring_spent"] is False
    with pytest.raises(RuntimeError, match="automatic rerun is prohibited"):
        _claim_lease(path, "a" * 64)


def test_scripted_verdict_requires_both_seed_effect_and_quality_passes() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    registration["evaluation"]["bootstrap"]["replicates"] = 1_000
    documents = [f"doc_{index // 2}" for index in range(8)]
    strata = ["general"] * 4 + ["code"] * 4
    positions = torch.arange(8)
    arm_rows = {}
    arm_summaries = {}
    for seed in (0, 1):
        control = torch.ones(8)
        full = control + 0.1 + seed * 0.01
        for arm, accepted in (
            ("full_a2", full),
            ("draft_only_control", control),
        ):
            name = f"seed_{seed}_{arm}"
            arm_rows[name] = {"accepted_length": accepted}
            arm_summaries[name] = {
                "retention": 0.999,
                "retention_wilson_95_lower": 0.995,
                "baseline_correct": 1000,
                "retained_correct": 999,
            }
    result = summarize(
        arm_summaries=arm_summaries,
        arm_rows=arm_rows,
        documents=documents,
        strata=strata,
        positions=positions,
        registration=registration,
    )
    assert result["primary_pass_both_seeds"] is True
    assert result["quality_pass_both_seeds"] is True
    assert result["scripted_verdict"] == "CONFIRMED_WITH_MEASURED_PARETO"


def test_runner_is_eval_only_and_uses_inherited_option_b_evaluator() -> None:
    source = (ROOT / "eval/eval_paper2_phase2_e1_confirmation.py").read_text(
        encoding="utf-8"
    )
    assert "from training.run_paper2_phase2_a2 import" in source
    assert "evaluate" in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "requires_grad_(False)" in source
    assert "read_once_scoring_spent=True" in source
    assert "eval_e_touched\": False" in source


def test_bootstrap_target_exposes_locked_e1_confirmation() -> None:
    for path in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py",
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md",
    ):
        source = path.read_text(encoding="utf-8")
        assert "paper2_phase2_e1_confirmation" in source
        assert "STAGE5_PAPER2_PHASE2_E1_CONFIRMATION_CELL.py" in source
        assert "failed score-bearing pass cannot rerun without strategy review" in source
    cell = (ROOT / "colab/STAGE5_PAPER2_PHASE2_E1_CONFIRMATION_CELL.py").read_text(
        encoding="utf-8"
    )
    assert "memory >= 70000" in cell
    assert "lock commit ebe4ea4b before scorer construction" in cell
