from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from training.paper2_stage2b_autopsy import (
    decision_mapping,
    margin_correlation_receipt,
    stable_dev2_subsample,
    validate_autopsy_lock,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "training/paper2_stage2b_autopsy_lock.json"


def test_autopsy_draft_is_score_only_and_sealed() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_autopsy_lock(lock, require_signature=False)
    with pytest.raises(RuntimeError, match="unsigned"):
        validate_autopsy_lock(lock, require_signature=True)
    assert lock["optimizer_steps_allowed"] == 0
    assert lock["training_authorized"] is False
    assert lock["sealed_partitions"]["remain_sealed"] is True


def test_autopsy_lock_rejects_training_or_seal_contact() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    training = copy.deepcopy(lock)
    training["optimizer_steps_allowed"] = 1
    with pytest.raises(RuntimeError, match="score-only"):
        validate_autopsy_lock(training, require_signature=False)
    unsealed = copy.deepcopy(lock)
    unsealed["sealed_partitions"]["confirm_scored"] = True
    with pytest.raises(RuntimeError, match="sealed-partition"):
        validate_autopsy_lock(unsealed, require_signature=False)


def test_dev2_subsample_is_deterministic_and_stratified() -> None:
    rows = [
        {"item_id": f"{battery}-{index}", "battery": battery}
        for battery, count in (("gsm8k", 80), ("mbpp", 10), ("mmlu", 8), ("tier1", 2))
        for index in range(count)
    ]
    first = stable_dev2_subsample(rows, size=25)
    second = stable_dev2_subsample(list(reversed(rows)), size=25)
    assert first == second
    counts = Counter(row["battery"] for row in first)
    assert len(first) == 25
    assert set(counts) == {"gsm8k", "mbpp", "mmlu", "tier1"}


def test_margin_correlation_reports_pearson_and_spearman() -> None:
    rows = [
        {"per_loop_mean_teacher_token_margin": [value, 0.0, 0.0, 2.0 * value]}
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    receipt = margin_correlation_receipt(rows)
    assert receipt["pearson"] == pytest.approx(1.0)
    assert receipt["spearman"] == pytest.approx(1.0)


def test_decision_mapping_composes_hypotheses() -> None:
    assert decision_mapping({"h_b_magnitude": True, "h_a_attractor": True}) == [
        "radius_control_successor",
        "task_preservation_anchor_required",
    ]


def test_autopsy_runner_contains_no_optimizer_or_sealed_partition_path() -> None:
    evaluator = (ROOT / "eval/eval_paper2_stage2b_autopsy.py").read_text(encoding="utf-8")
    orchestrator = (ROOT / "colab/run_stage5_paper2_stage2b_autopsy.py").read_text(
        encoding="utf-8"
    )
    assert "torch.optim" not in evaluator
    assert "optimizer.step" not in evaluator
    assert "stage5_paper2_phase3_confirm" not in orchestrator.lower()
    assert "stage5_paper2_eval_e" not in orchestrator.lower()
    assert "validate_autopsy_lock(lock, require_signature=True)" in orchestrator
    assert '"optimizer_steps": 0' in evaluator
    assert '"optimizer_steps": 0' in orchestrator
