from __future__ import annotations

import json
from pathlib import Path

from training.run_paper2_phase3_p34 import _task_guardrail


ROOT = Path(__file__).resolve().parents[1]


def test_task_guardrail_uses_paired_task_differences() -> None:
    lock = {"guardrails": {
        "tier_s_one_sided_alpha": 0.1,
        "tier_s_decision_margin": -0.03,
        "tier_w_drop_class": -0.03,
    }}
    rows = [
        {"base_correct": True, "augmented_correct": False} for _ in range(60)
    ] + [
        {"base_correct": True, "augmented_correct": True} for _ in range(40)
    ]
    read = _task_guardrail(rows, lock)
    assert read["mean_augmented_minus_base"] == -0.6
    assert read["tier_s_condition"]
    assert read["tier_w_condition"]


def test_runner_is_sealed_partition_blind_and_resumable() -> None:
    source = (ROOT / "training/run_paper2_phase3_p34.py").read_text(encoding="utf-8")
    assert "task_rows_look_" in source
    assert "checkpoint_step_" in source
    assert '"confirm_scored": False' in source
    assert '"eval_e_scored": False' in source
    lock = json.loads((ROOT / "training/paper2_phase3_p34_preregistration.json").read_text())
    assert lock["guardrails"]["look_count"] == 20


def test_colab_target_wires_the_approved_three_arm_campaign() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_PAPER2_PHASE3_P34_CELL.py").read_text(encoding="utf-8")
    launcher = (ROOT / "colab/run_stage5_paper2_phase3_p34.py").read_text(encoding="utf-8")
    assert '"paper2_phase3_p34"' in bootstrap
    assert "paper2_phase3_p34_campaign_v1" in cell
    assert 'parser.add_argument("--arm"' in launcher
    assert "I1_ID" in launcher
