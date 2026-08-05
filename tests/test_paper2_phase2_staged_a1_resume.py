from __future__ import annotations

import json
from pathlib import Path

import torch

from training.run_paper2_phase2_staged_a1 import (
    AMENDMENT_LOCK_COMMIT,
    _calibration_measurement_batches,
)


ROOT = Path(__file__).resolve().parents[1]


def test_amendment_lock_and_audit_receipt_are_bound() -> None:
    registration = json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    amendment = registration["resume_amendment_20260805"]
    assert AMENDMENT_LOCK_COMMIT == "01ae5f28c0b52f4635dc15298bd07269bd853055"
    assert amendment["audit_decision"] == "resume_saved_step_200"
    assert amendment["audit_receipt_lf_sha256"] == (
        "5cd62b5b0c4cf951b20c17e65a826386269291b16c62bd07d71d69a18d706039"
    )


def test_resume_uses_exact_51_calibration_measurement_batches() -> None:
    registration = json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    train = torch.arange(1000)
    batches = _calibration_measurement_batches(train, seed=0, registration=registration)
    assert len(batches) == 51
    assert all(batch.numel() == 128 for batch in batches)


def test_resume_source_encodes_amended_contract_and_no_extension() -> None:
    source = (ROOT / "training/run_paper2_phase2_staged_a1.py").read_text(encoding="utf-8")
    assert "amended_flow_share_contract_miss" in source
    assert "amended_probe_share_contract_miss_recalibration_review" in source
    assert "counterfactual_preserve_share_role" in source
    assert "counterfactual_preserve_loss_value_alarm_log_only" in source
    assert "fixed_51_batch_dev_population_shift_descriptive" in source
    assert 'verdict = "a1_gate_candidate_pass"' in source
    assert "shares_within_absolute_tolerance" not in source
    assert "staged_a1_extension" not in source
    assert '"automatic_extension_disabled": True' in source
    assert '"a2_launched": False' in source


def test_resume_runner_preserves_source_checkpoint_and_uses_separate_output() -> None:
    runner = (ROOT / "colab/run_stage5_paper2_phase2_staged_a1_resume.py").read_text(
        encoding="utf-8"
    )
    assert 'RUN_ID = "stage5_paper2_phase2_staged_a1_resume_20260805"' in runner
    assert 'SOURCE_RUN_ID = "stage5_paper2_phase2_staged_a1_20260805"' in runner
    assert "a1_resume_amended.pt" in (
        ROOT / "training/run_paper2_phase2_staged_a1.py"
    ).read_text(encoding="utf-8")
    assert "stage_resume_lineage" in runner
    assert "training.run_paper2_phase2_staged_a1" in runner
