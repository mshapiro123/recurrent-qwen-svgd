from __future__ import annotations

from pathlib import Path


def test_adapter_budget_target_is_wired_with_safety_markers() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_ADAPTER_BUDGET_ARM_CELL.py").read_text(encoding="utf-8")
    runner = Path("colab/run_stage5_adapter_budget_arm.py").read_text(encoding="utf-8")

    assert '"adapter_budget_arm_e"' in bootstrap
    assert "STAGE5_ADAPTER_BUDGET_ARM_CELL_VERSION" in cell
    assert "tests/test_adapter_budget_arm.py" in cell
    assert "accepted_returncodes={0, 2}" in cell
    assert "fresh_base_qwen_surgery" in runner
    assert "same_reader_final_rows.jsonl" in runner
    assert "pretrained_base_hash_unchanged" in runner
    assert "track_loop_dose" in runner
    assert "adapter_budget_depth_profile" in runner
    assert "EXPECTED_PRETRAINED_BASE_SHA256" in runner
    assert "EXPECTED_IMMUTABLE_SHA256" in runner
    assert "IMMUTABLE_DATASET_HASH_MODE" in runner
    assert "normalized_text_sha256" in runner


def test_runner_uses_frozen_phase_a_rows_and_r16_only() -> None:
    runner = Path("colab/run_stage5_adapter_budget_arm.py").read_text(encoding="utf-8")

    assert "stage5_synthetic_depth_frozen_eval_v2_depth14" in runner
    assert "stage5_same_reader_final_symbol_20260707_021010" in runner
    assert '"rank": 16' in runner
    assert '"alpha": 32' in runner
    assert "rank sweep" not in runner.lower()
