from __future__ import annotations

from training.adapter_verbal_transference import (
    guardrail_near_miss_context,
    classify_regression,
    first_threshold_crossing,
    paired_binary_test,
    score_transference,
    summarize_archived_active_diagonal,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_archived_diagonal_is_reduced_to_same_reader_total() -> None:
    result = summarize_archived_active_diagonal({"1": 0.5, "2": 0.25}, rows_per_depth=8)

    assert result == {"correct": 6, "total": 16, "accuracy": 0.375}


def test_paired_binary_test_reports_exact_discordance() -> None:
    result = paired_binary_test([True, True, False, False], [False, True, True, False])

    assert result["t_only"] == 1
    assert result["s_only"] == 1
    assert result["ties"] == 2
    assert result["two_sided_p"] == 1.0


def test_transference_requires_positive_delta_and_paired_support() -> None:
    positive = score_transference(
        t_hits=[True] * 10 + [False] * 2,
        s_hits=[False] * 10 + [False] * 2,
        alpha=0.05,
    )
    null = score_transference(
        t_hits=[True, False, True, False],
        s_hits=[False, True, False, False],
        alpha=0.05,
    )

    assert positive["verdict"] == "positive"
    assert null["verdict"] == "null"


def test_first_threshold_crossing_returns_first_registered_checkpoint() -> None:
    curve = {"0": 0.15, "1000": 0.65, "2000": 0.72, "3000": 0.80}

    assert first_threshold_crossing(curve, threshold=0.71) == 2000
    assert first_threshold_crossing({"0": 0.1, "1000": 0.7}, threshold=0.71) is None


def test_regression_classification_uses_locked_retention_and_e4_collapse_bands() -> None:
    retained = classify_regression({"0": {"1": 0.99}, "1000": {"1": 0.94}})
    partial = classify_regression({"0": {"1": 0.99}, "1000": {"1": 0.75}})
    collapsed = classify_regression({"0": {"1": 0.99}, "1000": {"1": 0.09}})

    assert retained["verdict"] == "retained"
    assert partial["verdict"] == "partial"
    assert collapsed["verdict"] == "collapsed"


def test_guardrail_near_miss_reports_discrete_resolution_and_paired_evidence() -> None:
    result = guardrail_near_miss_context(
        baseline_hits=[True, True, True, False],
        observed_hits=[True, False, True, True],
        hard_stop_delta=-0.24,
    )

    assert result["baseline"] == {"correct": 3, "total": 4, "accuracy": 0.75}
    assert result["observed"] == {"correct": 3, "total": 4, "accuracy": 0.75}
    assert result["item_resolution_points"] == 25.0
    assert result["hard_stop_triggered"] is False
    assert result["paired"]["baseline_only"] == 1
    assert result["paired"]["observed_only"] == 1


def test_guardrail_near_miss_preserves_strict_locked_stop() -> None:
    result = guardrail_near_miss_context(
        baseline_hits=[True] * 60 + [False] * 4,
        observed_hits=[True] * 58 + [False] * 6,
        hard_stop_delta=-0.03,
    )

    assert result["accuracy_delta"] == -0.03125
    assert result["hard_stop_triggered"] is True
    assert result["boundary_excess_points"] == 0.125
    assert result["interpretation"] == "near_boundary_discrete_hard_stop"


def test_runner_keeps_pointer_held_out_and_synthetic_floor_in_measurement_mode() -> None:
    source = (ROOT / "colab/run_stage5_adapter_verbal_transference.py").read_text(encoding="utf-8")

    assert 'TRAIN_DATA = DATA_ROOT / "rung0_train_mix_chain_symbol_sft.jsonl"' in source
    assert '"training_mix": "2048 relay plus 2048 synthetic rehearsal; pointer held out"' in source
    assert '"synthetic_regression_floor_enforced": False' in source
    assert '"reject_muon": True' in source
    assert '"checkpoint_backup_every": 1000' in source


def test_bootstrap_exposes_e3b_target_with_locked_markers() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    launcher = (ROOT / "colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py").read_text(encoding="utf-8")

    assert '"adapter_verbal_transference_e3b"' in bootstrap
    assert "stage5_adapter_verbal_transference_e3b_20260720" in bootstrap
    assert '"adapter_verbal_transference_e3b": "colab/run_stage5_adapter_verbal_transference.py"' in launcher


def test_bootstrap_exposes_e3b_salvage_without_training_override() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    launcher = (ROOT / "colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py").read_text(encoding="utf-8")
    salvage = (ROOT / "colab/run_stage5_adapter_verbal_transference_salvage.py").read_text(
        encoding="utf-8"
    )

    assert '"adapter_verbal_transference_e3b_salvage"' in bootstrap
    assert (
        '"adapter_verbal_transference_e3b_salvage": '
        '"colab/run_stage5_adapter_verbal_transference_salvage.py"'
    ) in launcher
    assert "MATCHED_STEPS = (0, 1000, 2000, 3000)" in salvage
    assert "ARM_T_STEPS = (0, 1000, 2000, 3000, 4000, 5000, 6000)" in salvage
    assert "train_arm(" not in salvage
    assert '"planned_endpoint_available": False' in salvage
