from __future__ import annotations

from pathlib import Path

import pytest

from training.adapter_parity_battery import (
    ARM_E_FINAL_SHA256,
    derive_frontier,
    score_e2_persistence,
    score_e3a_transfer,
    score_e4_retention,
    score_e4_checkpoint_series,
    validate_e4_source,
)


def test_e3a_reporting_bands_are_locked() -> None:
    assert score_e3a_transfer(correct=70, total=100)["band"] == "strong"
    assert score_e3a_transfer(correct=40, total=100)["band"] == "partial"
    assert score_e3a_transfer(correct=39, total=100)["band"] == "minimal"


def test_e2_strong_partial_and_failed_readings() -> None:
    strong = score_e2_persistence(
        diagonal_correct=596,
        diagonal_total=640,
        continue_count=327,
        hold_count=1,
        above_total=384,
    )
    assert strong["verdict"] == "strong"
    assert strong["e4_authorized"] is True

    partial = score_e2_persistence(
        diagonal_correct=596,
        diagonal_total=640,
        continue_count=250,
        hold_count=20,
        above_total=384,
    )
    assert partial["verdict"] == "partial"
    assert partial["e4_authorized"] is True

    failed = score_e2_persistence(
        diagonal_correct=595,
        diagonal_total=640,
        continue_count=380,
        hold_count=0,
        above_total=384,
    )
    assert failed["verdict"] == "failed"
    assert failed["e4_authorized"] is False


def test_e4_source_requires_exact_arm_e_lineage_and_healthy_e2() -> None:
    receipt = validate_e4_source(
        {
            "kind": "stage5_adapter_parity_e2",
            "source_checkpoint_sha256": ARM_E_FINAL_SHA256,
            "decision": {"e4_authorized": True},
            "final_checkpoint_sha256": "f" * 64,
        }
    )
    assert receipt["status"] == "authorized"

    with pytest.raises(RuntimeError, match="not authorized"):
        validate_e4_source(
            {
                "kind": "stage5_adapter_parity_e2",
                "source_checkpoint_sha256": ARM_E_FINAL_SHA256,
                "decision": {"e4_authorized": False},
                "final_checkpoint_sha256": "f" * 64,
            }
        )


def test_e4_readings_distinguish_hold_move_and_vanish() -> None:
    common = {
        "inverse_correct": 46,
        "inverse_total": 64,
        "synthetic_min_by_checkpoint": [0.94, 0.93],
        "natural_baseline_accuracy": 0.90,
        "natural_min_accuracy": 0.87,
    }
    assert score_e4_retention(**common)["verdict"] == "wall_vanishes"

    moved = score_e4_retention(
        **{**common, "synthetic_min_by_checkpoint": [0.94, 0.92]}
    )
    assert moved["verdict"] == "wall_moves"

    held = score_e4_retention(**{**common, "inverse_correct": 45})
    assert held["verdict"] == "wall_holds"

    series = score_e4_checkpoint_series(
        [
            {
                "step": 100,
                "inverse_correct": 50,
                "inverse_total": 64,
                "synthetic_min": 0.94,
                "natural_baseline": 0.90,
                "natural_accuracy": 0.89,
            },
            {
                "step": 200,
                "inverse_correct": 50,
                "inverse_total": 64,
                "synthetic_min": 0.92,
                "natural_baseline": 0.90,
                "natural_accuracy": 0.89,
            },
        ]
    )
    assert series["verdict"] == "wall_moves"


def test_frontier_derivation_records_interpolation() -> None:
    receipt = derive_frontier(
        last_above_depth=11,
        last_above_accuracy=111 / 128,
        first_below_depth=12,
        first_below_accuracy=75 / 128,
        threshold=0.71,
    )
    assert receipt["frontier"] == pytest.approx(11.56, abs=0.01)
    assert receipt["frontier_to_support_ratio"] == pytest.approx(1.44, abs=0.01)


def test_three_adapter_parity_targets_are_wired() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py").read_text(encoding="utf-8")
    for target in ("adapter_parity_e3a", "adapter_parity_e2", "adapter_parity_e4"):
        assert f'"{target}"' in bootstrap
        assert f'"{target}"' in cell
    assert ARM_E_FINAL_SHA256 in bootstrap
    assert "accepted_returncodes={0, 2}" in cell


def test_adapter_continuations_keep_lora_live_and_muon_rejected() -> None:
    common = Path("colab/adapter_parity_common.py").read_text(encoding="utf-8")
    e2 = Path("colab/run_stage5_adapter_parity_e2.py").read_text(encoding="utf-8")
    e4 = Path("colab/run_stage5_adapter_parity_e4.py").read_text(encoding="utf-8")
    assert '"training_mode": "frozen_lora"' in common
    assert '"require_frozen_base_hash": True' in common
    assert '"rank": ARM_E_RANK' in common
    assert '"chain_anneal_hold_frac": 1.0' in e2
    assert '"reject_muon": True' in e2
    assert '"reject_muon": True' in e4
    assert "validate_e4_source(e2)" in e4
    assert "restore_arm_e_checkpoint" in e4
