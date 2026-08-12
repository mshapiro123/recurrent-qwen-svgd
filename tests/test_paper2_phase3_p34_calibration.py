from __future__ import annotations

import pytest

from eval.eval_paper2_phase3_p34_calibration import (
    chi_receipt,
    clopper_pearson_upper,
    task_noise_model,
)


def _task_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (0, 1):
        for look in (1, 2, 3):
            for index in range(8):
                rows.append(
                    {
                        "partition": "dev",
                        "battery_role": "target_primary",
                        "seed": seed,
                        "look": look,
                        "item_id": f"item-{index}",
                        "base_correct": index % 2 == 0,
                        "augmented_correct": (index + look + seed) % 3 == 0,
                    }
                )
    return rows


def test_task_noise_model_uses_fixed_paired_panel() -> None:
    result = task_noise_model(_task_rows(), expected_rows=8, bootstrap_draws=100)
    assert result["source_condition_count"] == 6
    assert 0.0 <= result["paired_discordance"] <= 1.0
    assert len(result["mean_augmented_minus_base"]) == 6
    assert result["autocorrelation_bootstrap_upper_95"] >= result["adjacent_checkpoint_autocorrelation"]


def test_task_noise_model_rejects_sealed_rows() -> None:
    rows = _task_rows()
    rows[0]["partition"] = "confirm"
    with pytest.raises(RuntimeError, match="non-DEV"):
        task_noise_model(rows, expected_rows=8)


def test_chi_requires_all_rungs_and_explicit_margin() -> None:
    summaries = [
        {
            "gate_ceiling": ceiling,
            "rows": 1000,
            "collateral_flips": index,
            "estimator": "oracle_write_audit_v1",
        }
        for index, ceiling in enumerate((0.02, 0.08, 0.20, 0.50))
    ]
    receipt = chi_receipt(summaries, margin=0.01)
    assert receipt["estimator_same_clause"]
    assert receipt["chi_max"] > 0.01
    assert clopper_pearson_upper(flips=0, rows=1000) > 0.0
