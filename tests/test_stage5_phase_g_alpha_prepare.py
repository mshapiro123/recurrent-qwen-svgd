from __future__ import annotations

from colab.run_stage5_phase_g_alpha_prepare import preparation_summary
from training.phase_g_alpha_spec import preregistration_payload


def test_preparation_summary_passes_only_with_disjoint_valid_frozen_splits() -> None:
    manifest = {
        "status": "passed",
        "calibration_test_id_overlap": 0,
        "splits": {"calibration": {"status": "passed"}, "test": {"status": "passed"}},
    }

    summary = preparation_summary(manifest, preregistration_payload())

    assert summary["status"] == "passed"
    assert summary["phase_g_alpha_preparation_complete"] is True
    assert summary["power_rule_locked"] is True
    assert summary["only_remaining_blank"] is None


def test_preparation_summary_blocks_row_overlap() -> None:
    manifest = {
        "status": "passed",
        "calibration_test_id_overlap": 1,
        "splits": {"calibration": {"status": "passed"}, "test": {"status": "passed"}},
    }

    summary = preparation_summary(manifest, preregistration_payload())

    assert summary["status"] == "blocked"
