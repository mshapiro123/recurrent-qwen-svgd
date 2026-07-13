from __future__ import annotations

from pathlib import Path

from colab.run_stage5_phase_g_curriculum_autopsy import (
    EXPECTED_TEST_SHA,
    EXPECTED_TRAIN_SHA,
    construction_audit,
    generate_locked_data,
    sampling_recalibration,
    seen_training_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def test_locked_data_regenerates_exact_prior_manifests_and_prepares_inverse_control(tmp_path) -> None:
    receipt = generate_locked_data(tmp_path)

    assert receipt["canonical"]["train"]["row_sha256"] == EXPECTED_TRAIN_SHA
    assert receipt["canonical"]["test"]["row_sha256"] == EXPECTED_TEST_SHA
    assert receipt["inverse_control"]["status"] == "prepared_not_trained"
    assert (tmp_path / "train_injective_inverse_given.jsonl").exists()
    assert receipt["training_exposure_selection"]["manifest"]["depth_counts"] == {
        str(depth): 16 for depth in range(1, 9)
    }


def test_seen_training_indices_are_seed_reproducible_and_unique_within_epoch() -> None:
    first = seen_training_indices(dataset_size=2048, seed=81_001, steps=1000)
    second = seen_training_indices(dataset_size=2048, seed=81_001, steps=1000)

    assert first == second
    assert len(first) == len(set(first)) == 1000


def test_static_construction_audit_rejects_end_reader_hold_story() -> None:
    audit = construction_audit()

    assert all(audit["checks"].values())
    assert "no trained hold objective" in audit["conclusion"]


def test_sampling_recalibration_flags_k20_below_uniform() -> None:
    source = {
        "injective_smoke": {
            "overall": {
                "sampling": {
                    "1": {"mean_coverage": 0.18},
                    "20": {"mean_coverage": 0.5703125},
                }
            }
        }
    }

    receipt = sampling_recalibration(source)

    assert receipt["sampling"]["20"]["uniform_expected_coverage"] > 0.64
    assert receipt["k20_beats_uniform"] is False


def test_bootstrap_exposes_read_only_curriculum_autopsy_target() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_PHASE_G_CURRICULUM_AUTOPSY_CELL.py").read_text(encoding="utf-8")

    assert '"phase_g_curriculum_autopsy"' in bootstrap
    assert "eval/eval_abductive_curriculum_autopsy.py" in bootstrap
    assert "next_training_disabled_pending_strategy_review" in cell
    assert "colab/run_stage5_phase_g_curriculum_autopsy.py" in cell
