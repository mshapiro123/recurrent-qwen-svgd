from __future__ import annotations

import pytest

from colab.run_stage5_natural_surface_transfer import (
    active_diag_min,
    checkpoint_candidates_from_summary,
    existing_row_manifest,
    score_experiment,
    verify_expected_init_checkpoint,
    write_training_config,
    write_jsonl,
)
from colab.run_stage5_natural_surface_checkpoint_curve import (
    best_by_metric,
    checkpoint_path_for_step,
    parse_steps,
)


def eval_payload(*values: float) -> dict:
    return {"active_diagonal": {str(index): value for index, value in enumerate(values, start=1)}}


def test_checkpoint_candidates_prefer_requested_eval_step_then_final_drive_backup() -> None:
    summary = {
        "checkpoint_evals": [
            {"step": 2000, "checkpoint": "local-2000.pt"},
            {"step": 6000, "checkpoint_drive_backup": "/drive/final-6000.pt", "checkpoint": "local-6000.pt"},
        ],
        "final_checkpoint_drive_backup": "/drive/final.pt",
        "final_checkpoint": "local-final.pt",
    }

    candidates = checkpoint_candidates_from_summary(summary, preferred_step=6000)

    assert candidates[:2] == ["/drive/final-6000.pt", "local-6000.pt"]
    assert "/drive/final.pt" in candidates
    assert "local-final.pt" in candidates


def test_active_diag_min_can_focus_on_train_or_extrap_depth_ranges() -> None:
    summary = eval_payload(1.0, 0.9, 0.8, 0.7)

    assert active_diag_min(summary, depths=range(1, 3)) == 0.9
    assert active_diag_min(summary, depths=range(3, 5)) == 0.7


def test_score_experiment_keeps_frozen_and_post_train_reads_separate() -> None:
    frozen = {
        "n24": {
            "relay": eval_payload(1.0, 0.9, 0.8, 0.7),
            "pointer": eval_payload(0.5, 0.4, 0.3, 0.2),
            "synthetic_rehearsal": eval_payload(1.0, 1.0),
        }
    }
    post = {
        "relay": eval_payload(1.0, 1.0, 0.9, 0.8),
        "pointer": eval_payload(0.7, 0.6, 0.5, 0.4),
        "synthetic_rehearsal": eval_payload(0.95, 0.9),
    }

    scored = score_experiment(frozen=frozen, post=post, train_depth_max=2, eval_depth_max=4)

    assert scored["experiment_0"]["relay_depth_1_to_8_min"] == 0.7
    assert scored["experiment_1"]["relay_train_depth_min"] == 1.0
    assert scored["experiment_1"]["relay_extrap_depth_min"] == 0.8
    assert scored["experiment_1"]["synthetic_rehearsal_min_delta"] == pytest.approx(-0.1)


def test_existing_row_manifest_reuses_compacted_sample(tmp_path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    sample_path = tmp_path / "rows_sample.jsonl"
    write_jsonl(sample_path, [{"id": "a"}, {"id": "b"}])

    manifest = existing_row_manifest(rows_path)

    assert manifest["status"] == "reused_sample"
    assert manifest["sample_rows"] == 2


def test_verify_expected_init_checkpoint_accepts_n24_step6000(monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_RUN_ID", "stage5_n24_support12_rung_20260707_140139")
    monkeypatch.setenv("STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_STEP", "6000")

    verify_expected_init_checkpoint(
        {
            "source_run_id": "stage5_n24_support12_rung_20260707_140139",
            "preferred_step": 6000,
            "selected_checkpoint_sha256": "abc123",
        }
    )


def test_verify_expected_init_checkpoint_rejects_wrong_pointer(monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_RUN_ID", "stage5_n24_support12_rung_20260707_140139")
    monkeypatch.setenv("STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_STEP", "6000")

    with pytest.raises(RuntimeError, match="Wrong natural-transfer init run_id"):
        verify_expected_init_checkpoint(
            {
                "source_run_id": "stage5_support6_seed26_plateau_20260708_130654_dose2000",
                "preferred_step": 6000,
                "selected_checkpoint_sha256": "abc123",
            }
        )


def test_checkpoint_curve_step_parser_sorts_and_deduplicates() -> None:
    assert parse_steps("6000,2000,4000,2000") == [2000, 4000, 6000]


def test_checkpoint_curve_requires_local_training_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="same Colab runtime"):
        checkpoint_path_for_step(tmp_path / "source_run", 2000)


def test_checkpoint_curve_best_by_metric_picks_best_step() -> None:
    rows = [
        {"step": 2000, "decision_read": {"relay_train_depth_min": 0.4, "synthetic_rehearsal_min": 0.99}},
        {"step": 4000, "decision_read": {"relay_train_depth_min": 0.7, "synthetic_rehearsal_min": 0.95}},
    ]

    best = best_by_metric(rows)

    assert best["relay_train_depth_min"] == {"step": 4000, "value": 0.7}
    assert best["synthetic_rehearsal_min"] == {"step": 2000, "value": 0.99}


def test_natural_transfer_training_config_accepts_explicit_save_steps(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_NATURAL_TRANSFER_SAVE_STEPS", "1000,1500,2000")

    config_path = write_training_config(
        tmp_path,
        train_jsonl=tmp_path / "train.jsonl",
        resume_from=tmp_path / "resume.pt",
        max_steps=8000,
        max_loops=8,
        dtype="bfloat16",
    )

    text = config_path.read_text(encoding="utf-8")
    assert "save_steps:" in text
    assert "- 1000" in text
    assert "- 1500" in text
    assert "- 2000" in text
