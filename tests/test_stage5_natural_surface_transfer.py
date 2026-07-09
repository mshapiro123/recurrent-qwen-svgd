from __future__ import annotations

import pytest

from colab.run_stage5_natural_surface_transfer import (
    active_diag_min,
    checkpoint_candidates_from_summary,
    existing_row_manifest,
    score_experiment,
    write_jsonl,
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
