from __future__ import annotations

import json

from colab.assess_stage5_programmatic_depth_repair import assess


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_programmatic_depth_assessment_passes_loss_lift_with_loop_calibration(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_programmatic_depth_repair as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    val_sft = tmp_path / "outputs" / "stage5" / "run" / "val_sft.jsonl"
    val_sft.parent.mkdir(parents=True)
    _write_jsonl(
        val_sft,
        [
            {"target_loop_count": 1},
            {"target_loop_count": 3},
            {"target_loop_count": 4},
        ],
    )
    summary = {
        "val_sft": val_sft.relative_to(tmp_path).as_posix(),
        "best_checkpoint": "outputs/stage5/run/phase1/phase1_step_100.pt",
        "start_eval": {"loss": 3.0, "mean_expected_loops": 2.9},
        "best_eval": {"loss": 2.5, "mean_expected_loops": 2.8},
    }

    payload = assess(summary, source_summary=tmp_path / "outputs" / "stage5" / "run" / "summary.json")

    assert payload["status"] == "programmatic_depth_lift"
    assert payload["passed"] is True
    assert payload["checkpoint"] == "outputs/stage5/run/phase1/phase1_step_100.pt"
    assert payload["evidence"]["target_loop_mean"] == 8 / 3
    assert payload["evidence"]["loss_delta"] == -0.5


def test_programmatic_depth_assessment_flags_loop_warning(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_programmatic_depth_repair as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    val_sft = tmp_path / "val_sft.jsonl"
    _write_jsonl(val_sft, [{"target_loop_count": 1}, {"target_loop_count": 1}])
    summary = {
        "val_sft": val_sft.as_posix(),
        "best_checkpoint": "outputs/stage5/run/phase1/phase1_step_100.pt",
        "start_eval": {"loss": 3.0, "mean_expected_loops": 1.1},
        "best_eval": {"loss": 2.9, "mean_expected_loops": 2.0},
    }

    payload = assess(summary, source_summary=tmp_path / "summary.json")

    assert payload["status"] == "programmatic_depth_lift_loop_warning"
    assert payload["passed"] is False


def test_programmatic_depth_assessment_rejects_missing_checkpoint(tmp_path) -> None:
    summary = {
        "start_eval": {"loss": 3.0, "mean_expected_loops": 2.0},
        "best_eval": {"loss": 2.0, "mean_expected_loops": 2.0},
    }

    payload = assess(summary, source_summary=tmp_path / "summary.json")

    assert payload["status"] == "invalid_metrics"
    assert payload["passed"] is False
