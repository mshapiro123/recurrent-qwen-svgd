from __future__ import annotations

import json
from pathlib import Path

from colab.run_stage5_phase_g_alpha import lock_margin, prepare_data
from eval.eval_phase_g_alpha import read_resume_cache


def test_phase_g_runner_has_drive_backed_stable_resume_contract() -> None:
    source = Path("colab/run_stage5_phase_g_alpha.py").read_text(encoding="utf-8")

    assert "stage5_phase_g_alpha_guided_width_20260717" in source
    assert "sync_receipts_to_drive" in source
    assert "resume_cache_path" in source
    assert "resume_completed_phase_g_training" in source


def test_phase_g_runner_regenerates_exact_frozen_test_rows(tmp_path) -> None:
    manifests = prepare_data(tmp_path)

    assert manifests["test"]["rows"] == 512
    assert (
        manifests["test"]["row_sha256"]
        == "eb80ef24637aee511a3e35607e87ae2530842ce11c551e6fa90ecda4d4115ef8"
    )
    assert manifests["train"]["rows"] == 2048
    assert manifests["calibration"]["rows"] == 512


def test_phase_g_margin_lock_is_prospective_and_has_program_floor(tmp_path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    rows = [
        {
            "id": f"row_{index}",
            "scores": {"A": 3.0, "B": 1.0, "C": 0.0},
            "reachable_symbols": ["A", "B"],
        }
        for index in range(32)
    ]
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = lock_margin(rows_path, tmp_path / "margin.json")

    assert result["status"] == "locked_before_guided_training"
    assert result["locked_absolute_mean_coverage_margin"] >= 0.05
    assert result["paired_rows"] == 32


def test_phase_g_resume_cache_discards_only_torn_final_record(tmp_path) -> None:
    cache = tmp_path / "row_cache.jsonl"
    cache.write_text('{"id":"complete"}\n{"id":"torn"', encoding="utf-8")

    rows = read_resume_cache(cache)

    assert rows == [{"id": "complete"}]
    assert cache.read_text(encoding="utf-8") == '{"id": "complete"}\n'
