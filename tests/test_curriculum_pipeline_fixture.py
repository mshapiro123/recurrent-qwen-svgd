from __future__ import annotations

import json

from training.run_curriculum_pipeline_fixture import main, run_fixture_pipeline


def test_fixture_pipeline_writes_end_to_end_artifacts(tmp_path) -> None:
    summary = run_fixture_pipeline(tmp_path)

    assert summary["typed_records"] == 1
    assert summary["positive_sft_rows"] == 2
    assert summary["mode_counts"] == {"wide": 1}
    assert (tmp_path / "typed_records.jsonl").exists()
    assert (tmp_path / "positive_sft.jsonl").exists()
    assert (tmp_path / "summary.json").exists()

    typed_rows = [json.loads(line) for line in (tmp_path / "typed_records.jsonl").read_text(encoding="utf-8").splitlines()]
    sft_rows = [json.loads(line) for line in (tmp_path / "positive_sft.jsonl").read_text(encoding="utf-8").splitlines()]

    assert typed_rows[0]["mode"] == "wide"
    assert typed_rows[0]["width_signature"]["width"] == 2
    assert {row["trace_role"] for row in sft_rows} == {"positive_wide"}


def test_fixture_pipeline_cli(tmp_path) -> None:
    assert main(["--output_dir", str(tmp_path), "--overwrite"]) == 0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["typed_records"] == 1
    assert summary["positive_sft_rows"] == 2

