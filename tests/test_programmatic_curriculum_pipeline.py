from __future__ import annotations

import json

from training.check_curriculum_sft_gate import build_gate_payload, parse_args
from training.run_programmatic_curriculum_pipeline import main


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_programmatic_curriculum_pipeline_writes_gate_ready_work_dir(tmp_path) -> None:
    work_dir = tmp_path / "programmatic_direct_deep"

    assert main(
        [
            "--work_dir",
            str(work_dir),
            "--num_direct",
            "3",
            "--num_deep_narrow",
            "4",
            "--seed",
            "21",
        ]
    ) == 0

    summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "programmatic_curriculum_pipeline"
    assert summary["status"] == "complete"
    assert summary["counts"]["typed_records"] == 7
    assert summary["counts"]["positive_sft_rows"] == 7

    typed_rows = read_jsonl(work_dir / "typed_records.jsonl")
    sft_rows = read_jsonl(work_dir / "positive_sft.jsonl")
    assert {row["mode"] for row in typed_rows} == {"direct", "deep_narrow"}
    assert {row["curriculum_mode"] for row in sft_rows} == {"direct", "deep_narrow"}
    assert all(row["trace_role"].startswith("positive_") for row in sft_rows)

    gate = build_gate_payload(
        parse_args(
            [
                "--summary_json",
                str(work_dir / "summary.json"),
                "--min_positive_rows",
                "7",
                "--min_mode_rows",
                "direct=3,deep_narrow=4",
                "--max_loop_target",
                "4",
            ]
        )
    )
    assert gate["go"] is True
    assert gate["issues"] == []
    assert gate["checks"]["positive_sft"]["mode_requirements"] == {
        "deep_narrow": {"required": 4, "observed": 4, "passed": True},
        "direct": {"required": 3, "observed": 3, "passed": True},
    }


def test_programmatic_curriculum_pipeline_gate_blocks_wrong_mode_requirement(tmp_path) -> None:
    work_dir = tmp_path / "programmatic_direct_deep"
    assert main(
        [
            "--work_dir",
            str(work_dir),
            "--num_direct",
            "2",
            "--num_deep_narrow",
            "2",
        ]
    ) == 0

    gate = build_gate_payload(
        parse_args(
            [
                "--summary_json",
                str(work_dir / "summary.json"),
                "--min_positive_rows",
                "4",
                "--min_mode_rows",
                "wide=1",
            ]
        )
    )

    assert gate["go"] is False
    assert any(issue["code"] == "too_few_mode_rows" for issue in gate["issues"])
