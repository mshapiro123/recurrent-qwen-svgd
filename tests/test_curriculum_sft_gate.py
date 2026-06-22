from __future__ import annotations

import json
from pathlib import Path

from training.check_curriculum_sft_gate import build_gate_payload, main, parse_args, parse_min_mode_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def typed_record() -> dict:
    return {
        "id": "p1",
        "domain": "math",
        "statement": "A rectangle has side lengths 6 and 7. Find its area.",
        "answer": {
            "value": "42",
            "normalized": "42",
            "verified_by": ["cross_model", "numeric"],
            "confidence": "high",
        },
        "difficulty": {"pass_rate": 0.0, "reference_model": "weak-reference"},
        "width_signature": {"methods": ["algebra", "bounded_enumeration"], "width": 2},
        "depth": {"per_method": {"algebra": 3, "bounded_enumeration": 3}, "min_steps": 3},
        "mode": "wide",
        "target_loop_count": 2,
        "decontaminated": True,
        "traces": [
            {
                "role": "positive_wide",
                "method": "algebra",
                "correct": True,
                "natural": True,
                "steps": 3,
                "source_model": "solver-a",
                "answer_match": {
                    "matched": True,
                    "source": "method_constrained_answer_line",
                    "parsed_answer": "42",
                    "parsed_answer_normalized": "42",
                    "verified_answer_normalized": "42",
                },
                "text": "Multiply the sides: 6 * 7 = 42.\nANSWER: 42",
            },
            {
                "role": "verifier_rationalization",
                "correct": False,
                "text": "Assume the answer is 43.\nANSWER: 43",
            },
        ],
    }


def positive_sft_row() -> dict:
    return {
        "prompt": "<|im_start|>user\nFind area<|im_end|>\n<|im_start|>assistant\n",
        "completion": "Multiply the sides: 6 * 7 = 42.\nANSWER: 42",
        "trace_role": "positive_wide",
        "curriculum_id": "p1",
        "curriculum_mode": "wide",
        "routing_type": "wide",
        "target_loop_count": 2,
    }


def write_complete_work_dir(work_dir: Path, *, strict: bool = True) -> Path:
    artifacts = {
        "summary": work_dir / "summary.json",
        "typed_records": work_dir / "typed_records.jsonl",
        "typed_records_report": work_dir / "typed_records_report.json",
        "positive_sft": work_dir / "positive_sft.jsonl",
        "positive_sft_report": work_dir / "positive_sft_report.json",
        "verified_candidates_report": work_dir / "verified_candidates_report.json",
        "decontam_report": work_dir / "decontam_report.json",
        "method_solutions_report": work_dir / "method_solutions_report.json",
        "naturalness_report": work_dir / "naturalness_report.json",
        "depth_report": work_dir / "depth_report.json",
        "difficulty_report": work_dir / "difficulty_report.json",
    }
    write_jsonl(artifacts["typed_records"], [typed_record()])
    write_jsonl(artifacts["positive_sft"], [positive_sft_row()])
    write_json(
        artifacts["typed_records_report"],
        {
            "records": 1,
            "validation_issues": [],
            "unsafe_auxiliary_traces": 0,
            "distinctness_required": True,
            "min_natural_agree": 2,
            "min_distinct_agree": 2,
            "mode_counts": {"wide": 1},
        },
    )
    write_json(
        artifacts["positive_sft_report"],
        {
            "exported_examples": 1,
            "invalid_records": 0,
            "issues": [],
            "exported_role_counts": {"positive_wide": 1},
        },
    )
    write_json(
        artifacts["verified_candidates_report"],
        {"verified": 1, "require_programmatic_answer_check": strict},
    )
    write_json(artifacts["decontam_report"], {"accepted": 1, "rejected": 0})
    write_json(artifacts["method_solutions_report"], {"solution_candidates": 2})
    write_json(artifacts["naturalness_report"], {"judgments": 4})
    write_json(artifacts["depth_report"], {"measurements": 4})
    write_json(artifacts["difficulty_report"], {"measured": 1})
    summary = {
        "kind": "curriculum_pipeline_from_artifacts",
        "status": "complete",
        "artifacts": {
            name: {"path": str(path), "exists": path.exists(), "lines": 1 if path.suffix == ".jsonl" else 0}
            for name, path in artifacts.items()
        },
        "counts": {"positive_sft_rows": 1, "typed_records": 1},
    }
    write_json(artifacts["summary"], summary)
    return artifacts["summary"]


def test_curriculum_sft_gate_allows_complete_strict_shard(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run")

    payload = build_gate_payload(parse_args(["--summary_json", str(summary)]))

    assert payload["go"] is True
    assert payload["status"] == "go_train_recurrent_sft"
    assert payload["issues"] == []
    assert payload["checks"]["positive_sft"]["role_counts"] == {"positive_wide": 1}
    assert payload["checks"]["typed_records"]["positive_missing_answer_match"] == 0
    assert payload["checks"]["positive_sft"]["mode_requirements"] == {}


def test_parse_min_mode_rows_accepts_repeated_and_csv_values() -> None:
    assert parse_min_mode_rows(["direct=8,deep_narrow:12", "wide=4"]) == {
        "direct": 8,
        "deep_narrow": 12,
        "wide": 4,
    }


def test_curriculum_sft_gate_allows_matching_mode_requirement(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run")

    payload = build_gate_payload(parse_args(["--summary_json", str(summary), "--min_mode_rows", "wide=1"]))

    assert payload["go"] is True
    assert payload["checks"]["positive_sft"]["mode_requirements"] == {
        "wide": {"required": 1, "observed": 1, "passed": True}
    }


def test_curriculum_sft_gate_blocks_missing_mode_requirement(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run")

    payload = build_gate_payload(parse_args(["--summary_json", str(summary), "--min_mode_rows", "direct=1"]))

    assert payload["go"] is False
    assert payload["checks"]["positive_sft"]["mode_requirements"] == {
        "direct": {"required": 1, "observed": 0, "passed": False}
    }
    assert any(issue["code"] == "too_few_mode_rows" for issue in payload["issues"])


def test_curriculum_sft_gate_blocks_incomplete_pipeline(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["status"] = "pending_judgment_responses"
    write_json(summary, payload)

    result = build_gate_payload(parse_args(["--summary_json", str(summary)]))

    assert result["go"] is False
    assert any(issue["code"] == "pipeline_not_complete" for issue in result["issues"])


def test_curriculum_sft_gate_blocks_cross_model_only_generated_answers(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run", strict=False)

    result = build_gate_payload(parse_args(["--summary_json", str(summary)]))

    assert result["go"] is False
    assert any(issue["code"] == "programmatic_check_not_required" for issue in result["issues"])


def test_curriculum_sft_gate_blocks_positive_trace_without_answer_match(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run")
    typed_path = tmp_path / "run" / "typed_records.jsonl"
    record = typed_record()
    record["traces"][0].pop("answer_match")
    write_jsonl(typed_path, [record])

    result = build_gate_payload(parse_args(["--summary_json", str(summary)]))

    assert result["go"] is False
    assert any(issue["code"] == "positive_trace_missing_answer_match" for issue in result["issues"])


def test_curriculum_sft_gate_blocks_low_judge_agreement(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run")
    report_path = tmp_path / "run" / "typed_records_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["min_natural_agree"] = 1
    report["min_distinct_agree"] = 1
    write_json(report_path, report)

    result = build_gate_payload(parse_args(["--summary_json", str(summary)]))

    assert result["go"] is False
    assert any(issue["code"] == "naturalness_agreement_too_low" for issue in result["issues"])
    assert any(issue["code"] == "distinctness_agreement_too_low" for issue in result["issues"])


def test_curriculum_sft_gate_cli_writes_reports(tmp_path) -> None:
    summary = write_complete_work_dir(tmp_path / "run")
    output_json = tmp_path / "gate.json"
    output_md = tmp_path / "gate.md"

    assert main(
        [
            "--summary_json",
            str(summary),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
            "--fail_on_no_go",
        ]
    ) == 0

    assert json.loads(output_json.read_text(encoding="utf-8"))["go"] is True
    assert "Curriculum SFT Gate" in output_md.read_text(encoding="utf-8")
