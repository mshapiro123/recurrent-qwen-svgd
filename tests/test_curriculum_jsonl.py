from __future__ import annotations

import json

import pytest

from training.prepare_curriculum_jsonl import (
    convert_curriculum_records,
    positive_trace_to_causal_example,
    validate_curriculum_record,
)


def curriculum_record() -> dict:
    return {
        "id": "math-001",
        "domain": "math",
        "statement": "A train travels 120 miles in 3 hours. What is its average speed?",
        "answer": {"value": "40 mph", "verified_by": ["cross_model", "numeric"], "confidence": "high"},
        "difficulty": {"pass_rate": 0.7, "reference_model": "weak-ref"},
        "width_signature": {"methods": ["algebra"], "width": 1},
        "depth": {"per_method": {"algebra": 3}, "min_steps": 3},
        "mode": "direct",
        "decontaminated": True,
        "traces": [
            {
                "role": "positive_depth",
                "method": "algebra",
                "correct": True,
                "natural": True,
                "steps": 3,
                "source_model": "teacher-a",
                "logical_source_model": "logical-teacher-a",
                "answer_match": {
                    "matched": True,
                    "source": "method_constrained_answer_line",
                    "parsed_answer": "40 mph",
                    "parsed_answer_normalized": "40 mph",
                    "verified_answer_normalized": "40 mph",
                },
                "text": "Divide distance by time: 120 / 3 = 40 mph.",
            },
            {
                "role": "negative_contrastive",
                "correct": False,
                "error_type": "unit_slip",
                "source_model": "teacher-b",
                "text": "The answer is 120 mph.",
            },
            {
                "role": "verifier_rationalization",
                "correct": False,
                "first_error_step": 1,
                "source_model": "teacher-b",
                "text": "Assume the speed is 120 mph, so it is 120 mph.",
            },
        ],
    }


def test_curriculum_converter_exports_only_positive_traces() -> None:
    examples, report = convert_curriculum_records([curriculum_record()])

    assert len(examples) == 1
    assert examples[0]["trace_role"] == "positive_depth"
    assert "120 mph" not in examples[0]["completion"]
    assert examples[0]["target_loop_count"] == 1
    assert examples[0]["routing_type"] == "direct"
    assert examples[0]["answer_match"]["matched"] is True
    assert examples[0]["source_model"] == "teacher-a"
    assert examples[0]["logical_source_model"] == "logical-teacher-a"
    assert report["role_counts"] == {
        "negative_contrastive": 1,
        "positive_depth": 1,
        "verifier_rationalization": 1,
    }
    assert report["exported_role_counts"] == {"positive_depth": 1}
    assert report["exported_source_model_counts"] == {"teacher-a": 1}


def test_positive_trace_with_false_correctness_is_validation_error() -> None:
    record = curriculum_record()
    record["traces"][0]["correct"] = False

    issues = validate_curriculum_record(record)

    assert any("positive trace must have correct=true" in issue for issue in issues)


def test_converter_fails_on_invalid_positive_trace_by_default() -> None:
    record = curriculum_record()
    record["traces"][0]["correct"] = False

    with pytest.raises(ValueError, match="positive trace must have correct=true"):
        convert_curriculum_records([record])


def test_allow_validation_issues_skips_invalid_records_by_default() -> None:
    record = curriculum_record()
    record["decontaminated"] = False

    examples, report = convert_curriculum_records([record], fail_on_validation=False)

    assert examples == []
    assert report["invalid_records"] == 1
    assert report["skipped_invalid_records"] == 1
    assert report["exported_examples"] == 0
    assert any("decontaminated must be true" in issue for issue in report["issues"])


def test_debug_export_invalid_records_is_explicit() -> None:
    record = curriculum_record()
    record["decontaminated"] = False

    examples, report = convert_curriculum_records(
        [record],
        fail_on_validation=False,
        export_invalid_records=True,
    )

    assert len(examples) == 1
    assert report["invalid_records"] == 1
    assert report["skipped_invalid_records"] == 0
    assert report["export_invalid_records"] is True


def test_positive_trace_must_be_natural_and_stepped() -> None:
    record = curriculum_record()
    record["traces"][0]["natural"] = False
    record["traces"][0].pop("steps")

    issues = validate_curriculum_record(record)

    assert any("positive trace must have natural=true" in issue for issue in issues)
    assert any("positive trace steps must be a positive integer" in issue for issue in issues)


def test_positive_trace_requires_source_model_provenance() -> None:
    record = curriculum_record()
    record["traces"][0].pop("source_model")

    issues = validate_curriculum_record(record)

    assert any("positive trace missing source_model provenance" in issue for issue in issues)


def test_positive_trace_requires_answer_match_proof() -> None:
    record = curriculum_record()
    record["traces"][0].pop("answer_match")

    issues = validate_curriculum_record(record)

    assert any("positive trace missing answer_match proof" in issue for issue in issues)


def test_answer_requires_trusted_verification_anchor() -> None:
    record = curriculum_record()
    record["answer"]["verified_by"] = ["self_report"]

    issues = validate_curriculum_record(record)

    assert any("answer must be verified by cross_model or constructed" in issue for issue in issues)


def test_wide_mode_requires_method_consistent_positive_wide_trace() -> None:
    record = curriculum_record()
    record["mode"] = "wide"
    record["width_signature"] = {"methods": ["algebra", "unit_cancellation"], "width": 2}
    record["depth"] = {"per_method": {"algebra": 3, "unit_cancellation": 4}, "min_steps": 3}
    record["traces"][0]["role"] = "positive_wide"
    record["traces"][0]["method"] = "fake_method"

    issues = validate_curriculum_record(record)

    assert any("positive_wide method must appear in width_signature.methods" in issue for issue in issues)


def test_verifier_detection_is_not_exported_to_positive_sft() -> None:
    record = curriculum_record()
    record["traces"].append(
        {
            "role": "verifier_detection",
            "detected": True,
            "source_model": "judge-a",
            "text": "The proposed answer is incorrect at step 2.",
        }
    )

    examples, report = convert_curriculum_records([record])

    assert len(examples) == 1
    assert "verifier_detection" in report["role_counts"]
    assert report["exported_role_counts"] == {"positive_depth": 1}


def test_negative_and_verifier_traces_have_required_supervision_fields() -> None:
    record = curriculum_record()
    record["traces"][1].pop("correct")
    record["traces"][2]["text"] = ""
    record["traces"].append(
        {
            "role": "verifier_detection",
            "source_model": "judge-a",
            "text": "The proposed solution is wrong.",
        }
    )

    issues = validate_curriculum_record(record)

    assert any("negative_contrastive must have correct=false" in issue for issue in issues)
    assert any("verifier_rationalization missing text" in issue for issue in issues)
    assert any("verifier_detection must have detected=true|false" in issue for issue in issues)


def test_wide_mode_rejects_single_method_width() -> None:
    record = curriculum_record()
    record["mode"] = "wide"

    issues = validate_curriculum_record(record)

    assert any("wide mode requires width >= 2" in issue for issue in issues)


def test_converter_can_filter_modes_without_leaking_negatives() -> None:
    direct = curriculum_record()
    wide = curriculum_record()
    wide["id"] = "math-002"
    wide["mode"] = "wide"
    wide["width_signature"] = {"methods": ["algebra", "unit_cancellation"], "width": 2}
    wide["depth"] = {"per_method": {"algebra": 3, "unit_cancellation": 4}, "min_steps": 3}
    wide["traces"][0]["role"] = "positive_wide"

    examples, report = convert_curriculum_records([direct, wide], modes={"wide"})

    assert len(examples) == 1
    assert examples[0]["curriculum_id"] == "math-002"
    assert examples[0]["trace_role"] == "positive_wide"
    assert report["mode_counts"] == {"wide": 1}


def test_positive_trace_to_causal_example_uses_plain_prompt_when_requested() -> None:
    record = curriculum_record()
    trace = record["traces"][0]

    example = positive_trace_to_causal_example(record, trace, prompt_style="plain")

    assert example["prompt"].startswith("A train travels")
    assert example["completion"].startswith("Divide distance")


def test_cli_writes_safe_positive_jsonl(tmp_path) -> None:
    from training.prepare_curriculum_jsonl import main

    input_jsonl = tmp_path / "curriculum.jsonl"
    output_jsonl = tmp_path / "train.jsonl"
    report_json = tmp_path / "report.json"
    input_jsonl.write_text(json.dumps(curriculum_record()) + "\n", encoding="utf-8")

    assert main(
        [
            "--input_jsonl",
            str(input_jsonl),
            "--output_jsonl",
            str(output_jsonl),
            "--report_json",
            str(report_json),
        ]
    ) == 0

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["trace_role"] == "positive_depth"
    assert report["exported_examples"] == 1


def test_cli_debug_export_invalid_requires_allow_validation_issues(tmp_path) -> None:
    from training.prepare_curriculum_jsonl import main

    input_jsonl = tmp_path / "curriculum.jsonl"
    output_jsonl = tmp_path / "train.jsonl"
    input_jsonl.write_text(json.dumps(curriculum_record()) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--input_jsonl",
                str(input_jsonl),
                "--output_jsonl",
                str(output_jsonl),
                "--export_invalid_records",
            ]
        )

    assert exc.value.code == 2
