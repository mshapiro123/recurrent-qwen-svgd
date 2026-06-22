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
    assert report["role_counts"] == {
        "negative_contrastive": 1,
        "positive_depth": 1,
        "verifier_rationalization": 1,
    }
    assert report["exported_role_counts"] == {"positive_depth": 1}


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


def test_positive_trace_must_be_natural_and_stepped() -> None:
    record = curriculum_record()
    record["traces"][0]["natural"] = False
    record["traces"][0].pop("steps")

    issues = validate_curriculum_record(record)

    assert any("positive trace must have natural=true" in issue for issue in issues)
    assert any("positive trace steps must be a positive integer" in issue for issue in issues)


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
