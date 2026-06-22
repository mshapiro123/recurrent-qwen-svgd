from __future__ import annotations

import json

from training.assemble_curriculum_records import assemble_curriculum_records, main
from training.prepare_curriculum_jsonl import convert_curriculum_records


def verified_candidate(*, decontaminated: bool = True) -> dict:
    return {
        "id": "p1",
        "domain": "math",
        "statement": "Find the area of a rectangle with sides 6 and 7.",
        "answer": {"value": "42", "normalized": "42", "verified_by": ["cross_model"], "confidence": "high"},
        "candidate_methods": ["algebra", "bounded_enumeration"],
        "decontaminated": decontaminated,
    }


def solution(solution_id: str, *, method: str, text: str = "Solve cleanly.\nANSWER: 42") -> dict:
    return {
        "id": solution_id,
        "record_id": "p1",
        "domain": "math",
        "statement": "Find the area of a rectangle with sides 6 and 7.",
        "method": method,
        "source_model": "opus-test",
        "text": text,
        "solution": text,
        "answer": {"value": "42", "normalized": "42", "verified_by": ["cross_model"]},
        "correct": True,
    }


def natural(solution_id: str, *, method: str, actually_uses: str | None = None) -> dict:
    return {
        "solution_id": solution_id,
        "record_id": "p1",
        "method": method,
        "judge_model": "judge-a",
        "natural": True,
        "actually_uses": actually_uses or method,
        "reason": "natural",
    }


def depth(solution_id: str, *, method: str, count: int) -> dict:
    return {
        "solution_id": solution_id,
        "record_id": "p1",
        "method": method,
        "judge_model": "judge-a",
        "steps": [f"step {idx}" for idx in range(1, count + 1)],
        "count": count,
    }


def test_assemble_wide_record_and_exports_positive_sft() -> None:
    records, report = assemble_curriculum_records(
        [verified_candidate()],
        [
            solution("s-algebra", method="algebra"),
            solution("s-enum", method="bounded_enumeration"),
        ],
        [
            natural("s-algebra", method="algebra"),
            natural("s-enum", method="bounded_enumeration"),
        ],
        [
            depth("s-algebra", method="algebra", count=3),
            depth("s-enum", method="bounded_enumeration", count=4),
        ],
        deep_threshold=5,
    )

    assert report["records"] == 1
    assert report["mode_counts"] == {"wide": 1}
    record = records[0]
    assert record["mode"] == "wide"
    assert record["width_signature"] == {"methods": ["algebra", "bounded_enumeration"], "width": 2}
    assert {trace["role"] for trace in record["traces"]} == {"positive_wide"}

    examples, export_report = convert_curriculum_records(records)
    assert len(examples) == 2
    assert export_report["exported_role_counts"] == {"positive_wide": 2}


def test_assemble_rejects_not_decontaminated_by_default() -> None:
    records, report = assemble_curriculum_records(
        [verified_candidate(decontaminated=False)],
        [solution("s-algebra", method="algebra")],
        [natural("s-algebra", method="algebra")],
        [depth("s-algebra", method="algebra", count=3)],
    )

    assert records == []
    assert report["rejected_records"] == [{"id": "p1", "reason": "not_decontaminated"}]


def test_assemble_rejects_method_mismatch_when_required() -> None:
    records, report = assemble_curriculum_records(
        [verified_candidate()],
        [solution("s-algebra", method="algebra")],
        [natural("s-algebra", method="algebra", actually_uses="bounded_enumeration")],
        [depth("s-algebra", method="algebra", count=3)],
    )

    assert records == []
    assert report["rejected_records"][0]["reason"] == "no_natural_depth_measured_solution"


def test_assemble_deep_narrow_role_and_target_loop() -> None:
    records, report = assemble_curriculum_records(
        [verified_candidate()],
        [solution("s-algebra", method="algebra")],
        [natural("s-algebra", method="algebra")],
        [depth("s-algebra", method="algebra", count=8)],
        deep_threshold=5,
        max_target_loops=4,
    )

    assert report["mode_counts"] == {"deep_narrow": 1}
    assert records[0]["mode"] == "deep_narrow"
    assert records[0]["target_loop_count"] == 4
    assert records[0]["traces"][0]["role"] == "positive_depth"


def test_cli_assembles_curriculum_records(tmp_path) -> None:
    verified_path = tmp_path / "verified.jsonl"
    solutions_path = tmp_path / "solutions.jsonl"
    natural_path = tmp_path / "natural.jsonl"
    depth_path = tmp_path / "depth.jsonl"
    output_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.json"

    verified_path.write_text(json.dumps(verified_candidate()) + "\n", encoding="utf-8")
    solutions_path.write_text(json.dumps(solution("s-algebra", method="algebra")) + "\n", encoding="utf-8")
    natural_path.write_text(json.dumps(natural("s-algebra", method="algebra")) + "\n", encoding="utf-8")
    depth_path.write_text(json.dumps(depth("s-algebra", method="algebra", count=3)) + "\n", encoding="utf-8")

    assert main(
        [
            "--verified_candidates_jsonl",
            str(verified_path),
            "--solution_candidates_jsonl",
            str(solutions_path),
            "--naturalness_jsonl",
            str(natural_path),
            "--depth_jsonl",
            str(depth_path),
            "--output_jsonl",
            str(output_path),
            "--report_json",
            str(report_path),
        ]
    ) == 0

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rows[0]["id"] == "p1"
    assert report["records"] == 1

