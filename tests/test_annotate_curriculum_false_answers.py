from __future__ import annotations

import json

from training.annotate_curriculum_false_answers import annotate_false_answers, main, read_jsonl


def write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_false_answer_uses_wrong_claimed_answer_first() -> None:
    annotated, rejected, report = annotate_false_answers(
        [
            {
                "id": "p1",
                "claimed_answer": "43",
                "answer": {"value": "42", "normalized": "42", "verified_by": ["cross_model"]},
            }
        ]
    )

    assert rejected == []
    assert annotated[0]["false_answer"] == "43"
    assert annotated[0]["false_answer_metadata"]["source"] == "existing_wrong_answer"
    assert report["source_counts"] == {"existing_wrong_answer": 1}


def test_false_answer_generates_numeric_near_miss_with_units() -> None:
    annotated, rejected, report = annotate_false_answers(
        [{"id": "p1", "answer": {"value": "40 miles per hour", "verified_by": ["cross_model"]}}]
    )

    assert rejected == []
    assert annotated[0]["false_answer"] == "41 miles per hour"
    assert annotated[0]["false_answer_metadata"]["normalized_false_answer"] == "41 miles per hour"
    assert report["source_counts"] == {"numeric_near_miss": 1}


def test_false_answer_preserves_money_prefix_and_decimals() -> None:
    annotated, _, _ = annotate_false_answers(
        [{"id": "p1", "answer": {"value": "$40.00", "verified_by": ["cross_model"]}}]
    )

    assert annotated[0]["false_answer"] == "$41.00"


def test_false_answer_keeps_unannotated_by_default_or_drops() -> None:
    candidates = [{"id": "p1", "answer": {"value": "stare", "verified_by": ["cross_model"]}}]

    kept, kept_rejected, kept_report = annotate_false_answers(candidates)
    dropped, dropped_rejected, dropped_report = annotate_false_answers(candidates, drop_unannotated=True)

    assert len(kept) == 1
    assert kept[0].get("false_answer") is None
    assert kept_rejected[0]["reason"] == "could_not_construct_false_answer"
    assert kept_report["with_false_answer"] == 0
    assert dropped == []
    assert dropped_rejected[0]["id"] == "p1"
    assert dropped_report["annotated"] == 0


def test_cli_writes_false_answer_artifacts(tmp_path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(candidates_path, [{"id": "p1", "answer": {"value": "42", "verified_by": ["cross_model"]}}])

    assert main(
        [
            "--candidates_jsonl",
            str(candidates_path),
            "--output_jsonl",
            str(output_path),
            "--rejected_jsonl",
            str(rejected_path),
            "--report_json",
            str(report_path),
        ]
    ) == 0

    rows = read_jsonl(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rows[0]["false_answer"] == "43"
    assert read_jsonl(rejected_path) == []
    assert report["with_false_answer"] == 1
