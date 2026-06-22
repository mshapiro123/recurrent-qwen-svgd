from __future__ import annotations

import json

from training.annotate_curriculum_difficulty import annotate_difficulty, main, read_jsonl


def write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_annotate_difficulty_computes_reference_pass_rate() -> None:
    annotated, rejected, report = annotate_difficulty(
        [{"id": "p1", "statement": "What is 2+2?"}],
        [
            {"record_id": "p1", "correct": True},
            {"record_id": "p1", "correct": False},
            {"record_id": "p1", "correct": True},
        ],
        reference_model="weak-ref",
        min_samples=3,
    )

    assert rejected == []
    assert annotated[0]["difficulty"] == {
        "pass_rate": 2 / 3,
        "reference_model": "weak-ref",
        "samples": 3,
        "correct": 2,
        "measured": True,
    }
    assert annotated[0]["difficulty_pass_rate"] == 2 / 3
    assert report["measured"] == 1


def test_annotate_difficulty_marks_unmeasured_when_samples_insufficient() -> None:
    annotated, rejected, report = annotate_difficulty(
        [{"id": "p1", "statement": "What is 2+2?"}],
        [{"record_id": "p1", "correct": True}],
        reference_model="weak-ref",
        min_samples=2,
    )

    assert rejected == [{"id": "p1", "reason": "insufficient_reference_samples", "samples": 1, "min_samples": 2}]
    assert annotated[0]["difficulty"] == {
        "pass_rate": None,
        "reference_model": "weak-ref",
        "samples": 1,
        "measured": False,
    }
    assert report["measured"] == 0


def test_annotate_difficulty_can_drop_unmeasured() -> None:
    annotated, rejected, report = annotate_difficulty(
        [{"id": "p1", "statement": "What is 2+2?"}],
        [],
        reference_model="weak-ref",
        min_samples=1,
        drop_unmeasured=True,
    )

    assert annotated == []
    assert rejected[0]["reason"] == "insufficient_reference_samples"
    assert report["annotated"] == 0


def test_cli_writes_annotated_candidates_and_report(tmp_path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    report_path = tmp_path / "report.json"

    write_jsonl(candidates_path, [{"id": "p1", "statement": "What is 3+4?"}])
    write_jsonl(
        attempts_path,
        [
            {"record_id": "p1", "is_correct": True},
            {"record_id": "p1", "is_correct": False},
        ],
    )

    assert main(
        [
            "--candidates_jsonl",
            str(candidates_path),
            "--attempts_jsonl",
            str(attempts_path),
            "--output_jsonl",
            str(output_path),
            "--rejected_jsonl",
            str(rejected_path),
            "--report_json",
            str(report_path),
            "--reference_model",
            "weak-ref",
            "--min_samples",
            "2",
        ]
    ) == 0

    rows = read_jsonl(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rows[0]["difficulty"]["pass_rate"] == 0.5
    assert read_jsonl(rejected_path) == []
    assert report["mean_pass_rate"] == 0.5
