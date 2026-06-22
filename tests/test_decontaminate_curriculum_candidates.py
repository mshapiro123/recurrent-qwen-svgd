from __future__ import annotations

import json

from training.decontaminate_curriculum_candidates import (
    annotate_candidates,
    build_reference_index,
    main,
    read_jsonl,
)


def write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_decontamination_rejects_exact_overlap(tmp_path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    references_path = tmp_path / "references.jsonl"
    output_path = tmp_path / "clean.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    report_path = tmp_path / "report.json"

    text = "A train travels 120 miles in 3 hours. What is its average speed?"
    write_jsonl(candidates_path, [{"id": "candidate-1", "statement": text}])
    write_jsonl(references_path, [{"name": "train speed", "prompt": text}])

    assert main(
        [
            "--candidates_jsonl",
            str(candidates_path),
            "--references_jsonl",
            str(references_path),
            "--output_jsonl",
            str(output_path),
            "--rejected_jsonl",
            str(rejected_path),
            "--report_json",
            str(report_path),
            "--ngram_size",
            "3",
            "--threshold",
            "0.5",
        ]
    ) == 0

    assert read_jsonl(output_path) == []
    rejected = read_jsonl(rejected_path)
    assert rejected[0]["id"] == "candidate-1"
    assert rejected[0]["decontaminated"] is False
    assert rejected[0]["decontamination"]["matched_reference_id"] == "train speed"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["accepted"] == 0
    assert report["rejected"] == 1


def test_decontamination_rejects_reference_embedded_in_longer_statement(tmp_path) -> None:
    references_path = tmp_path / "references.jsonl"
    write_jsonl(
        references_path,
        [
            {
                "id": "eval-1",
                "prompt": "The pharmacy has 20 tubs in storage and needs 100 total tubs for the week.",
            }
        ],
    )
    references = build_reference_index([references_path], text_fields=("statement", "prompt"), ngram_size=4)
    accepted, rejected, report = annotate_candidates(
        [
            {
                "id": "candidate-1",
                "statement": (
                    "Solve carefully. The pharmacy has 20 tubs in storage and needs 100 total tubs "
                    "for the week. It buys some from a new vendor."
                ),
            }
        ],
        references,
        text_fields=("statement", "prompt"),
        ngram_size=4,
        threshold=0.5,
    )

    assert accepted == []
    assert rejected[0]["decontaminated"] is False
    assert rejected[0]["decontamination"]["reference_containment"] == 1.0
    assert report["rejected"] == 1


def test_decontamination_accepts_low_overlap_and_preserves_fields(tmp_path) -> None:
    references_path = tmp_path / "references.jsonl"
    write_jsonl(references_path, [{"id": "eval-1", "prompt": "What is 17 plus 28?"}])
    references = build_reference_index([references_path], text_fields=("statement", "prompt"), ngram_size=3)

    accepted, rejected, report = annotate_candidates(
        [
            {
                "id": "candidate-2",
                "statement": "A square has side length 9. Find its area.",
                "domain": "math",
                "claimed_answer": "81",
            }
        ],
        references,
        text_fields=("statement", "prompt"),
        ngram_size=3,
        threshold=0.5,
    )

    assert rejected == []
    assert accepted[0]["id"] == "candidate-2"
    assert accepted[0]["domain"] == "math"
    assert accepted[0]["decontaminated"] is True
    assert accepted[0]["decontamination"]["score"] == 0.0
    assert report["accepted"] == 1


def test_default_min_ngrams_catch_short_prompt_with_wrapper(tmp_path) -> None:
    references_path = tmp_path / "references.jsonl"
    write_jsonl(references_path, [{"id": "eval-short", "prompt": "Solve exactly. What is 17 + 28?"}])
    references = build_reference_index([references_path], text_fields=("statement", "prompt"), ngram_size=5)

    accepted, rejected, _ = annotate_candidates(
        [{"id": "candidate-short", "statement": "What is 17 + 28?"}],
        references,
        text_fields=("statement", "prompt"),
        ngram_size=5,
        threshold=0.5,
    )

    assert accepted == []
    assert rejected[0]["decontaminated"] is False
    assert rejected[0]["decontamination"]["score"] >= 0.5


def test_cli_accepts_repeated_and_comma_separated_reference_paths(tmp_path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    ref_a = tmp_path / "ref_a.jsonl"
    ref_b = tmp_path / "ref_b.jsonl"
    output_path = tmp_path / "clean.jsonl"
    annotated_path = tmp_path / "annotated.jsonl"

    write_jsonl(candidates_path, [{"id": "clean", "statement": "How many sides does a triangle have?"}])
    write_jsonl(ref_a, [{"id": "eval-a", "prompt": "What is 2 + 2?"}])
    write_jsonl(ref_b, [{"id": "eval-b", "question": "Which particle has a negative electric charge?"}])

    assert main(
        [
            "--candidates_jsonl",
            str(candidates_path),
            "--references_jsonl",
            f"{ref_a},{ref_b}",
            "--output_jsonl",
            str(output_path),
            "--annotated_jsonl",
            str(annotated_path),
            "--ngram_size",
            "3",
        ]
    ) == 0

    assert len(read_jsonl(output_path)) == 1
    assert read_jsonl(annotated_path)[0]["decontaminated"] is True
