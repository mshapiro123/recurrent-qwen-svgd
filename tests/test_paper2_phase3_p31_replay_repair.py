from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.repair_paper2_phase3_p31_replay_duplicates import repair_replay_duplicates


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def source_row(item_id: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "battery": "gsm8k",
        "battery_role": "target_primary",
        "partition": "verified_train",
        "document_id": f"doc-{item_id}",
        "content_sha256": item_id * 64,
        "reader": "reader-v1",
    }


def score_row(item_id: str, *, prediction: str, correct: bool) -> dict[str, object]:
    return {
        **source_row(item_id),
        "kind": "paper2_phase3_p31_model_score_v1",
        "model_key": "teacher_14b",
        "model": "model",
        "revision": "revision",
        "generation_batch_size": 8,
        "prediction": prediction,
        "correct": correct,
        "generated_text": f"answer {prediction}",
    }


def test_repair_archives_and_keeps_first_durable_write(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    score_path = tmp_path / "scores.jsonl"
    archive_path = tmp_path / "archive" / "scores.jsonl"
    write_jsonl(source_path, [source_row("a"), source_row("b")])
    write_jsonl(
        score_path,
        [
            score_row("a", prediction="1", correct=True),
            score_row("a", prediction="2", correct=False),
            score_row("b", prediction="3", correct=True),
        ],
    )
    receipt = repair_replay_duplicates(
        score_path=score_path,
        source_path=source_path,
        archive_path=archive_path,
    )
    repaired = [json.loads(line) for line in score_path.read_text().splitlines()]
    assert [row["item_id"] for row in repaired] == ["a", "b"]
    assert repaired[0]["prediction"] == "1"
    assert receipt["duplicate_groups"] == 1
    assert receipt["prediction_conflict_groups"] == 1
    assert receipt["correctness_conflict_groups"] == 1
    assert receipt["original_sha256"] == receipt["archive_sha256"]


def test_repair_rejects_immutable_lineage_drift(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    score_path = tmp_path / "scores.jsonl"
    write_jsonl(source_path, [source_row("a")])
    changed = score_row("a", prediction="2", correct=False)
    changed["revision"] = "changed"
    write_jsonl(score_path, [score_row("a", prediction="1", correct=True), changed])
    with pytest.raises(RuntimeError, match="immutable lineage"):
        repair_replay_duplicates(
            score_path=score_path,
            source_path=source_path,
            archive_path=tmp_path / "archive.jsonl",
        )
