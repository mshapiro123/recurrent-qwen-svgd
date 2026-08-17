from __future__ import annotations

import json

from eval.cache_paper2_stage2a_training import (
    MODEL_SPECS,
    TEACHER_KEY,
    TeacherForcedExample,
    answer_token_positions,
    build_population,
    write_jsonl,
)


class StubTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"] + "\nAssistant:"


def _identity(battery: str, item_id: str) -> dict[str, str]:
    return {
        "battery": battery,
        "item_id": item_id,
        "content_sha256": f"sha-{item_id}",
    }


def test_answer_token_positions_respects_mbpp_whole_token_boundaries() -> None:
    offsets = [(0, 5), (5, 10), (10, 14)]
    positions, exact = answer_token_positions(
        offsets, span_start=6, span_end=14, strict_boundaries=True
    )
    assert positions == []
    assert exact is False

    positions, exact = answer_token_positions(
        offsets, span_start=5, span_end=14, strict_boundaries=True
    )
    assert positions == [1, 2]
    assert exact is True


def test_arc_population_allows_whitespace_bearing_answer_token() -> None:
    source = {
        **_identity("arc_challenge", "arc-1"),
        "prompt": {
            "question": "Which letter?",
            "choice_labels": ["A", "B", "C", "D"],
            "choice_text": ["one", "two", "three", "four"],
        },
        "answer": "B",
    }
    teacher = {
        **_identity("arc_challenge", "arc-1"),
        "correct": True,
        "prediction": "B",
        "revision": MODEL_SPECS[TEACHER_KEY]["revision"],
    }
    firm = {
        **_identity("arc_challenge", "arc-1"),
        "partition": "verified_train",
        "stage2a_firm_knowledge_admitted": True,
    }
    examples, owners, receipt = build_population(
        firm_rows=[firm],
        memory_rows=[firm],
        source_rows=[source],
        teacher_rows=[teacher],
        tokenizer=StubTokenizer(),
    )

    assert len(examples) == 1
    assert isinstance(examples[0], TeacherForcedExample)
    assert examples[0].strict_boundaries is False
    assert owners[0]["retrieval_contract"] == "leave_one_out"
    assert receipt["rows"] == 1


def test_population_manifest_writer_is_jsonl(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    rows = [{"item_id": "a"}, {"item_id": "b"}]
    write_jsonl(path, rows)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == rows
