from __future__ import annotations

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_stage2a_content_geometry import (
    concurrence_value,
    join_sources,
    make_firm_rows,
)


def source(item_id: str, *, battery: str = "gsm8k") -> dict[str, object]:
    return {
        "item_id": item_id,
        "battery": battery,
        "partition": "verified_train",
        "document_id": f"doc-{item_id}",
        "content_sha256": "a" * 64,
        "reader": "reader-v1",
        "prompt": "question",
        "answer": "42",
    }


def test_stage2a_32b_lineage_is_pinned() -> None:
    assert MODEL_SPECS["verifier_32b"] == {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    }


def test_join_ignores_sealed_source_rows_and_checks_14b_reader() -> None:
    sources = [source("train"), source("sealed") | {"partition": "confirm"}]
    merged = [source("train") | {"teacher_14b_correct": True, "base_correct": False}]
    score = source("train") | {
        "model_key": "teacher_14b",
        "prediction": "42",
        "correct": True,
    }
    joined = join_sources(source_rows=sources, merged_rows=merged, teacher_scores=[score])
    assert [row["item_id"] for row in joined] == ["train"]
    assert joined[0]["teacher_14b_normalized_answer"] == "42"


def test_mbpp_concurrence_is_functional_not_source_text_identity() -> None:
    first = source("code", battery="mbpp") | {
        "prediction": "def f(): return 1",
        "correct": True,
    }
    second = source("code", battery="mbpp") | {
        "prediction": "def f():\n    return 1",
        "correct": True,
    }
    assert concurrence_value(first) == "passes_required_tests"
    assert concurrence_value(second) == "passes_required_tests"


def test_firm_rows_bind_verifier_output_and_reader() -> None:
    row = source("one") | {
        "teacher_14b_correct": True,
        "teacher_14b_normalized_answer": "42",
        "teacher_14b_output_sha256": "a" * 64,
    }
    verifier = source("one") | {"prediction": "42", "correct": True}
    actual = make_firm_rows([row], [verifier])
    assert actual[0]["teacher_32b_normalized_answer"] == "42"
    assert actual[0]["correctness_reader"] == "reader-v1"
    assert len(actual[0]["teacher_32b_output_sha256"]) == 64
