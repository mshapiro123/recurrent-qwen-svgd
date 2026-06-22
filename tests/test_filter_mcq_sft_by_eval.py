from __future__ import annotations

from training.filter_mcq_sft_by_eval import select_rows


def _mcq(row_id: str, answer: str = "A") -> dict:
    return {
        "id": row_id,
        "question": f"Question {row_id}?",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "answer": answer,
    }


def _eval(row_id: str, answer: str, prediction: str, answer_score: float, other_score: float = 0.0) -> dict:
    scores = {"A": other_score, "B": other_score, "C": other_score, "D": other_score}
    scores[answer] = answer_score
    return {
        "id": row_id,
        "answer": answer,
        "prediction": prediction,
        "hit": prediction == answer,
        "scores": scores,
    }


def test_select_rows_keeps_base_correct_high_margin_examples() -> None:
    mcq_rows = [_mcq("keep", "A"), _mcq("wrong", "B"), _mcq("low", "C")]
    eval_rows = [
        _eval("keep", "A", "A", 2.0),
        _eval("wrong", "B", "A", 3.0),
        _eval("low", "C", "C", 0.2),
    ]

    rows, summary = select_rows(
        mcq_rows,
        eval_rows,
        min_base_margin=1.0,
        balance_labels=False,
    )

    assert [row["arc_id"] for row in rows] == ["keep"]
    assert rows[0]["target_loop_count"] == 1
    assert rows[0]["routing_type"] == "direct_base_preserve"
    assert rows[0]["base_margin"] == 2.0
    assert summary["skipped"] == {"base_wrong": 1, "low_margin": 1}


def test_select_rows_balances_answer_labels() -> None:
    mcq_rows = [
        _mcq("a1", "A"),
        _mcq("a2", "A"),
        _mcq("b1", "B"),
        _mcq("b2", "B"),
        _mcq("c1", "C"),
    ]
    eval_rows = [
        _eval("a1", "A", "A", 2.0),
        _eval("a2", "A", "A", 2.0),
        _eval("b1", "B", "B", 2.0),
        _eval("b2", "B", "B", 2.0),
        _eval("c1", "C", "C", 2.0),
    ]

    rows, summary = select_rows(
        mcq_rows,
        eval_rows,
        min_base_margin=1.0,
        balance_labels=True,
        seed=0,
    )

    assert summary["pre_balance_answer_counts"] == {"A": 2, "B": 2, "C": 1}
    assert summary["answer_counts"] == {"A": 1, "B": 1, "C": 1}
    assert sorted(row["answer"] for row in rows) == ["A", "B", "C"]
