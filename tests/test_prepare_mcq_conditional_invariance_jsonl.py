from training.prepare_mcq_conditional_invariance_jsonl import build_rows


def test_build_rows_teaches_semantic_answer_across_permutations() -> None:
    mcq_rows = [
        {
            "id": "q1",
            "question": "Which option is correct?",
            "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "answer": "B",
        }
    ]
    diagnosis = {
        "summary": {"benchmark": "toy"},
        "rows": [
            {
                "id": "q1",
                "change": "loss",
                "candidate_order_sensitive": True,
                "candidate_content_prediction": "A",
                "candidate_cyclic_prediction": "B",
                "candidate_permutation_prediction_counts": {"A": 2, "B": 2},
            }
        ],
    }

    rows, summary = build_rows(
        mcq_rows,
        diagnosis,
        rows_per_item=2,
        semantic_repeat=1,
        label_repeat=1,
        seed=123,
    )

    assert summary["selected_ids"] == 1
    assert summary["output_rows"] == 4
    assert summary["routing_type_counts"] == {
        "conditional_invariance_label": 2,
        "conditional_invariance_semantic": 2,
    }
    semantic = [row for row in rows if row["routing_type"] == "conditional_invariance_semantic"]
    labels = [row for row in rows if row["routing_type"] == "conditional_invariance_label"]
    assert {row["completion"] for row in semantic} == {" beta"}
    assert {row["completion"] for row in labels} == {" B", " C"}
    assert all("\nA. " in row["prompt"] for row in rows)
    assert {row["target_loop_count"] for row in rows} == {1}


def test_build_rows_ignores_order_stable_losses_by_default() -> None:
    mcq_rows = [
        {
            "id": "q1",
            "question": "Question?",
            "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "answer": "B",
        }
    ]
    diagnosis = {
        "rows": [
            {
                "id": "q1",
                "change": "loss",
                "candidate_order_sensitive": False,
            }
        ],
    }

    rows, summary = build_rows(mcq_rows, diagnosis)

    assert rows == []
    assert summary["selected_ids"] == 0


def test_build_rows_can_include_order_sensitive_wins() -> None:
    mcq_rows = [
        {
            "id": "q1",
            "question": "Question?",
            "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "answer": "B",
        }
    ]
    diagnosis = {
        "rows": [
            {
                "id": "q1",
                "change": "win",
                "candidate_order_sensitive": True,
            }
        ],
    }

    rows, summary = build_rows(mcq_rows, diagnosis, include_wins=True, rows_per_item=1)

    assert summary["selected_ids"] == 1
    assert len(rows) == 3
