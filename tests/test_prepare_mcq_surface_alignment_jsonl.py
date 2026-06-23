from training.prepare_mcq_surface_alignment_jsonl import build_rows


def test_build_rows_emits_content_and_cyclic_surfaces() -> None:
    mcq_rows = [
        {
            "id": "q1",
            "question": "Which option is correct?",
            "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "answer": "B",
        }
    ]
    diagnosis = {
        "summary": {
            "benchmark": "toy",
            "stable_rescue_examples": [
                {
                    "id": "q1",
                    "candidate_content_prediction": "A",
                    "candidate_cyclic_prediction": "B",
                    "candidate_content_answer_rank": 2,
                }
            ],
        }
    }

    rows, summary = build_rows(mcq_rows, diagnosis, cyclic_rows_per_item=2, seed=123)

    assert summary["selected_ids"] == 1
    assert summary["output_rows"] == 3
    assert summary["routing_type_counts"] == {
        "surface_content_align": 1,
        "surface_cyclic_preserve": 2,
    }
    content_rows = [row for row in rows if row["routing_type"] == "surface_content_align"]
    cyclic_rows = [row for row in rows if row["routing_type"] == "surface_cyclic_preserve"]
    assert len(content_rows) == 1
    assert content_rows[0]["prompt"] == "Which option is correct?\nAnswer:"
    assert content_rows[0]["completion"] == " beta"
    assert all("\nA. " in row["prompt"] for row in cyclic_rows)
    assert {row["target_loop_count"] for row in rows} == {1}


def test_build_rows_can_include_unrescued_losses() -> None:
    mcq_rows = [
        {
            "id": "q1",
            "question": "Question 1?",
            "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "answer": "B",
        },
        {
            "id": "q2",
            "question": "Question 2?",
            "choices": {"A": "red", "B": "blue", "C": "green", "D": "gold"},
            "answer": "C",
        },
    ]
    diagnosis = {
        "summary": {
            "benchmark": "toy",
            "stable_rescue_examples": [{"id": "q1"}],
            "unrescued_loss_examples": [{"id": "q2"}],
        }
    }

    rows, summary = build_rows(
        mcq_rows,
        diagnosis,
        include_unrescued=True,
        cyclic_rows_per_item=1,
        content_repeat=2,
        cyclic_repeat=1,
    )

    assert summary["selected_ids"] == 2
    assert summary["routing_type_counts"] == {
        "surface_content_align": 4,
        "surface_cyclic_preserve": 2,
    }
    assert {row["arc_id"] for row in rows} == {"q1", "q2"}


def test_build_rows_reports_missing_mcq_rows() -> None:
    diagnosis = {"summary": {"stable_rescue_examples": [{"id": "missing"}]}}

    rows, summary = build_rows([], diagnosis)

    assert rows == []
    assert summary["skipped"] == {"missing_mcq": 1}
