import json

from eval.analyze_mcq_surface_mismatch import analyze, summarize


def row(
    row_id: str,
    *,
    answer: str,
    prediction: str,
    hit: bool,
    scores: dict[str, float] | None = None,
    counts: dict[str, int] | None = None,
) -> dict:
    payload = {
        "id": row_id,
        "answer": answer,
        "prediction": prediction,
        "hit": hit,
        "scores": scores or {"A": -2.0, "B": -1.0, "C": -3.0, "D": -4.0},
    }
    if counts is not None:
        payload["permutation_prediction_counts"] = counts
    return payload


def test_summarize_prefers_surface_alignment_for_stable_cyclic_rescues() -> None:
    rows = [
        {
            "base_content_hit": True,
            "candidate_content_hit": False,
            "candidate_cyclic_hit": True,
            "content_loss": True,
            "content_win": False,
            "cyclic_rescues_content_loss": True,
            "stable_cyclic_rescue": True,
            "candidate_order_sensitive": False,
            "content_cyclic_disagree": True,
            "candidate_content_prediction": "A",
            "candidate_cyclic_prediction": "B",
            "candidate_content_answer_rank": 2,
        },
        {
            "base_content_hit": True,
            "candidate_content_hit": False,
            "candidate_cyclic_hit": True,
            "content_loss": True,
            "content_win": False,
            "cyclic_rescues_content_loss": True,
            "stable_cyclic_rescue": True,
            "candidate_order_sensitive": False,
            "content_cyclic_disagree": True,
            "candidate_content_prediction": "C",
            "candidate_cyclic_prediction": "B",
            "candidate_content_answer_rank": 2,
        },
    ]
    summary = summarize(rows, benchmark="toy")
    assert summary["recommendation"] == "prioritize_content_cyclic_surface_alignment"
    assert summary["content_losses_stably_rescued_by_cyclic_fraction"] == 1.0


def test_summarize_prefers_distillation_for_unrescued_losses() -> None:
    rows = [
        {
            "base_content_hit": True,
            "candidate_content_hit": False,
            "candidate_cyclic_hit": False,
            "content_loss": True,
            "content_win": False,
            "cyclic_rescues_content_loss": False,
            "stable_cyclic_rescue": False,
            "candidate_order_sensitive": False,
            "content_cyclic_disagree": False,
            "candidate_content_prediction": "A",
            "candidate_cyclic_prediction": "A",
            "candidate_content_answer_rank": 4,
        },
        {
            "base_content_hit": True,
            "candidate_content_hit": False,
            "candidate_cyclic_hit": False,
            "content_loss": True,
            "content_win": False,
            "cyclic_rescues_content_loss": False,
            "stable_cyclic_rescue": False,
            "candidate_order_sensitive": False,
            "content_cyclic_disagree": False,
            "candidate_content_prediction": "C",
            "candidate_cyclic_prediction": "C",
            "candidate_content_answer_rank": 3,
        },
    ]
    summary = summarize(rows, benchmark="toy")
    assert summary["recommendation"] == "prioritize_direct_distillation_or_data_repair"
    assert summary["content_losses_unrescued_by_cyclic_fraction"] == 1.0


def test_analyze_reads_surfaces_and_ranks_answer(tmp_path) -> None:
    base_content = tmp_path / "base_content.jsonl"
    candidate_content = tmp_path / "candidate_content.jsonl"
    candidate_cyclic = tmp_path / "candidate_cyclic.jsonl"
    base_cyclic = tmp_path / "base_cyclic.jsonl"

    base_content.write_text(
        json.dumps(
            row(
                "x",
                answer="B",
                prediction="B",
                hit=True,
                scores={"A": -2.0, "B": -0.1, "C": -3.0, "D": -4.0},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_content.write_text(
        json.dumps(
            row(
                "x",
                answer="B",
                prediction="A",
                hit=False,
                scores={"A": -0.2, "B": -0.5, "C": -3.0, "D": -4.0},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_cyclic.write_text(
        json.dumps(
            row(
                "x",
                answer="B",
                prediction="B",
                hit=True,
                scores={"A": -2.0, "B": -0.2, "C": -3.0, "D": -4.0},
                counts={"B": 4},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    base_cyclic.write_text(
        json.dumps(
            row(
                "x",
                answer="B",
                prediction="B",
                hit=True,
                scores={"A": -2.0, "B": -0.2, "C": -3.0, "D": -4.0},
                counts={"B": 4},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    payload = analyze(
        benchmark="toy",
        base_content_path=base_content,
        candidate_content_path=candidate_content,
        candidate_cyclic_path=candidate_cyclic,
        base_cyclic_path=base_cyclic,
    )

    assert payload["summary"]["recommendation"] == "prioritize_content_cyclic_surface_alignment"
    assert payload["rows"][0]["candidate_content_answer_rank"] == 2
    assert payload["rows"][0]["candidate_content_answer_margin"] == -0.3
