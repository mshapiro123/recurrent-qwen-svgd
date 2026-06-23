from eval.analyze_mcq_order_sensitivity import analyze, summarize


def test_summarize_flags_order_sensitive_content_loss() -> None:
    rows = [
        {
            "change": "loss",
            "candidate_order_sensitive": True,
            "cyclic_rescues_content_loss": True,
            "base_content_hit": True,
        },
        {
            "change": "loss",
            "candidate_order_sensitive": True,
            "cyclic_rescues_content_loss": False,
            "base_content_hit": True,
        },
        {
            "change": "tie_correct",
            "candidate_order_sensitive": False,
            "cyclic_rescues_content_loss": False,
            "base_content_hit": True,
        },
        {
            "change": "win",
            "candidate_order_sensitive": False,
            "cyclic_rescues_content_loss": False,
            "base_content_hit": False,
        },
    ]

    summary = summarize(rows, benchmark="toy")

    assert summary["content_delta"] == -1
    assert summary["content_losses_order_sensitive"] == 2
    assert summary["content_losses_order_sensitive_fraction"] == 1.0
    assert summary["content_losses_rescued_by_cyclic"] == 1
    assert summary["likely_order_sensitivity_issue"] is True
    assert summary["recommendation"] == "prioritize_conditional_invariance_repair"


def test_summarize_prefers_distillation_when_losses_are_stable() -> None:
    rows = [
        {
            "change": "loss",
            "candidate_order_sensitive": False,
            "cyclic_rescues_content_loss": False,
            "base_content_hit": True,
        },
        {
            "change": "tie_correct",
            "candidate_order_sensitive": True,
            "cyclic_rescues_content_loss": False,
            "base_content_hit": True,
        },
        {
            "change": "tie_correct",
            "candidate_order_sensitive": False,
            "cyclic_rescues_content_loss": False,
            "base_content_hit": True,
        },
    ]

    summary = summarize(rows, benchmark="toy")

    assert summary["content_losses_order_sensitive"] == 0
    assert summary["content_losses_rescued_by_cyclic"] == 0
    assert summary["likely_order_sensitivity_issue"] is False
    assert summary["recommendation"] == "prioritize_direct_distillation_or_data_repair"


def test_summarize_separates_cyclic_rescue_from_order_sensitivity() -> None:
    rows = [
        {
            "change": "loss",
            "candidate_order_sensitive": False,
            "cyclic_rescues_content_loss": True,
            "base_content_hit": True,
        },
        {
            "change": "loss",
            "candidate_order_sensitive": False,
            "cyclic_rescues_content_loss": True,
            "base_content_hit": True,
        },
        {
            "change": "tie_correct",
            "candidate_order_sensitive": False,
            "cyclic_rescues_content_loss": False,
            "base_content_hit": True,
        },
    ]

    summary = summarize(rows, benchmark="toy")

    assert summary["likely_order_sensitivity_issue"] is False
    assert summary["likely_content_route_specific_issue"] is True
    assert (
        summary["recommendation"]
        == "diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation"
    )


def test_analyze_pairs_content_and_cyclic_rows(tmp_path) -> None:
    base_content = tmp_path / "base_content.jsonl"
    cand_content = tmp_path / "cand_content.jsonl"
    cand_cyclic = tmp_path / "cand_cyclic.jsonl"
    base_cyclic = tmp_path / "base_cyclic.jsonl"

    base_content.write_text(
        '{"id":"a","prediction":"A","answer":"A","hit":true}\n'
        '{"id":"b","prediction":"B","answer":"B","hit":true}\n',
        encoding="utf-8",
    )
    cand_content.write_text(
        '{"id":"a","prediction":"B","answer":"A","hit":false}\n'
        '{"id":"b","prediction":"B","answer":"B","hit":true}\n',
        encoding="utf-8",
    )
    cand_cyclic.write_text(
        '{"id":"a","prediction":"A","answer":"A","hit":true,"permutation_prediction_counts":{"A":2,"B":2}}\n'
        '{"id":"b","prediction":"B","answer":"B","hit":true,"permutation_prediction_counts":{"B":4}}\n',
        encoding="utf-8",
    )
    base_cyclic.write_text(
        '{"id":"a","prediction":"A","answer":"A","hit":true,"permutation_prediction_counts":{"A":4}}\n'
        '{"id":"b","prediction":"B","answer":"B","hit":true,"permutation_prediction_counts":{"B":4}}\n',
        encoding="utf-8",
    )

    payload = analyze(
        benchmark="toy",
        base_content_path=base_content,
        candidate_content_path=cand_content,
        candidate_cyclic_path=cand_cyclic,
        base_cyclic_path=base_cyclic,
    )

    assert payload["summary"]["content_losses"] == 1
    assert payload["summary"]["content_losses_order_sensitive"] == 1
    assert payload["summary"]["content_losses_rescued_by_cyclic"] == 1
    assert payload["rows"][0]["candidate_order_sensitive"] is True
