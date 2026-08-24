from analysis.analyze_paper2_bicameral_w1_diagnostics import paired_summary


def _row(item: str, delta: float, *, arm: str) -> dict:
    return {
        "arm": arm,
        "baseline_margin": 0.5,
        "battery": "gsm8k",
        "evaluator": "EV-LADDER-1",
        "item_id": item,
        "margin_delta": delta,
        "schedule": "sequential_shared_middle_v1",
        "seed": 0,
    }


def test_paired_summary_uses_same_rows_and_reports_advantage() -> None:
    own = [_row("a", 2.0, arm="l0c"), _row("b", 1.0, arm="l0c")]
    control = [_row("b", 0.25, arm="l5_c"), _row("a", 0.5, arm="l5_c")]
    result = paired_summary(own, control, bootstrap_seed=3)
    assert result["rows"] == 2
    assert result["paired_advantage"]["mean"] == 1.125
    assert result["paired_advantage_positive_fraction"] == 1.0
    assert result["by_battery"]["gsm8k"]["rows"] == 2
