from eval.eval_branching_relations import summarize_rows


def test_branching_summary_keeps_harmed_depth_visible() -> None:
    rows = [
        {
            "depth": depth,
            "valid": index < (100 if depth < 4 else 60),
            "reachable_set_stratum": "2" if depth == 1 else "3-4",
        }
        for depth in range(1, 5)
        for index in range(128)
    ]
    summary = summarize_rows(rows)
    assert summary["gate"]["pooled_accuracy"] >= 0.70
    assert summary["gate"]["by_depth"]["4"]["accuracy"] < 0.55
    assert summary["gate"]["passed"] is False
