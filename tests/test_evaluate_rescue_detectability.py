from __future__ import annotations

import eval.analyze_depth_sweep as depth
import eval.evaluate_rescue_detectability as detect
import eval.evaluate_rescue_selector_transfer as transfer
from tests.test_evaluate_rescue_selector_transfer import write_sweep


def test_best_detectability_row_prefers_cleared_null_margin() -> None:
    rows = [
        {
            "available": True,
            "shrinkage": 0.1,
            "clears_null_p95": False,
            "observed_minus_null_p95": 0.2,
            "observed_alignment": 0.8,
        },
        {
            "available": True,
            "shrinkage": 1.0,
            "clears_null_p95": True,
            "observed_minus_null_p95": 0.01,
            "observed_alignment": 0.4,
        },
    ]

    assert detect.best_detectability_row(rows)["shrinkage"] == 1.0


def test_analyze_detectability_writes_selector_safe_gate_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(depth, "ROOT", tmp_path)
    monkeypatch.setattr(transfer, "ROOT", tmp_path)
    monkeypatch.setattr(detect, "ROOT", tmp_path)
    sweep = write_sweep(tmp_path, "discovery")

    payload = detect.analyze_detectability(
        sweep_summary=sweep,
        benchmark="toy",
        score_target="content_question_only",
        aggregate="mean",
        shrinkages=[1.0],
        repeats=4,
        permutations=4,
        sample_fraction=1.0,
        seed=1,
        run_id="toy_detectability",
    )

    assert payload["kind"] == "stage5_rescue_detectability_gate"
    assert payload["category_counts"]["rescuable"] == 1
    assert payload["detectability_by_shrinkage"][0]["available"] is False
    assert payload["detectability_by_shrinkage"][0]["reason"] == "insufficient_positive_or_negative_examples"
    assert payload["supervised_probe_discovery"]
