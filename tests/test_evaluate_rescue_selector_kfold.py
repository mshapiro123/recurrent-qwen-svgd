from __future__ import annotations

import eval.analyze_depth_sweep as depth
import eval.evaluate_rescue_selector_kfold as kfold
import eval.evaluate_rescue_selector_transfer as transfer
from tests.test_evaluate_rescue_selector_transfer import write_sweep


def test_kfold_rescue_selector_reports_conservative_transfer(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(depth, "ROOT", tmp_path)
    monkeypatch.setattr(transfer, "ROOT", tmp_path)
    monkeypatch.setattr(kfold, "ROOT", tmp_path)
    sweep = write_sweep(tmp_path, "pooled")

    payload = kfold.analyze_kfold(
        sweep_summary=sweep,
        benchmarks=["toy"],
        score_target="content_question_only",
        aggregate="mean",
        folds=2,
        seed=1,
        shrinkages=[1.0],
        primary_shrinkage=1.0,
        run_id="toy_kfold",
    )

    assert payload["kind"] == "stage5_rescue_selector_kfold"
    assert payload["pooled"]["total"] == 4
    assert payload["pooled"]["category_counts"]["rescuable"] == 1
    assert payload["aggregate_policy_results"]
    assert payload["primary_conservative_result"]["policy_label"] in {"zero_harm", "harm_budget_1"}
    assert payload["primary_conservative_result"]["shrinkage"] == 1.0


def test_stable_fold_is_repeatable() -> None:
    example = {"benchmark": "toy", "id": "abc"}

    assert kfold.stable_fold(example, folds=5, seed=17) == kfold.stable_fold(example, folds=5, seed=17)


def test_unavailable_policy_defaults_to_loop1() -> None:
    row = kfold.zero_result(10, 4, 6, policy_label="zero_harm", shrinkage=1.0)

    assert row["correct"] == 4
    assert row["routed_deep"] == 0
    assert row["rescue_captured"] == 0
    assert row["harm_triggered"] == 0
