from __future__ import annotations

import json
from pathlib import Path

import eval.analyze_depth_sweep as depth
import eval.evaluate_rescue_selector_kfold as kfold
import eval.evaluate_rescue_selector_transfer as transfer


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(row_id: str, prediction: str, *, answer: str = "A", hit: bool, margin: float) -> dict:
    return {
        "id": row_id,
        "prediction": prediction,
        "answer": answer,
        "hit": hit,
        "aggregate": "mean",
        "scores": {"A": margin, "B": 0.0} if prediction == "A" else {"A": 0.0, "B": margin},
        "loop_diagnostics": {"mean_expected_loops": 1.5, "mean_halt_entropy": 0.5},
    }


def write_loop(root: Path, run_id: str, recurrent_rows: list[dict]) -> None:
    run_dir = root / "outputs" / "stage5" / run_id
    base_path = run_dir / "toy_base_content_question_only.jsonl"
    rec_path = run_dir / "toy_recurrent_content_question_only.jsonl"
    base_rows = [
        row("rescue", "B", hit=False, margin=0.1),
        row("harm", "A", hit=True, margin=3.0),
        row("stable_correct", "A", hit=True, margin=2.0),
        row("stable_wrong", "B", hit=False, margin=1.0),
    ]
    write_jsonl(base_path, base_rows)
    write_jsonl(rec_path, recurrent_rows)
    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "benchmarks": ["toy"],
            "results": [
                {
                    "benchmark": "toy",
                    "arm": "base",
                    "score_target": "content_question_only",
                    "output_jsonl": str(base_path),
                },
                {
                    "benchmark": "toy",
                    "arm": "recurrent",
                    "score_target": "content_question_only",
                    "output_jsonl": str(rec_path),
                },
            ],
        },
    )


def write_sweep(root: Path, name: str) -> Path:
    loop1_rows = [
        row("rescue", "B", hit=False, margin=0.2),
        row("harm", "A", hit=True, margin=3.0),
        row("stable_correct", "A", hit=True, margin=2.0),
        row("stable_wrong", "B", hit=False, margin=1.0),
    ]
    loop2_rows = [
        row("rescue", "A", hit=True, margin=1.5),
        row("harm", "B", hit=False, margin=0.4),
        row("stable_correct", "A", hit=True, margin=2.1),
        row("stable_wrong", "B", hit=False, margin=1.1),
    ]
    write_loop(root, f"{name}_loop1", loop1_rows)
    write_loop(root, f"{name}_loop2", loop2_rows)
    sweep = root / "outputs" / "stage5" / name / "summary.json"
    write_json(sweep, {"run_id": name, "loop_run_ids": [f"{name}_loop1", f"{name}_loop2"]})
    return sweep


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
