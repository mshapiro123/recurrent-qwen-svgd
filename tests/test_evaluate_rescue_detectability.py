from __future__ import annotations

import json
from pathlib import Path

import eval.analyze_depth_sweep as depth
import eval.evaluate_rescue_detectability as detect
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
