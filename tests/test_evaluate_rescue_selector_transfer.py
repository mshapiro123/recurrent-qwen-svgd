from __future__ import annotations

import json
from pathlib import Path

import eval.analyze_depth_sweep as depth
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


def write_loop(root: Path, run_id: str, loop: int, recurrent_rows: list[dict]) -> None:
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
    write_loop(root, f"{name}_loop1", 1, loop1_rows)
    write_loop(root, f"{name}_loop2", 2, loop2_rows)
    sweep = root / "outputs" / "stage5" / name / "summary.json"
    write_json(sweep, {"run_id": name, "loop_run_ids": [f"{name}_loop1", f"{name}_loop2"]})
    return sweep


def test_transfer_curve_applies_discovery_thresholds_to_heldout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(depth, "ROOT", tmp_path)
    monkeypatch.setattr(transfer, "ROOT", tmp_path)

    discovery = write_sweep(tmp_path, "discovery")
    heldout = write_sweep(tmp_path, "heldout")

    payload = transfer.analyze_transfer(
        discovery_sweep_summary=discovery,
        heldout_sweep_summary=heldout,
        discovery_benchmark="toy",
        score_target="content_question_only",
        aggregate="mean",
        include_manual_base_margin_thresholds=[0.5],
    )

    assert payload["discovery"]["category_counts"]["rescuable"] == 1
    assert payload["discovery"]["selected_policies"]
    toy = payload["heldout"]["toy"]
    assert toy["category_counts"]["harmable"] == 1
    assert toy["transferred_curve_summary"]["points"] > 0
    assert toy["transferred_curve_summary"]["zero_harm"]["harm_triggered"] == 0
    assert any(row["policy_label"].startswith("manual_base_margin_low") for row in toy["policy_results"])


def test_best_result_prefers_less_harm_on_tie() -> None:
    rows = [
        {"delta_vs_loop1": 1, "rescue_captured": 2, "harm_triggered": 1, "routed_deep": 4},
        {"delta_vs_loop1": 1, "rescue_captured": 2, "harm_triggered": 0, "routed_deep": 4},
    ]

    result = transfer.best_result(rows, label="zero-ish")

    assert result["harm_triggered"] == 0
