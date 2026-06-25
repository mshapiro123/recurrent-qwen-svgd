from __future__ import annotations

import json
from pathlib import Path

from eval.evaluate_depth_router_transfer import transfer_summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(row_id: str, *, answer: str, prediction: str, scores: dict[str, float]) -> dict:
    return {
        "id": row_id,
        "answer": answer,
        "prediction": prediction,
        "hit": prediction == answer,
        "scores": scores,
        "aggregate": "mean",
    }


def write_loop_run(root: Path, run_id: str, benchmark: str, loop: int, rows: list[dict], base_rows: list[dict]) -> None:
    run_dir = root / run_id
    base_jsonl = run_dir / f"{benchmark}_base_content_question_only.jsonl"
    recurrent_jsonl = run_dir / f"{benchmark}_recurrent_content_question_only.jsonl"
    write_jsonl(base_jsonl, base_rows)
    write_jsonl(recurrent_jsonl, rows)
    write_json(
        run_dir / "summary.json",
        {
            "kind": "stage5_benchmark_suite",
            "run_id": run_id,
            "benchmarks": [benchmark],
            "recurrent_forced_loop_count": loop,
            "results": [
                {
                    "benchmark": benchmark,
                    "arm": "base",
                    "score_target": "content_question_only",
                    "output_jsonl": str(base_jsonl),
                },
                {
                    "benchmark": benchmark,
                    "arm": "recurrent",
                    "score_target": "content_question_only",
                    "output_jsonl": str(recurrent_jsonl),
                },
            ],
        },
    )


def write_sweep(tmp_path: Path, name: str, benchmark: str, loop1_rows: list[dict], loop2_rows: list[dict], base_rows: list[dict]) -> Path:
    root = tmp_path / "outputs" / "stage5"
    run1 = f"{name}_loop1"
    run2 = f"{name}_loop2"
    write_loop_run(root, run1, benchmark, 1, loop1_rows, base_rows)
    write_loop_run(root, run2, benchmark, 2, loop2_rows, base_rows)
    summary = root / name / "summary.json"
    write_json(
        summary,
        {
            "kind": "stage5_forced_depth_diagnostic",
            "run_id": name,
            "loop_run_ids": [run1, run2],
            "loops": [1, 2],
        },
    )
    return summary


def test_depth_router_transfer_reports_heldout_gap_capture(tmp_path) -> None:
    benchmark = "toy_arc"
    base_rows = [
        row("a", answer="A", prediction="B", scores={"A": 0.0, "B": 0.1}),
        row("b", answer="A", prediction="B", scores={"A": 0.0, "B": 0.1}),
        row("c", answer="A", prediction="A", scores={"A": 1.0, "B": 0.9}),
        row("d", answer="A", prediction="A", scores={"A": 1.0, "B": 0.9}),
    ]
    loop1_rows = [
        row("a", answer="A", prediction="B", scores={"A": 0.0, "B": 0.1}),
        row("b", answer="A", prediction="B", scores={"A": 0.0, "B": 0.1}),
        row("c", answer="A", prediction="A", scores={"A": 1.0, "B": 0.9}),
        row("d", answer="A", prediction="A", scores={"A": 1.0, "B": 0.9}),
    ]
    loop2_rows = [
        row("a", answer="A", prediction="A", scores={"A": 2.0, "B": 0.0}),
        row("b", answer="A", prediction="A", scores={"A": 2.0, "B": 0.0}),
        row("c", answer="A", prediction="A", scores={"A": 2.0, "B": 0.0}),
        row("d", answer="A", prediction="A", scores={"A": 2.0, "B": 0.0}),
    ]
    discovery = write_sweep(tmp_path, "discovery", benchmark, loop1_rows, loop2_rows, base_rows)
    heldout = write_sweep(tmp_path, "heldout", benchmark, loop1_rows, loop2_rows, base_rows)

    payload = transfer_summary(
        discovery_sweep=discovery,
        heldout_sweep=heldout,
        train_benchmark=benchmark,
        score_target="content_question_only",
        aggregate="mean",
        min_oracle_gap_capture=0.2,
    )

    result = payload["heldout"][benchmark]
    assert payload["gate_status"] == "router_transfer_passed"
    assert result["selected_selector"]["delta_vs_loop1"] == 2
    assert result["oracle_gap_capture"] == 1.0
