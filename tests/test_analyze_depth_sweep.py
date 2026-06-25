import json
from pathlib import Path

from eval.analyze_depth_sweep import analyze


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(row_id, prediction, answer, hit, scores):
    return {
        "id": row_id,
        "prediction": prediction,
        "answer": answer,
        "hit": hit,
        "scores": scores,
    }


def write_loop(tmp_path: Path, run_id: str, loop: int) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / run_id
    base_path = run_dir / "toy_base_label.jsonl"
    rec_path = run_dir / "toy_recurrent_label.jsonl"
    base_content_path = run_dir / "toy_base_content_question_only.jsonl"
    rec_content_path = run_dir / "toy_recurrent_content_question_only.jsonl"
    base_rows = [
        row("a", "A", "A", True, {"A": 2.0, "B": 0.0}),
        row("b", "B", "A", False, {"A": 0.0, "B": 1.0}),
        row("c", "A", "A", True, {"A": 1.0, "B": 0.9}),
    ]
    recurrent_by_loop = {
        1: [
            row("a", "A", "A", True, {"A": 2.1, "B": 0.0}),
            row("b", "B", "A", False, {"A": 0.0, "B": 1.1}),
            row("c", "A", "A", True, {"A": 1.2, "B": 0.4}),
        ],
        2: [
            row("a", "B", "A", False, {"A": 0.0, "B": 1.0}),
            row("b", "A", "A", True, {"A": 1.0, "B": 0.0}),
            row("c", "A", "A", True, {"A": 1.0, "B": 0.8}),
        ],
    }
    write_jsonl(base_path, base_rows)
    write_jsonl(rec_path, recurrent_by_loop[loop])
    write_jsonl(base_content_path, base_rows)
    write_jsonl(rec_content_path, recurrent_by_loop[loop])
    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "benchmarks": ["toy"],
            "results": [
                {
                    "benchmark": "toy",
                    "arm": "base",
                    "score_target": "label",
                    "output_jsonl": str(base_path),
                },
                {
                    "benchmark": "toy",
                    "arm": "recurrent",
                    "score_target": "label",
                    "output_jsonl": str(rec_path),
                },
                {
                    "benchmark": "toy",
                    "arm": "base",
                    "score_target": "content_question_only",
                    "output_jsonl": str(base_content_path),
                },
                {
                    "benchmark": "toy",
                    "arm": "recurrent",
                    "score_target": "content_question_only",
                    "output_jsonl": str(rec_content_path),
                },
            ],
        },
    )


def test_analyze_depth_sweep_oracle_and_router(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import eval.analyze_depth_sweep as mod

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    run_ids = ["sweep_loop1", "sweep_loop2"]
    write_loop(tmp_path, run_ids[0], 1)
    write_loop(tmp_path, run_ids[1], 2)
    sweep = tmp_path / "outputs" / "stage5" / "sweep" / "summary.json"
    write_json(sweep, {"run_id": "sweep", "loop_run_ids": run_ids})

    payload = analyze(sweep)
    toy = payload["benchmarks"]["toy"]
    interactions = toy["depth_interactions"]

    assert interactions["loop1_correct"] == 2
    assert interactions["any_recurrent_correct"] == 3
    assert interactions["deeper_unique_over_loop1"] == 1
    assert interactions["loop1_harmed_by_any_deeper"] == 1

    loop2 = toy["loop_summaries"][1]
    assert loop2["loop"] == 2
    assert loop2["loop_correct"] == 2
    assert loop2["wins_vs_base"] == 1
    assert loop2["losses_vs_base"] == 1

    assert toy["threshold_router_top10"][0]["total"] == 3


def test_analyze_depth_sweep_supports_content_score_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import eval.analyze_depth_sweep as mod

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    run_ids = ["sweep_loop1", "sweep_loop2"]
    write_loop(tmp_path, run_ids[0], 1)
    write_loop(tmp_path, run_ids[1], 2)
    sweep = tmp_path / "outputs" / "stage5" / "sweep" / "summary.json"
    write_json(sweep, {"run_id": "sweep", "loop_run_ids": run_ids})

    payload = analyze(sweep, score_target="content_question_only", aggregate="mean")

    assert payload["score_target"] == "content_question_only"
    assert payload["aggregate"] == "mean"
    assert payload["benchmarks"]["toy"]["depth_interactions"]["deeper_unique_over_loop1"] == 1
