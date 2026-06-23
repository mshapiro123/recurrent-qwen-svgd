from __future__ import annotations

import json

from colab import assess_stage5_mcq_recipe_control as module


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def rows(hits: list[bool], *, aggregate: str = "mean") -> list[dict[str, object]]:
    return [
        {
            "id": f"item_{idx}",
            "aggregate": aggregate,
            "hit": hit,
            "prediction": "A" if hit else "B",
            "answer": "A",
        }
        for idx, hit in enumerate(hits)
    ]


def make_dense_and_recurrent(tmp_path, *, recurrent_summary_wrapper: bool = False):
    dense_dir = tmp_path / "outputs" / "stage5" / "dense"
    recurrent_dir = tmp_path / "outputs" / "stage5" / "recurrent_benchmark"

    specs = {
        ("arc_challenge", "cyclic_label_aggregated"): {
            "aggregate": "permutation_mean",
            "dense": [False, False, True, False],
            "recurrent": [True, False, True, True],
        },
        ("arc_challenge", "content_question_only"): {
            "aggregate": "mean",
            "dense": [False, True, True, False],
            "recurrent": [True, True, True, False],
        },
        ("arc_easy", "cyclic_label_aggregated"): {
            "aggregate": "permutation_mean",
            "dense": [True, True, False, False],
            "recurrent": [True, True, True, False],
        },
        ("arc_easy", "content_question_only"): {
            "aggregate": "mean",
            "dense": [True, True, False, False],
            "recurrent": [True, False, False, False],
        },
    }

    artifacts: dict[str, dict[str, object]] = {}
    for (benchmark, score_target), spec in specs.items():
        dense_path = dense_dir / f"{benchmark}_dense_{score_target}.jsonl"
        recurrent_path = recurrent_dir / f"{benchmark}_recurrent_{score_target}.jsonl"
        write_jsonl(dense_path, rows(spec["dense"], aggregate=spec["aggregate"]))
        write_jsonl(recurrent_path, rows(spec["recurrent"], aggregate=spec["aggregate"]))
        artifacts.setdefault(benchmark, {"data_jsonl": f"data/{benchmark}.jsonl"})
        artifacts[benchmark][score_target] = {
            "base": f"outputs/stage5/dense/{benchmark}_base_{score_target}.jsonl",
            "dense": module.path_for_cli(dense_path),
        }

    dense_summary = {
        "run_id": "dense",
        "kind": "stage5_dense_mcq_trace_sft_control",
        "source_summary": "outputs/source/summary.json",
        "config": {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "benchmarks": ["arc_easy", "arc_challenge"],
            "score_targets": ["content_question_only", "cyclic_label_aggregated"],
            "aggregates": ["mean"],
        },
        "artifacts": artifacts,
    }
    recurrent_summary = {
        "run_id": "recurrent",
        "kind": "stage5_benchmark_suite",
        "source_summary": "outputs/source/summary.json",
        "benchmarks": ["arc_easy", "arc_challenge"],
        "score_targets": ["content_question_only", "cyclic_label_aggregated"],
        "aggregates": ["mean"],
        "paired_comparisons": {
            "arc_easy": {
                "content_question_only": {
                    "mean": {"correct_delta_recurrent_vs_base": -8},
                },
                "cyclic_label_aggregated": {
                    "permutation_mean": {"correct_delta_recurrent_vs_base": 2},
                },
            }
        },
    }

    dense_summary_path = dense_dir / "summary.json"
    recurrent_summary_path = recurrent_dir / "summary.json"
    write_json(dense_summary_path, dense_summary)
    write_json(recurrent_summary_path, recurrent_summary)

    if recurrent_summary_wrapper:
        wrapper = tmp_path / "outputs" / "stage5" / "wrapper" / "summary.json"
        write_json(wrapper, {"kind": "assessment", "benchmark_summary": module.path_for_cli(recurrent_summary_path)})
        recurrent_summary_path = wrapper

    return dense_summary_path, recurrent_summary_path


def test_mcq_recipe_control_reports_hard_tail_lift(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    dense_summary, recurrent_summary = make_dense_and_recurrent(tmp_path)

    assessment = module.assess_mcq_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_primary_examples=4,
    )

    assert assessment["status"] == "hard_tail_lift_vs_dense"
    assert assessment["passed"] is True
    primary = assessment["decision_evidence"]["primary"]
    assert primary["dense_correct"] == 1
    assert primary["recurrent_correct"] == 3
    assert primary["correct_delta_recurrent_vs_dense"] == 2
    assert primary["wins"] == 2
    assert assessment["surface_notes_recurrent_vs_base"]["arc_easy"]["pattern"] == "content_down_cyclic_up"


def test_mcq_recipe_control_resolves_wrapper_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    dense_summary, recurrent_summary = make_dense_and_recurrent(tmp_path, recurrent_summary_wrapper=True)

    assessment = module.assess_mcq_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_primary_examples=4,
    )

    assert assessment["recurrent_summary"] == "outputs/stage5/recurrent_benchmark/summary.json"
    assert assessment["status"] == "hard_tail_lift_vs_dense"


def test_mcq_recipe_control_flags_metadata_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    dense_summary, recurrent_summary = make_dense_and_recurrent(tmp_path)
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["score_targets"] = ["content_question_only"]
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = module.assess_mcq_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_primary_examples=4,
    )

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["score_targets"] == {
        "dense": ["content_question_only", "cyclic_label_aggregated"],
        "recurrent": ["content_question_only"],
    }


def test_mcq_recipe_control_writes_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    dense_summary, recurrent_summary = make_dense_and_recurrent(tmp_path)
    output_json = tmp_path / "out" / "summary.json"
    output_md = tmp_path / "out" / "summary.md"

    payload = module.assess_mcq_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_primary_examples=4,
    )
    module.write_report(payload, output_json=output_json, output_md=output_md)

    assert json.loads(output_json.read_text(encoding="utf-8"))["kind"] == "stage5_mcq_recipe_control_assessment"
    assert "ARC-Challenge cyclic" in output_md.read_text(encoding="utf-8")
