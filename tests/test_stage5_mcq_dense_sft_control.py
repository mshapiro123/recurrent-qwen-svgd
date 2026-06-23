from __future__ import annotations

import json

from colab import run_stage5_mcq_dense_sft_control as module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_source_positive_sft_path_prefers_dataset_source(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "dataset": {
            "source_positive_sft": "data/curriculum/run/positive_sft.jsonl",
        },
        "gate": {
            "artifacts": {
                "positive_sft": "/wrong/positive_sft.jsonl",
            }
        },
    }

    assert module.source_positive_sft_path(payload) == tmp_path / "data" / "curriculum" / "run" / "positive_sft.jsonl"


def test_source_positive_sft_path_falls_back_to_gate_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "gate": {
            "artifacts": {
                "positive_sft": "data/curriculum/trace/positive_sft.jsonl",
            }
        },
    }

    assert module.source_positive_sft_path(payload) == tmp_path / "data" / "curriculum" / "trace" / "positive_sft.jsonl"


def test_paired_dense_vs_base_reports_wins_losses_and_ties(tmp_path) -> None:
    base = tmp_path / "base.jsonl"
    dense = tmp_path / "dense.jsonl"
    write_jsonl(
        base,
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": False},
            {"id": "c", "aggregate": "mean", "hit": True},
            {"id": "d", "aggregate": "mean", "hit": False},
        ],
    )
    write_jsonl(
        dense,
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": True},
            {"id": "c", "aggregate": "mean", "hit": False},
            {"id": "d", "aggregate": "mean", "hit": False},
        ],
    )

    paired = module.paired_dense_vs_base(base, dense)["mean"]

    assert paired["paired_examples"] == 4
    assert paired["base_correct"] == 2
    assert paired["dense_correct"] == 2
    assert paired["correct_delta_dense_vs_base"] == 0
    assert paired["wins"] == 1
    assert paired["losses"] == 1
    assert paired["ties"] == 2


def test_write_summary_updates_current_pointer(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "dense"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    payload = {
        "run_id": "dense",
        "source_summary": "outputs/source/summary.json",
        "dataset": {"rows": 3},
        "dense_checkpoint": "outputs/dense/dense_lora.pt",
        "config": {"dense_lora_layer_range": "6,18"},
        "paired_comparisons": {
            "arc_easy": {
                "content_question_only": {
                    "mean": {
                        "dense_correct": 2,
                        "base_correct": 1,
                        "paired_examples": 3,
                        "correct_delta_dense_vs_base": 1,
                        "wins": 1,
                        "losses": 0,
                        "ties": 2,
                        "sign_test_p_value": 1.0,
                    }
                }
            }
        },
    }

    module.write_summary(payload)

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip() == (
        "outputs/stage5/dense/summary.json"
    )


def test_write_summary_points_front_of_queue_to_recipe_assessment(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "dense"
    assessment = run_dir / "mcq_recipe_control_assessment.json"
    assessment.parent.mkdir(parents=True)
    assessment.write_text(json.dumps({"kind": "stage5_mcq_recipe_control_assessment"}), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    payload = {
        "run_id": "dense",
        "source_summary": "outputs/source/summary.json",
        "dataset": {"rows": 3},
        "dense_checkpoint": "outputs/dense/dense_lora.pt",
        "config": {"dense_lora_layer_range": "6,18"},
        "paired_comparisons": {},
        "recipe_control_assessment": {
            "ran": True,
            "summary_json": "outputs/stage5/dense/mcq_recipe_control_assessment.json",
        },
    }

    module.write_summary(payload)

    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip() == (
        "outputs/stage5/dense/mcq_recipe_control_assessment.json"
    )
