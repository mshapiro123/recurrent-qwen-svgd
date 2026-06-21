from __future__ import annotations

import json

from colab.assess_stage5_recipe_control import (
    assess_recipe_control,
    latest_summary,
    main,
)


def _summary(selected: list[bool], *, best: list[bool] | None = None, hard_count: int = 6) -> dict[str, object]:
    best = best or selected
    examples = []
    for idx, hit in enumerate(selected):
        examples.append(
            {
                "task_id": f"task_{idx}",
                "test_index": 0,
                "has_target": True,
                "selected_exact": hit,
                "best_of_k_exact": best[idx],
                "first_exact": hit,
                "difficulty_bucket": "hard" if idx < hard_count else "easy",
            }
        )
    return {
        "summary": {
            "selected_exact": sum(1 for value in selected if value),
            "best_of_k_exact": sum(1 for value in best if value),
            "first_exact": sum(1 for value in selected if value),
            "examples_with_targets": len(selected),
        },
        "examples": examples,
    }


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metadata(eval_limit: int = 20) -> dict[str, object]:
    return {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.5,
        "arc_version": "1",
        "train_split": "training",
        "eval_split": "evaluation",
        "train_task_limit": 100,
        "eval_task_limit": eval_limit,
        "color_augmentations": 2,
        "geometry_augmentations": "all",
        "trace_mode": "symbolic_program",
        "trace_filter": "covered",
        "synthetic_tasks": 0,
        "candidate_distill_jsonls": [],
        "grid_format": "compact",
        "program_parse_mode": "fallback",
        "selection_strategy": "heuristic",
        "train_steps": 300,
        "learning_rate": 8e-6,
        "distillation": {"enabled": False, "weight": 0.1, "temperature": 1.0, "on": "response"},
        "include_symbolic_candidates": False,
        "eval_checkpoint_ladder": False,
    }


def _make_dense_and_recurrent(
    tmp_path,
    *,
    recurrent_selected: list[bool],
    dense_selected: list[bool],
    recurrent_best: list[bool] | None = None,
    dense_best: list[bool] | None = None,
):
    dense_dir = tmp_path / "outputs" / "stage5" / "dense"
    recurrent_dir = tmp_path / "outputs" / "stage5" / "recurrent"
    base = _summary([False] * len(dense_selected))
    dense = _summary(dense_selected, best=dense_best)
    start = _summary([False] * len(dense_selected))
    recurrent = _summary(recurrent_selected, best=recurrent_best)
    _write_json(dense_dir / "base_summary.json", base)
    _write_json(dense_dir / "dense_tuned_summary.json", dense)
    _write_json(
        dense_dir / "summary.json",
        {
            "run_id": "dense",
            "kind": "dense_sft_control",
            "metadata": _metadata(),
            "base": base,
            "dense_tuned": dense,
            "phase1_start": start,
        },
    )
    _write_json(recurrent_dir / "phase1_start_summary.json", start)
    _write_json(recurrent_dir / "phase1_arc_agi_tuned_summary.json", recurrent)
    _write_json(
        recurrent_dir / "summary.json",
        {
            "run_id": "recurrent",
            "metadata": _metadata(),
            "phase1_start": start["summary"],
            "phase1_arc_agi_tuned": recurrent["summary"],
            "tuned_checkpoint": "outputs/stage5/recurrent/phase1.pt",
        },
    )
    return dense_dir / "summary.json", recurrent_dir / "summary.json"


def test_recipe_control_passes_hard_tail_recurrent_lift(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False, False, False, False, False, False] + [False] * 14,
        recurrent_selected=[True, True, True, False, False, False] + [False] * 14,
    )

    assessment = assess_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_total_examples=20,
        min_hard_examples=5,
    )

    assert assessment["status"] == "passed"
    assert assessment["passed"] is True
    assert assessment["decision_evidence"]["aggregate"]["delta_exact"] == 3
    assert assessment["decision_evidence"]["hard"]["delta_exact"] == 3


def test_recipe_control_flags_selector_conversion_when_best_of_k_lifts(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[False] * 20,
        dense_best=[False] * 20,
        recurrent_best=[True, True, True, False, False, False] + [False] * 14,
    )

    assessment = assess_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_total_examples=20,
        min_hard_examples=5,
    )

    assert assessment["status"] == "needs_selector_conversion"
    assert assessment["passed"] is False
    assert assessment["decision_evidence"]["aggregate"]["delta_exact"] == 0
    assert assessment["decision_evidence"]["aggregate_best_of_k"]["delta_exact"] == 3
    assert assessment["decision_evidence"]["hard_best_of_k"]["delta_exact"] == 3


def test_recipe_control_does_not_pass_without_hard_tail_selected_lift(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[False, False, False, False, False, False] + [True, True, True] + [False] * 11,
    )

    assessment = assess_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_total_examples=20,
        min_hard_examples=5,
    )

    assert assessment["status"] == "needs_more_evidence"
    assert assessment["passed"] is False
    assert assessment["decision_evidence"]["aggregate"]["delta_exact"] == 3
    assert assessment["decision_evidence"]["hard"]["delta_exact"] == 0


def test_recipe_control_does_not_request_selector_conversion_without_hard_tail_candidate_lift(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[False] * 20,
        dense_best=[False] * 20,
        recurrent_best=[False, False, False, False, False, False] + [True, True, True] + [False] * 11,
    )

    assessment = assess_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        min_total_examples=20,
        min_hard_examples=5,
    )

    assert assessment["status"] == "needs_more_evidence"
    assert assessment["passed"] is False
    assert assessment["decision_evidence"]["aggregate_best_of_k"]["delta_exact"] == 3
    assert assessment["decision_evidence"]["hard_best_of_k"]["delta_exact"] == 0


def test_recipe_control_flags_metadata_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["trace_mode"] = "none"
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["trace_mode"] == {
        "dense": "symbolic_program",
        "recurrent": "none",
    }


def test_recipe_control_flags_base_model_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["model_name"] = "Qwen/Qwen2.5-1.5B-Instruct"
    payload["metadata"]["params_b"] = 1.5
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["model_name"] == {
        "dense": "Qwen/Qwen2.5-0.5B-Instruct",
        "recurrent": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    assert assessment["metadata_differences"]["params_b"] == {"dense": "0.5", "recurrent": "1.5"}


def test_recipe_control_flags_eval_split_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["eval_split"] = "training"
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["eval_split"] == {"dense": "evaluation", "recurrent": "training"}


def test_recipe_control_flags_synthetic_task_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["synthetic_tasks"] = 200
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["synthetic_tasks"] == {"dense": "0", "recurrent": "200"}


def test_recipe_control_flags_candidate_distill_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["candidate_distill_jsonls"] = ["outputs/stage5/candidates.jsonl"]
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["candidate_distill_jsonls"] == {
        "dense": "[]",
        "recurrent": '["outputs/stage5/candidates.jsonl"]',
    }


def test_recipe_control_flags_distillation_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["distillation"] = {"enabled": True, "weight": 0.2, "temperature": 2.0, "on": "full"}
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    diff = assessment["metadata_differences"]["distillation"]
    assert diff["dense"] != diff["recurrent"]
    assert '"enabled": false' in diff["dense"]
    assert '"enabled": true' in diff["recurrent"]


def test_recipe_control_flags_symbolic_candidates_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["include_symbolic_candidates"] = True
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["include_symbolic_candidates"] == {
        "dense": "False",
        "recurrent": "True",
    }


def test_recipe_control_flags_checkpoint_ladder_mismatch(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    payload = json.loads(recurrent_summary.read_text(encoding="utf-8"))
    payload["metadata"]["eval_checkpoint_ladder"] = True
    recurrent_summary.write_text(json.dumps(payload), encoding="utf-8")

    assessment = assess_recipe_control(dense_summary_path=dense_summary, recurrent_summary_path=recurrent_summary)

    assert assessment["status"] == "needs_review"
    assert assessment["metadata_differences"]["eval_checkpoint_ladder"] == {"dense": "False", "recurrent": "True"}


def test_recipe_control_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    output_json = tmp_path / "assessment.json"
    output_md = tmp_path / "assessment.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_recipe_control.py",
            "--dense_summary_json",
            str(dense_summary),
            "--recurrent_summary_json",
            str(recurrent_summary),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "passed"
    assert "Same-Recipe Architecture Assessment" in output_md.read_text(encoding="utf-8")


def test_recipe_control_latest_summary_finds_expected_types(tmp_path) -> None:
    dense_summary, recurrent_summary = _make_dense_and_recurrent(
        tmp_path,
        dense_selected=[False] * 20,
        recurrent_selected=[True] * 20,
    )
    scan_root = tmp_path / "outputs" / "stage5"

    assert latest_summary(scan_root, lambda payload: payload.get("kind") == "dense_sft_control") == dense_summary
    assert latest_summary(scan_root, lambda payload: bool(payload.get("tuned_checkpoint"))) == recurrent_summary
