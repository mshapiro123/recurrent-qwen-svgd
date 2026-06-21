from __future__ import annotations

import json

from colab.assess_stage5_recipe_selector_conversion import assess_recipe_selector_conversion, main


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _eval_summary(selected: list[bool], *, best: list[bool] | None = None, hard_count: int = 6) -> dict:
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


def _recipe_and_selector(
    tmp_path,
    *,
    dense_selected: list[bool],
    selector_selected: list[bool],
):
    dense_dir = tmp_path / "outputs" / "stage5" / "dense"
    selector_dir = tmp_path / "outputs" / "stage5" / "selector"
    dense_eval = _eval_summary(dense_selected)
    selector_eval = _eval_summary(selector_selected)
    dense_summary = dense_dir / "summary.json"
    selector_output = selector_dir / "recovered__selector_reliability_vote_summary.json"
    recipe_summary = tmp_path / "outputs" / "stage5" / "recipe" / "summary.json"
    selector_summary = selector_dir / "summary.json"

    _write(dense_dir / "dense_tuned_summary.json", dense_eval)
    _write(
        dense_summary,
        {
            "run_id": "dense",
            "kind": "dense_sft_control",
            "dense_tuned": dense_eval["summary"],
        },
    )
    _write(selector_output, selector_eval)
    _write(
        recipe_summary,
        {
            "run_id": "recipe",
            "gate": "stage5_same_recipe_architecture",
            "status": "needs_selector_conversion",
            "passed": False,
            "dense_summary": str(dense_summary),
        },
    )
    _write(
        selector_summary,
        {
            "run_id": "selector",
            "source_run_dir": str(selector_dir),
            "strategies": ["reliability_vote"],
            "rows": [
                {
                    "label": "recovered",
                    "selection_strategy": "reliability_vote",
                    "selected_exact": sum(selector_selected),
                    "best_of_k_exact": sum(selector_selected),
                    "examples": len(selector_selected),
                    "valid_candidate_rate": 1.0,
                    "output_summary_json": str(selector_output),
                }
            ],
            "best_by_label": {},
        },
    )
    return recipe_summary, selector_summary


def test_recipe_selector_conversion_passes_when_selector_beats_dense_with_hard_support(tmp_path) -> None:
    recipe, selector = _recipe_and_selector(
        tmp_path,
        dense_selected=[False] * 20,
        selector_selected=[True, True, True, False, False, False] + [False] * 14,
    )

    payload = assess_recipe_selector_conversion(
        recipe_control_summary=recipe,
        selector_rescore_summary=selector,
        min_total_examples=20,
        min_hard_examples=5,
    )

    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["passing_selectors"] == [{"label": "recovered", "selection_strategy": "reliability_vote"}]


def test_recipe_selector_conversion_fails_when_selector_does_not_beat_dense(tmp_path) -> None:
    recipe, selector = _recipe_and_selector(
        tmp_path,
        dense_selected=[True, True, False, False, False, False] + [False] * 14,
        selector_selected=[True, False, False, False, False, False] + [False] * 14,
    )

    payload = assess_recipe_selector_conversion(
        recipe_control_summary=recipe,
        selector_rescore_summary=selector,
        min_total_examples=20,
        min_hard_examples=5,
    )

    assert payload["status"] == "failed"
    assert payload["passed"] is False


def test_recipe_selector_conversion_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    recipe, selector = _recipe_and_selector(
        tmp_path,
        dense_selected=[False] * 20,
        selector_selected=[True, True, True, False, False, False] + [False] * 14,
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_recipe_selector_conversion.py",
            "--recipe_control_summary",
            str(recipe),
            "--selector_rescore_summary",
            str(selector),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "passed"
    assert "Same-Recipe Selector Conversion" in output_md.read_text(encoding="utf-8")
