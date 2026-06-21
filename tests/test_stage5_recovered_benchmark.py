from __future__ import annotations

from colab.run_stage5_arc_agi_recovered_benchmark import (
    comparison_specs,
    eval_arc,
    final_stage_row,
    gap_closure,
    metric_delta,
    recovered_checkpoint_from_curriculum,
)


def test_final_stage_row_returns_last_stage() -> None:
    curriculum = {
        "stages": [
            {"stage": {"name": "warmup"}},
            {"stage": {"name": "mixed"}},
        ]
    }

    assert final_stage_row(curriculum)["stage"]["name"] == "mixed"


def test_recovered_checkpoint_from_curriculum_prefers_final_stage_selected_checkpoint() -> None:
    curriculum = {
        "final_checkpoint": "final.pt",
        "stages": [
            {
                "selected_checkpoint": {"checkpoint": "stage0.pt"},
            },
            {
                "selected_checkpoint": {"checkpoint": "stage1.pt"},
            },
        ],
    }

    checkpoint = recovered_checkpoint_from_curriculum(curriculum)

    assert checkpoint.name == "stage1.pt"


def test_metric_delta_tracks_core_arc_metrics() -> None:
    delta = metric_delta(
        {
            "first_exact": 2,
            "selected_exact": 3,
            "best_of_k_exact": 4,
            "tasks_solved_best_of_k": 5,
        },
        {
            "first_exact": 1,
            "selected_exact": 3,
            "best_of_k_exact": 2,
            "tasks_solved_best_of_k": 7,
        },
    )

    assert delta == {
        "first_exact_delta": 1,
        "selected_exact_delta": 0,
        "best_of_k_exact_delta": 2,
        "tasks_solved_best_of_k_delta": -2,
    }


def test_gap_closure_tracks_recovery_from_surgical_regression() -> None:
    closure = gap_closure(
        {"selected_exact": 10, "best_of_k_exact": 12},
        {"selected_exact": 4, "best_of_k_exact": 8},
        {"selected_exact": 7, "best_of_k_exact": 13},
    )

    assert closure["selected_exact"] == {
        "base": 10,
        "phase1_start": 4,
        "recovered": 7,
        "initial_gap_to_base": 6,
        "recovered_gain_from_start": 3,
        "remaining_gap_to_base": 3,
        "closure_fraction": 0.5,
        "status": "partially_closed",
    }
    assert closure["best_of_k_exact"]["closure_fraction"] == 1.25
    assert closure["best_of_k_exact"]["status"] == "closed_or_surpassed"


def test_gap_closure_marks_start_at_or_above_base() -> None:
    closure = gap_closure(
        {"selected_exact": 5},
        {"selected_exact": 6},
        {"selected_exact": 7},
    )

    assert closure["selected_exact"]["initial_gap_to_base"] == -1
    assert closure["selected_exact"]["closure_fraction"] is None
    assert closure["selected_exact"]["status"] == "start_at_or_above_base"


def test_comparison_specs_cover_base_start_and_recovered_pairs() -> None:
    assert comparison_specs() == [
        ("phase1_start_vs_base", "base", "phase1_start"),
        ("recovered_vs_start", "phase1_start", "recovered"),
        ("recovered_vs_base", "base", "recovered"),
    ]


def test_eval_arc_omits_limit_for_full_split(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_arc_agi_recovered_benchmark as module

    commands: list[list[str]] = []
    summary_json = module.RUN_DIR / "base_summary.json"

    def fake_run(cmd: list[str], **kwargs) -> None:
        commands.append(cmd)
        summary_json.write_text('{"summary": {}}', encoding="utf-8")

    monkeypatch.setattr(module, "LIMIT", None)
    monkeypatch.setattr(module, "run", fake_run)

    eval_arc("base", mode="base", tasks_path=tmp_path)

    assert commands
    assert "--limit" not in commands[0]


def test_eval_arc_uses_difficulty_stratified_slice_without_limit(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_arc_agi_recovered_benchmark as module

    commands: list[list[str]] = []
    summary_json = module.RUN_DIR / "base_summary.json"

    def fake_run(cmd: list[str], **kwargs) -> None:
        commands.append(cmd)
        summary_json.write_text('{"summary": {}}', encoding="utf-8")

    monkeypatch.setattr(module, "LIMIT", 100)
    monkeypatch.setattr(module, "DIFFICULTY_BUCKETS", "easy,medium,hard")
    monkeypatch.setattr(module, "EXAMPLES_PER_DIFFICULTY", 20)
    monkeypatch.setattr(module, "run", fake_run)

    eval_arc("base", mode="base", tasks_path=tmp_path)

    assert commands
    assert "--limit" not in commands[0]
    assert commands[0][commands[0].index("--difficulty_buckets") + 1] == "easy,medium,hard"
    assert commands[0][commands[0].index("--examples_per_difficulty") + 1] == "20"
