from __future__ import annotations

from colab.run_stage5_arc_agi_curriculum import (
    compact_family_summary,
    parse_curriculum_stages,
    stage_best_checkpoint,
)


def test_parse_curriculum_stages() -> None:
    stages = parse_curriculum_stages("warmup:constant_output,geometry_color:10:20;mix:all:30:40")

    assert [stage.name for stage in stages] == ["warmup", "mix"]
    assert stages[0].synthetic_modes == "constant_output,geometry_color"
    assert stages[0].synthetic_tasks == 10
    assert stages[1].train_steps == 40


def test_stage_best_checkpoint_prefers_best_ladder_row() -> None:
    summary = {
        "tuned_checkpoint": "final.pt",
        "phase1_arc_agi_tuned": {"selected_exact": 1, "best_of_k_exact": 1},
        "best_checkpoint": {
            "checkpoint": "best.pt",
            "step": 150,
            "summary": {"selected_exact": 2, "best_of_k_exact": 3},
            "eval_diagnostics": {"task_family_summary": {"crop": {"selected_exact": 2}}},
        },
    }

    best = stage_best_checkpoint(summary)

    assert best["source"] == "best_checkpoint"
    assert best["checkpoint"] == "best.pt"
    assert best["step"] == 150
    assert best["eval_diagnostics"]["task_family_summary"]["crop"]["selected_exact"] == 2


def test_stage_best_checkpoint_falls_back_to_final() -> None:
    summary = {
        "tuned_checkpoint": "final.pt",
        "phase1_arc_agi_tuned": {"selected_exact": 1, "best_of_k_exact": 1},
        "eval_diagnostics": {
            "phase1_arc_agi_tuned": {
                "task_family_summary": {"frame_object": {"selected_exact": 1}},
            }
        },
    }

    best = stage_best_checkpoint(summary)

    assert best["source"] == "final_checkpoint"
    assert best["checkpoint"] == "final.pt"
    assert best["eval_diagnostics"]["task_family_summary"]["frame_object"]["selected_exact"] == 1


def test_compact_family_summary_keeps_core_fields() -> None:
    summary = {
        "move_recolor": {
            "selected_exact": 3,
            "best_of_k_exact": 4,
            "examples_with_targets": 5,
            "tasks_solved_best_of_k": 2,
            "tasks_with_targets": 3,
            "valid_candidate_rate": 0.75,
            "extra": "ignored",
        }
    }

    compact = compact_family_summary(summary)

    assert compact == {
        "move_recolor": {
            "selected_exact": 3,
            "best_of_k_exact": 4,
            "examples_with_targets": 5,
            "tasks_solved_best_of_k": 2,
            "tasks_with_targets": 3,
            "valid_candidate_rate": 0.75,
        }
    }
