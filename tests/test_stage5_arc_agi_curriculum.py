from __future__ import annotations

from colab.run_stage5_arc_agi_curriculum import (
    candidate_distill_env,
    candidate_distill_jsonls,
    candidate_distill_row_count,
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


def test_candidate_distill_jsonls_parses_nonempty_items() -> None:
    assert candidate_distill_jsonls("a.jsonl, b.jsonl,,") == ["a.jsonl", "b.jsonl"]


def test_candidate_distill_env_disables_empty_source() -> None:
    assert candidate_distill_env("")["STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS"] == ""


def test_candidate_distill_env_threads_nonempty_source_and_recipe() -> None:
    env = candidate_distill_env("a.jsonl", choice="all_exact", completion_source="trace_then_canonical_grid")

    assert env == {
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS": "a.jsonl",
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE": "all_exact",
        "STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE": "trace_then_canonical_grid",
    }


def test_candidate_distill_row_count_sums_child_outputs() -> None:
    summary = {"candidate_distill_info": [{"rows": 2}, {"rows": "3"}]}

    assert candidate_distill_row_count(summary) == 5


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
