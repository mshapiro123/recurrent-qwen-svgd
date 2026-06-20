from __future__ import annotations

import json

from colab.run_stage5_arc_agi_post_curriculum_particle_gate import (
    curriculum_context,
    final_stage_row,
)


def test_final_stage_row_returns_last_stage() -> None:
    curriculum = {
        "stages": [
            {"stage": {"name": "warmup"}},
            {"stage": {"name": "mixed"}},
        ]
    }

    assert final_stage_row(curriculum)["stage"]["name"] == "mixed"


def test_curriculum_context_reads_final_checkpoint_and_eval_path(tmp_path) -> None:
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    checkpoint = tmp_path / "phase1_step_200.pt"
    checkpoint.write_bytes(b"checkpoint")
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text("{}", encoding="utf-8")

    child_summary = {
        "metadata": {
            "eval_path": str(tasks_path),
            "eval_task_limit": 12,
            "program_parse_mode": "prefer",
            "grid_format": "compact",
        }
    }
    (child_dir / "summary.json").write_text(json.dumps(child_summary), encoding="utf-8")

    curriculum = {
        "stages": [
            {
                "child_run_dir": str(child_dir),
                "selected_checkpoint": {
                    "checkpoint": str(checkpoint),
                    "summary": {"selected_exact": 2, "best_of_k_exact": 3},
                    "eval_diagnostics": {
                        "task_family_summary": {
                            "move_recolor": {"selected_exact": 2},
                        }
                    },
                },
            }
        ]
    }
    summary_path = tmp_path / "curriculum_summary.json"
    summary_path.write_text(json.dumps(curriculum), encoding="utf-8")

    context = curriculum_context(summary_path)

    assert context["checkpoint"] == checkpoint
    assert context["tasks_path"] == tasks_path
    assert context["eval_limit"] == 12
    assert context["program_parse_mode"] == "prefer"
    assert context["reference_summary"]["best_of_k_exact"] == 3
    assert context["reference_task_family_summary"]["move_recolor"]["selected_exact"] == 2
