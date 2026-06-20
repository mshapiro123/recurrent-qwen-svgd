from __future__ import annotations

import json

from colab.run_stage5_arc_agi_autopilot_followup import (
    compare_recovered_to_base,
    curriculum_summary_path,
    run_recovered_benchmark,
    should_run_eval_followups,
)


def test_curriculum_summary_path_prefers_child_run_id() -> None:
    autopilot = {"child_run_ids": {"curriculum": "parent_curriculum"}}

    path = curriculum_summary_path(autopilot)

    assert path is not None
    assert path.as_posix().endswith("outputs/stage5/parent_curriculum/summary.json")


def test_curriculum_summary_path_falls_back_to_recommended_action_command(tmp_path) -> None:
    summary = tmp_path / "curriculum_summary.json"
    command = f"STAGE5_ARC_AGI_CURRICULUM_SUMMARY={summary} python colab/run_stage5_arc_agi_tta_sweep.py"
    autopilot = {"compact": {"recommended_next_actions": [{"command": command}]}}

    assert curriculum_summary_path(autopilot) == summary


def test_should_run_eval_followups_rejects_failed_candidate_gate() -> None:
    autopilot = {"compact": {"candidate_distillation_passed": False, "final_checkpoint": "ckpt.pt"}}

    should_run, reason = should_run_eval_followups(autopilot)

    assert should_run is False
    assert "candidate distillation" in reason


def test_should_run_eval_followups_requires_existing_curriculum_summary(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"final_checkpoint": "ckpt.pt"}), encoding="utf-8")
    autopilot = {
        "compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "ckpt.pt",
            "recommended_next_actions": [
                {
                    "command": (
                        f"STAGE5_ARC_AGI_CURRICULUM_SUMMARY={summary} "
                        "python colab/run_stage5_arc_agi_recovered_benchmark.py"
                    )
                }
            ],
        }
    }

    should_run, reason = should_run_eval_followups(autopilot)

    assert should_run is True
    assert reason == "recovered checkpoint available"


def test_compare_recovered_to_base_returns_benchmark_delta() -> None:
    benchmark = {
        "deltas": {
            "recovered_vs_base": {
                "selected_exact_delta": 2,
                "best_of_k_exact_delta": 3,
            }
        }
    }

    assert compare_recovered_to_base(benchmark) == {
        "selected_exact_delta": 2,
        "best_of_k_exact_delta": 3,
    }
    assert compare_recovered_to_base(None) is None


def test_run_recovered_benchmark_passes_full_limit_to_child(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_arc_agi_autopilot_followup as module

    captured: dict[str, str] = {}

    def fake_run_child(label: str, script: str, env_updates: dict[str, str]) -> dict[str, object]:
        captured.update(env_updates)
        return {}

    monkeypatch.setattr(module, "LIMIT", None)
    monkeypatch.setattr(module, "run_child", fake_run_child)

    run_recovered_benchmark(tmp_path / "summary.json")

    assert captured["STAGE5_ARC_AGI_LIMIT"] == "full"
    assert captured["STAGE5_ARC_AGI_RECOVERED_BENCHMARK_RUN_ID"].endswith("_limitfull")
