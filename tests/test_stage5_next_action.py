from __future__ import annotations

import subprocess
import sys

import colab.run_stage5_next_action as module
from colab.run_stage5_next_action import (
    action_fingerprint,
    execute_action_loop,
    is_repeat_action,
    parse_action_command,
    selected_action,
)


def test_parse_action_command_extracts_env_and_python_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_ARC_AGI_LIMIT=100 STAGE5_ARC_AGI_SELECTION_STRATEGY=self_consistency "
        "python colab/run_stage5_arc_agi_recovered_benchmark.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {
        "STAGE5_ARC_AGI_LIMIT": "100",
        "STAGE5_ARC_AGI_SELECTION_STRATEGY": "self_consistency",
    }
    assert parsed.argv == [sys.executable, "colab/run_stage5_arc_agi_recovered_benchmark.py"]


def test_parse_action_command_keeps_quoted_env_values() -> None:
    parsed = parse_action_command(
        "STAGE5_ARC_AGI_CURRICULUM_STAGES='focus:move_recolor,frame_object:260:320;mixed:all:340:420' "
        "python colab/run_stage5_arc_agi_curriculum_particle_autopilot.py"
    )

    assert parsed.env["STAGE5_ARC_AGI_CURRICULUM_STAGES"] == (
        "focus:move_recolor,frame_object:260:320;mixed:all:340:420"
    )
    assert parsed.argv[1] == "colab/run_stage5_arc_agi_curriculum_particle_autopilot.py"


def test_parse_action_command_allows_readonly_cat() -> None:
    parsed = parse_action_command("cat outputs/stage5/run/summary.md")

    assert parsed.kind == "cat"
    assert parsed.argv == ["cat", "outputs/stage5/run/summary.md"]


def test_parse_action_command_allows_gate1_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_GATE1_ASSESSMENT_RUN_ID=gate1 "
        "python colab/assess_stage5_gate1.py --summary_json outputs/stage5/run/summary.json"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_GATE1_ASSESSMENT_RUN_ID": "gate1"}
    assert parsed.argv == [
        sys.executable,
        "colab/assess_stage5_gate1.py",
        "--summary_json",
        "outputs/stage5/run/summary.json",
    ]


def test_parse_action_command_allows_gate2_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_GATE2_ASSESSMENT_RUN_ID=gate2 "
        "python colab/assess_stage5_gate2.py --summary_json outputs/stage5/run/summary.json"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_GATE2_ASSESSMENT_RUN_ID": "gate2"}
    assert parsed.argv == [
        sys.executable,
        "colab/assess_stage5_gate2.py",
        "--summary_json",
        "outputs/stage5/run/summary.json",
    ]


def test_parse_action_command_allows_recipe_control_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_RECIPE_CONTROL_ASSESSMENT_RUN_ID=recipe "
        "python colab/assess_stage5_recipe_control.py --recurrent_summary_json outputs/stage5/run/summary.json"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_RECIPE_CONTROL_ASSESSMENT_RUN_ID": "recipe"}
    assert parsed.argv == [
        sys.executable,
        "colab/assess_stage5_recipe_control.py",
        "--recurrent_summary_json",
        "outputs/stage5/run/summary.json",
    ]


def test_parse_action_command_allows_dense_and_recurrent_sft_runners() -> None:
    dense = parse_action_command("python colab/run_stage5_arc_agi_dense_sft.py")
    recurrent = parse_action_command("python colab/run_stage5_arc_agi_sft.py")

    assert dense.argv == [sys.executable, "colab/run_stage5_arc_agi_dense_sft.py"]
    assert recurrent.argv == [sys.executable, "colab/run_stage5_arc_agi_sft.py"]


def test_parse_action_command_rejects_arbitrary_shell() -> None:
    try:
        parse_action_command("rm -rf outputs")
    except ValueError as exc:
        assert "Unsupported planner executable" in str(exc)
    else:
        raise AssertionError("arbitrary shell command should be rejected")


def test_parse_action_command_rejects_non_allowlisted_python() -> None:
    try:
        parse_action_command("python scripts/not_a_stage5_runner.py")
    except ValueError as exc:
        assert "not allowlisted" in str(exc)
    else:
        raise AssertionError("non-allowlisted python script should be rejected")


def test_selected_action_uses_configured_index(monkeypatch) -> None:
    action = selected_action({"actions": [{"name": "first"}, {"name": "second"}]}, action_index=1)

    assert action["name"] == "second"


def test_repeat_guard_uses_command_fingerprint() -> None:
    action = {"command": "python colab/plan_stage5_next_run.py"}
    seen = {action_fingerprint(action)}

    assert is_repeat_action(action, seen) is True
    assert is_repeat_action(action, seen, allow_repeat=True) is False


def test_execute_action_loop_dry_run_stops_after_one_plan(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run_planner(*, step=None):
        calls.append(step)
        return (
            tmp_path / "plan.json",
            {"actions": [{"name": "Plan only", "command": "python colab/plan_stage5_next_run.py"}]},
            subprocess.CompletedProcess([], 0),
        )

    monkeypatch.setattr(module, "run_planner", fake_run_planner)

    steps = execute_action_loop(execute=False, max_actions=3)

    assert calls == [None]
    assert len(steps) == 1
    assert steps[0]["execution"]["dry_run"] is True


def test_execute_action_loop_stops_on_repeated_action(monkeypatch, tmp_path) -> None:
    calls = []
    executed = []
    action = {"name": "Repeat", "command": "python colab/plan_stage5_next_run.py"}

    def fake_run_planner(*, step=None):
        calls.append(step)
        return tmp_path / f"plan_{step}.json", {"actions": [action]}, subprocess.CompletedProcess([], 0)

    def fake_execute(parsed, *, log_name="selected_action.log"):
        executed.append((parsed.argv, log_name))
        return subprocess.CompletedProcess(parsed.argv, 0)

    monkeypatch.setattr(module, "run_planner", fake_run_planner)
    monkeypatch.setattr(module, "execute_parsed_command", fake_execute)

    steps = execute_action_loop(execute=True, max_actions=3)

    assert calls == [0, 1]
    assert len(executed) == 1
    assert len(steps) == 2
    assert steps[0]["execution"]["executed"] is True
    assert steps[1]["repeat_detected"] is True
    assert steps[1]["execution"]["stopped"] is True


def test_execute_action_loop_rejects_nonpositive_max_actions() -> None:
    try:
        execute_action_loop(execute=False, max_actions=0)
    except ValueError as exc:
        assert "MAX_ACTIONS must be >= 1" in str(exc)
    else:
        raise AssertionError("nonpositive max_actions should fail")
