from __future__ import annotations

import sys

from colab.run_stage5_next_action import parse_action_command, selected_action


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
    # ACTION_INDEX is read at import time, so this helper test sticks to the
    # default index and verifies the plan contract directly.
    action = selected_action({"actions": [{"name": "first"}, {"name": "second"}]})

    assert action["name"] == "first"
