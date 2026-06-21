from __future__ import annotations

import subprocess
import sys

import colab.run_stage5_next_action as module
from colab.run_stage5_next_action import (
    action_fingerprint,
    bootstrap_plan,
    execute_action_loop,
    is_repeat_action,
    parse_action_command,
    run_planner,
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


def test_parse_action_command_allows_recipe_selector_conversion_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_RECIPE_SELECTOR_CONVERSION_RUN_ID=conversion python "
        "colab/assess_stage5_recipe_selector_conversion.py "
        "--recipe_control_summary outputs/stage5/recipe/summary.json "
        "--selector_rescore_summary outputs/stage5/selector/summary.json"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_RECIPE_SELECTOR_CONVERSION_RUN_ID": "conversion"}
    assert parsed.argv == [
        sys.executable,
        "colab/assess_stage5_recipe_selector_conversion.py",
        "--recipe_control_summary",
        "outputs/stage5/recipe/summary.json",
        "--selector_rescore_summary",
        "outputs/stage5/selector/summary.json",
    ]


def test_parse_action_command_allows_release_gate_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_RELEASE_GATE_RUN_ID=release python colab/assess_stage5_release_gate.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_RELEASE_GATE_RUN_ID": "release"}
    assert parsed.argv == [sys.executable, "colab/assess_stage5_release_gate.py"]


def test_parse_action_command_allows_benchmark_suite_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_BENCHMARK_ASSESS_RUN_ID=assess python colab/assess_stage5_benchmark_suite.py --summary_json outputs/stage5/suite/summary.json"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_BENCHMARK_ASSESS_RUN_ID": "assess"}
    assert parsed.argv == [
        sys.executable,
        "colab/assess_stage5_benchmark_suite.py",
        "--summary_json",
        "outputs/stage5/suite/summary.json",
    ]


def test_parse_action_command_allows_claim_packet_builder() -> None:
    parsed = parse_action_command("STAGE5_CLAIM_PACKET_RUN_ID=claim python colab/build_stage5_claim_packet.py")

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_CLAIM_PACKET_RUN_ID": "claim"}
    assert parsed.argv == [sys.executable, "colab/build_stage5_claim_packet.py"]


def test_parse_action_command_allows_arc_agi_sota_comparison_builder() -> None:
    parsed = parse_action_command(
        "STAGE5_ARC_AGI_SOTA_COMPARISON_RUN_ID=sota python colab/build_stage5_arc_agi_sota_comparison.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_ARC_AGI_SOTA_COMPARISON_RUN_ID": "sota"}
    assert parsed.argv == [sys.executable, "colab/build_stage5_arc_agi_sota_comparison.py"]


def test_parse_action_command_allows_arc_agi_baseline_registry_validator() -> None:
    parsed = parse_action_command(
        "STAGE5_ARC_AGI_BASELINE_REGISTRY_RUN_ID=registry python colab/validate_arc_agi_baseline_registry.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_ARC_AGI_BASELINE_REGISTRY_RUN_ID": "registry"}
    assert parsed.argv == [sys.executable, "colab/validate_arc_agi_baseline_registry.py"]


def test_parse_action_command_allows_reproduced_baseline_registry_builder() -> None:
    parsed = parse_action_command(
        "python colab/build_stage5_arc_agi_reproduced_baseline_registry.py "
        "--summary_json outputs/stage5/candidate/summary.json "
        "--labels base "
        "--output_json config/arc_agi_same_size_baselines.json "
        "--validation_json outputs/stage5/registry/summary.json"
    )

    assert parsed.kind == "python"
    assert parsed.argv == [
        sys.executable,
        "colab/build_stage5_arc_agi_reproduced_baseline_registry.py",
        "--summary_json",
        "outputs/stage5/candidate/summary.json",
        "--labels",
        "base",
        "--output_json",
        "config/arc_agi_same_size_baselines.json",
        "--validation_json",
        "outputs/stage5/registry/summary.json",
    ]


def test_parse_action_command_allows_benchmark_suite_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_BENCHMARK_SUITE_RUN_ID=bench python colab/run_stage5_benchmark_suite.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_BENCHMARK_SUITE_RUN_ID": "bench"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_benchmark_suite.py"]


def test_parse_action_command_allows_candidate_gate_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_ARC_AGI_GATE_RUN_ID=gate python colab/run_stage5_arc_agi_candidate_gate.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_ARC_AGI_GATE_RUN_ID": "gate"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_arc_agi_candidate_gate.py"]


def test_parse_action_command_allows_selector_replication_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_SELECTOR_REPLICATION_RUN_ID=selector_rep python colab/assess_stage5_selector_replication.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_SELECTOR_REPLICATION_RUN_ID": "selector_rep"}
    assert parsed.argv == [sys.executable, "colab/assess_stage5_selector_replication.py"]


def test_parse_action_command_allows_phase1_recovery_ladder() -> None:
    parsed = parse_action_command(
        "STAGE5_RUN_ID=recovery python colab/run_stage5_phase1_recovery_ladder.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_RUN_ID": "recovery"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_phase1_recovery_ladder.py"]


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


def test_bootstrap_plan_runs_candidate_gate_first() -> None:
    plan = bootstrap_plan("bootstrap_run", reason="no summaries")

    action = plan["actions"][0]
    assert plan["source_kind"] == "bootstrap"
    assert action["name"] == "Run Stage 5 ARC-AGI candidate gate"
    assert "python colab/run_stage5_arc_agi_candidate_gate.py" in action["command"]


def test_run_planner_bootstraps_when_no_stage5_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "RUN_ID", "bootstrap")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "")

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        return subprocess.CompletedProcess(
            cmd,
            1,
            "FileNotFoundError: No Stage 5 result summary found. Set STAGE5_ARC_AGI_NEXT_PLAN_SOURCE_SUMMARY.",
            None,
        )

    monkeypatch.setattr(module, "run", fake_run)

    path, plan, proc = run_planner()

    assert proc.returncode == 1
    assert path.exists()
    assert plan["source_kind"] == "bootstrap"
    assert plan["actions"][0]["name"] == "Run Stage 5 ARC-AGI candidate gate"
