from __future__ import annotations

import json
import subprocess
import sys

import colab.run_stage5_next_action as module
from colab.run_stage5_next_action import (
    action_fingerprint,
    a100_execution_guard,
    bootstrap_plan,
    execute_action_loop,
    local_only_runtime_guard,
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


def test_parse_action_command_allows_competence_recovery_scripts() -> None:
    for script in [
        "colab/run_stage5_balanced_arc_mix_gate.py",
        "colab/run_stage5_competence_preserving_pipeline.py",
        "colab/run_stage5_recovery_full_assessment.py",
    ]:
        parsed = parse_action_command(f"STAGE5_RUN_ID=test python {script}")

        assert parsed.kind == "python"
        assert parsed.env == {"STAGE5_RUN_ID": "test"}
        assert parsed.argv == [sys.executable, script]


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


def test_parse_action_command_allows_routing_diagnostic_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_ROUTING_DIAGNOSTIC_RUN_ID=route python colab/run_stage5_routing_diagnostic.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_ROUTING_DIAGNOSTIC_RUN_ID": "route"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_routing_diagnostic.py"]


def test_parse_action_command_allows_routing_repair_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_ROUTING_REPAIR_RUN_ID=repair python colab/run_stage5_routing_repair.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_ROUTING_REPAIR_RUN_ID": "repair"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_routing_repair.py"]


def test_parse_action_command_allows_programmatic_depth_repair_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_PROGRAMMATIC_DEPTH_RUN_ID=depth python colab/run_stage5_programmatic_depth_repair.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_PROGRAMMATIC_DEPTH_RUN_ID": "depth"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_programmatic_depth_repair.py"]


def test_parse_action_command_allows_curriculum_sft_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_CURRICULUM_WORK_DIR=data/curriculum/run_001 "
        "STAGE5_CURRICULUM_MIN_MODE_ROWS=direct=64,deep_narrow=64 "
        "STAGE5_CURRICULUM_PHASE1_STEPS=150 "
        "python colab/run_stage5_curriculum_sft.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {
        "STAGE5_CURRICULUM_WORK_DIR": "data/curriculum/run_001",
        "STAGE5_CURRICULUM_MIN_MODE_ROWS": "direct=64,deep_narrow=64",
        "STAGE5_CURRICULUM_PHASE1_STEPS": "150",
    }
    assert parsed.argv == [sys.executable, "colab/run_stage5_curriculum_sft.py"]


def test_parse_action_command_allows_curriculum_sft_gate_checker() -> None:
    parsed = parse_action_command(
        "python training/check_curriculum_sft_gate.py "
        "--summary_json data/curriculum/run_001/summary.json "
        "--output_json data/curriculum/run_001/curriculum_sft_gate.json "
        "--min_mode_rows direct=64,deep_narrow=64 "
        "--fail_on_no_go"
    )

    assert parsed.kind == "python"
    assert parsed.env == {}
    assert parsed.argv == [
        sys.executable,
        "training/check_curriculum_sft_gate.py",
        "--summary_json",
        "data/curriculum/run_001/summary.json",
        "--output_json",
        "data/curriculum/run_001/curriculum_sft_gate.json",
        "--min_mode_rows",
        "direct=64,deep_narrow=64",
        "--fail_on_no_go",
    ]


def test_parse_action_command_allows_programmatic_curriculum_pipeline() -> None:
    parsed = parse_action_command(
        "python training/run_programmatic_curriculum_pipeline.py "
        "--work_dir data/curriculum/programmatic_direct_deep_001 "
        "--num_direct 64 "
        "--num_deep_narrow 64"
    )

    assert parsed.kind == "python"
    assert parsed.env == {}
    assert parsed.argv == [
        sys.executable,
        "training/run_programmatic_curriculum_pipeline.py",
        "--work_dir",
        "data/curriculum/programmatic_direct_deep_001",
        "--num_direct",
        "64",
        "--num_deep_narrow",
        "64",
    ]


def test_parse_action_command_allows_programmatic_depth_assessor() -> None:
    parsed = parse_action_command(
        "STAGE5_PROGRAMMATIC_DEPTH_ASSESS_RUN_ID=assess python colab/assess_stage5_programmatic_depth_repair.py --summary_json outputs/stage5/run/summary.json"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_PROGRAMMATIC_DEPTH_ASSESS_RUN_ID": "assess"}
    assert parsed.argv == [
        sys.executable,
        "colab/assess_stage5_programmatic_depth_repair.py",
        "--summary_json",
        "outputs/stage5/run/summary.json",
    ]


def test_parse_action_command_allows_reasoning_dataset_audit_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_DATASET_AUDIT_RUN_ID=audit python colab/run_stage5_reasoning_dataset_audit.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_DATASET_AUDIT_RUN_ID": "audit"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_reasoning_dataset_audit.py"]


def test_parse_action_command_allows_reasoning_dataset_pipeline_runner() -> None:
    parsed = parse_action_command(
        "STAGE5_REASONING_DATASET_PIPELINE_RUN_ID=pipe python colab/run_stage5_reasoning_dataset_pipeline.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE5_REASONING_DATASET_PIPELINE_RUN_ID": "pipe"}
    assert parsed.argv == [sys.executable, "colab/run_stage5_reasoning_dataset_pipeline.py"]


def test_parse_action_command_allows_stage4_opus_finetune_runner() -> None:
    parsed = parse_action_command(
        "STAGE4_RUN_ID=opus OPUS_DATASET_ADAPTER=qwen_text python colab/run_stage4_opus_finetune.py"
    )

    assert parsed.kind == "python"
    assert parsed.env == {"STAGE4_RUN_ID": "opus", "OPUS_DATASET_ADAPTER": "qwen_text"}
    assert parsed.argv == [sys.executable, "colab/run_stage4_opus_finetune.py"]


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


def test_parse_action_command_allows_distill_sft_gate_runner() -> None:
    parsed = parse_action_command("python colab/run_stage5_arc_agi_distill_sft_gate.py")

    assert parsed.argv == [sys.executable, "colab/run_stage5_arc_agi_distill_sft_gate.py"]


def test_paid_gpu_arc_runners_are_a100_guarded() -> None:
    expected = {
        "colab/run_stage5_arc_agi_candidate_gate.py",
        "colab/run_stage5_arc_agi_trace_sft_gate.py",
        "colab/run_stage5_arc_agi_distill_sft_gate.py",
        "colab/run_stage5_arc_agi_dense_sft.py",
        "colab/run_stage5_arc_agi_sft.py",
        "colab/run_stage5_arc_agi_candidate_distill_gate.py",
        "colab/run_stage5_arc_agi_curriculum_particle_autopilot.py",
        "colab/run_stage5_arc_agi_autopilot_followup.py",
        "colab/run_stage5_arc_agi_recovered_benchmark.py",
        "colab/run_stage5_arc_agi_recovery_particle_gate.py",
        "colab/run_stage5_arc_agi_tta_sweep.py",
        "colab/run_stage5_phase1_recovery_ladder.py",
        "colab/run_stage5_recovered_phase1_arc_gate.py",
        "colab/run_stage5_recovered_phase1_particle_arc_gate.py",
        "colab/run_stage5_recovered_phase2_smoke.py",
    }

    assert expected <= module.GUARDED_A100_SCRIPTS


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


def test_a100_guard_allows_bounded_arc_mix_proxy(tmp_path) -> None:
    checkpoint = tmp_path / "outputs" / "stage5" / "run" / "phase1" / "phase1_step_1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test")
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_recovery_full_assessment", '
        '"status": "needs_competence_recovery", '
        f'"selected_checkpoint": "{checkpoint.as_posix()}"'
        "}",
        encoding="utf-8",
    )
    action = {
        "name": "Run another competence-preserving ARC-mix proxy gate",
        "command": "python colab/run_stage5_balanced_arc_mix_gate.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is True
    assert guard["status"] == "go_bounded_proxy"
    assert guard["checkpoint_preflight"]["available"] is True


def test_a100_guard_blocks_full_assessment_after_calibration_warning(tmp_path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_balanced_arc_mix_gate", "status": "proxy_lift_calibration_warning", "passed": false}',
        encoding="utf-8",
    )
    action = {
        "name": "Run full balanced assessment for ARC-mix checkpoint",
        "command": "python colab/run_stage5_recovery_full_assessment.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is False
    assert guard["status"] == "calibration_warning_no_go"


def test_a100_guard_skips_local_assessment_actions() -> None:
    action = {"name": "Plan", "command": "python colab/plan_stage5_next_run.py"}
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": None}, parsed)

    assert guard["checked"] is False
    assert guard["allowed"] is True


def test_a100_guard_allows_bootstrap_candidate_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.checkpoint_availability_for_path",
        lambda checkpoint: {
            "checkpoint": str(checkpoint),
            "available": True,
            "exists": True,
            "drive_candidate_exists": False,
        },
    )
    action = {
        "name": "Run Stage 5 ARC-AGI candidate gate",
        "command": "python colab/run_stage5_arc_agi_candidate_gate.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_kind": "bootstrap", "source_summary": None}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is True
    assert guard["status"] == "go_arc_agi_candidate_gate"
    assert guard["spend_class"] == "bounded_arc_agi_candidate_gate"


def test_a100_guard_blocks_guarded_arc_runner_without_readable_source() -> None:
    action = {
        "name": "Run dense ARC-AGI SFT control",
        "command": "python colab/run_stage5_arc_agi_dense_sft.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": None}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is False
    assert guard["status"] == "missing_source_summary"


def test_a100_guard_allows_bounded_routing_diagnostic(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.default_recovered_checkpoint_availability",
        lambda: {
            "checkpoint": "outputs/stage5/recovered/phase1/phase1_step_125.pt",
            "available": True,
            "exists": True,
            "drive_candidate_exists": False,
        },
    )
    checkpoint = tmp_path / "outputs" / "stage5" / "run" / "phase1" / "phase1_step_1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test")
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_balanced_arc_mix_gate", '
        '"status": "no_proxy_lift", '
        f'"selected_checkpoint": "{checkpoint.as_posix()}"'
        "}",
        encoding="utf-8",
    )
    action = {
        "name": "Run bounded depth/width routing diagnostic",
        "command": "python colab/run_stage5_routing_diagnostic.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is True
    assert guard["status"] == "go_routing_diagnostic"
    assert guard["spend_class"] == "bounded_routing_diagnostic"


def test_a100_guard_blocks_routing_diagnostic_when_recovered_checkpoint_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.default_recovered_checkpoint_availability",
        lambda: {
            "checkpoint": "outputs/stage5/recovered/phase1/phase1_step_125.pt",
            "available": False,
            "exists": False,
            "drive_candidate_exists": False,
        },
    )
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_balanced_arc_mix_gate", '
        '"status": "no_proxy_lift"}',
        encoding="utf-8",
    )
    action = {
        "name": "Run bounded depth/width routing diagnostic",
        "command": "python colab/run_stage5_routing_diagnostic.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is False
    assert guard["status"] == "routing_checkpoint_missing_no_go"


def test_a100_guard_allows_routing_diagnostic_after_calibration_warning(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.default_recovered_checkpoint_availability",
        lambda: {
            "checkpoint": "outputs/stage5/recovered/phase1/phase1_step_125.pt",
            "available": True,
            "exists": False,
            "drive_candidate_exists": True,
        },
    )
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_balanced_arc_mix_gate", '
        '"status": "no_proxy_lift", '
        '"best_arm": {"best_checkpoint": {"comparison_to_base": {"calibration_ok": false}}}}',
        encoding="utf-8",
    )
    action = {
        "name": "Run bounded depth/width routing diagnostic",
        "command": "python colab/run_stage5_routing_diagnostic.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is True
    assert guard["status"] == "go_routing_diagnostic"
    assert guard["checkpoint_preflight"]["available"] is True


def test_a100_guard_allows_routing_repair_for_repair_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.routing_repair_checkpoint_availability",
        lambda source_payload=None: {
            "checkpoint": "outputs/stage5/recovered/phase1/phase1_step_125.pt",
            "available": True,
            "exists": False,
            "drive_candidate_exists": True,
        },
    )
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_routing_diagnostic_assessment", "status": "needs_direct_halting_repair"}',
        encoding="utf-8",
    )
    action = {
        "name": "Run bounded direct-mode halting Phase 1 repair",
        "command": "python colab/run_stage5_routing_repair.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is True
    assert guard["status"] == "go_routing_repair"
    assert guard["spend_class"] == "bounded_routing_repair"


def test_a100_guard_blocks_routing_repair_when_no_repair_needed(tmp_path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_routing_diagnostic_assessment", "status": "routing_diagnostic_pass"}',
        encoding="utf-8",
    )
    action = {
        "name": "Run bounded direct-mode halting Phase 1 repair",
        "command": "python colab/run_stage5_routing_repair.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is False
    assert guard["status"] == "routing_repair_blocked"


def test_a100_guard_allows_stage4_opus_after_promoted_audit(tmp_path) -> None:
    source = tmp_path / "audit" / "summary.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
                {
                    "kind": "stage5_reasoning_dataset_audit",
                    "recommendations": [
                        {
                            "status": "promote_to_small_train_mix",
                            "key": "opus47_sft",
                            "converted_rows": 900,
                            "conversion_rate": 0.9,
                        }
                    ],
                }
            ),
        encoding="utf-8",
    )
    action = {
        "name": "Run audited modified-Opus recurrent fine-tune",
        "command": "python colab/run_stage4_opus_finetune.py",
    }
    parsed = parse_action_command(action["command"])

    guard = a100_execution_guard(action, {"source_summary": str(source)}, parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is True
    assert guard["status"] == "go_stage4_opus_finetune"
    assert guard["checkpoint_preflight"]["available"] is True


def test_local_only_guard_blocks_dataset_audit_on_gpu(monkeypatch) -> None:
    parsed = parse_action_command("python colab/run_stage5_reasoning_dataset_audit.py")
    monkeypatch.setattr(module, "gpu_runtime_attached", lambda: True)
    monkeypatch.setenv("STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_LOCAL_ONLY_ON_GPU", "0")

    guard = local_only_runtime_guard(parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is False
    assert guard["status"] == "local_only_gpu_no_go"


def test_local_only_guard_allows_dataset_audit_without_gpu(monkeypatch) -> None:
    parsed = parse_action_command("python colab/run_stage5_reasoning_dataset_audit.py")
    monkeypatch.setattr(module, "gpu_runtime_attached", lambda: False)

    guard = local_only_runtime_guard(parsed)

    assert guard["checked"] is True
    assert guard["allowed"] is True
    assert guard["status"] == "local_only_no_gpu_ok"


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


def test_execute_action_loop_blocks_guarded_a100_action(monkeypatch, tmp_path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        '{"kind": "stage5_balanced_arc_mix_gate", "status": "proxy_lift_calibration_warning", "passed": false}',
        encoding="utf-8",
    )
    action = {
        "name": "Run full balanced assessment for ARC-mix checkpoint",
        "command": "python colab/run_stage5_recovery_full_assessment.py",
    }

    def fake_run_planner(*, step=None):
        return tmp_path / "plan.json", {"source_summary": str(source), "actions": [action]}, subprocess.CompletedProcess([], 0)

    def fake_execute(parsed, *, log_name="selected_action.log"):
        raise AssertionError("guarded A100 action should not execute")

    monkeypatch.setattr(module, "run_planner", fake_run_planner)
    monkeypatch.setattr(module, "execute_parsed_command", fake_execute)

    steps = execute_action_loop(execute=True, max_actions=1)

    assert len(steps) == 1
    assert steps[0]["a100_guard"]["status"] == "calibration_warning_no_go"
    assert steps[0]["execution"]["executed"] is False
    assert steps[0]["execution"]["stopped"] is True


def test_execute_action_loop_blocks_local_only_action_on_gpu(monkeypatch, tmp_path) -> None:
    action = {
        "name": "Run reasoning dataset audit",
        "command": "python colab/run_stage5_reasoning_dataset_audit.py",
    }

    def fake_run_planner(*, step=None):
        return tmp_path / "plan.json", {"source_summary": None, "actions": [action]}, subprocess.CompletedProcess([], 0)

    def fake_execute(parsed, *, log_name="selected_action.log"):
        raise AssertionError("local-only action should not execute on a GPU runtime")

    monkeypatch.setattr(module, "run_planner", fake_run_planner)
    monkeypatch.setattr(module, "execute_parsed_command", fake_execute)
    monkeypatch.setattr(module, "gpu_runtime_attached", lambda: True)

    steps = execute_action_loop(execute=True, max_actions=1)

    assert len(steps) == 1
    assert steps[0]["local_runtime_guard"]["status"] == "local_only_gpu_no_go"
    assert steps[0]["execution"]["executed"] is False
    assert steps[0]["execution"]["stopped"] is True


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
