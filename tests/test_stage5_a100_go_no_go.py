from __future__ import annotations

from colab.check_stage5_a100_go_no_go import (
    apply_checkpoint_guard,
    checkpoint_from_payload,
    classify_action,
    command_script,
    source_has_calibration_warning,
)


def test_command_script_extracts_python_runner() -> None:
    command = "FOO=bar python colab/run_stage5_balanced_arc_mix_gate.py --x"

    assert command_script(command) == "colab/run_stage5_balanced_arc_mix_gate.py"


def test_bounded_arc_mix_proxy_is_go() -> None:
    decision = classify_action(
        {
            "name": "Run another competence-preserving ARC-mix proxy gate",
            "command": "STAGE5_X=1 python colab/run_stage5_balanced_arc_mix_gate.py",
        },
        source_payload={"status": "needs_competence_recovery"},
    )

    assert decision["go"] is True
    assert decision["status"] == "go_bounded_proxy"
    assert decision["spend_class"] == "single_arc_mix_proxy"


def test_calibration_warning_blocks_a100() -> None:
    decision = classify_action(
        {
            "name": "Inspect ARC-mix proxy gate",
            "command": "cat outputs/stage5/run/summary.md",
        },
        source_payload={"status": "proxy_lift_calibration_warning", "passed": False},
    )

    assert decision["go"] is False
    assert decision["status"] == "calibration_warning_no_go"


def test_nested_calibration_warning_blocks_a100() -> None:
    payload = {
        "status": "proxy_lift",
        "best_arm": {
            "best_checkpoint": {
                "comparison_to_base": {"calibration_ok": False},
            },
        },
    }

    assert source_has_calibration_warning(payload) is True


def test_clean_proxy_pass_allows_full_confirmation() -> None:
    decision = classify_action(
        {
            "name": "Run full balanced assessment for ARC-mix checkpoint",
            "command": "STAGE5_X=1 python colab/run_stage5_recovery_full_assessment.py",
        },
        source_payload={"status": "proxy_lift", "passed": True},
    )

    assert decision["go"] is True
    assert decision["status"] == "go_full_confirmation"


def test_checkpoint_from_payload_uses_selected_checkpoint() -> None:
    payload = {"selected_checkpoint": "outputs/stage5/run/phase1/phase1_step_1.pt"}

    assert checkpoint_from_payload(payload) == "outputs/stage5/run/phase1/phase1_step_1.pt"


def test_checkpoint_from_payload_uses_nested_best_checkpoint() -> None:
    payload = {
        "best_arm": {
            "best_checkpoint": {"checkpoint": "outputs/stage5/run/phase1/phase1_step_2.pt"},
        },
    }

    assert checkpoint_from_payload(payload) == "outputs/stage5/run/phase1/phase1_step_2.pt"


def test_checkpoint_guard_blocks_go_without_checkpoint() -> None:
    decision, checkpoint = apply_checkpoint_guard(
        {"go": True, "status": "go_bounded_proxy", "spend_class": "single_arc_mix_proxy"},
        source_payload={"status": "needs_competence_recovery"},
    )

    assert checkpoint["available"] is False
    assert decision["go"] is False
    assert decision["status"] == "checkpoint_missing_no_go"


def test_unpassed_proxy_blocks_full_confirmation() -> None:
    decision = classify_action(
        {
            "name": "Run full balanced assessment for ARC-mix checkpoint",
            "command": "STAGE5_X=1 python colab/run_stage5_recovery_full_assessment.py",
        },
        source_payload={"status": "proxy_lift", "passed": False},
    )

    assert decision["go"] is False
    assert decision["status"] == "full_assessment_blocked"


def test_inspection_action_is_no_gpu() -> None:
    decision = classify_action(
        {
            "name": "Inspect ARC-mix proxy gate",
            "command": "cat outputs/stage5/run/summary.md",
        },
        source_payload={"status": "no_proxy_lift"},
    )

    assert decision["go"] is False
    assert decision["status"] == "no_gpu_action"
