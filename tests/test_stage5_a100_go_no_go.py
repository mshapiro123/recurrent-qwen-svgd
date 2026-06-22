from __future__ import annotations

from colab.check_stage5_a100_go_no_go import (
    apply_checkpoint_guard,
    checkpoint_from_payload,
    classify_action,
    command_script,
    curriculum_sft_checkpoint_availability,
    promoted_stage4_opus_sources,
    routing_repair_checkpoint_availability,
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


def test_decision_calibration_repair_blocks_a100() -> None:
    assert source_has_calibration_warning({"decision": "stop_for_calibration_repair"}) is True


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


def test_checkpoint_from_payload_uses_phase1_checkpoint() -> None:
    payload = {"phase1_checkpoint": "outputs/stage5/run/phase1/phase1_step_150.pt"}

    assert checkpoint_from_payload(payload) == "outputs/stage5/run/phase1/phase1_step_150.pt"


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


def test_routing_repair_checkpoint_preflight_can_block(monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.routing_repair_checkpoint_availability",
        lambda: {
            "checkpoint": "outputs/stage5/recovered/phase1/phase1_step_125.pt",
            "available": False,
            "exists": False,
            "drive_candidate_exists": False,
        },
    )

    decision, checkpoint = apply_checkpoint_guard(
        {"go": True, "status": "go_routing_repair", "spend_class": "bounded_routing_repair"},
        source_payload={"status": "needs_direct_halting_repair"},
    )

    assert checkpoint["available"] is False
    assert decision["go"] is False
    assert decision["status"] == "routing_checkpoint_missing_no_go"


def test_routing_repair_checkpoint_preflight_allows_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.routing_repair_checkpoint_availability",
        lambda: {
            "checkpoint": "outputs/stage5/recovered/phase1/phase1_step_125.pt",
            "available": True,
            "exists": True,
            "drive_candidate_exists": False,
        },
    )

    decision, checkpoint = apply_checkpoint_guard(
        {"go": True, "status": "go_routing_repair", "spend_class": "bounded_routing_repair"},
        source_payload={"status": "needs_direct_halting_repair"},
    )

    assert checkpoint["available"] is True
    assert decision["go"] is True
    assert decision["status"] == "go_routing_repair"


def test_programmatic_depth_repair_uses_recovered_checkpoint_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.routing_repair_checkpoint_availability",
        lambda: {
            "checkpoint": "outputs/stage5/recovered/phase1/phase1_step_125.pt",
            "available": True,
            "exists": True,
            "drive_candidate_exists": False,
        },
    )

    decision = classify_action(
        {
            "name": "Run constructed depth repair",
            "command": "python colab/run_stage5_programmatic_depth_repair.py",
        },
        source_payload={"status": "needs_direct_halting_repair"},
    )
    guarded, checkpoint = apply_checkpoint_guard(
        decision,
        source_payload={"status": "needs_direct_halting_repair"},
    )

    assert checkpoint["available"] is True
    assert guarded["go"] is True
    assert guarded["status"] == "go_programmatic_depth_repair"
    assert guarded["spend_class"] == "bounded_programmatic_depth_repair"


def test_programmatic_depth_repair_blocks_when_no_repair_needed() -> None:
    decision = classify_action(
        {
            "name": "Run constructed depth repair",
            "command": "python colab/run_stage5_programmatic_depth_repair.py",
        },
        source_payload={"status": "proxy_lift", "passed": True},
    )

    assert decision["go"] is False
    assert decision["status"] == "programmatic_depth_repair_blocked"


def test_stage4_opus_finetune_allowed_only_after_dataset_promotion() -> None:
    action = {
        "name": "Run audited modified-Opus recurrent fine-tune",
        "command": "python colab/run_stage4_opus_finetune.py",
    }

    blocked = classify_action(action, source_payload={"kind": "stage5_reasoning_dataset_audit", "recommendations": []})
    allowed = classify_action(
        action,
        source_payload={
            "kind": "stage5_reasoning_dataset_audit",
            "recommendations": [
                {
                    "status": "promote_to_small_train_mix",
                    "key": "opus47_sft",
                    "converted_rows": 900,
                    "conversion_rate": 0.9,
                }
            ],
        },
    )

    assert blocked["go"] is False
    assert blocked["status"] == "stage4_opus_finetune_blocked"
    assert allowed["go"] is True
    assert allowed["status"] == "go_stage4_opus_finetune"
    assert allowed["spend_class"] == "bounded_stage4_opus_finetune"


def test_stage4_opus_finetune_blocks_unapproved_promoted_opus_source() -> None:
    action = {
        "name": "Run audited modified-Opus recurrent fine-tune",
        "command": "python colab/run_stage4_opus_finetune.py",
    }
    decision = classify_action(
        action,
        source_payload={
            "kind": "stage5_reasoning_dataset_audit",
            "recommendations": [
                {
                    "status": "promote_to_small_train_mix",
                    "key": "jackrong_opus47_trace_inversion",
                    "dataset_id": "Jackrong/Claude-opus-4.7-TraceInversion-5000x",
                    "converted_rows": 800,
                    "conversion_rate": 0.8,
                }
            ],
        },
    )

    assert promoted_stage4_opus_sources(
        {
            "recommendations": [
                {
                    "status": "promote_to_small_train_mix",
                    "key": "jackrong_opus47_trace_inversion",
                    "converted_rows": 800,
                    "conversion_rate": 0.8,
                }
            ]
        }
    ) == []
    assert decision["go"] is False
    assert decision["status"] == "stage4_opus_finetune_blocked"


def test_stage4_opus_finetune_checkpoint_guard_does_not_require_recovered_checkpoint() -> None:
    decision, checkpoint = apply_checkpoint_guard(
        {
            "go": True,
            "status": "go_stage4_opus_finetune",
            "spend_class": "bounded_stage4_opus_finetune",
        },
        source_payload={
            "kind": "stage5_reasoning_dataset_audit",
            "recommendations": [
                {
                    "status": "promote_to_small_train_mix",
                    "key": "opus47_sft",
                    "converted_rows": 900,
                    "conversion_rate": 0.9,
                }
            ],
        },
    )

    assert decision["go"] is True
    assert checkpoint["available"] is True
    assert checkpoint["checkpoint"] is None


def test_curriculum_sft_allowed_only_after_green_sft_gate() -> None:
    action = {
        "name": "Run generated curriculum SFT",
        "command": "python colab/run_stage5_curriculum_sft.py",
    }

    blocked = classify_action(action, source_payload={"kind": "curriculum_sft_gate", "go": False})
    allowed = classify_action(action, source_payload={"kind": "curriculum_sft_gate", "go": True})

    assert blocked["go"] is False
    assert blocked["status"] == "curriculum_sft_blocked"
    assert allowed["go"] is True
    assert allowed["status"] == "go_curriculum_sft"
    assert allowed["spend_class"] == "bounded_curriculum_sft"


def test_curriculum_sft_checkpoint_guard_without_resume_starts_from_base(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_CURRICULUM_RESUME_FROM", raising=False)

    decision, checkpoint = apply_checkpoint_guard(
        {
            "go": True,
            "status": "go_curriculum_sft",
            "spend_class": "bounded_curriculum_sft",
        },
        source_payload={"kind": "curriculum_sft_gate", "go": True},
    )

    assert decision["go"] is True
    assert checkpoint["available"] is True
    assert checkpoint["checkpoint"] is None
    assert "starts from the base model" in checkpoint["reason"]


def test_curriculum_sft_checkpoint_guard_blocks_missing_resume(monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_CURRICULUM_RESUME_FROM", "outputs/stage5/run/phase1/missing.pt")

    decision, checkpoint = apply_checkpoint_guard(
        {
            "go": True,
            "status": "go_curriculum_sft",
            "spend_class": "bounded_curriculum_sft",
        },
        source_payload={"kind": "curriculum_sft_gate", "go": True},
    )

    assert checkpoint["available"] is False
    assert checkpoint["checkpoint"].endswith("outputs/stage5/run/phase1/missing.pt")
    assert decision["go"] is False
    assert decision["status"] == "checkpoint_missing_no_go"


def test_curriculum_sft_checkpoint_availability_accepts_existing_resume(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "outputs" / "stage5" / "run" / "phase1" / "phase1_step_150.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("STAGE5_CURRICULUM_RESUME_FROM", str(checkpoint))

    status = curriculum_sft_checkpoint_availability()

    assert status["available"] is True
    assert status["exists"] is True


def test_routing_checkpoint_availability_reports_default_checkpoint() -> None:
    status = routing_repair_checkpoint_availability()

    assert status["checkpoint"].endswith("/phase1/phase1_step_125.pt")
    assert "Routing diagnostic/repair requires" in status["reason"]


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
