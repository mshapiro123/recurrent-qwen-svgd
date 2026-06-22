from __future__ import annotations

from colab.check_stage5_a100_go_no_go import (
    apply_checkpoint_guard,
    build_payload,
    checkpoint_from_payload,
    classify_action,
    command_env_assignments,
    command_script,
    curriculum_sft_checkpoint_availability,
    curriculum_sft_input_availability,
    normalize_min_mode_rows,
    promoted_stage4_opus_sources,
    routing_repair_profile_preflight,
    routing_repair_checkpoint_availability,
    source_has_calibration_warning,
    source_is_clean_full_confirmation_proxy,
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


def test_routing_repair_proxy_pass_allows_full_confirmation() -> None:
    source_payload = {
        "kind": "stage5_routing_repair",
        "status": "repair_proxy_lift",
        "passed": True,
        "proxy_alignment": {"ok": True},
    }

    assert source_is_clean_full_confirmation_proxy(source_payload) is True
    decision = classify_action(
        {
            "name": "Run full balanced assessment for routing-repair checkpoint",
            "command": "STAGE5_X=1 python colab/run_stage5_recovery_full_assessment.py",
        },
        source_payload=source_payload,
    )

    assert decision["go"] is True
    assert decision["status"] == "go_full_confirmation"


def test_routing_repair_proxy_misalignment_blocks_full_confirmation() -> None:
    source_payload = {
        "kind": "stage5_routing_repair",
        "status": "repair_proxy_lift",
        "passed": True,
        "proxy_alignment": {"ok": False},
    }

    assert source_is_clean_full_confirmation_proxy(source_payload) is False
    decision = classify_action(
        {
            "name": "Run full balanced assessment for routing-repair checkpoint",
            "command": "STAGE5_X=1 python colab/run_stage5_recovery_full_assessment.py",
        },
        source_payload=source_payload,
    )

    assert decision["go"] is False
    assert decision["status"] == "full_assessment_blocked"


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
        lambda source_payload=None: {
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


def test_routing_repair_profile_preflight_exposes_expected_proxy() -> None:
    status = routing_repair_profile_preflight({"status": "needs_direct_halting_repair"})

    assert status["checked"] is True
    assert status["repair_mode"] == "direct_halting"
    assert status["expected_arc_eval_config"] == "ARC-Easy"
    assert status["arms"] == "arc_mix_response_w02_lr2e6"


def test_build_payload_includes_routing_repair_profile(monkeypatch, tmp_path) -> None:
    import colab.check_stage5_a100_go_no_go as module

    source = tmp_path / "summary.json"
    source.write_text('{"status": "needs_direct_halting_repair"}', encoding="utf-8")
    monkeypatch.setattr(
        module,
        "plan_next_actions",
        lambda payload, source_summary: [
            {"name": "Run routing repair", "command": "python colab/run_stage5_routing_repair.py"}
        ],
    )
    monkeypatch.setattr(
        module,
        "apply_checkpoint_guard",
        lambda decision, source_payload: (decision, {"available": True}),
    )

    payload = build_payload(source)

    assert payload["routing_repair_profile"]["repair_mode"] == "direct_halting"
    assert payload["routing_repair_profile"]["expected_arc_eval_config"] == "ARC-Easy"


def test_routing_repair_checkpoint_preflight_allows_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "colab.check_stage5_a100_go_no_go.routing_repair_checkpoint_availability",
        lambda source_payload=None: {
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


def test_programmatic_depth_repair_uses_source_checkpoint_preflight(tmp_path) -> None:
    resume = tmp_path / "outputs" / "stage5" / "repair" / "phase1" / "phase1_step_50.pt"
    resume.parent.mkdir(parents=True)
    resume.write_bytes(b"checkpoint")

    decision = classify_action(
        {
            "name": "Run constructed depth repair",
            "command": "python colab/run_stage5_programmatic_depth_repair.py",
        },
        source_payload={"status": "needs_direct_halting_repair"},
    )
    guarded, checkpoint = apply_checkpoint_guard(
        decision,
        source_payload={
            "status": "needs_direct_halting_repair",
            "selected_checkpoint": str(resume),
        },
    )

    assert checkpoint["available"] is True
    assert checkpoint["checkpoint"].endswith("phase1/phase1_step_50.pt")
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
        "command": (
            "STAGE5_CURRICULUM_MIN_MODE_ROWS=direct=1000,deep_narrow=1000 "
            "python colab/run_stage5_curriculum_sft.py"
        ),
    }

    blocked = classify_action(action, source_payload={"kind": "curriculum_sft_gate", "go": False})
    allowed = classify_action(action, source_payload={"kind": "curriculum_sft_gate", "go": True})

    assert blocked["go"] is False
    assert blocked["status"] == "curriculum_sft_blocked"
    assert allowed["go"] is True
    assert allowed["status"] == "go_curriculum_sft"
    assert allowed["spend_class"] == "bounded_curriculum_sft"


def test_curriculum_sft_blocks_missing_mode_gate() -> None:
    action = {
        "name": "Run generated curriculum SFT",
        "command": "python colab/run_stage5_curriculum_sft.py",
    }

    decision = classify_action(action, source_payload={"kind": "curriculum_sft_gate", "go": True})

    assert decision["go"] is False
    assert decision["status"] == "curriculum_sft_mode_gate_mismatch"
    assert "STAGE5_CURRICULUM_MIN_MODE_ROWS" in decision["reason"]


def test_curriculum_sft_requires_gate_specific_mode_rows() -> None:
    source_payload = {
        "kind": "curriculum_sft_gate",
        "go": True,
        "checks": {
            "positive_sft": {
                "mode_requirements": {
                    "wide": {"required": 64, "observed": 80, "passed": True},
                }
            }
        },
    }
    wrong = {
        "name": "Run generated curriculum SFT",
        "command": (
            "STAGE5_CURRICULUM_MIN_MODE_ROWS=direct=1000,deep_narrow=1000 "
            "python colab/run_stage5_curriculum_sft.py"
        ),
    }
    right = {
        "name": "Run generated curriculum SFT",
        "command": "STAGE5_CURRICULUM_MIN_MODE_ROWS=wide=64 python colab/run_stage5_curriculum_sft.py",
    }

    blocked = classify_action(wrong, source_payload=source_payload)
    allowed = classify_action(right, source_payload=source_payload)

    assert blocked["go"] is False
    assert blocked["status"] == "curriculum_sft_mode_gate_mismatch"
    assert allowed["go"] is True
    assert allowed["status"] == "go_curriculum_sft"


def test_curriculum_mode_row_helpers_normalize_command_env() -> None:
    command = (
        "STAGE5_CURRICULUM_MIN_MODE_ROWS=deep_narrow=8,direct=16 "
        "python colab/run_stage5_curriculum_sft.py"
    )

    assert command_env_assignments(command)["STAGE5_CURRICULUM_MIN_MODE_ROWS"] == "deep_narrow=8,direct=16"
    assert normalize_min_mode_rows("direct=16,deep_narrow=8") == "deep_narrow=8,direct=16"


def test_candidate_gate_allowed_only_for_bootstrap_source() -> None:
    action = {
        "name": "Run Stage 5 ARC-AGI candidate gate",
        "command": "python colab/run_stage5_arc_agi_candidate_gate.py",
    }

    allowed = classify_action(action, source_payload={"source_kind": "bootstrap"})
    blocked = classify_action(action, source_payload={"kind": "stage5_reasoning_dataset_audit"})

    assert allowed["go"] is True
    assert allowed["status"] == "go_arc_agi_candidate_gate"
    assert allowed["spend_class"] == "bounded_arc_agi_candidate_gate"
    assert allowed["checkpoint"].endswith("outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt")
    assert blocked["go"] is False
    assert blocked["status"] == "candidate_gate_blocked"


def test_arc_gate_chain_classifies_by_source_kind() -> None:
    trace = classify_action(
        {"name": "Compare trace targets", "command": "python colab/run_stage5_arc_agi_trace_sft_gate.py"},
        source_payload={"kind": "stage5_arc_agi_candidate_gate"},
    )
    distill = classify_action(
        {"name": "Compare distillation", "command": "python colab/run_stage5_arc_agi_distill_sft_gate.py"},
        source_payload={"kind": "trace_sft_gate"},
    )
    dense = classify_action(
        {"name": "Run dense control", "command": "python colab/run_stage5_arc_agi_dense_sft.py"},
        source_payload={"kind": "trace_sft_gate"},
    )
    recurrent = classify_action(
        {"name": "Run matched recurrent", "command": "python colab/run_stage5_arc_agi_sft.py"},
        source_payload={"kind": "dense_sft_control"},
    )

    assert trace["status"] == "go_trace_sft_gate"
    assert distill["status"] == "go_distill_sft_gate"
    assert dense["status"] == "go_dense_arc_sft_control"
    assert recurrent["status"] == "go_matched_recurrent_arc_sft"


def test_arc_gate_chain_blocks_wrong_source_kind() -> None:
    action = {"name": "Run matched recurrent", "command": "python colab/run_stage5_arc_agi_sft.py"}

    decision = classify_action(action, source_payload={"kind": "stage5_arc_agi_candidate_gate"})

    assert decision["go"] is False
    assert decision["status"] == "matched_recurrent_arc_sft_blocked"


def test_candidate_distill_and_autopilot_are_guarded_paid_actions() -> None:
    candidate = classify_action(
        {
            "name": "Run candidate-distill diagnostic",
            "command": "python colab/run_stage5_arc_agi_candidate_distill_gate.py",
        },
        source_payload={"recovered_benchmark": {}, "selected_checkpoint": "outputs/stage5/run/phase1/phase1.pt"},
    )
    autopilot = classify_action(
        {
            "name": "Scale deterministic curriculum",
            "command": "python colab/run_stage5_arc_agi_curriculum_particle_autopilot.py",
        },
        source_payload={"recovered_benchmark": {}, "selected_checkpoint": "outputs/stage5/run/phase1/phase1.pt"},
    )

    assert candidate["go"] is True
    assert candidate["status"] == "go_candidate_distill_gate"
    assert autopilot["go"] is True
    assert autopilot["status"] == "go_curriculum_particle_autopilot"


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
    assert checkpoint["input_preflight"]["available"] is True


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


def test_curriculum_sft_input_preflight_blocks_missing_work_dir(monkeypatch, tmp_path) -> None:
    work_dir = tmp_path / "data" / "curriculum" / "run_001"
    source_payload = {
        "kind": "curriculum_sft_gate",
        "go": True,
        "work_dir": str(work_dir),
        "summary_json": str(work_dir / "summary.json"),
        "artifacts": {"positive_sft": str(work_dir / "positive_sft.jsonl")},
    }
    monkeypatch.setenv("STAGE5_CURRICULUM_INPUT_BACKUP_DIR", str(tmp_path / "drive" / "curriculum_runs"))

    decision, preflight = apply_checkpoint_guard(
        {
            "go": True,
            "status": "go_curriculum_sft",
            "spend_class": "bounded_curriculum_sft",
        },
        source_payload=source_payload,
    )

    assert preflight["available"] is False
    assert preflight["input_preflight"]["available"] is False
    assert decision["go"] is False
    assert decision["status"] == "curriculum_input_missing_no_go"


def test_curriculum_sft_input_preflight_allows_drive_backup(monkeypatch, tmp_path) -> None:
    work_dir = tmp_path / "workspace" / "data" / "curriculum" / "run_001"
    backup_root = tmp_path / "drive" / "curriculum_runs"
    backup = backup_root / "run_001"
    backup.mkdir(parents=True)
    (backup / "summary.json").write_text("{}", encoding="utf-8")
    (backup / "positive_sft.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("STAGE5_CURRICULUM_INPUT_BACKUP_DIR", str(backup_root))

    status = curriculum_sft_input_availability(
        {
            "kind": "curriculum_sft_gate",
            "go": True,
            "work_dir": str(work_dir),
            "summary_json": str(work_dir / "summary.json"),
            "artifacts": {"positive_sft": str(work_dir / "positive_sft.jsonl")},
        }
    )

    assert status["available"] is True
    assert status["local_available"] is False
    assert status["drive_candidate_exists"] is True


def test_curriculum_sft_input_preflight_allows_published_gate_drive_backup(monkeypatch, tmp_path) -> None:
    import colab.check_stage5_a100_go_no_go as guard

    repo_root = tmp_path / "repo"
    backup_root = tmp_path / "drive" / "curriculum_runs"
    backup = backup_root / "programmatic_direct_deep_001"
    backup.mkdir(parents=True)
    (backup / "summary.json").write_text("{}", encoding="utf-8")
    (backup / "positive_sft.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(guard, "ROOT", repo_root)
    monkeypatch.setenv("STAGE5_CURRICULUM_INPUT_BACKUP_DIR", str(backup_root))

    status = guard.curriculum_sft_input_availability(
        {
            "kind": "curriculum_sft_gate",
            "go": True,
            "work_dir": "data/curriculum/programmatic_direct_deep_001",
            "summary_json": "data/curriculum/programmatic_direct_deep_001/summary.json",
            "artifacts": {
                "positive_sft": "data/curriculum/programmatic_direct_deep_001/positive_sft.jsonl"
            },
        }
    )

    assert status["available"] is True
    assert status["local_available"] is False
    assert status["drive_candidate_exists"] is True
    assert status["first_existing_drive_candidate"].endswith("programmatic_direct_deep_001")


def test_curriculum_sft_input_preflight_allows_local_artifacts(tmp_path) -> None:
    work_dir = tmp_path / "data" / "curriculum" / "run_001"
    work_dir.mkdir(parents=True)
    (work_dir / "summary.json").write_text("{}", encoding="utf-8")
    (work_dir / "positive_sft.jsonl").write_text("{}\n", encoding="utf-8")

    status = curriculum_sft_input_availability(
        {
            "kind": "curriculum_sft_gate",
            "go": True,
            "work_dir": str(work_dir),
            "summary_json": str(work_dir / "summary.json"),
            "artifacts": {"positive_sft": str(work_dir / "positive_sft.jsonl")},
        }
    )

    assert status["available"] is True
    assert status["local_available"] is True


def test_routing_checkpoint_availability_reports_default_checkpoint() -> None:
    status = routing_repair_checkpoint_availability()

    assert status["checkpoint"].endswith("/phase1/phase1_step_125.pt")
    assert "Routing diagnostic/repair requires" in status["reason"]


def test_routing_checkpoint_availability_uses_benchmark_summary(tmp_path) -> None:
    checkpoint = tmp_path / "outputs" / "stage5" / "bench" / "phase1" / "phase1_step_75.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    benchmark = tmp_path / "benchmark_summary.json"
    benchmark.write_text(
        f'{{"kind": "stage5_benchmark_suite", "checkpoint": "{checkpoint.as_posix()}"}}',
        encoding="utf-8",
    )

    status = routing_repair_checkpoint_availability({"benchmark_summary": str(benchmark)})

    assert status["available"] is True
    assert status["source"] == "benchmark_summary"
    assert status["checkpoint"].endswith("phase1/phase1_step_75.pt")


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
