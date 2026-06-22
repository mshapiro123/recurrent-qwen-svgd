from __future__ import annotations

import json

import colab.plan_stage5_next_run as planner
from colab.plan_stage5_next_run import (
    best_recovered_tta_row,
    evidence_fragment,
    next_validation_limit,
    paired_delta_or_aggregate,
    paired_metric,
    plan_next_actions,
    recovery_particle_actions,
    selector_rescore_actions,
    source_kind,
)


def _summary(selected: int, best: int, examples: int = 50) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "examples_with_targets": examples,
        "valid_candidate_rate": 1.0,
    }


def _paired(delta: int, wins: int, losses: int, ties: int) -> dict[str, object]:
    return {
        "delta_exact": delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "bootstrap_delta_accuracy_ci95": {"low": -0.1, "high": 0.2},
    }


def _planner_readable_summary(path, *, status: str = "needs_direct_halting_repair") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"kind": "stage5_routing_diagnostic_assessment", "status": status}),
        encoding="utf-8",
    )


def test_resolve_source_summary_prefers_explicit_source_over_current_pointer(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "explicit" / "summary.json"
    current = tmp_path / "current" / "summary.json"
    _planner_readable_summary(explicit)
    _planner_readable_summary(current)
    pointer = tmp_path / "stage5_current_source_summary.txt"
    pointer.write_text(current.as_posix(), encoding="utf-8")

    monkeypatch.setattr(planner, "CURRENT_SOURCE_SUMMARY", "")
    monkeypatch.setattr(planner, "CURRENT_SOURCE_SUMMARY_FILE", pointer)

    assert planner.resolve_source_summary(explicit.as_posix()) == explicit


def test_resolve_source_summary_uses_current_pointer_before_mtime_latest(tmp_path, monkeypatch) -> None:
    current = tmp_path / "current" / "summary.json"
    latest = tmp_path / "outputs" / "stage5" / "newer_old_branch" / "summary.json"
    _planner_readable_summary(current)
    _planner_readable_summary(latest)
    pointer = tmp_path / "stage5_current_source_summary.txt"
    pointer.write_text(current.as_posix(), encoding="utf-8")

    monkeypatch.setattr(planner, "CURRENT_SOURCE_SUMMARY", "")
    monkeypatch.setattr(planner, "CURRENT_SOURCE_SUMMARY_FILE", pointer)
    monkeypatch.setattr(planner, "ROOT", tmp_path)

    assert planner.resolve_source_summary() == current


def test_configured_current_summary_fails_loudly_when_pointer_is_missing(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "missing" / "summary.json"
    pointer = tmp_path / "stage5_current_source_summary.txt"
    pointer.write_text(missing.as_posix(), encoding="utf-8")

    monkeypatch.setattr(planner, "CURRENT_SOURCE_SUMMARY", "")
    monkeypatch.setattr(planner, "CURRENT_SOURCE_SUMMARY_FILE", pointer)

    try:
        planner.configured_current_summary()
    except FileNotFoundError as exc:
        assert "Configured current Stage 5 source summary does not exist" in str(exc)
    else:
        raise AssertionError("missing current source pointer should fail loudly")


def test_committed_current_source_summary_points_at_valid_stage5_source() -> None:
    path = planner.configured_current_summary()

    assert path is not None
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert source_kind(payload) != "unknown"


def test_failed_candidate_distillation_recommends_baseline_curriculum(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "compact": {
            "candidate_distillation_passed": False,
            "final_checkpoint": None,
            "particle_passed": False,
        }
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run baseline curriculum without candidate distillation"
    assert "RUN_CANDIDATE_DISTILL_GATE=0" in actions[0]["command"]


def test_failed_arc_mix_recommends_routing_diagnostic(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "no_proxy_lift",
        "decision": "stop_and_revise_objective",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run bounded depth/width routing diagnostic"
    assert "python colab/run_stage5_routing_diagnostic.py" in actions[0]["command"]


def test_arc_mix_calibration_warning_recommends_routing_diagnostic(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "proxy_lift_calibration_warning",
        "decision": "stop_for_calibration_repair",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run bounded depth/width routing diagnostic"
    assert "direct-mode loop depth" in actions[0]["reason"]


def test_routing_diagnostic_direct_status_recommends_cpu_curriculum_gate(tmp_path) -> None:
    source = tmp_path / "routing" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_routing_diagnostic_assessment",
        "status": "needs_direct_halting_repair",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run CPU programmatic direct/deep curriculum gate"
    assert "python colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py" in actions[0]["command"]
    assert "1000 direct / 1000 deep-narrow" in actions[0]["reason"]


def test_green_curriculum_sft_gate_recommends_guarded_sft_runner(tmp_path) -> None:
    source = tmp_path / "curriculum_sft_gate.json"
    payload = {
        "kind": "curriculum_sft_gate",
        "go": True,
        "status": "go_train_recurrent_sft",
        "work_dir": "data/curriculum/run_001",
        "summary_json": "data/curriculum/run_001/summary.json",
        "checks": {
            "positive_sft": {
                "rows": 24,
                "mode_requirements": {
                    "direct": {"required": 16, "observed": 18, "passed": True},
                    "deep_narrow": {"required": 8, "observed": 9, "passed": True},
                },
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert source_kind(payload) == "curriculum_sft_gate"
    assert actions[0]["name"] == "Run generated curriculum recurrent SFT"
    assert "python colab/run_stage5_curriculum_sft.py" in actions[0]["command"]
    assert "STAGE5_CURRICULUM_WORK_DIR=data/curriculum/run_001" in actions[0]["command"]
    assert "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS=24" in actions[0]["command"]
    assert "STAGE5_CURRICULUM_MIN_MODE_ROWS=deep_narrow=8,direct=16" in actions[0]["command"]


def test_red_curriculum_sft_gate_recommends_inspection(tmp_path) -> None:
    source = tmp_path / "curriculum_sft_gate.json"
    payload = {
        "kind": "curriculum_sft_gate",
        "go": False,
        "status": "no_go",
        "issues": [{"code": "too_few_positive_sft_lines"}],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect generated curriculum SFT gate"
    assert actions[0]["command"].startswith("cat ")
    assert "run_stage5_curriculum_sft.py" not in actions[0]["command"]


def test_complete_curriculum_pipeline_recommends_no_gpu_sft_gate(tmp_path) -> None:
    source = tmp_path / "curriculum" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "curriculum_pipeline_from_artifacts",
        "status": "complete",
        "work_dir": "data/curriculum/run_001",
        "counts": {"positive_sft_rows": 24},
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert source_kind(payload) == "curriculum_pipeline"
    assert actions[0]["name"] == "Run generated curriculum SFT safety gate"
    assert "python training/check_curriculum_sft_gate.py" in actions[0]["command"]
    assert "--summary_json" in actions[0]["command"]
    assert "--min_mode_rows direct=1000,deep_narrow=1000" in actions[0]["command"]
    assert "curriculum_sft_gate.json" in actions[0]["command"]
    assert "run_stage5_curriculum_sft.py" not in actions[0]["command"]


def test_pending_curriculum_pipeline_recommends_inspection_not_gpu(tmp_path) -> None:
    source = tmp_path / "curriculum" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "curriculum_pipeline_from_artifacts",
        "status": "pending_method_or_perturbation_responses",
        "next_action": "Run provider responses for jobs_methods.jsonl",
        "pending_responses": [
            {
                "name": "method_solve",
                "expected_rows": 8,
                "existing_rows": 3,
                "remaining_rows": 5,
            },
            {
                "name": "perturbation",
                "expected_rows": 16,
                "existing_rows": 0,
                "remaining_rows": 16,
            },
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Continue generated curriculum responses: method_solve, perturbation"
    assert actions[0]["command"].startswith("cat ")
    assert "Remaining response rows: 21" in actions[0]["reason"]
    assert "method_solve: 5 remaining" in actions[0]["reason"]
    assert "run_stage5_curriculum_sft.py" not in actions[0]["command"]


def test_pending_curriculum_pipeline_without_structured_responses_recommends_inspection(tmp_path) -> None:
    source = tmp_path / "curriculum" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "curriculum_pipeline_from_artifacts",
        "status": "pending_seed_responses",
        "next_action": "Run provider responses for jobs_seed.jsonl",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect generated curriculum pipeline `pending_seed_responses`"
    assert actions[0]["command"].startswith("cat ")


def test_complete_capability_ladder_curriculum_recommends_observed_count_sft_gate(tmp_path) -> None:
    source = tmp_path / "capability_ladder" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "capability_ladder_curriculum_pipeline",
        "status": "complete",
        "counts": {
            "typed_records": 30,
            "positive_sft_rows": 30,
            "mode_counts": {"direct": 12, "deep_narrow": 18},
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert source_kind(payload) == "capability_ladder_curriculum"
    assert actions[0]["name"] == "Run capability-ladder SFT safety gate"
    assert "python training/check_curriculum_sft_gate.py" in actions[0]["command"]
    assert "--work_dir" in actions[0]["command"]
    assert "capability_ladder/curriculum_sft_gate.json" in actions[0]["command"].replace("\\", "/")
    assert "--min_positive_rows 30" in actions[0]["command"]
    assert "--min_mode_rows deep_narrow=18,direct=12" in actions[0]["command"]
    assert "run_stage5_curriculum_sft.py" not in actions[0]["command"]


def test_incomplete_capability_ladder_curriculum_recommends_inspection_not_gpu(tmp_path) -> None:
    source = tmp_path / "capability_ladder" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "capability_ladder_curriculum_pipeline",
        "status": "complete_with_skips_warning",
        "counts": {"typed_records": 0, "positive_sft_rows": 0},
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect capability-ladder curriculum `complete_with_skips_warning`"
    assert actions[0]["command"].startswith("cat ")
    assert "run_stage5_curriculum_sft.py" not in actions[0]["command"]


def test_routing_diagnostic_deep_status_recommends_cpu_curriculum_gate(tmp_path) -> None:
    source = tmp_path / "routing" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_routing_diagnostic_assessment",
        "status": "needs_deep_narrow_recovery",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run CPU programmatic direct/deep curriculum gate"
    assert "python colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py" in actions[0]["command"]
    assert "deep-narrow deterministic" in actions[0]["reason"]


def test_routing_diagnostic_pass_recommends_larger_confirmation(tmp_path) -> None:
    source = tmp_path / "routing" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_routing_diagnostic_assessment",
        "status": "routing_diagnostic_pass",
        "benchmark_summary": "outputs/stage5/routing/benchmark_run/summary.json",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run larger routing confirmation benchmark"
    assert "python colab/run_stage5_benchmark_suite.py" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_SOURCE_SUMMARY=outputs/stage5/routing/benchmark_run/summary.json" in actions[0]["command"]


def test_routing_repair_pass_recommends_full_assessment_from_wrapper_summary(tmp_path) -> None:
    source = tmp_path / "routing_repair" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_routing_repair",
        "status": "repair_proxy_lift",
        "passed": True,
        "arc_mix_summary": "outputs/stage5/repair_child/summary.json",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run full balanced assessment for routing-repair checkpoint"
    assert "python colab/run_stage5_recovery_full_assessment.py" in actions[0]["command"]
    assert f"STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY={source.as_posix()}" in actions[0]["command"]
    assert "outputs/stage5/repair_child/summary.json" not in actions[0]["command"]


def test_routing_repair_failure_recommends_programmatic_depth_repair(tmp_path) -> None:
    source = tmp_path / "routing_repair" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_routing_repair",
        "status": "repair_no_proxy_lift",
        "next_step": "Stop and revise direct-loop supervision.",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run bounded programmatic direct/deep repair"
    assert "python colab/run_stage5_programmatic_depth_repair.py" in actions[0]["command"]
    assert f"STAGE5_PROGRAMMATIC_SOURCE_SUMMARY={source.as_posix()}" in actions[0]["command"]
    assert actions[1]["name"] == "Inspect routing repair `repair_no_proxy_lift`"


def test_routing_repair_misaligned_proxy_recommends_inspection_only(tmp_path) -> None:
    source = tmp_path / "repair" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_routing_repair",
        "status": "repair_proxy_misaligned",
        "passed": False,
        "proxy_alignment": {"ok": False, "expected_arc_eval_config": "ARC-Easy", "actual_arc_eval_config": "ARC-Challenge"},
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect routing repair `repair_proxy_misaligned`"
    assert actions[0]["command"].startswith("cat ")
    assert "run_stage5_recovery_full_assessment.py" not in actions[0]["command"]
    assert "run_stage5_programmatic_depth_repair.py" not in actions[0]["command"]


def test_routing_repair_unexpected_pass_status_recommends_inspection_only(tmp_path) -> None:
    source = tmp_path / "repair" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_routing_repair",
        "status": "repair_unknown",
        "passed": True,
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect routing repair `repair_unknown`"
    assert actions[0]["command"].startswith("cat ")
    assert "run_stage5_recovery_full_assessment.py" not in actions[0]["command"]
    assert "run_stage5_programmatic_depth_repair.py" not in actions[0]["command"]


def test_programmatic_depth_repair_complete_recommends_no_gpu_assessment(tmp_path) -> None:
    source = tmp_path / "programmatic" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_programmatic_depth_repair",
        "status": "complete",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess programmatic direct/deep repair"
    assert "python colab/assess_stage5_programmatic_depth_repair.py" in actions[0]["command"]
    assert f"--summary_json {source.as_posix()}" in actions[0]["command"]


def test_programmatic_depth_assessment_pass_recommends_routing_diagnostic(tmp_path) -> None:
    source = tmp_path / "programmatic_assess" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_programmatic_depth_assessment",
        "status": "programmatic_depth_lift",
        "passed": True,
        "checkpoint": "outputs/stage5/programmatic/phase1/phase1_step_100.pt",
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run ARC routing diagnostic for programmatic-depth checkpoint"
    assert "python colab/run_stage5_routing_diagnostic.py" in actions[0]["command"]
    assert "STAGE5_RECOVERED_PHASE1_CHECKPOINT=outputs/stage5/programmatic/phase1/phase1_step_100.pt" in actions[0]["command"]
    assert f"STAGE5_RECOVERED_SOURCE_SUMMARY={source.as_posix()}" in actions[0]["command"]


def test_programmatic_depth_assessment_failure_inspects(tmp_path) -> None:
    source = tmp_path / "programmatic_assess" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_programmatic_depth_assessment",
        "status": "programmatic_depth_no_lift",
        "passed": False,
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect programmatic depth assessment `programmatic_depth_no_lift`"
    assert "summary.md" in actions[0]["command"]


def test_generic_candidate_distillation_pass_adds_selector_exact_gate(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "candidate_distillation_evidence": {
                "candidate_distill_rows": 12,
                "candidate_distill_selector_generated_rows": 0,
            },
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        }
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    selector_gate = next(action for action in actions if action["name"] == "Run selector-exact candidate-distillation gate")

    assert "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE=selector_exact" in selector_gate["command"]
    assert "STAGE5_ARC_AGI_CANDIDATE_DISTILL_SELECTION_STRATEGY=cell_vote" in selector_gate["command"]
    assert "claim-level selector evidence" in selector_gate["reason"]


def test_selector_candidate_distillation_pass_does_not_repeat_selector_gate(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "candidate_distillation_evidence": {
                "candidate_distill_rows": 12,
                "candidate_distill_selector_generated_rows": 5,
            },
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        }
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert "Run selector-exact candidate-distillation gate" not in [action["name"] for action in actions]


def test_smoke_win_recommends_confirmation_and_export(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(2, 3)},
            "phase1_start": {"summary": _summary(1, 1)},
            "recovered": {"summary": _summary(3, 4)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": 1,
                    "best_of_k_exact_delta": 1,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    names = [action["name"] for action in actions]

    assert any(name.startswith("Confirm recovered-vs-base") for name in names)
    assert "Export recovered adapter to Hugging Face" in names


def test_confirmation_win_at_large_limit_recommends_full_split(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(20, 22, examples=400)},
            "phase1_start": {"summary": _summary(16, 18, examples=400)},
            "recovered": {"summary": _summary(21, 23, examples=400)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": 1,
                    "best_of_k_exact_delta": 1,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 5,
                    "best_of_k_exact_delta": 5,
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    confirm = next(action for action in actions if action["name"].startswith("Confirm recovered-vs-base"))

    assert confirm["name"] == "Confirm recovered-vs-base at ARC limit full"
    assert "STAGE5_ARC_AGI_FOLLOWUP_LIMIT=full" in confirm["command"]
    assert "full ARC split" in confirm["reason"]


def test_paired_evidence_overrides_misleading_aggregate_win(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(2, 3)},
            "phase1_start": {"summary": _summary(1, 1)},
            "recovered": {"summary": _summary(3, 4)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": 1,
                    "best_of_k_exact_delta": 1,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
            "paired_comparisons": {
                "recovered_vs_base": {
                    "metrics": {
                        "selected_exact": _paired(-1, wins=0, losses=1, ties=49),
                        "best_of_k_exact": _paired(-1, wins=0, losses=1, ties=49),
                    }
                },
                "recovered_vs_start": {
                    "metrics": {
                        "selected_exact": _paired(2, wins=2, losses=0, ties=48),
                        "best_of_k_exact": _paired(3, wins=3, losses=0, ties=47),
                    }
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    names = [action["name"] for action in actions]

    assert not any(name.startswith("Confirm recovered-vs-base") for name in names)
    assert actions[0]["name"] == "Scale deterministic curriculum"
    assert "paired delta 2" in actions[0]["reason"]


def test_partial_recovery_recommends_scaled_curriculum(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 2)},
            "recovered": {"summary": _summary(4, 5)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -4,
                    "best_of_k_exact_delta": -4,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Scale deterministic curriculum"
    assert "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT=160" in actions[0]["command"]
    assert "Gap closure: selected 33.33%, best-of-K 42.86%." in actions[0]["reason"]


def test_recovery_analysis_focuses_scaled_curriculum_on_worst_families(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 2)},
            "recovered": {"summary": _summary(4, 5)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -4,
                    "best_of_k_exact_delta": -4,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
            "recovery_analysis": {
                "recommendations": [{"area": "family_targeted_sft", "reason": "focus"}],
                "family_gaps": {
                    "recovered_vs_base": [
                        {"family": "arc", "selected_delta": -10, "best_of_k_delta": -10, "paired_examples": 20},
                        {"family": "move_recolor", "selected_delta": -3, "best_of_k_delta": -2, "paired_examples": 6},
                        {"family": "frame_object", "selected_delta": -2, "best_of_k_delta": -2, "paired_examples": 4},
                    ]
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    scale = next(action for action in actions if action["name"] == "Scale deterministic curriculum")

    assert "STAGE5_ARC_AGI_CURRICULUM_STAGES='focus:move_recolor,frame_object" in scale["command"]
    assert "move_recolor, frame_object" in scale["reason"]


def test_recovery_analysis_selector_miss_recommends_no_gpu_rescore(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "run_id": "bench_run",
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 2)},
            "recovered": {"summary": _summary(4, 5)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -4,
                    "best_of_k_exact_delta": -4,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
            "recovery_analysis": {
                "recommendations": [{"area": "selector_or_tta", "reason": "selector misses found"}],
                "family_gaps": {"recovered_vs_base": []},
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    rescore = next(action for action in actions if action["name"] == "Rescore recovered candidates with selector variants")

    assert "colab/run_stage5_arc_agi_rescore_selectors.py" in rescore["command"]
    assert "STAGE5_ARC_AGI_RESCORE_SOURCE_RUN_DIR=outputs/stage5/bench_run" in rescore["command"]
    assert "STAGE5_ARC_AGI_RESCORE_SOURCE_GLOB=recovered_candidates.jsonl" in rescore["command"]
    assert "STAGE5_ARC_AGI_RESCORE_STRATEGIES=self_consistency,reliability_vote,symbolic_priority,cell_vote" in rescore["command"]
    assert "STAGE5_ARC_AGI_RESCORE_WRITE_JSONL=1" in rescore["command"]


def test_recovery_analysis_format_failures_recommend_format_branch(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(4, 5)},
            "recovered": {"summary": _summary(3, 4)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -5,
                    "best_of_k_exact_delta": -5,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": -1,
                    "best_of_k_exact_delta": -1,
                },
            },
            "recovery_analysis": {
                "recommendations": [{"area": "format_parse", "reason": "no valid grids"}],
                "family_gaps": {"recovered_vs_base": []},
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)
    format_action = next(action for action in actions if action["name"] == "Run output-format recovery curriculum")

    assert "format:constant_output,geometry_color,crop_non_background" in format_action["command"]
    assert "no valid grids" in format_action["reason"]


def test_no_recovery_recommends_training_target_diagnostics(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(4, 5)},
            "recovered": {"summary": _summary(3, 4)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -5,
                    "best_of_k_exact_delta": -5,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": -1,
                    "best_of_k_exact_delta": -1,
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run trace/candidate-distillation diagnostics before more SFT"
    assert "run_stage5_arc_agi_candidate_distill_gate.py" in actions[0]["command"]


def test_best_recovered_tta_row_prefers_best_then_selected() -> None:
    tta = {
        "rows": [
            {"arm": "recovered", "tta_variant": "none", "best_of_k_exact": 2, "selected_exact": 2},
            {"arm": "recovered", "tta_variant": "all", "best_of_k_exact": 4, "selected_exact": 1},
            {"arm": "recovered", "tta_variant": "rotations", "best_of_k_exact": 4, "selected_exact": 3},
        ]
    }

    assert best_recovered_tta_row(tta)["tta_variant"] == "rotations"


def test_tta_replicate_uses_paired_evidence_when_available(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "autopilot_compact": {
            "candidate_distillation_passed": True,
            "final_checkpoint": "outputs/stage5/run/final.pt",
            "particle_passed": False,
        },
        "recovered_benchmark": {
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 2)},
            "recovered": {"summary": _summary(4, 5)},
            "deltas": {
                "recovered_vs_base": {
                    "selected_exact_delta": -4,
                    "best_of_k_exact_delta": -4,
                },
                "recovered_vs_start": {
                    "selected_exact_delta": 2,
                    "best_of_k_exact_delta": 3,
                },
            },
        },
        "tta_sweep": {
            "rows": [
                {"arm": "recovered", "tta_variant": "none", "best_of_k_exact": 2, "selected_exact": 1},
                {"arm": "recovered", "tta_variant": "all", "best_of_k_exact": 4, "selected_exact": 3},
            ],
            "paired_comparisons": {
                "recovered__tta_all_vs_none": {
                    "metrics": {
                        "best_of_k_exact": _paired(-1, wins=0, losses=1, ties=49),
                    }
                }
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert not any(action["name"].startswith("Replicate recovered TTA") for action in actions)


def test_selector_rescore_summary_requests_gate1_assessment(tmp_path) -> None:
    source = tmp_path / "selector_summary.json"
    payload = {
        "source_run_dir": str(tmp_path / "source_benchmark"),
        "strategies": ["reliability_vote"],
        "rows": [
            {
                "label": "recovered",
                "selection_strategy": "reliability_vote",
                "examples": 50,
                "selected_exact": 12,
                "best_of_k_exact": 13,
                "valid_candidate_rate": 0.9,
                "selected_delta_vs_source": 2,
            }
        ],
        "best_by_label": {},
        "paired_comparisons": {
            "recovered__selector_reliability_vote_vs_source": {
                "metrics": {"selected_exact": _paired(2, wins=2, losses=0, ties=48)}
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    expected_source = str(source).replace("\\", "/")
    assert actions[0]["name"] == "Assess Gate 1 selector/TTA evidence"
    assert "colab/assess_stage5_gate1.py" in actions[0]["command"]
    assert f"--summary_json {expected_source}" in actions[0]["command"]


def test_direct_tta_sweep_summary_requests_gate1_assessment(tmp_path) -> None:
    source = tmp_path / "tta_summary.json"
    payload = {
        "run_id": "tta_sweep",
        "rows": [
            {
                "arm": "recovered",
                "tta_variant": "none",
                "examples_with_targets": 50,
                "selected_exact": 10,
                "best_of_k_exact": 11,
            },
            {
                "arm": "recovered",
                "tta_variant": "rotations",
                "examples_with_targets": 50,
                "selected_exact": 10,
                "best_of_k_exact": 13,
            },
        ],
        "deltas": {"recovered": {"best_of_k_exact_delta": 2}},
        "paired_comparisons": {
            "recovered__tta_rotations_vs_none": {
                "metrics": {"best_of_k_exact": _paired(2, wins=2, losses=0, ties=48)}
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess Gate 1 selector/TTA evidence"
    assert "colab/assess_stage5_gate1.py" in actions[0]["command"]
    assert source_kind(payload) == "tta_sweep"


def test_selector_rescore_summary_promotes_paired_lift(tmp_path) -> None:
    source_run = tmp_path / "source_benchmark"
    source_run.mkdir()
    source_benchmark = {
        "metadata": {
            "curriculum_summary": "outputs/stage5/curriculum/summary.json",
            "phase1_start_checkpoint": "outputs/stage4/phase1.pt",
            "recovered_checkpoint": "outputs/stage5/recovered.pt",
            "arc_version": "1",
            "arc_split": "evaluation",
            "grid_format": "compact",
            "program_parse_mode": "fallback",
            "difficulty_buckets": "easy,medium,hard",
            "examples_per_difficulty": 20,
        }
    }
    (source_run / "summary.json").write_text(json.dumps(source_benchmark), encoding="utf-8")
    source = tmp_path / "selector_summary.json"
    payload = {
        "run_id": "selector_rescore",
        "source_run_dir": str(source_run),
        "strategies": ["self_consistency", "symbolic_priority"],
        "rows": [
            {
                "label": "recovered",
                "selection_strategy": "original:heuristic",
                "examples": 50,
                "selected_exact": 10,
                "best_of_k_exact": 13,
                "valid_candidate_rate": 0.9,
                "selected_delta_vs_source": 0,
            },
            {
                "label": "recovered",
                "selection_strategy": "self_consistency",
                "examples": 50,
                "selected_exact": 12,
                "best_of_k_exact": 13,
                "valid_candidate_rate": 0.9,
                "selected_delta_vs_source": 2,
            },
            {
                "label": "recovered",
                "selection_strategy": "symbolic_priority",
                "examples": 50,
                "selected_exact": 11,
                "best_of_k_exact": 14,
                "valid_candidate_rate": 1.0,
                "selected_delta_vs_source": 1,
            },
        ],
        "best_by_label": {},
        "paired_comparisons": {
            "recovered__selector_self_consistency_vs_source": {
                "metrics": {"selected_exact": _paired(2, wins=2, losses=0, ties=48)}
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = selector_rescore_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Promote selector `self_consistency` for `recovered` benchmark"
    assert "STAGE5_ARC_AGI_SELECTION_STRATEGY=self_consistency" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_LIMIT=100" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_RECOVERED_CKPT=outputs/stage5/recovered.pt" in actions[0]["command"]
    assert "colab/run_stage5_arc_agi_recovered_benchmark.py" in actions[0]["command"]
    assert "paired delta 2" in actions[0]["reason"]


def test_selector_rescore_summary_validates_hard_tail_lift(tmp_path) -> None:
    source_run = tmp_path / "source_benchmark"
    source_run.mkdir()
    source_benchmark = {
        "metadata": {
            "curriculum_summary": "outputs/stage5/curriculum/summary.json",
            "phase1_start_checkpoint": "outputs/stage4/phase1.pt",
            "recovered_checkpoint": "outputs/stage5/recovered.pt",
            "arc_version": "1",
            "arc_split": "evaluation",
            "grid_format": "compact",
            "program_parse_mode": "fallback",
            "difficulty_buckets": "easy,medium,hard",
            "examples_per_difficulty": 20,
        }
    }
    (source_run / "summary.json").write_text(json.dumps(source_benchmark), encoding="utf-8")
    source = tmp_path / "selector_summary.json"
    payload = {
        "run_id": "selector_rescore",
        "source_run_dir": str(source_run),
        "strategies": ["self_consistency"],
        "rows": [
            {
                "label": "recovered",
                "selection_strategy": "self_consistency",
                "examples": 50,
                "selected_exact": 10,
                "best_of_k_exact": 13,
                "valid_candidate_rate": 0.9,
                "selected_delta_vs_source": 0,
            }
        ],
        "best_by_label": {},
        "paired_comparisons": {
            "recovered__selector_self_consistency_vs_source": {
                "metrics": {"selected_exact": _paired(0, wins=1, losses=1, ties=48)},
                "difficulty_metrics": {
                    "selected_exact": {
                        "hard": _paired(2, wins=2, losses=0, ties=8),
                        "medium": _paired(-2, wins=0, losses=2, ties=28),
                    }
                },
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = selector_rescore_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Validate hard-tail selector `self_consistency` for `recovered` benchmark"
    assert "STAGE5_ARC_AGI_SELECTION_STRATEGY=self_consistency" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_DIFFICULTY_BUCKETS=easy,medium,hard" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EXAMPLES_PER_DIFFICULTY=20" in actions[0]["command"]
    assert "Hard-bucket evidence: paired delta 2" in actions[0]["reason"]
    assert "Aggregate evidence: paired delta 0" in actions[0]["reason"]


def test_selector_rescore_summary_flags_hard_tail_tradeoff(tmp_path) -> None:
    source = tmp_path / "selector_summary.json"
    payload = {
        "source_run_dir": str(tmp_path / "missing_source"),
        "strategies": ["self_consistency"],
        "rows": [
            {
                "label": "recovered",
                "selection_strategy": "self_consistency",
                "examples": 50,
                "selected_exact": 9,
                "best_of_k_exact": 13,
                "valid_candidate_rate": 0.9,
                "selected_delta_vs_source": -1,
            }
        ],
        "best_by_label": {},
        "paired_comparisons": {
            "recovered__selector_self_consistency_vs_source": {
                "metrics": {"selected_exact": _paired(-1, wins=1, losses=2, ties=47)},
                "difficulty_metrics": {
                    "selected_exact": {
                        "hard": _paired(2, wins=2, losses=0, ties=8),
                    }
                },
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = selector_rescore_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect hard-tail selector tradeoff `self_consistency`"
    assert "hard difficulty bucket" in actions[0]["reason"]
    assert "Aggregate evidence: paired delta -1" in actions[0]["reason"]


def test_selector_rescore_summary_without_paired_lift_replans_source(tmp_path) -> None:
    source_run = tmp_path / "source_benchmark"
    source_run.mkdir()
    (source_run / "summary.json").write_text(json.dumps({"recovered_benchmark": {}}), encoding="utf-8")
    source = tmp_path / "selector_summary.json"
    payload = {
        "source_run_dir": str(source_run),
        "strategies": ["self_consistency"],
        "rows": [
            {
                "label": "recovered",
                "selection_strategy": "self_consistency",
                "examples": 100,
                "selected_exact": 10,
                "best_of_k_exact": 13,
                "valid_candidate_rate": 0.9,
                "selected_delta_vs_source": 0,
            }
        ],
        "best_by_label": {},
        "paired_comparisons": {
            "recovered__selector_self_consistency_vs_source": {
                "metrics": {"selected_exact": _paired(0, wins=0, losses=0, ties=50)}
            }
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = selector_rescore_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Defer selector changes and continue recovery plan"
    assert "colab/plan_stage5_next_run.py" in actions[0]["command"]
    assert "summary.json" in actions[0]["command"]


def test_recovery_particle_gate_failed_recovery_recommends_trace_training_gate(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "settings": {"eval_task_limit": 20},
        "recovery_decision": {
            "passed": False,
            "evidence": {
                "phase1_tuned_vs_start": {"selected_delta": -1, "best_of_k_delta": 0},
            },
        },
        "particle_decision": {"passed": False, "evidence": {}},
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Compare ARC trace-training targets"
    assert "run_stage5_arc_agi_trace_sft_gate.py" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_SFT_GATE_ARMS" in actions[0]["command"]
    assert "symbolic_state_trace_covered" in actions[0]["command"]
    assert "selected_delta" in actions[0]["reason"]


def test_recovery_particle_gate_recovery_without_particle_recommends_benchmark(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "settings": {
            "eval_task_limit": 20,
            "program_parse_mode": "prefer",
            "selection_strategy": "self_consistency",
        },
        "sft_summary": {"metadata": {"phase1_checkpoint": "outputs/stage4/phase1.pt"}},
        "recovered_checkpoint": {
            "checkpoint": "outputs/stage5/recovered.pt",
            "summary": _summary(4, 5),
        },
        "recovery_decision": {
            "passed": True,
            "evidence": {
                "phase1_tuned_vs_start": {"selected_delta": 2, "best_of_k_delta": 3},
                "phase1_tuned_vs_base": {"selected_delta": -4, "best_of_k_delta": -3},
            },
        },
        "particle_decision": {"passed": False, "evidence": {"best_replicated_variant": None}},
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Benchmark recovered recurrent against base at ARC limit 100"
    assert "STAGE5_ARC_AGI_RECOVERED_CKPT=outputs/stage5/recovered.pt" in actions[0]["command"]
    assert "STAGE5_PHASE1_CKPT=outputs/stage4/phase1.pt" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_LIMIT=100" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE=prefer" in actions[0]["command"]
    assert actions[1]["name"] == "Defer particle/SVGD training pressure"


def test_recovery_particle_gate_particle_pass_recommends_replicated_particle_gate(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "settings": {
            "eval_task_limit": 100,
            "synthetic_tasks": 200,
            "synthetic_modes": "all",
            "train_steps": 300,
            "train_task_limit": 100,
            "trace_mode": "symbolic_program",
            "trace_filter": "covered",
            "program_parse_mode": "prefer",
            "selection_strategy": "heuristic",
            "particle_variants": [
                {"name": "k4_noise001_rep05", "noise": 0.01, "repulsion": 0.5},
            ],
        },
        "sft_summary": {"metadata": {"phase1_checkpoint": "outputs/stage4/phase1.pt"}},
        "recovered_checkpoint": {
            "checkpoint": "outputs/stage5/recovered.pt",
            "summary": _summary(4, 5),
        },
        "recovery_decision": {
            "passed": True,
            "evidence": {
                "phase1_tuned_vs_start": {"selected_delta": 2, "best_of_k_delta": 3},
                "phase1_tuned_vs_base": {"selected_delta": -4, "best_of_k_delta": -3},
            },
        },
        "particle_decision": {
            "passed": True,
            "evidence": {"best_replicated_variant": "k4_noise001_rep05"},
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = recovery_particle_actions(payload, source_summary=source)
    particle = next(action for action in actions if action["name"].startswith("Replicate particle value"))

    assert "k4_noise001_rep05" in particle["name"]
    assert "run_stage5_arc_agi_recovery_particle_gate.py" in particle["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=400" in particle["command"]
    assert "STAGE5_ARC_AGI_PARTICLE_SEEDS=0,1,2,3,4" in particle["command"]
    assert "STAGE5_ARC_AGI_PARTICLE_VARIANTS=k4_noise001_rep05:0.01:0.5" in particle["command"]


def test_recovery_particle_gate_particle_pass_requests_gate2_assessment(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "recovery_decision": {"passed": True, "evidence": {}},
        "particle_decision": {
            "passed": True,
            "evidence": {
                "best_replicated_variant": "k4_noise001_rep05",
                "variants": {
                    "k4_noise001_rep05": {
                        "passed": True,
                        "evaluated_seed_count": 5,
                        "non_negative_seed_count": 5,
                        "mean_delta_vs_tuned": {"selected_delta": 1, "best_of_k_delta": 2},
                    }
                },
            },
        },
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess Gate 2 particle mechanism evidence"
    assert "colab/assess_stage5_gate2.py" in actions[0]["command"]


def test_gate2_passed_delegates_to_particle_replication(tmp_path) -> None:
    source = tmp_path / "particle" / "summary.json"
    source.parent.mkdir()
    source_payload = {
        "settings": {
            "eval_task_limit": 100,
            "synthetic_tasks": 200,
            "synthetic_modes": "all",
            "train_steps": 300,
            "train_task_limit": 100,
            "trace_mode": "symbolic_program",
            "trace_filter": "covered",
            "program_parse_mode": "prefer",
            "selection_strategy": "heuristic",
            "particle_variants": [
                {"name": "k4_noise001_rep05", "noise": 0.01, "repulsion": 0.5},
            ],
        },
        "recovery_decision": {"passed": True, "evidence": {}},
        "particle_decision": {
            "passed": True,
            "evidence": {"best_replicated_variant": "k4_noise001_rep05"},
        },
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    gate2 = tmp_path / "gate2" / "summary.json"
    gate2.parent.mkdir()
    payload = {
        "gate": "stage5_gate2_particle_mechanism",
        "status": "passed",
        "passed": True,
        "reason": "replicated selected lift",
        "next_step": "replicate larger",
        "source_summary": str(source),
        "source_kind": "recovery_particle_gate",
    }
    gate2.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=gate2)

    assert actions[0]["name"].startswith("Gate 2 passed: Replicate particle value `k4_noise001_rep05`")
    assert "replicated selected lift" in actions[0]["reason"]
    assert "run_stage5_arc_agi_recovery_particle_gate.py" in actions[0]["command"]


def test_gate2_selector_conversion_stops_for_inspection(tmp_path) -> None:
    gate2 = tmp_path / "gate2" / "summary.json"
    gate2.parent.mkdir()
    payload = {
        "gate": "stage5_gate2_particle_mechanism",
        "status": "needs_selector_conversion",
        "passed": False,
        "reason": "coverage improved but selected accuracy did not",
        "next_step": "run selector work",
    }
    gate2.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=gate2)

    assert actions[0]["name"] == "Inspect Gate 2 assessment `needs_selector_conversion`"
    assert actions[0]["command"].startswith("cat ")


def test_gate1_passed_without_previous_gate1_runs_confirmation_selector_benchmark(tmp_path) -> None:
    source_run = tmp_path / "source_benchmark"
    source_run.mkdir()
    (source_run / "summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "recovered_checkpoint": "outputs/stage5/recovered.pt",
                    "phase1_start_checkpoint": "outputs/stage4/phase1.pt",
                    "grid_format": "compact",
                    "program_parse_mode": "fallback",
                }
            }
        ),
        encoding="utf-8",
    )
    selector = tmp_path / "selector" / "summary.json"
    selector.parent.mkdir()
    selector.write_text(
        json.dumps(
            {
                "source_run_dir": str(source_run),
                "strategies": ["reliability_vote"],
                "rows": [
                    {
                        "label": "recovered",
                        "selection_strategy": "reliability_vote",
                        "examples": 50,
                        "selected_exact": 12,
                        "best_of_k_exact": 13,
                        "valid_candidate_rate": 0.9,
                        "selected_delta_vs_source": 2,
                    }
                ],
                "best_by_label": {},
                "paired_comparisons": {
                    "recovered__selector_reliability_vote_vs_source": {
                        "metrics": {"selected_exact": _paired(2, wins=2, losses=0, ties=48)}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    gate1 = tmp_path / "gate1" / "summary.json"
    gate1.parent.mkdir()
    payload = {
        "gate": "stage5_gate1_selector_tta",
        "status": "passed",
        "passed": True,
        "reason": "hard-tail lift",
        "next_step": "replicate",
        "source_summary": str(selector),
        "source_kind": "selector_rescore",
    }
    gate1.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=gate1)

    assert actions[0]["name"].startswith("Gate 1 discovery passed: Confirm selector `reliability_vote`")
    assert "hard-tail lift" in actions[0]["reason"]
    assert "confirmation run, not a final selector promotion" in actions[0]["reason"]
    assert "STAGE5_ARC_AGI_SELECTION_STRATEGY=reliability_vote" in actions[0]["command"]


def test_gate1_passed_with_previous_gate1_runs_selector_replication(tmp_path) -> None:
    previous = tmp_path / "previous_gate1" / "summary.json"
    previous.parent.mkdir()
    previous_payload = {
        "gate": "stage5_gate1_selector_tta",
        "status": "passed",
        "passed": True,
        "passing_comparisons": ["recovered__selector_reliability_vote_vs_source"],
    }
    previous.write_text(json.dumps(previous_payload), encoding="utf-8")
    gate1 = tmp_path / "gate1" / "summary.json"
    gate1.parent.mkdir()
    payload = {
        "gate": "stage5_gate1_selector_tta",
        "status": "passed",
        "passed": True,
        "reason": "hard-tail lift",
        "next_step": "replicate",
        "passing_comparisons": ["recovered__selector_reliability_vote_vs_source"],
    }
    gate1.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=gate1)

    assert actions[0]["name"] == "Assess selector replication across Gate 1 slices"
    assert "assess_stage5_selector_replication.py" in actions[0]["command"]
    assert previous.as_posix() in actions[0]["command"]
    assert gate1.as_posix() in actions[0]["command"]


def test_gate1_needs_more_evidence_does_not_promote_source_action(tmp_path) -> None:
    selector = tmp_path / "selector" / "summary.json"
    selector.parent.mkdir()
    selector.write_text(
        json.dumps(
            {
                "strategies": ["reliability_vote"],
                "rows": [
                    {
                        "label": "recovered",
                        "selection_strategy": "reliability_vote",
                        "examples": 50,
                        "selected_exact": 12,
                        "best_of_k_exact": 13,
                        "valid_candidate_rate": 0.9,
                        "selected_delta_vs_source": 2,
                    }
                ],
                "best_by_label": {},
                "paired_comparisons": {
                    "recovered__selector_reliability_vote_vs_source": {
                        "metrics": {"selected_exact": _paired(2, wins=2, losses=0, ties=48)}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    gate1 = tmp_path / "gate1" / "summary.json"
    gate1.parent.mkdir()
    payload = {
        "gate": "stage5_gate1_selector_tta",
        "status": "needs_more_evidence",
        "passed": False,
        "reason": "aggregate lift without hard-bucket evidence",
        "next_step": "run stratified slice",
        "source_summary": str(selector),
        "source_kind": "selector_rescore",
    }
    gate1.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=gate1)

    assert actions[0]["name"] == "Inspect Gate 1 assessment `needs_more_evidence`"
    assert "STAGE5_ARC_AGI_SELECTION_STRATEGY=reliability_vote" not in actions[0]["command"]


def test_gate1_needs_review_recommends_inspection(tmp_path) -> None:
    gate1 = tmp_path / "gate1" / "summary.json"
    gate1.parent.mkdir()
    (gate1.parent / "summary.md").write_text("# Gate 1\n", encoding="utf-8")
    payload = {
        "gate": "stage5_gate1_selector_tta",
        "status": "needs_review",
        "passed": False,
        "reason": "hard-tail lift with aggregate harm",
        "next_step": "inspect",
    }
    gate1.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=gate1)

    assert actions[0]["name"] == "Inspect Gate 1 assessment `needs_review`"
    assert actions[0]["command"].startswith("cat ")
    assert "summary.md" in actions[0]["command"]


def test_selector_replication_passed_routes_to_inspection(tmp_path) -> None:
    source = tmp_path / "selector_replication" / "summary.json"
    source.parent.mkdir()
    (source.parent / "summary.md").write_text("# Selector replication\n", encoding="utf-8")
    payload = {
        "gate": "stage5_selector_replication",
        "status": "passed",
        "passed": True,
        "replicated_comparisons": ["recovered__selector_reliability_vote_vs_source"],
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect replicated selector evidence"
    assert "summary.md" in actions[0]["command"]


def test_selector_replication_missing_confirmation_routes_to_inspection(tmp_path) -> None:
    source = tmp_path / "selector_replication" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_selector_replication",
        "status": "needs_confirmation",
        "passed": False,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect selector replication `needs_confirmation`"


def test_paired_metric_helpers_fall_back_to_aggregate() -> None:
    payload = {
        "deltas": {"recovered_vs_base": {"selected_exact_delta": 2}},
        "paired_comparisons": {
            "recovered_vs_base": {"metrics": {"selected_exact": _paired(1, wins=2, losses=1, ties=47)}}
        },
    }

    assert paired_metric(payload, "recovered_vs_base", "selected_exact")["delta_exact"] == 1
    assert (
        paired_delta_or_aggregate(
            payload,
            comparison="recovered_vs_base",
            metric_name="selected_exact",
            aggregate_group="recovered_vs_base",
            aggregate_key="selected_exact_delta",
        )
        == 1
    )
    assert (
        paired_delta_or_aggregate(
            payload,
            comparison="missing",
            metric_name="selected_exact",
            aggregate_group="recovered_vs_base",
            aggregate_key="selected_exact_delta",
        )
        == 2
    )
    assert "paired delta 1" in evidence_fragment(paired_metric(payload, "recovered_vs_base", "selected_exact"), 2)


def test_next_validation_limit_graduates_smoke_to_confirm_to_full() -> None:
    assert next_validation_limit(50) == 100
    assert next_validation_limit(100) == 400
    assert next_validation_limit(400) is None


def test_source_kind_classifies_followup_and_autopilot() -> None:
    assert source_kind({"recovered_benchmark": {}}) == "followup"
    assert source_kind({"compact": {}}) == "autopilot"
    assert source_kind({"best_by_label": {}, "rows": []}) == "selector_rescore"
    assert source_kind({"kind": "dense_sft_control"}) == "dense_sft_control"
    assert source_kind({"recovery_decision": {}, "particle_decision": {}}) == "recovery_particle_gate"
    assert source_kind({"gate": "stage5_gate1_selector_tta"}) == "gate1_assessment"
    assert source_kind({"gate": "stage5_gate2_particle_mechanism"}) == "gate2_assessment"
    assert source_kind({"gate": "stage5_selector_replication"}) == "selector_replication"
    assert source_kind({"gate": "stage5_same_recipe_selector_conversion"}) == "recipe_selector_conversion"
    assert source_kind({"gate": "stage5_same_recipe_architecture"}) == "recipe_control_assessment"
    assert source_kind({"gate": "stage5_release_benchmark_readiness"}) == "release_gate"
    assert source_kind({"kind": "stage5_benchmark_suite"}) == "benchmark_suite"
    assert source_kind({"gate": "stage5_broader_benchmark_suite"}) == "benchmark_suite_assessment"
    assert source_kind({"kind": "stage5_balanced_arc_mix_gate"}) == "balanced_arc_mix_gate"
    assert source_kind({"kind": "stage5_routing_diagnostic_assessment"}) == "routing_diagnostic"
    assert source_kind({"gate": "stage5_claim_readiness"}) == "claim_readiness"
    assert source_kind({"gate": "stage5_arc_agi_baseline_registry"}) == "arc_agi_baseline_registry"
    assert source_kind({"gate": "stage5_arc_agi_sota_comparison"}) == "arc_agi_sota_comparison"
    assert source_kind({"kind": "stage5_arc_agi_candidate_gate"}) == "candidate_gate"
    assert source_kind({"kind": "stage5_reasoning_dataset_audit"}) == "reasoning_dataset_audit"
    assert source_kind({"kind": "stage4_opus_finetune"}) == "stage4_opus_finetune"
    assert source_kind({"kind": "trace_sft_gate"}) == "trace_sft_gate"
    assert source_kind({"kind": "distill_sft_gate"}) == "distill_sft_gate"
    assert source_kind({"phase1_arc_agi_tuned": {}, "tuned_checkpoint": "ckpt.pt"}) == "recurrent_sft"
    assert source_kind({"rows": [], "deltas": {}, "paired_comparisons": {}}) == "tta_sweep"


def test_reasoning_dataset_audit_promotes_opus_finetune(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "run_id": "audit_run",
        "kind": "stage5_reasoning_dataset_audit",
        "recommendations": [
            {
                "key": "opus47_sft",
                "dataset_id": "lordx64/reasoning-distill-opus-4-7-max-sft",
                "status": "promote_to_small_train_mix",
                "converted_rows": 900,
                "conversion_rate": 0.9,
            },
            {
                "key": "fable5_pi_agent",
                "dataset_id": "Glint-Research/Fable-5-traces",
                "status": "hold_for_agent_tool_filter",
                "converted_rows": 100,
                "conversion_rate": 0.1,
            },
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run audited modified-Opus recurrent fine-tune"
    assert "STAGE4_RUN_ID=audit_run_audited_opus_finetune" in actions[0]["command"]
    assert "OPUS_DATASET_ID=lordx64/reasoning-distill-opus-4-7-max-sft" in actions[0]["command"]
    assert "OPUS_DATASET_ADAPTER=qwen_text" in actions[0]["command"]
    assert "python colab/run_stage4_opus_finetune.py" in actions[0]["command"]


def test_reasoning_dataset_audit_inspects_unapproved_promoted_opus_source(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "run_id": "audit_run",
        "kind": "stage5_reasoning_dataset_audit",
        "recommendations": [
            {
                "key": "jackrong_opus47_trace_inversion",
                "dataset_id": "Jackrong/Claude-opus-4.7-TraceInversion-5000x",
                "status": "promote_to_small_train_mix",
                "converted_rows": 900,
                "conversion_rate": 0.9,
            },
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect promoted non-Opus trace source"
    assert actions[0]["command"].startswith("cat ")
    assert "run_stage4_opus_finetune.py" not in actions[0]["command"]


def test_reasoning_dataset_audit_holds_fable_without_training(tmp_path) -> None:
    source = tmp_path / "summary.json"
    payload = {
        "kind": "stage5_reasoning_dataset_audit",
        "recommendations": [
            {
                "key": "fable5_pi_agent",
                "dataset_id": "Glint-Research/Fable-5-traces",
                "status": "hold_for_agent_tool_filter",
                "converted_rows": 700,
                "conversion_rate": 0.7,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect Fable/tool-trace audit before training"
    assert actions[0]["command"].startswith("cat ")


def test_stage4_opus_finetune_runs_benchmark_suite(tmp_path) -> None:
    source = tmp_path / "stage4" / "summary.json"
    source.parent.mkdir()
    payload = {
        "run_id": "stage4_opus",
        "kind": "stage4_opus_finetune",
        "checkpoint": "outputs/stage4/stage4_opus/phase1_step_500.pt",
        "phase1_checkpoint": "outputs/stage4/stage4_opus/phase1_step_500.pt",
        "phase2_checkpoint": "outputs/stage4/stage4_opus/phase2_step_100.pt",
        "arc_ladder": {
            "phase1_gap_to_base": -0.12,
            "phase2_best_lift_over_phase1": 0.03,
            "phase2_best_gap_to_base": -0.09,
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run benchmark suite for Stage 4 recurrent checkpoint"
    assert "STAGE5_BENCHMARK_SUITE_RUN_ID=stage4_opus_benchmark_suite" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_SOURCE_SUMMARY=" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_CHECKPOINT=outputs/stage4/stage4_opus/phase1_step_500.pt" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_RECURRENT_MODE=phase1" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_NUM_TRAJECTORIES=1" in actions[0]["command"]
    assert "python colab/run_stage5_benchmark_suite.py" in actions[0]["command"]


def test_legacy_stage4_opus_finetune_summary_runs_benchmark_suite(tmp_path) -> None:
    source = tmp_path / "legacy_stage4" / "summary.json"
    source.parent.mkdir()
    payload = {
        "run_id": "legacy_stage4",
        "phase1_checkpoint": "outputs/stage4/legacy/phase1.pt",
        "phase2_checkpoint": "outputs/stage4/legacy/phase2.pt",
        "arc_ladder": {},
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run benchmark suite for Stage 4 recurrent checkpoint"
    assert "STAGE5_BENCHMARK_SUITE_RUN_ID=legacy_stage4_benchmark_suite" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_CHECKPOINT=outputs/stage4/legacy/phase1.pt" in actions[0]["command"]


def test_curriculum_sft_summary_runs_benchmark_suite(tmp_path) -> None:
    source = tmp_path / "curriculum_sft" / "summary.json"
    source.parent.mkdir()
    payload = {
        "run_id": "curriculum_sft_run",
        "kind": "stage5_curriculum_sft",
        "phase1_checkpoint": "outputs/stage5/curriculum_sft_run/phase1/phase1_step_150.pt",
        "dataset": {"train_rows": 40, "val_rows": 5},
        "phase1_val": {"loss": 2.5, "mean_expected_loops": 3.0},
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert source_kind(payload) == "curriculum_sft"
    assert actions[0]["name"] == "Run benchmark suite for generated-curriculum recurrent checkpoint"
    assert "STAGE5_BENCHMARK_SUITE_RUN_ID=curriculum_sft_run_benchmark_suite" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_SOURCE_SUMMARY=" in actions[0]["command"]
    assert (
        "STAGE5_BENCHMARK_CHECKPOINT=outputs/stage5/curriculum_sft_run/phase1/phase1_step_150.pt"
        in actions[0]["command"]
    )
    assert "STAGE5_BENCHMARK_RECURRENT_MODE=phase1" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_NUM_TRAJECTORIES=1" in actions[0]["command"]
    assert "python colab/run_stage5_benchmark_suite.py" in actions[0]["command"]


def test_candidate_gate_plans_trace_sft_when_symbolic_hybrid_signal_exists(tmp_path) -> None:
    source = tmp_path / "candidate_gate" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_arc_agi_candidate_gate",
        "metadata": {
            "arc_version": "1",
            "arc_split": "evaluation",
            "limit": 20,
            "phase1_checkpoint": "outputs/stage4/phase1.pt",
            "grid_format": "compact",
            "selection_strategy": "heuristic",
        },
        "symbolic_coverage": {"exact_symbolic": 1},
        "rows": [
            {"variant": "base_model_only", "best": 1},
            {"variant": "base_hybrid_symbolic_first", "best": 1},
            {"variant": "phase1_model_only", "best": 1},
            {"variant": "phase1_hybrid_symbolic_first", "best": 2},
        ],
        "results": [],
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Compare ARC trace-training targets"
    assert "python colab/run_stage5_arc_agi_trace_sft_gate.py" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_VERSION=1" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_SPLIT=evaluation" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=20" in actions[0]["command"]
    assert "STAGE5_PHASE1_CKPT=outputs/stage4/phase1.pt" in actions[0]["command"]


def test_candidate_gate_without_symbolic_signal_starts_dense_control(tmp_path) -> None:
    source = tmp_path / "candidate_gate" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_candidate_gate",
        "metadata": {
            "arc_version": "1",
            "arc_split": "evaluation",
            "limit": 20,
            "grid_format": "compact",
        },
        "symbolic_coverage": {"exact_symbolic": 0},
        "rows": [
            {"variant": "base_model_only", "best": 1},
            {"variant": "base_hybrid_symbolic_first", "best": 0},
            {"variant": "phase1_model_only", "best": 1},
            {"variant": "phase1_hybrid_symbolic_first", "best": 0},
        ],
        "results": [],
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run dense ARC-AGI SFT control"
    assert "python colab/run_stage5_arc_agi_dense_sft.py" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_MODE=none" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=20" in actions[0]["command"]


def test_trace_sft_gate_plans_distill_when_trace_matches_grid(tmp_path) -> None:
    source = tmp_path / "trace_gate" / "summary.json"
    source.parent.mkdir()
    metadata = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "arc_version": "1",
        "train_split": "training",
        "eval_split": "evaluation",
        "train_task_limit": 80,
        "eval_task_limit": 12,
        "trace_mode": "symbolic_program",
        "trace_filter": "covered",
        "grid_format": "compact",
        "program_parse_mode": "fallback",
        "selection_strategy": "heuristic",
        "train_steps": 120,
        "learning_rate": 8e-6,
        "distillation": {"enabled": False, "weight": 0.1, "temperature": 1.0, "on": "response"},
    }
    payload = {
        "kind": "trace_sft_gate",
        "arms": [
            {"label": "grid_only", "trace_mode": "none", "trace_filter": "all"},
            {"label": "symbolic_program_trace_covered", "trace_mode": "symbolic_program", "trace_filter": "covered"},
        ],
        "results": {
            "grid_only": {"metadata": {**metadata, "trace_mode": "none", "trace_filter": "all"}},
            "symbolic_program_trace_covered": {"metadata": metadata},
        },
        "comparison": {
            "grid_only": {"best_best": 2, "best_selected": 1, "tuned_best": 2, "tuned_selected": 1},
            "symbolic_program_trace_covered": {
                "best_best": 2,
                "best_selected": 2,
                "tuned_best": 2,
                "tuned_selected": 2,
            },
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Compare base-logit distillation for selected ARC trace recipe"
    assert "python colab/run_stage5_arc_agi_distill_sft_gate.py" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_MODE=symbolic_program" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_FILTER=covered" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=12" in actions[0]["command"]


def test_trace_sft_gate_without_trace_signal_runs_dense_grid_control(tmp_path) -> None:
    source = tmp_path / "trace_gate" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_trace_sft_gate",
        "arms": [
            {"label": "grid_only", "trace_mode": "none", "trace_filter": "all"},
            {"label": "symbolic_state_trace_covered", "trace_mode": "symbolic_state_trace", "trace_filter": "covered"},
        ],
        "results": {
            "grid_only": {
                "metadata": {
                    "arc_version": "1",
                    "eval_split": "evaluation",
                    "eval_task_limit": 10,
                    "trace_mode": "none",
                    "trace_filter": "all",
                }
            }
        },
        "comparison": {
            "grid_only": {"best_best": 3, "best_selected": 2},
            "symbolic_state_trace_covered": {"best_best": 1, "best_selected": 1},
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run dense ARC-AGI SFT control for grid-only recipe"
    assert "python colab/run_stage5_arc_agi_dense_sft.py" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_MODE=none" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_FILTER=all" in actions[0]["command"]


def test_distill_sft_gate_plans_dense_control_with_selected_distill_arm(tmp_path) -> None:
    source = tmp_path / "distill_gate" / "summary.json"
    source.parent.mkdir()
    metadata = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "arc_version": "1",
        "train_split": "training",
        "eval_split": "evaluation",
        "train_task_limit": 80,
        "eval_task_limit": 12,
        "trace_mode": "symbolic_program",
        "trace_filter": "covered",
        "grid_format": "compact",
        "distillation": {"enabled": True, "weight": 0.1, "temperature": 1.0, "on": "response"},
    }
    payload = {
        "kind": "distill_sft_gate",
        "distill_off": {"metadata": {**metadata, "distillation": {"enabled": False}}},
        "distill_on": {"metadata": metadata},
        "comparison": {
            "distill_off": {"best_best": 2, "best_selected": 1, "tuned_best": 2, "tuned_selected": 1},
            "distill_on": {"best_best": 3, "best_selected": 2, "tuned_best": 3, "tuned_selected": 2},
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run dense ARC-AGI SFT control for selected recipe"
    assert "python colab/run_stage5_arc_agi_dense_sft.py" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_DENSE_DISTILL=1" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_MODE=symbolic_program" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=12" in actions[0]["command"]


def test_dense_sft_control_plans_matched_recurrent_recipe(tmp_path) -> None:
    source = tmp_path / "dense" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "dense_sft_control",
        "metadata": {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "arc_version": "1",
            "train_split": "training",
            "eval_split": "evaluation",
            "train_task_limit": 80,
            "eval_task_limit": 12,
            "color_augmentations": 3,
            "geometry_augmentations": "rotations",
            "trace_mode": "symbolic_program",
            "trace_filter": "covered",
            "synthetic_tasks": 0,
            "candidate_distill_jsonls": [],
            "grid_format": "compact",
            "program_parse_mode": "fallback",
            "selection_strategy": "heuristic",
            "train_steps": 300,
            "learning_rate": 8e-6,
            "distillation": {"enabled": False, "weight": 0.1, "temperature": 1.0, "on": "response"},
            "include_symbolic_candidates": False,
            "eval_checkpoint_ladder": False,
        },
        "deltas": {
            "dense_tuned_vs_base": {"selected_exact_delta": 2},
            "phase1_start_vs_base": {"selected_exact_delta": -1},
        },
        "paired_comparisons": {
            "dense_tuned_vs_base": {"metrics": {"selected_exact": _paired(2, wins=2, losses=0, ties=10)}},
            "phase1_start_vs_base": {"metrics": {"selected_exact": _paired(-1, wins=0, losses=1, ties=11)}},
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run matched recurrent ARC-AGI SFT control"
    assert "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT=80" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=12" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_COLOR_AUGS=3" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_GEOMETRY_AUGS=rotations" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_MODE=symbolic_program" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE=fallback" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRAIN_STEPS=300" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_LR=8e-06" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_DISTILL=0" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_INCLUDE_SYMBOLIC=0" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER=0" in actions[0]["command"]
    assert "python colab/run_stage5_arc_agi_sft.py" in actions[0]["command"]
    assert "Dense-vs-base evidence: paired delta 2" in actions[0]["reason"]


def test_dense_sft_control_plans_matched_recurrent_distillation(tmp_path) -> None:
    source = tmp_path / "dense" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "dense_sft_control",
        "metadata": {
            "train_task_limit": 80,
            "eval_task_limit": 12,
            "trace_mode": "symbolic_program",
            "trace_filter": "covered",
            "grid_format": "compact",
            "distillation": {"enabled": True, "weight": 0.2, "temperature": 2.0, "on": "full"},
            "eval_checkpoint_ladder": False,
        },
        "deltas": {
            "dense_tuned_vs_base": {"selected_exact_delta": 0},
            "phase1_start_vs_base": {"selected_exact_delta": 0},
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert "STAGE5_ARC_AGI_DISTILL=1" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_DISTILL_WEIGHT=0.2" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_DISTILL_TEMPERATURE=2.0" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_DISTILL_ON=full" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER=0" in actions[0]["command"]


def test_recurrent_sft_summary_plans_same_recipe_assessment(tmp_path) -> None:
    source = tmp_path / "recurrent" / "summary.json"
    source.parent.mkdir()
    payload = {
        "phase1_arc_agi_tuned": {"selected_exact": 3, "best_of_k_exact": 3},
        "tuned_checkpoint": "outputs/stage5/recurrent/phase1.pt",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess same-recipe recurrent-vs-dense control"
    assert "python colab/assess_stage5_recipe_control.py" in actions[0]["command"]
    assert "--recurrent_summary_json" in actions[0]["command"]


def test_recipe_control_assessment_passed_replicates_dense_control(tmp_path) -> None:
    source = tmp_path / "recipe" / "summary.json"
    dense = tmp_path / "dense" / "summary.json"
    source.parent.mkdir()
    dense.parent.mkdir()
    dense.write_text(
        json.dumps(
            {
                "kind": "dense_sft_control",
                "metadata": {
                    "arc_version": "1",
                    "train_task_limit": 80,
                    "eval_task_limit": 40,
                    "trace_mode": "symbolic_program",
                    "trace_filter": "covered",
                    "grid_format": "compact",
                    "program_parse_mode": "fallback",
                    "selection_strategy": "heuristic",
                    "train_steps": 300,
                    "learning_rate": 8e-6,
                    "distillation": {"enabled": False, "weight": 0.1, "temperature": 1.0, "on": "response"},
                    "include_symbolic_candidates": False,
                    "eval_checkpoint_ladder": False,
                },
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "passed",
        "dense_summary": str(dense),
        "evidence": {
            "recurrent_vs_dense": {
                "candidate_summary": {"examples_with_targets": 40},
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Replicate dense control at ARC limit 100"
    assert "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT=80" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=100" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_TRACE_MODE=symbolic_program" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE=fallback" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_DISTILL=0" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER=0" in actions[0]["command"]
    assert "python colab/run_stage5_arc_agi_dense_sft.py" in actions[0]["command"]


def test_recipe_control_assessment_full_pass_runs_release_gate(tmp_path) -> None:
    source = tmp_path / "recipe" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "passed",
        "evidence": {
            "recurrent_vs_dense": {
                "candidate_summary": {"examples_with_targets": 400},
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run release gate after full same-recipe pass"
    assert "STAGE5_RELEASE_GATE_RUN_ID=" in actions[0]["command"]
    assert "python colab/assess_stage5_release_gate.py" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=full" not in actions[0]["command"]
    assert "run_stage5_arc_agi_dense_sft.py" not in actions[0]["command"]


def test_recipe_control_assessment_full_needs_more_evidence_inspects(tmp_path) -> None:
    source = tmp_path / "recipe" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "needs_more_evidence",
        "evidence": {
            "recurrent_vs_dense": {
                "candidate_summary": {"examples_with_targets": 400},
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect full-split same-recipe evidence"
    assert "summary.md" in actions[0]["command"]
    assert "run_stage5_arc_agi_dense_sft.py" not in actions[0]["command"]


def test_recipe_control_assessment_selector_conversion_runs_rescore(tmp_path) -> None:
    source = tmp_path / "recipe" / "summary.json"
    recurrent = tmp_path / "recurrent" / "summary.json"
    source.parent.mkdir()
    recurrent.parent.mkdir()
    payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "needs_selector_conversion",
        "recurrent_summary": str(recurrent),
        "evidence": {
            "recurrent_vs_dense": {
                "candidate_summary": {"examples_with_targets": 40},
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Rescore recurrent candidates with selectors"
    assert "STAGE5_ARC_AGI_RESCORE_SOURCE_RUN_DIR" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_RESCORE_RECIPE_CONTROL_SUMMARY" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_RESCORE_WRITE_JSONL=1" in actions[0]["command"]
    assert "python colab/run_stage5_arc_agi_rescore_selectors.py" in actions[0]["command"]


def test_recipe_control_metadata_mismatch_reruns_recurrent_matched_to_dense(tmp_path) -> None:
    source = tmp_path / "recipe" / "summary.json"
    dense = tmp_path / "dense" / "summary.json"
    source.parent.mkdir()
    dense.parent.mkdir()
    dense.write_text(
        json.dumps(
            {
                "kind": "dense_sft_control",
                "metadata": {
                    "arc_version": "1",
                    "train_task_limit": 80,
                    "eval_task_limit": 40,
                    "trace_mode": "symbolic_program",
                    "trace_filter": "covered",
                    "grid_format": "compact",
                    "program_parse_mode": "fallback",
                    "selection_strategy": "heuristic",
                    "train_steps": 300,
                    "learning_rate": 8e-6,
                    "distillation": {"enabled": False, "weight": 0.1, "temperature": 1.0, "on": "response"},
                    "include_symbolic_candidates": False,
                    "eval_checkpoint_ladder": False,
                },
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "needs_review",
        "dense_summary": str(dense),
        "recurrent_summary": str(tmp_path / "recurrent" / "summary.json"),
        "metadata_differences": {
            "eval_checkpoint_ladder": {"dense": "False", "recurrent": "True"},
            "synthetic_tasks": {"dense": "0", "recurrent": "200"},
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Rerun recurrent ARC-AGI SFT matched to dense recipe"
    assert "Mismatched fields: eval_checkpoint_ladder, synthetic_tasks" in actions[0]["reason"]
    assert "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT=80" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=40" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER=0" in actions[0]["command"]
    assert "STAGE5_ARC_AGI_SYNTHETIC_TASKS=0" not in actions[0]["command"]
    assert "python colab/run_stage5_arc_agi_sft.py" in actions[0]["command"]


def test_recipe_selector_rescore_runs_conversion_assessment(tmp_path) -> None:
    recipe = tmp_path / "recipe" / "summary.json"
    source = tmp_path / "selector" / "summary.json"
    recipe.parent.mkdir()
    source.parent.mkdir()
    payload = {
        "run_id": "selector_rescore",
        "recipe_control_summary": str(recipe),
        "strategies": ["reliability_vote"],
        "rows": [
            {
                "label": "recovered",
                "selection_strategy": "reliability_vote",
                "output_summary_json": str(tmp_path / "selector" / "recovered_summary.json"),
            }
        ],
        "best_by_label": {},
        "paired_comparisons": {},
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess same-recipe selector conversion"
    assert "assess_stage5_recipe_selector_conversion.py" in actions[0]["command"]
    assert str(recipe) in actions[0]["command"]
    assert source.as_posix() in actions[0]["command"]


def test_recipe_selector_conversion_passed_runs_release_gate(tmp_path) -> None:
    source = tmp_path / "conversion" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_same_recipe_selector_conversion",
        "kind": "recipe_selector_conversion",
        "status": "passed",
        "passed": True,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run release gate with selector-conversion evidence"
    assert "STAGE5_RELEASE_GATE_RUN_ID=" in actions[0]["command"]
    assert "python colab/assess_stage5_release_gate.py" in actions[0]["command"]


def test_recipe_selector_conversion_with_candidates_adds_selector_exact_sft(tmp_path) -> None:
    source = tmp_path / "conversion" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_same_recipe_selector_conversion",
        "kind": "recipe_selector_conversion",
        "status": "passed",
        "passed": True,
        "best_selector": {"label": "recovered", "selection_strategy": "cell_vote"},
        "selector_evidence": [
            {
                "label": "recovered",
                "selection_strategy": "cell_vote",
                "selector_candidates_jsonl": "outputs/stage5/rescore/recovered__selector_cell_vote_candidates.jsonl",
            }
        ],
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run release gate with selector-conversion evidence"
    assert actions[1]["name"] == "Run selector-exact candidate-distillation SFT"
    assert "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE=selector_exact" in actions[1]["command"]
    assert "STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS=outputs/stage5/rescore/recovered__selector_cell_vote_candidates.jsonl" in actions[1]["command"]
    assert "python colab/run_stage5_arc_agi_sft.py" in actions[1]["command"]


def test_recipe_control_assessment_failed_inspects_markdown(tmp_path) -> None:
    source = tmp_path / "recipe" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "failed",
        "evidence": {
            "recurrent_vs_dense": {
                "candidate_summary": {"examples_with_targets": 40},
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect same-recipe assessment `failed`"
    assert "summary.md" in actions[0]["command"]


def test_release_gate_needs_hf_export_runs_exporter(tmp_path) -> None:
    source = tmp_path / "release" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_release_benchmark_readiness",
        "status": "needs_hf_export",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Export recurrent adapter with release-gate evidence"
    assert "python colab/run_stage5_publish_hf_adapter.py" in actions[0]["command"]


def test_release_gate_ready_runs_benchmark_suite(tmp_path) -> None:
    source = tmp_path / "release" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_release_benchmark_readiness",
        "status": "ready_for_broader_benchmarks",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run broader Stage 5 benchmark suite"
    assert "python colab/run_stage5_benchmark_suite.py" in actions[0]["command"]


def test_release_gate_other_status_inspects_markdown(tmp_path) -> None:
    source = tmp_path / "release" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_release_benchmark_readiness",
        "status": "needs_selector_conversion",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect release gate `needs_selector_conversion`"
    assert "summary.md" in actions[0]["command"]


def test_benchmark_suite_summary_inspects_markdown(tmp_path) -> None:
    source = tmp_path / "benchmark_suite" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_benchmark_suite",
        "status": "completed",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess broader benchmark suite `completed`"
    assert "python colab/assess_stage5_benchmark_suite.py" in actions[0]["command"]
    assert "--summary_json" in actions[0]["command"]


def test_benchmark_suite_assessment_negative_runs_competence_pipeline(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(planner, "A100_BUDGET_PROFILE", "gate")
    source = tmp_path / "benchmark_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_broader_benchmark_suite",
        "status": "needs_recurrent_recovery",
        "checkpoint": "outputs/stage5/balanced/phase1_step_150.pt",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run competence-preserving recurrent recovery pipeline"
    assert "python colab/run_stage5_competence_preserving_pipeline.py" in actions[0]["command"]
    assert "STAGE5_COMPETENCE_SOURCE_SUMMARY=" in actions[0]["command"]


def test_benchmark_suite_assessment_credit_saver_runs_short_recovery_probe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(planner, "A100_BUDGET_PROFILE", "credit_saver")
    source = tmp_path / "benchmark_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_broader_benchmark_suite",
        "status": "needs_recurrent_recovery",
        "checkpoint": "outputs/stage5/balanced/phase1_step_150.pt",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run competence-preserving ARC-mix proxy gate"
    assert "Credit-saving probe" in actions[0]["reason"]
    assert "python colab/run_stage5_balanced_arc_mix_gate.py" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_ARMS=arc_mix_response_w01_lr2e6" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_ARC_EVAL_LIMIT=128" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_MIN_MARGIN_DELTA=-0.05" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT=16" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_SOURCE_SUMMARY=" in actions[0]["command"]


def test_benchmark_suite_assessment_needs_review_with_negative_arc_runs_short_recovery_probe(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(planner, "A100_BUDGET_PROFILE", "credit_saver")
    source = tmp_path / "benchmark_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_broader_benchmark_suite",
        "status": "needs_review",
        "benchmarks": [
            {
                "benchmark": "arc_challenge",
                "present": True,
                "required_examples": 128,
                "paired_examples": 128,
                "correct_delta_recurrent_vs_base": -10,
            },
            {
                "benchmark": "gpqa_lite",
                "present": False,
                "required_examples": 16,
                "paired_examples": 0,
                "correct_delta_recurrent_vs_base": 0,
            },
        ],
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run competence-preserving ARC-mix proxy gate"
    assert "python colab/run_stage5_balanced_arc_mix_gate.py" in actions[0]["command"]


def test_benchmark_suite_assessment_credit_saver_keeps_source_summary_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(planner, "A100_BUDGET_PROFILE", "credit_saver")
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    suite = tmp_path / "outputs" / "stage5" / "suite" / "summary.json"
    suite.parent.mkdir(parents=True)
    suite.write_text(
        json.dumps(
            {
                "kind": "stage5_benchmark_suite",
                "checkpoint": "outputs/stage5/balanced/phase1_step_150.pt",
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "outputs" / "stage5" / "assessment" / "summary.json"
    source.parent.mkdir(parents=True)
    payload = {
        "gate": "stage5_broader_benchmark_suite",
        "status": "needs_recurrent_recovery",
        "source_summary": "outputs\\stage5\\suite\\summary.json",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert "python colab/run_stage5_balanced_arc_mix_gate.py" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_SOURCE_SUMMARY=outputs/stage5/assessment/summary.json" in actions[0]["command"]


def test_balanced_arc_mix_passed_runs_full_assessment(tmp_path) -> None:
    source = tmp_path / "arc_mix" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "proxy_lift",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run full balanced assessment for ARC-mix checkpoint"
    assert "python colab/run_stage5_recovery_full_assessment.py" in actions[0]["command"]
    assert "STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY=" in actions[0]["command"]


def test_balanced_arc_mix_decision_runs_full_assessment(tmp_path) -> None:
    source = tmp_path / "arc_mix" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "proxy_lift",
        "decision": "run_full_balanced_assessment",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run full balanced assessment for ARC-mix checkpoint"
    assert "python colab/run_stage5_recovery_full_assessment.py" in actions[0]["command"]


def test_balanced_arc_mix_decision_blocks_calibration_warning(tmp_path) -> None:
    source = tmp_path / "arc_mix" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "proxy_lift",
        "decision": "stop_for_calibration_repair",
        "blocked_reason": "Proxy lifted accuracy but failed calibration.",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run bounded depth/width routing diagnostic"
    assert "failed calibration" in actions[0]["reason"]
    assert "run_stage5_recovery_full_assessment.py" not in actions[0]["command"]
    assert "python colab/run_stage5_routing_diagnostic.py" in actions[0]["command"]


def test_balanced_arc_mix_routing_diagnostic_uses_best_checkpoint(tmp_path) -> None:
    source = tmp_path / "arc_mix" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "proxy_lift_calibration_warning",
        "decision": "stop_for_calibration_repair",
        "best_arm": {
            "best_checkpoint": {
                "checkpoint": "outputs/stage5/arc_mix/arm/phase1/phase1_step_50.pt"
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run bounded depth/width routing diagnostic"
    assert "STAGE5_RECOVERED_PHASE1_CHECKPOINT=outputs/stage5/arc_mix/arm/phase1/phase1_step_50.pt" in actions[0]["command"]
    assert "STAGE5_RECOVERED_SOURCE_SUMMARY=" in actions[0]["command"]
    assert "python colab/run_stage5_routing_diagnostic.py" in actions[0]["command"]


def test_recovery_full_assessment_nonnegative_runs_broader_benchmarks(tmp_path) -> None:
    source = tmp_path / "full_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_recovery_full_assessment",
        "status": "balanced_nonnegative",
        "balanced_assessment": {
            "status": "balanced_nonnegative",
            "best_checkpoint": {
                "checkpoint": "outputs/stage5/full/phase1_step_100.pt",
                "micro_correct_delta": 1,
            },
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run broader benchmark suite for balanced checkpoint"
    assert "python colab/run_stage5_benchmark_suite.py" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_CHECKPOINT=outputs/stage5/full/phase1_step_100.pt" in actions[0]["command"]
    assert "STAGE5_BENCHMARK_SOURCE_SUMMARY=" in actions[0]["command"]


def test_recovery_full_assessment_negative_runs_recovery_proxy(tmp_path) -> None:
    source = tmp_path / "full_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_recovery_full_assessment",
        "status": "needs_competence_recovery",
        "balanced_assessment": {
            "status": "needs_competence_recovery",
            "best_checkpoint": {
                "checkpoint": "outputs/stage5/full/phase1_step_100.pt",
                "micro_correct_delta": -1,
            },
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run another competence-preserving ARC-mix proxy gate"
    assert "python colab/run_stage5_balanced_arc_mix_gate.py" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_SOURCE_SUMMARY=" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_MIN_MARGIN_DELTA=-0.05" in actions[0]["command"]
    assert "STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT=16" in actions[0]["command"]


def test_recovery_full_assessment_child_failure_inspects_only(tmp_path) -> None:
    source = tmp_path / "full_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_recovery_full_assessment",
        "status": "benchmark_child_failed",
        "passed": False,
        "child_returncode": 7,
        "child_summary_path": "outputs/stage5/full_assessment_balanced_full/summary.json",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect failed full balanced assessment child `benchmark_child_failed`"
    assert actions[0]["command"].startswith("cat ")
    assert "return code `7`" in actions[0]["reason"]
    assert "outputs/stage5/full_assessment_balanced_full/summary.json" in actions[0]["reason"]
    assert "run_stage5_balanced_arc_mix_gate.py" not in actions[0]["command"]
    assert "run_stage5_benchmark_suite.py" not in actions[0]["command"]


def test_parse_args_accepts_source_summary() -> None:
    from colab.plan_stage5_next_run import parse_args

    args = parse_args(["--source-summary", "outputs/stage5/run/summary.json"])

    assert args.source_summary == "outputs/stage5/run/summary.json"


def test_balanced_arc_mix_failed_inspects_summary(tmp_path) -> None:
    source = tmp_path / "arc_mix" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "no_proxy_lift",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run bounded depth/width routing diagnostic"
    assert "python colab/run_stage5_routing_diagnostic.py" in actions[0]["command"]


def test_balanced_arc_mix_calibration_warning_does_not_run_full_assessment(tmp_path) -> None:
    source = tmp_path / "arc_mix" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "stage5_balanced_arc_mix_gate",
        "status": "proxy_lift_calibration_warning",
        "passed": False,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Run bounded depth/width routing diagnostic"
    assert "run_stage5_recovery_full_assessment.py" not in actions[0]["command"]
    assert "python colab/run_stage5_routing_diagnostic.py" in actions[0]["command"]


def test_benchmark_suite_assessment_passed_builds_claim_packet(tmp_path) -> None:
    source = tmp_path / "benchmark_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_broader_benchmark_suite",
        "status": "passed",
        "passed": True,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Build Stage 5 claim readiness packet"
    assert "python colab/build_stage5_claim_packet.py" in actions[0]["command"]


def test_benchmark_suite_assessment_low_coverage_expands_suite(tmp_path) -> None:
    source = tmp_path / "benchmark_assessment" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_broader_benchmark_suite",
        "status": "needs_benchmark_confirmation",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Expand broader benchmark suite confirmation"
    assert "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT=256" in actions[0]["command"]
    assert "python colab/run_stage5_benchmark_suite.py" in actions[0]["command"]


def test_claim_readiness_missing_export_runs_exporter(tmp_path) -> None:
    source = tmp_path / "claim" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_claim_readiness",
        "status": "needs_hf_export",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Export recurrent adapter for claim packet"
    assert "python colab/run_stage5_publish_hf_adapter.py" in actions[0]["command"]


def test_claim_readiness_missing_selector_replication_runs_assessor(tmp_path) -> None:
    source = tmp_path / "claim" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_claim_readiness",
        "status": "needs_selector_replication",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess selector replication for claim packet"
    assert "python colab/assess_stage5_selector_replication.py" in actions[0]["command"]


def test_claim_readiness_missing_particle_gate_runs_gate2_assessor(tmp_path) -> None:
    source = tmp_path / "claim" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_claim_readiness",
        "status": "needs_particle_mechanism_gate",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Assess Gate 2 particle mechanism for claim packet"
    assert "python colab/assess_stage5_gate2.py" in actions[0]["command"]


def test_claim_readiness_release_candidate_inspects_markdown(tmp_path) -> None:
    source = tmp_path / "claim" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_claim_readiness",
        "status": "ready_for_release_candidate_not_sota",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Build ARC-AGI same-size comparison artifact"
    assert "python colab/build_stage5_arc_agi_sota_comparison.py" in actions[0]["command"]


def test_claim_readiness_with_reproduced_control_requests_public_baselines(tmp_path) -> None:
    source = tmp_path / "claim" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_claim_readiness",
        "status": "ready_for_release_candidate_not_sota",
        "artifacts": {
            "arc_agi_comparison": {
                "summary": {
                    "status": "passed_reproduced_control",
                    "comparison_scope": "reproduced_control",
                }
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Add public-source ARC-AGI same-size baselines"
    assert "summary.md" in actions[0]["command"]


def test_claim_readiness_sota_export_linkage_runs_hf_exporter(tmp_path) -> None:
    source = tmp_path / "claim" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_claim_readiness",
        "status": "ready_for_release_candidate_needs_sota_export_linkage",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Rebuild claim packet inputs with matched SOTA export linkage"
    assert "python colab/run_stage5_publish_hf_adapter.py" in actions[0]["command"]


def test_arc_agi_sota_comparison_passed_rebuilds_claim_packet(tmp_path) -> None:
    source = tmp_path / "arc_agi_sota" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_sota_comparison",
        "status": "passed",
        "passed": True,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Rebuild claim packet with ARC-AGI comparison"
    assert "python colab/build_stage5_claim_packet.py" in actions[0]["command"]


def test_arc_agi_sota_comparison_reproduced_control_rebuilds_claim_packet(tmp_path) -> None:
    source = tmp_path / "arc_agi_sota" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_sota_comparison",
        "status": "passed_reproduced_control",
        "passed": False,
        "control_passed": True,
        "comparison_scope": "reproduced_control",
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Rebuild claim packet with reproduced ARC-AGI control"
    assert "python colab/build_stage5_claim_packet.py" in actions[0]["command"]


def test_arc_agi_baseline_registry_passed_builds_sota_comparison(tmp_path) -> None:
    source = tmp_path / "arc_agi_registry" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_baseline_registry",
        "status": "passed",
        "passed": True,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Build ARC-AGI same-size comparison artifact"
    assert "python colab/build_stage5_arc_agi_sota_comparison.py" in actions[0]["command"]


def test_arc_agi_baseline_registry_missing_values_inspects_markdown(tmp_path) -> None:
    source = tmp_path / "arc_agi_registry" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_baseline_registry",
        "status": "needs_baseline_registry",
        "passed": False,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Inspect ARC-AGI baseline registry `needs_baseline_registry`"
    assert "summary.md" in actions[0]["command"]


def test_arc_agi_sota_comparison_missing_registry_runs_validator(tmp_path) -> None:
    source = tmp_path / "arc_agi_sota" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_sota_comparison",
        "status": "needs_baseline_registry",
        "passed": False,
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Validate ARC-AGI same-size baseline registry"
    assert "STAGE5_ARC_AGI_BASELINE_REGISTRY_RUN_ID=" in actions[0]["command"]
    assert "python colab/validate_arc_agi_baseline_registry.py" in actions[0]["command"]


def test_arc_agi_sota_comparison_missing_registry_builds_reproduced_registry_when_candidate_path_exists(
    tmp_path,
) -> None:
    source = tmp_path / "arc_agi_sota" / "summary.json"
    source.parent.mkdir()
    payload = {
        "gate": "stage5_arc_agi_sota_comparison",
        "status": "needs_baseline_registry",
        "passed": False,
        "candidate": {
            "path": "outputs/stage5/candidate/summary.json",
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Build reproduced ARC-AGI baseline registry"
    assert "python colab/build_stage5_arc_agi_reproduced_baseline_registry.py" in actions[0]["command"]
    assert "--summary_json outputs/stage5/candidate/summary.json" in actions[0]["command"]
    assert "--labels base" in actions[0]["command"]
    assert "--validation_json outputs/stage5/" in actions[0]["command"]
