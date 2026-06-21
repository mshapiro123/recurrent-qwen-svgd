from __future__ import annotations

import json

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
    assert "STAGE5_ARC_AGI_RESCORE_STRATEGIES=self_consistency,reliability_vote,symbolic_priority" in rescore["command"]


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


def test_gate1_passed_delegates_to_source_summary_action(tmp_path) -> None:
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

    assert actions[0]["name"].startswith("Gate 1 passed: Promote selector `reliability_vote`")
    assert "hard-tail lift" in actions[0]["reason"]
    assert "STAGE5_ARC_AGI_SELECTION_STRATEGY=reliability_vote" in actions[0]["command"]


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
    assert source_kind({"gate": "stage5_same_recipe_architecture"}) == "recipe_control_assessment"
    assert source_kind({"gate": "stage5_release_benchmark_readiness"}) == "release_gate"
    assert source_kind({"kind": "stage5_benchmark_suite"}) == "benchmark_suite"
    assert source_kind({"phase1_arc_agi_tuned": {}, "tuned_checkpoint": "ckpt.pt"}) == "recurrent_sft"
    assert source_kind({"rows": [], "deltas": {}, "paired_comparisons": {}}) == "tta_sweep"


def test_dense_sft_control_plans_matched_recurrent_recipe(tmp_path) -> None:
    source = tmp_path / "dense" / "summary.json"
    source.parent.mkdir()
    payload = {
        "kind": "dense_sft_control",
        "metadata": {
            "arc_version": "1",
            "train_task_limit": 80,
            "eval_task_limit": 12,
            "trace_mode": "symbolic_program",
            "trace_filter": "covered",
            "grid_format": "compact",
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
    assert "STAGE5_ARC_AGI_TRACE_MODE=symbolic_program" in actions[0]["command"]
    assert "python colab/run_stage5_arc_agi_sft.py" in actions[0]["command"]
    assert "Dense-vs-base evidence: paired delta 2" in actions[0]["reason"]


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
    source.parent.mkdir()
    payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "passed",
        "evidence": {
            "recurrent_vs_dense": {
                "candidate_summary": {"examples_with_targets": 40},
            }
        },
    }

    actions = plan_next_actions(payload, source_summary=source)

    assert actions[0]["name"] == "Replicate dense control at ARC limit 100"
    assert "python colab/run_stage5_arc_agi_dense_sft.py" in actions[0]["command"]


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
    assert "python colab/run_stage5_arc_agi_rescore_selectors.py" in actions[0]["command"]


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

    assert actions[0]["name"] == "Inspect broader benchmark suite `completed`"
    assert "summary.md" in actions[0]["command"]
