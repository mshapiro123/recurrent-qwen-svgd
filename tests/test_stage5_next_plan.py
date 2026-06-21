from __future__ import annotations

import json

from colab.plan_stage5_next_run import (
    best_recovered_tta_row,
    evidence_fragment,
    next_validation_limit,
    paired_delta_or_aggregate,
    paired_metric,
    plan_next_actions,
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

    actions = plan_next_actions(payload, source_summary=source)

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

    actions = plan_next_actions(payload, source_summary=source)

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

    actions = plan_next_actions(payload, source_summary=source)

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

    actions = plan_next_actions(payload, source_summary=source)

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

    actions = plan_next_actions(payload, source_summary=source)
    particle = next(action for action in actions if action["name"].startswith("Replicate particle value"))

    assert "k4_noise001_rep05" in particle["name"]
    assert "run_stage5_arc_agi_recovery_particle_gate.py" in particle["command"]
    assert "STAGE5_ARC_AGI_EVAL_TASK_LIMIT=400" in particle["command"]
    assert "STAGE5_ARC_AGI_PARTICLE_SEEDS=0,1,2,3,4" in particle["command"]
    assert "STAGE5_ARC_AGI_PARTICLE_VARIANTS=k4_noise001_rep05:0.01:0.5" in particle["command"]


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
    assert source_kind({"recovery_decision": {}, "particle_decision": {}}) == "recovery_particle_gate"
    assert source_kind({"gate": "stage5_gate1_selector_tta"}) == "gate1_assessment"
