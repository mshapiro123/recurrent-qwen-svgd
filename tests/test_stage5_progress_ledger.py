from __future__ import annotations

import json
import os
from pathlib import Path

from colab.summarize_stage5_progress import scan_progress, write_report


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary(selected: int, best: int, examples: int = 20) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_exact": selected,
        "examples_with_targets": examples,
        "selected_accuracy": selected / examples,
        "best_of_k_accuracy": best / examples,
        "valid_candidate_rate": 1.0,
    }


def test_progress_ledger_reads_recovered_benchmark_and_gaps(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "bench" / "summary.json"
    _write(
        source,
        {
            "run_id": "bench",
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 3)},
            "recovered": {"summary": _summary(6, 7)},
            "deltas": {},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 3
    assert payload["best_by_arm"]["base"]["selected_exact"] == 8
    assert payload["best_by_arm"]["recovered"]["best_of_k_exact"] == 7
    assert payload["recovered_vs_base_gaps"] == [
        {
            "run_id": "bench",
            "examples": 20,
            "selected_initial_gap_to_base": 6,
            "selected_gain_from_start": 4,
            "selected_delta_recovered_vs_base": -2,
            "selected_gap_closure_fraction": 4 / 6,
            "best_of_k_initial_gap_to_base": 6,
            "best_of_k_gain_from_start": 4,
            "best_of_k_delta_recovered_vs_base": -2,
            "best_of_k_gap_closure_fraction": 4 / 6,
            "path": str(source),
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_selector_rescore_rows(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "selector" / "summary.json"
    _write(
        source,
        {
            "run_id": "selector",
            "source_run_dir": "outputs/stage5/source",
            "strategies": ["self_consistency"],
            "rows": [
                {
                    "label": "recovered",
                    "selection_strategy": "self_consistency",
                    "examples": 50,
                    "selected_exact": 12,
                    "best_of_k_exact": 14,
                    "valid_candidate_rate": 0.9,
                }
            ],
            "best_by_label": {},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 1
    record = payload["records"][0]
    assert record["kind"] == "selector_rescore"
    assert record["arm"] == "recovered"
    assert record["label"] == "self_consistency"
    assert record["selected_exact"] == 12
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_recovery_particle_gate(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "particle" / "summary.json"
    _write(
        source,
        {
            "run_id": "particle",
            "recovered_checkpoint": {
                "checkpoint": "outputs/stage5/recovered.pt",
                "summary": _summary(3, 4, examples=10),
            },
            "recovery_decision": {"passed": True, "evidence": {}},
            "particle_decision": {
                "passed": True,
                "evidence": {
                    "variants": {
                        "svgd": {
                            "passed": True,
                            "mean_delta_vs_tuned": {
                                "selected_delta": 1,
                                "best_of_k_delta": 2,
                            },
                        }
                    }
                },
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 2
    recovered = next(record for record in payload["records"] if record["arm"] == "recovered")
    particle = next(record for record in payload["records"] if record["arm"] == "particle")
    assert recovered["selected_exact"] == 3
    assert particle["label"] == "svgd"
    assert particle["selected_delta_vs_recovered"] == 1.0
    assert particle["best_of_k_delta_vs_recovered"] == 2.0
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_dense_sft_control(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "dense" / "summary.json"
    _write(
        source,
        {
            "run_id": "dense",
            "kind": "dense_sft_control",
            "base": {"summary": _summary(5, 6, examples=10)},
            "dense_tuned": {"summary": _summary(7, 8, examples=10)},
            "phase1_start": {"summary": _summary(4, 5, examples=10)},
            "deltas": {},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 3
    dense = next(record for record in payload["records"] if record["arm"] == "dense_tuned")
    assert dense["kind"] == "dense_sft_control"
    assert dense["selected_exact"] == 7
    assert payload["best_by_arm"]["dense_tuned"]["best_of_k_exact"] == 8
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_recurrent_sft_summary(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "recurrent" / "summary.json"
    _write(
        source,
        {
            "run_id": "recurrent",
            "base": _summary(5, 6, examples=10),
            "phase1_start": _summary(4, 5, examples=10),
            "phase1_arc_agi_tuned": _summary(7, 8, examples=10),
            "tuned_checkpoint": "outputs/stage5/recurrent/phase1.pt",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 3
    tuned = next(record for record in payload["records"] if record["arm"] == "recurrent_tuned")
    assert tuned["kind"] == "recurrent_sft"
    assert tuned["selected_exact"] == 7
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_gate1_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "gate1" / "summary.json"
    _write(
        source,
        {
            "run_id": "gate1",
            "gate": "stage5_gate1_selector_tta",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/selector/summary.json",
            "source_kind": "selector_rescore",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "passing_comparisons": ["selector_vs_source"],
            "tradeoff_comparisons": [],
            "num_comparisons": 1,
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["gate1_assessments"] == [
        {
            "path": str(source),
            "run_id": "gate1",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/selector/summary.json",
            "source_kind": "selector_rescore",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "passing_comparisons": ["selector_vs_source"],
            "tradeoff_comparisons": [],
            "num_comparisons": 1,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_gate2_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "gate2" / "summary.json"
    _write(
        source,
        {
            "run_id": "gate2",
            "gate": "stage5_gate2_particle_mechanism",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/particle/summary.json",
            "source_kind": "recovery_particle_gate",
            "reason": "replicated selected lift",
            "next_step": "replicate",
            "best_variant": {
                "variant": "svgd",
                "selected_delta": 1,
                "best_of_k_delta": 2,
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["gate2_assessments"] == [
        {
            "path": str(source),
            "run_id": "gate2",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/particle/summary.json",
            "source_kind": "recovery_particle_gate",
            "reason": "replicated selected lift",
            "next_step": "replicate",
            "best_variant": "svgd",
            "selected_delta": 1.0,
            "best_of_k_delta": 2.0,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_selector_replications(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "selector_replication" / "summary.json"
    _write(
        source,
        {
            "run_id": "selector_replication",
            "gate": "stage5_selector_replication",
            "kind": "selector_replication",
            "status": "passed",
            "passed": True,
            "replicated_comparisons": ["recovered__selector_reliability_vote_vs_source"],
            "next_step": "use selector",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["selector_replications"] == [
        {
            "path": str(source),
            "run_id": "selector_replication",
            "status": "passed",
            "passed": True,
            "replicated_comparisons": ["recovered__selector_reliability_vote_vs_source"],
            "next_step": "use selector",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_recipe_selector_conversions(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "recipe_selector_conversion" / "summary.json"
    _write(
        source,
        {
            "run_id": "recipe_selector_conversion",
            "gate": "stage5_same_recipe_selector_conversion",
            "kind": "recipe_selector_conversion",
            "status": "passed",
            "passed": True,
            "passing_selectors": [{"label": "recovered", "selection_strategy": "reliability_vote"}],
            "best_selector": {"label": "recovered", "selection_strategy": "reliability_vote"},
            "selector_evidence": [
                {
                    "label": "recovered",
                    "selection_strategy": "reliability_vote",
                    "claim_level_selector": True,
                    "selector_generated_selected_exact": 2,
                    "selected_exceeds_best_of_k": 1,
                }
            ],
            "next_step": "reassess architecture",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recipe_selector_conversions"] == [
        {
            "path": str(source),
            "run_id": "recipe_selector_conversion",
            "status": "passed",
            "passed": True,
            "passing_selectors": [{"label": "recovered", "selection_strategy": "reliability_vote"}],
            "best_selector": {"label": "recovered", "selection_strategy": "reliability_vote"},
            "claim_level_selector": True,
            "selector_generated_selected_exact": 2,
            "selected_exceeds_best_of_k": 1,
            "next_step": "reassess architecture",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_recipe_control_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "recipe" / "summary.json"
    _write(
        source,
        {
            "run_id": "recipe",
            "gate": "stage5_same_recipe_architecture",
            "status": "passed",
            "passed": True,
            "dense_summary": "outputs/stage5/dense/summary.json",
            "recurrent_summary": "outputs/stage5/recurrent/summary.json",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "decision_evidence": {
                "aggregate": {"delta_exact": 1},
                "hard": {"delta_exact": 2},
                "aggregate_best_of_k": {"delta_exact": 3},
                "hard_best_of_k": {"delta_exact": 4},
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recipe_control_assessments"] == [
        {
            "path": str(source),
            "run_id": "recipe",
            "gate": "stage5_same_recipe_architecture",
            "status": "passed",
            "passed": True,
            "dense_summary": "outputs/stage5/dense/summary.json",
            "recurrent_summary": "outputs/stage5/recurrent/summary.json",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "aggregate_selected_delta": 1,
            "hard_selected_delta": 2,
            "aggregate_best_of_k_delta": 3,
            "hard_best_of_k_delta": 4,
            "primary_delta_recurrent_vs_dense": 0,
            "arc_challenge_content_delta_recurrent_vs_dense": 0,
            "arc_challenge_cyclic_delta_recurrent_vs_dense": 0,
            "arc_easy_content_delta_recurrent_vs_dense": 0,
            "arc_easy_cyclic_delta_recurrent_vs_dense": 0,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_mcq_recipe_control_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "mcq_recipe" / "summary.json"
    _write(
        source,
        {
            "run_id": "mcq_recipe",
            "kind": "stage5_mcq_recipe_control_assessment",
            "gate": "stage5_same_recipe_mcq_architecture",
            "status": "hard_tail_lift_vs_dense",
            "passed": True,
            "dense_summary": "outputs/stage5/dense/summary.json",
            "recurrent_summary": "outputs/stage5/recurrent/summary.json",
            "reason": "challenge cyclic lift",
            "next_step": "replicate",
            "decision_evidence": {
                "primary": {"correct_delta_recurrent_vs_dense": 2},
                "arc_challenge_content": {"correct_delta_recurrent_vs_dense": 1},
                "arc_challenge_cyclic": {"correct_delta_recurrent_vs_dense": 2},
                "arc_easy_content": {"correct_delta_recurrent_vs_dense": -1},
                "arc_easy_cyclic": {"correct_delta_recurrent_vs_dense": 0},
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recipe_control_assessments"] == [
        {
            "path": str(source),
            "run_id": "mcq_recipe",
            "gate": "stage5_same_recipe_mcq_architecture",
            "status": "hard_tail_lift_vs_dense",
            "passed": True,
            "dense_summary": "outputs/stage5/dense/summary.json",
            "recurrent_summary": "outputs/stage5/recurrent/summary.json",
            "reason": "challenge cyclic lift",
            "next_step": "replicate",
            "aggregate_selected_delta": 0,
            "hard_selected_delta": 0,
            "aggregate_best_of_k_delta": 0,
            "hard_best_of_k_delta": 0,
            "primary_delta_recurrent_vs_dense": 2,
            "arc_challenge_content_delta_recurrent_vs_dense": 1,
            "arc_challenge_cyclic_delta_recurrent_vs_dense": 2,
            "arc_easy_content_delta_recurrent_vs_dense": -1,
            "arc_easy_cyclic_delta_recurrent_vs_dense": 0,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_surface_alignment_repairs(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "surface" / "summary.json"
    _write(
        source,
        {
            "run_id": "surface",
            "kind": "stage5_surface_alignment_repair",
            "status": "surface_alignment_partial",
            "passed": False,
            "source_summary": "outputs/stage5/source/summary.json",
            "benchmark_summary": "outputs/stage5/surface_bench/summary.json",
            "checkpoint": "outputs/stage5/surface/phase1.pt",
            "surface_alignment_rows": 32,
            "surface_repair_assessment_status": "surface_repair_partial",
            "assessment_status": "failed",
            "next_step": "confirm before dense",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["surface_alignment_statuses"] == [
        {
            "path": str(source),
            "run_id": "surface",
            "status": "surface_alignment_partial",
            "passed": False,
            "source_summary": "outputs/stage5/source/summary.json",
            "benchmark_summary": "outputs/stage5/surface_bench/summary.json",
            "checkpoint": "outputs/stage5/surface/phase1.pt",
            "surface_alignment_rows": 32,
            "surface_repair_assessment_status": "surface_repair_partial",
            "assessment_status": "failed",
            "next_step": "confirm before dense",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_dense_mcq_controls(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "dense_mcq" / "summary.json"
    _write(
        source,
        {
            "run_id": "dense_mcq",
            "kind": "stage5_dense_mcq_trace_sft_control",
            "source_summary": "outputs/stage5/source/summary.json",
            "recurrent_benchmark_summary": "outputs/stage5/recurrent/summary.json",
            "dataset": {"train_rows": 64, "extra_train_rows": 12},
            "dense_checkpoint": "outputs/stage5/dense_mcq/dense.pt",
            "recipe_control_assessment": {
                "ran": True,
                "status": "mixed_hard_tail_signal_vs_dense",
                "passed": False,
                "summary_json": "outputs/stage5/dense_mcq/mcq_recipe_control_assessment.json",
                "next_step": "inspect challenge content",
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["dense_mcq_control_statuses"] == [
        {
            "path": str(source),
            "run_id": "dense_mcq",
            "source_summary": "outputs/stage5/source/summary.json",
            "recurrent_benchmark_summary": "outputs/stage5/recurrent/summary.json",
            "train_rows": 64,
            "extra_train_rows": 12,
            "dense_checkpoint": "outputs/stage5/dense_mcq/dense.pt",
            "assessment_ran": True,
            "assessment_status": "mixed_hard_tail_signal_vs_dense",
            "assessment_passed": False,
            "assessment_summary": "outputs/stage5/dense_mcq/mcq_recipe_control_assessment.json",
            "next_step": "inspect challenge content",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_release_gate_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "release" / "summary.json"
    _write(
        source,
        {
            "run_id": "release",
            "gate": "stage5_release_benchmark_readiness",
            "status": "needs_hf_export",
            "passed": False,
            "next_step": "export",
            "min_arc_examples": 100,
            "criteria": [
                {"name": "arc_benchmark_confirmation", "passed": True},
                {"name": "same_recipe_architecture", "passed": True},
                {"name": "hf_export_artifact", "passed": False},
            ],
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["release_gate_assessments"] == [
        {
            "path": str(source),
            "run_id": "release",
            "status": "needs_hf_export",
            "passed": False,
            "next_step": "export",
            "min_arc_examples": 100,
            "criteria": [
                {"name": "arc_benchmark_confirmation", "passed": True},
                {"name": "same_recipe_architecture", "passed": True},
                {"name": "hf_export_artifact", "passed": False},
            ],
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_benchmark_suite_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "bench_suite" / "summary.json"
    _write(
        source,
        {
            "run_id": "bench_suite",
            "kind": "stage5_benchmark_suite",
            "status": "completed",
            "checkpoint": "outputs/recurrent.pt",
            "benchmarks": ["arc_challenge"],
            "comparisons": {
                "arc_challenge": {
                    "label": {
                        "mean": {
                            "correct_delta_recurrent_vs_base": -2,
                            "accuracy_delta_recurrent_vs_base": -0.125,
                        }
                    }
                }
            },
            "paired_comparisons": {
                "arc_challenge": {
                    "label": {
                        "mean": {
                            "paired_examples": 16,
                            "wins": 1,
                            "losses": 3,
                            "ties": 12,
                            "sign_test_p_value": 0.625,
                        }
                    }
                }
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["benchmark_suite_assessments"] == [
        {
            "path": str(source),
            "run_id": "bench_suite",
            "status": "completed",
            "checkpoint": "outputs/recurrent.pt",
            "benchmarks": ["arc_challenge"],
            "deltas": [
                {
                    "benchmark": "arc_challenge",
                    "score_target": "label",
                    "aggregate": "mean",
                    "correct_delta_recurrent_vs_base": -2,
                    "accuracy_delta_recurrent_vs_base": -0.125,
                    "paired_examples": 16,
                    "wins": 1,
                    "losses": 3,
                    "ties": 12,
                    "sign_test_p_value": 0.625,
                }
            ],
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_broader_benchmark_gate_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "bench_gate" / "summary.json"
    _write(
        source,
        {
            "run_id": "bench_gate",
            "gate": "stage5_broader_benchmark_suite",
            "status": "needs_recurrent_recovery",
            "passed": False,
            "source_summary": "outputs/stage5/suite/summary.json",
            "next_step": "recover recurrent",
            "benchmarks": [
                {
                    "benchmark": "arc_challenge",
                    "correct_delta_recurrent_vs_base": -2,
                }
            ],
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["broader_benchmark_gate_assessments"] == [
        {
            "path": str(source),
            "run_id": "bench_gate",
            "status": "needs_recurrent_recovery",
            "passed": False,
            "source_summary": "outputs/stage5/suite/summary.json",
            "next_step": "recover recurrent",
            "benchmarks": [
                {
                    "benchmark": "arc_challenge",
                    "correct_delta_recurrent_vs_base": -2,
                }
            ],
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_full_balanced_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    proxy = scan_root / "arc_mix_proxy" / "summary.json"
    source = scan_root / "full_assessment" / "summary.json"
    _write(
        proxy,
        {
            "run_id": "arc_mix_proxy",
            "kind": "stage5_balanced_arc_mix_gate",
            "status": "proxy_lift",
            "passed": True,
        },
    )
    _write(
        source,
        {
            "run_id": "full_assessment",
            "kind": "stage5_recovery_full_assessment",
            "status": "needs_competence_recovery",
            "passed": False,
            "selected_checkpoint": "outputs/stage5/full/phase1_step_100.pt",
            "balanced_assessment": {
                "status": "needs_competence_recovery",
                "passed": False,
                "next_step": "recover more",
                "best_checkpoint": {
                    "label": "step_100",
                    "checkpoint": "outputs/stage5/full/phase1_step_100.pt",
                    "micro_correct_delta": -1,
                    "required_macro_accuracy_delta": -0.002,
                    "base_correct": 588,
                    "recurrent_correct": 587,
                    "total": 869,
                    "combined_wins": 34,
                    "combined_losses": 35,
                    "combined_ties": 800,
                },
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["balanced_full_assessments"] == [
        {
            "path": str(source),
            "run_id": "full_assessment",
            "kind": "stage5_recovery_full_assessment",
            "status": "needs_competence_recovery",
            "passed": False,
            "selected_checkpoint": "outputs/stage5/full/phase1_step_100.pt",
            "label": "step_100",
            "micro_correct_delta": -1,
            "macro_accuracy_delta": -0.002,
            "base_correct": 588,
            "recurrent_correct": 587,
            "total": 869,
            "combined_wins": 34,
            "combined_losses": 35,
            "combined_ties": 800,
            "child_returncode": None,
            "child_summary_path": None,
            "child_stdout_tail": None,
            "next_step": "recover more",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_full_assessment_child_failure(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "full_assessment_failed" / "summary.json"
    _write(
        source,
        {
            "run_id": "full_assessment_failed",
            "kind": "stage5_recovery_full_assessment",
            "status": "balanced_assessment_child_failed",
            "passed": False,
            "selected_checkpoint": "outputs/stage5/full/phase1_step_100.pt",
            "child_returncode": 8,
            "child_summary_path": "outputs/stage5/full/balanced_assessment/summary.json",
            "child_stdout_tail": "assessment died",
            "next_step": "inspect child output",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["balanced_full_assessments"] == [
        {
            "path": str(source),
            "run_id": "full_assessment_failed",
            "kind": "stage5_recovery_full_assessment",
            "status": "balanced_assessment_child_failed",
            "passed": False,
            "selected_checkpoint": "outputs/stage5/full/phase1_step_100.pt",
            "label": None,
            "micro_correct_delta": None,
            "macro_accuracy_delta": None,
            "base_correct": None,
            "recurrent_correct": None,
            "total": None,
            "combined_wins": None,
            "combined_losses": None,
            "combined_ties": None,
            "child_returncode": 8,
            "child_summary_path": "outputs/stage5/full/balanced_assessment/summary.json",
            "child_stdout_tail": "assessment died",
            "next_step": "inspect child output",
        }
    ]
    output_dir = tmp_path / "ledger"
    write_report(payload, output_dir)
    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "status `balanced_assessment_child_failed`" in report
    assert "child return `8`" in report
    assert "outputs/stage5/full/balanced_assessment/summary.json" in report


def test_progress_ledger_reports_claim_readiness_packets(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "claim" / "summary.json"
    _write(
        source,
        {
            "run_id": "claim",
            "gate": "stage5_claim_readiness",
            "status": "ready_for_release_candidate_not_sota",
            "passed": True,
            "claim_level": "release_candidate",
            "artifacts": {
                "sota_export_linkage": {
                    "passed": False,
                    "verified": False,
                    "matched_on": None,
                    "reason": "No authoritative ARC-AGI same-size SOTA comparison proves a SOTA claim.",
                }
            },
            "next_step": "write report without SOTA claim",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["claim_readiness_packets"] == [
        {
            "path": str(source),
            "run_id": "claim",
            "status": "ready_for_release_candidate_not_sota",
            "passed": True,
            "claim_level": "release_candidate",
            "sota_export_linkage_passed": False,
            "sota_export_linkage_verified": False,
            "sota_export_linkage_matched_on": None,
            "sota_export_linkage_reason": "No authoritative ARC-AGI same-size SOTA comparison proves a SOTA claim.",
            "next_step": "write report without SOTA claim",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_sota_export_linkage_status(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "claim_linkage" / "summary.json"
    _write(
        source,
        {
            "run_id": "claim_linkage",
            "gate": "stage5_claim_readiness",
            "status": "ready_for_release_candidate_needs_sota_export_linkage",
            "passed": True,
            "claim_level": "release_candidate",
            "artifacts": {
                "sota_export_linkage": {
                    "passed": False,
                    "verified": True,
                    "matched_on": "checkpoint",
                    "reason": "HF export checkpoint does not match the ARC-AGI candidate checkpoint.",
                }
            },
            "next_step": "rebuild matched export",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["claim_readiness_packets"] == [
        {
            "path": str(source),
            "run_id": "claim_linkage",
            "status": "ready_for_release_candidate_needs_sota_export_linkage",
            "passed": True,
            "claim_level": "release_candidate",
            "sota_export_linkage_passed": False,
            "sota_export_linkage_verified": True,
            "sota_export_linkage_matched_on": "checkpoint",
            "sota_export_linkage_reason": "HF export checkpoint does not match the ARC-AGI candidate checkpoint.",
            "next_step": "rebuild matched export",
        }
    ]


def test_progress_ledger_reports_arc_agi_sota_comparisons(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "arc_agi_sota" / "summary.json"
    _write(
        source,
        {
            "run_id": "arc_agi_sota",
            "gate": "stage5_arc_agi_sota_comparison",
            "kind": "arc_agi_sota_comparison",
            "status": "failed",
            "passed": False,
            "metric": "selected_accuracy",
            "candidate": {"accuracy": 0.08},
            "best_baseline": {"name": "same-size-baseline", "accuracy": 0.1},
            "delta_accuracy_vs_best_baseline": -0.02,
            "next_step": "train more",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["arc_agi_sota_comparisons"] == [
        {
            "path": str(source),
            "run_id": "arc_agi_sota",
            "status": "failed",
            "passed": False,
            "metric": "selected_accuracy",
            "candidate_accuracy": 0.08,
            "best_baseline": "same-size-baseline",
            "best_baseline_accuracy": 0.1,
            "delta_accuracy_vs_best_baseline": -0.02,
            "next_step": "train more",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_arc_agi_baseline_registries(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "arc_agi_registry" / "summary.json"
    _write(
        source,
        {
            "run_id": "arc_agi_registry",
            "gate": "stage5_arc_agi_baseline_registry",
            "kind": "arc_agi_baseline_registry",
            "status": "passed",
            "passed": True,
            "metric": "selected_accuracy",
            "valid_baseline_count": 1,
            "best_baseline": {"name": "same-size-baseline", "accuracy": 0.1},
            "next_step": "compare",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["arc_agi_baseline_registries"] == [
        {
            "path": str(source),
            "run_id": "arc_agi_registry",
            "status": "passed",
            "passed": True,
            "metric": "selected_accuracy",
            "valid_baseline_count": 1,
            "best_baseline": "same-size-baseline",
            "best_baseline_accuracy": 0.1,
            "next_step": "compare",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_arc_agi_candidate_gates(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "candidate_gate" / "summary.json"
    _write(
        source,
        {
            "run_id": "candidate_gate",
            "gate": "stage5_arc_agi_candidate_gate",
            "kind": "stage5_arc_agi_candidate_gate",
            "metadata": {
                "arc_version": "arc-agi-1",
                "arc_split": "evaluation",
                "limit": 8,
                "grid_format": "compact",
                "selection_strategy": "symbolic_first",
            },
            "symbolic_coverage": {
                "examples_with_targets": 8,
                "exact_symbolic": 2,
            },
            "rows": [
                {"variant": "phase1_model_only", "best": 1},
                {"variant": "phase1_hybrid_symbolic_first", "best": 3},
                {"variant": "base_model_only", "best": 4},
                {"variant": "base_hybrid_symbolic_first", "best": 5},
            ],
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["arc_agi_candidate_gates"] == [
        {
            "path": str(source),
            "run_id": "candidate_gate",
            "arc_version": "arc-agi-1",
            "arc_split": "evaluation",
            "limit": 8,
            "grid_format": "compact",
            "selection_strategy": "symbolic_first",
            "examples": 8,
            "symbolic_exact": 2,
            "phase1_model_best": 1,
            "phase1_hybrid_best": 3,
            "phase1_hybrid_best_delta": 2,
            "base_model_best": 4,
            "base_hybrid_best": 5,
            "base_hybrid_best_delta": 1,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_arc_agi_sft_recipe_gates(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    trace_source = scan_root / "trace_gate" / "summary.json"
    distill_source = scan_root / "distill_gate" / "summary.json"
    _write(
        trace_source,
        {
            "run_id": "trace_gate",
            "kind": "trace_sft_gate",
            "arms": [
                {"label": "grid_only", "trace_mode": "none", "trace_filter": "all"},
                {"label": "symbolic_program_trace_covered", "trace_mode": "symbolic_program", "trace_filter": "covered"},
            ],
            "comparison": {
                "grid_only": {"best_best": 2, "best_selected": 1},
                "symbolic_program_trace_covered": {"best_best": 3, "best_selected": 2},
            },
        },
    )
    _write(
        distill_source,
        {
            "run_id": "distill_gate",
            "gate": "stage5_arc_agi_distill_sft_gate",
            "comparison": {
                "distill_off": {"best_best": 3, "best_selected": 2},
                "distill_on": {"best_best": 4, "best_selected": 2},
            },
            "distill_off": {"metadata": {"distillation": {"enabled": False}}},
            "distill_on": {"metadata": {"distillation": {"enabled": True}}},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["arc_agi_sft_recipe_gates"] == [
        {
            "path": str(distill_source),
            "run_id": "distill_gate",
            "kind": "distill_sft_gate",
            "best_arm": "distill_on",
            "best_delta": 1,
            "selected_delta": 0,
        },
        {
            "path": str(trace_source),
            "run_id": "trace_gate",
            "kind": "trace_sft_gate",
            "best_arm": "symbolic_program_trace_covered",
            "best_delta": 1,
            "selected_delta": 1,
        },
    ]
    assert payload["recommended_next_plan_source"] in {str(trace_source), str(distill_source)}


def test_progress_ledger_prefers_passed_arc_mix_gate_over_newer_release_gate(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    arc_mix_source = scan_root / "arc_mix" / "summary.json"
    release_source = scan_root / "release" / "summary.json"
    _write(
        arc_mix_source,
        {
            "run_id": "arc_mix",
            "kind": "stage5_balanced_arc_mix_gate",
            "status": "proxy_lift",
            "passed": True,
        },
    )
    _write(
        release_source,
        {
            "run_id": "release",
            "gate": "stage5_release_benchmark_readiness",
            "status": "needs_benchmark_confirmation",
            "passed": False,
        },
    )
    os.utime(arc_mix_source, (1000, 1000))
    os.utime(release_source, (2000, 2000))

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recommended_next_plan_source"] == str(arc_mix_source)


def test_progress_ledger_reports_generated_curriculum_statuses(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    capability = scan_root / "capability_ladder" / "summary.json"
    probe = scan_root / "capability_ladder_probe" / "summary.json"
    trace_jobs = scan_root / "capability_ladder_trace_jobs" / "summary.json"
    trace_responses = scan_root / "capability_ladder_trace_responses" / "summary.json"
    trace_collection = scan_root / "capability_ladder_trace_collection" / "summary.json"
    pipeline = scan_root / "curriculum_pipeline" / "summary.json"
    gate = scan_root / "curriculum_gate" / "summary.json"
    sft = scan_root / "curriculum_sft" / "summary.json"
    _write(
        capability,
        {
            "run_id": "capability",
            "kind": "capability_ladder_curriculum_pipeline",
            "work_dir": "data/curriculum/capability_ladder_001",
            "status": "complete",
            "next_action": "Run training/check_curriculum_sft_gate.py before any GPU fine-tuning.",
            "counts": {"positive_sft_rows": 30, "mode_counts": {"direct": 12, "deep_narrow": 18}},
        },
    )
    _write(
        probe,
        {
            "run_id": "probe",
            "kind": "stage5_capability_ladder_mcq_probe",
            "status": "capability_ladder_probe_needs_review",
            "next_action": "Run no-GPU probe SFT gate.",
            "curriculum": {
                "work_dir": "data/curriculum/probe",
                "counts": {"positive_sft_rows": 9, "mode_counts": {"direct": 5, "deep_narrow": 4}},
            },
        },
    )
    _write(
        trace_jobs,
        {
            "run_id": "trace_jobs",
            "kind": "stage5_capability_ladder_trace_jobs",
            "status": "ready",
            "trace_jobs": {"jobs": 18, "selected_rows": 9, "by_target_loop": {"1": 5, "2": 4}},
            "next_action": "Run provider responses then collect traced rows.",
        },
    )
    _write(
        trace_responses,
        {
            "run_id": "trace_responses",
            "kind": "stage5_capability_ladder_trace_responses",
            "status": "responses_ready",
            "response_report": {"written": 8, "skipped": 1, "errors": 0, "timeouts": 0},
            "next_action": "Run capability_ladder_trace_collect_cpu to verify final answers.",
        },
    )
    _write(
        trace_collection,
        {
            "run_id": "trace_collection",
            "kind": "stage5_capability_ladder_trace_collection",
            "status": "trace_curriculum_gate_ready",
            "next_action": "If the gate is green, run the recurrent SFT gate/training path.",
            "curriculum": {
                "work_dir": "data/curriculum/traced",
                "counts": {"positive_sft_rows": 7, "mode_counts": {"direct": 3, "deep_narrow": 4}},
            },
            "gate": {"go": True},
        },
    )
    _write(
        pipeline,
        {
            "run_id": "pipeline",
            "kind": "curriculum_pipeline_from_artifacts",
            "work_dir": "data/curriculum/run_001",
            "status": "complete",
            "next_action": "Review typed_records.jsonl and positive_sft.jsonl before any GPU fine-tuning.",
            "counts": {"positive_sft_rows": 24},
        },
    )
    _write(
        gate,
        {
            "run_id": "gate",
            "kind": "curriculum_sft_gate",
            "go": True,
            "status": "go_train_recurrent_sft",
            "work_dir": "data/curriculum/run_001",
            "checks": {"positive_sft": {"rows": 24}},
        },
    )
    _write(
        sft,
        {
            "run_id": "sft",
            "kind": "stage5_curriculum_sft",
            "status": "validation_sane",
            "config": {"work_dir": "data/curriculum/run_001"},
            "dataset": {"rows": 24, "train_rows": 21, "val_rows": 3},
            "phase1_val": {"mean_expected_loops": 2.5, "expected_ce": 1.25},
            "phase1_checkpoint": "outputs/stage5/sft/phase1/phase1_step_150.pt",
            "validation_checks": {
                "status": "validation_sane",
                "issues": [],
                "depth_gradient": {"observed": True, "direct_mean_expected_loops": 1.2, "deep_mean_expected_loops": 2.5},
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    statuses = payload["curriculum_statuses"]
    by_kind = {row["kind"]: row for row in statuses}
    assert [row["kind"] for row in statuses] == [
        "capability_ladder_curriculum_pipeline",
        "stage5_capability_ladder_mcq_probe",
        "stage5_capability_ladder_trace_collection",
        "stage5_capability_ladder_trace_jobs",
        "stage5_capability_ladder_trace_responses",
        "curriculum_sft_gate",
        "curriculum_pipeline_from_artifacts",
        "stage5_curriculum_sft",
    ]
    assert by_kind["capability_ladder_curriculum_pipeline"]["positive_rows"] == 30
    assert by_kind["capability_ladder_curriculum_pipeline"]["next_action"].startswith("Run training/check_curriculum_sft_gate.py")
    assert by_kind["curriculum_sft_gate"]["go"] is True
    assert by_kind["curriculum_sft_gate"]["positive_rows"] == 24
    assert by_kind["stage5_capability_ladder_mcq_probe"]["positive_rows"] == 9
    assert by_kind["stage5_capability_ladder_mcq_probe"]["work_dir"] == "data/curriculum/probe"
    assert by_kind["stage5_capability_ladder_trace_jobs"]["positive_rows"] == 9
    assert by_kind["stage5_capability_ladder_trace_jobs"]["next_action"].startswith("Run provider responses")
    assert by_kind["stage5_capability_ladder_trace_responses"]["positive_rows"] == 9
    assert by_kind["stage5_capability_ladder_trace_responses"]["next_action"].startswith("Run capability_ladder_trace_collect_cpu")
    assert by_kind["stage5_capability_ladder_trace_collection"]["positive_rows"] == 7
    assert by_kind["stage5_capability_ladder_trace_collection"]["go"] is True
    assert by_kind["stage5_capability_ladder_trace_collection"]["work_dir"] == "data/curriculum/traced"
    assert by_kind["stage5_capability_ladder_trace_collection"]["next_action"].startswith("If the gate is green")
    assert by_kind["curriculum_pipeline_from_artifacts"]["next_action"].startswith("Review typed_records")
    assert by_kind["stage5_curriculum_sft"]["train_rows"] == 21
    assert by_kind["stage5_curriculum_sft"]["val_rows"] == 3
    assert by_kind["stage5_curriculum_sft"]["mean_expected_loops"] == 2.5
    assert by_kind["stage5_curriculum_sft"]["checkpoint"] == "outputs/stage5/sft/phase1/phase1_step_150.pt"
    assert by_kind["stage5_curriculum_sft"]["validation_status"] == "validation_sane"
    assert by_kind["stage5_curriculum_sft"]["validation_issues"] == []
    assert by_kind["stage5_curriculum_sft"]["depth_gradient_observed"] is True
    assert by_kind["stage5_curriculum_sft"]["next_action"].startswith("run bounded routing diagnostic")
    assert payload["recommended_next_plan_source"] == str(sft)


def test_progress_ledger_writes_generated_curriculum_markdown(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "curriculum_sft" / "summary.json"
    _write(
        source,
        {
            "run_id": "sft",
            "kind": "stage5_curriculum_sft",
            "status": "validation_sane",
            "config": {"work_dir": "data/curriculum/run_001"},
            "dataset": {"rows": 24, "train_rows": 21, "val_rows": 3},
            "phase1_val": {"mean_expected_loops": 2.5, "expected_ce": 1.25},
            "phase1_checkpoint": "outputs/stage5/sft/phase1/phase1_step_150.pt",
            "validation_checks": {
                "status": "validation_sane",
                "issues": [],
                "depth_gradient": {"observed": True},
            },
        },
    )
    payload = scan_progress(scan_root, run_id="ledger")
    output_dir = tmp_path / "ledger"

    write_report(payload, output_dir)

    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Generated Curriculum Pipeline" in report
    assert "`stage5_curriculum_sft`" in report
    assert "`validation_sane`" in report
    assert "`True`" in report
    assert "21/3" in report
    assert "`outputs/stage5/sft/phase1/phase1_step_150.pt`" in report


def test_progress_ledger_reports_direct_preservation_repairs(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "direct_preserve" / "summary.json"
    _write(
        source,
        {
            "run_id": "direct_preserve",
            "kind": "stage5_direct_preservation_probe",
            "status": "direct_route_matches_base",
            "passed": True,
            "source_summary": "outputs/stage5/source/summary.json",
            "resume_checkpoint": "outputs/stage5/source/phase1/phase1_step_200.pt",
            "data": {"direct_sft": {"selected_rows": 128}},
            "base_eval": {"correct": 80, "total": 128},
            "start_loop1_eval": {"correct": 72, "total": 128},
            "best_checkpoint": {
                "checkpoint": "outputs/stage5/direct_preserve/phase1_direct_preserve/phase1_step_100.pt",
                "trained": True,
                "loop1_eval": {"correct": 81, "total": 128},
                "loop4_eval": {"correct": 79, "total": 128},
                "comparison_to_base": {
                    "helped": 3,
                    "hurt": 2,
                    "mean_margin_delta": 0.125,
                    "max_abs_prediction_count_delta": 4,
                    "calibration_ok": True,
                },
            },
            "next_step": "Confirm the direct-route preservation checkpoint on a larger ARC-Easy/Challenge slice.",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")
    output_dir = tmp_path / "ledger"
    write_report(payload, output_dir)

    assert payload["direct_preservation_statuses"] == [
        {
            "path": str(source),
            "run_id": "direct_preserve",
            "status": "direct_route_matches_base",
            "passed": True,
            "source_summary": "outputs/stage5/source/summary.json",
            "resume_checkpoint": "outputs/stage5/source/phase1/phase1_step_200.pt",
            "checkpoint": "outputs/stage5/direct_preserve/phase1_direct_preserve/phase1_step_100.pt",
            "trained": True,
            "selected_rows": 128,
            "base_correct": 80,
            "start_loop1_correct": 72,
            "best_loop1_correct": 81,
            "best_loop4_correct": 79,
            "total": 128,
            "helped": 3,
            "hurt": 2,
            "mean_margin_delta": 0.125,
            "max_abs_prediction_count_delta": 4,
            "calibration_ok": True,
            "next_step": "Confirm the direct-route preservation checkpoint on a larger ARC-Easy/Challenge slice.",
        }
    ]
    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Direct Preservation Repairs" in report
    assert "`direct_route_matches_base`" in report
    assert "80/128" in report
    assert "72/128" in report
    assert "81/128" in report
    assert "3/2" in report


def test_progress_ledger_reports_reentry_norm_statuses(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "reentry_norm" / "summary.json"
    assessment = scan_root / "reentry_norm" / "reentry_assessment.json"
    _write(
        source,
        {
            "run_id": "reentry_norm",
            "kind": "stage5_reentry_norm_eval_only",
        },
    )
    _write(
        assessment,
        {
            "kind": "stage5_reentry_assessment",
            "source_kind": "stage5_reentry_norm_eval_only",
            "source_run_id": "reentry_norm",
            "stage": "norm",
            "status": "entry_rms_safe_for_smoke",
            "recommendation": "run_reentry_repair_smoke",
            "reason": "safe for trainable smoke",
            "metrics": {
                "candidate_hits_delta_entry_minus_none": 1,
                "best_hits_delta_entry_minus_none": 0,
                "loop8_output_over_entry_delta_entry_minus_none": -0.04,
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")
    output_dir = tmp_path / "ledger"
    write_report(payload, output_dir)

    assert payload["reentry_statuses"] == [
        {
            "path": str(source),
            "run_id": "reentry_norm",
            "kind": "stage5_reentry_norm_eval_only",
            "stage": "norm",
            "status": "entry_rms_safe_for_smoke",
            "recommendation": "run_reentry_repair_smoke",
            "reason": "safe for trainable smoke",
            "assessment_path": str(assessment),
            "bridge_gate": None,
            "bridge_live": None,
            "bridge_moved": None,
            "adapter_live": None,
            "adapter_moved": None,
            "loop1_best_hits_delta": None,
            "loop1_candidate_hits_delta": None,
            "candidate_hits_delta_entry_minus_none": 1,
            "best_hits_delta_entry_minus_none": 0,
            "loop8_output_over_entry_delta_entry_minus_none": -0.04,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)
    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Re-entry Phase 0" in report
    assert "`run_reentry_repair_smoke`" in report
    assert "entry_rms_safe_for_smoke" in report


def test_progress_ledger_preserves_zero_reentry_bridge_gate(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "reentry_drift" / "summary.json"
    assessment = scan_root / "reentry_drift" / "reentry_assessment.json"
    _write(
        source,
        {
            "run_id": "reentry_drift",
            "kind": "reentry_drift_diagnostic",
        },
    )
    _write(
        assessment,
        {
            "kind": "stage5_reentry_assessment",
            "stage": "drift",
            "status": "bridge_dead",
            "recommendation": "run_reentry_norm_then_repair_smoke",
            "metrics": {"bridge_gate": 0.0, "dead_bridge": True},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["reentry_statuses"][0]["bridge_gate"] == 0.0


def test_progress_ledger_reports_reentry_repair_smoke_statuses(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "reentry_repair" / "summary.json"
    assessment = scan_root / "reentry_repair" / "reentry_assessment.json"
    _write(
        source,
        {
            "run_id": "reentry_repair",
            "kind": "stage5_reentry_repair_smoke",
        },
    )
    _write(
        assessment,
        {
            "kind": "stage5_reentry_assessment",
            "source_kind": "stage5_reentry_repair_smoke",
            "source_run_id": "reentry_repair",
            "stage": "repair_smoke",
            "status": "bridge_repair_smoke_passed",
            "recommendation": "run_bounded_recovery_training_with_reentry_repair",
            "reason": "bridge and adapter moved",
            "metrics": {
                "post_bridge_gate": 1.0,
                "bridge_live": True,
                "bridge_moved": True,
                "adapter_live": True,
                "adapter_moved": True,
                "loop1_best_hits_delta": 0,
                "loop1_candidate_hits_delta": 0,
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")
    row = payload["reentry_statuses"][0]

    assert row["status"] == "bridge_repair_smoke_passed"
    assert row["recommendation"] == "run_bounded_recovery_training_with_reentry_repair"
    assert row["bridge_gate"] == 1.0
    assert row["bridge_live"] is True
    assert row["adapter_moved"] is True
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_reentry_recovery_training_as_next_source(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "reentry_recovery" / "summary.json"
    _write(
        source,
        {
            "run_id": "reentry_recovery",
            "kind": "stage5_reentry_recovery_training",
            "status": "validation_sane",
            "checkpoint": "outputs/stage5/reentry_recovery/phase1/phase1_step_75.pt",
            "phase1_checkpoint": "outputs/stage5/reentry_recovery/phase1/phase1_step_75.pt",
            "dataset": {"rows": 24, "train_rows": 21, "val_rows": 3},
            "phase1_val": {"mean_expected_loops": 2.4, "expected_ce": 1.5},
            "validation_checks": {
                "status": "validation_sane",
                "depth_gradient": {"observed": True},
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")
    row = next(
        item
        for item in payload["curriculum_statuses"]
        if item["kind"] == "stage5_reentry_recovery_training"
    )

    assert row["validation_status"] == "validation_sane"
    assert row["depth_gradient_observed"] is True
    assert row["checkpoint"] == "outputs/stage5/reentry_recovery/phase1/phase1_step_75.pt"
    assert row["next_action"] == "run debiased_benchmark_suite before dense control or breadth diagnostics"
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_does_not_recommend_failed_reentry_recovery_training(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    trace_collection = scan_root / "trace_collection" / "summary.json"
    recovery = scan_root / "reentry_recovery" / "summary.json"
    _write(
        trace_collection,
        {
            "run_id": "trace_collection",
            "kind": "stage5_capability_ladder_trace_collection",
            "status": "trace_curriculum_gate_ready",
            "curriculum": {"work_dir": "data/curriculum/traced", "counts": {"positive_sft_rows": 16}},
            "gate": {"go": True},
        },
    )
    _write(
        recovery,
        {
            "run_id": "reentry_recovery",
            "kind": "stage5_reentry_recovery_training",
            "status": "validation_needs_review",
            "dataset": {"rows": 24, "train_rows": 21, "val_rows": 3},
            "phase1_val": {"mean_expected_loops": 1.05, "expected_ce": 2.25},
            "phase1_checkpoint": "outputs/stage5/reentry_recovery/phase1/phase1_step_75.pt",
            "validation_checks": {
                "status": "validation_needs_review",
                "issues": ["depth gradient not observed"],
                "depth_gradient": {"observed": False},
            },
        },
    )
    os.utime(trace_collection, (1000, 1000))
    os.utime(recovery, (2000, 2000))

    payload = scan_progress(scan_root, run_id="ledger")
    row = next(
        item
        for item in payload["curriculum_statuses"]
        if item["kind"] == "stage5_reentry_recovery_training"
    )

    assert row["validation_status"] == "validation_needs_review"
    assert row["validation_issues"] == ["depth gradient not observed"]
    assert row["depth_gradient_observed"] is False
    assert row["next_action"] == "inspect re-entry recovery validation before benchmarking"
    assert payload["recommended_next_plan_source"] == str(trace_collection)


def test_progress_ledger_recommends_direct_preservation_probe_over_traced_assessment(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    traced = scan_root / "traced_assessment" / "summary.json"
    direct = scan_root / "direct_preserve" / "summary.json"
    _write(
        traced,
        {
            "run_id": "traced_assessment",
            "kind": "stage5_traced_sft_assessment",
            "status": "needs_direct_preservation_repair",
            "passed": False,
        },
    )
    _write(
        direct,
        {
            "run_id": "direct_preserve",
            "kind": "stage5_direct_preservation_probe",
            "status": "direct_route_matches_base",
            "passed": True,
            "data": {"direct_sft": {"selected_rows": 64}},
            "base_eval": {"correct": 42, "total": 64},
            "start_loop1_eval": {"correct": 39, "total": 64},
            "best_checkpoint": {
                "checkpoint": "outputs/stage5/direct_preserve/phase1/phase1_step_100.pt",
                "loop1_eval": {"correct": 43, "total": 64},
                "comparison_to_base": {"calibration_ok": True},
            },
        },
    )
    os.utime(traced, (2000, 2000))
    os.utime(direct, (1000, 1000))

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recommended_next_plan_source"] == str(direct)


def test_progress_ledger_prefers_current_pointer_over_older_high_priority_source(tmp_path, monkeypatch) -> None:
    import colab.summarize_stage5_progress as progress

    monkeypatch.setattr(progress, "ROOT", tmp_path)
    scan_root = tmp_path / "outputs" / "stage5"
    old_full = scan_root / "old_full" / "summary.json"
    current = scan_root / "current_traced" / "summary.json"
    _write(
        old_full,
        {
            "run_id": "old_full",
            "kind": "stage5_recovery_full_assessment",
            "status": "needs_competence_recovery",
        },
    )
    _write(
        current,
        {
            "run_id": "current_traced",
            "kind": "stage5_traced_sft_assessment",
            "status": "needs_direct_preservation_repair",
        },
    )
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("outputs/stage5/current_traced/summary.json\n", encoding="utf-8")
    os.utime(old_full, (3000, 3000))
    os.utime(current, (1000, 1000))

    payload = progress.scan_progress(scan_root, run_id="ledger")

    assert payload["recommended_next_plan_source"] == progress.path_for_cli(current)


def test_progress_ledger_does_not_recommend_failed_curriculum_sft(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    trace_collection = scan_root / "trace_collection" / "summary.json"
    sft = scan_root / "curriculum_sft" / "summary.json"
    _write(
        trace_collection,
        {
            "run_id": "trace_collection",
            "kind": "stage5_capability_ladder_trace_collection",
            "status": "trace_curriculum_gate_ready",
            "curriculum": {"work_dir": "data/curriculum/traced", "counts": {"positive_sft_rows": 16}},
            "gate": {"go": True},
        },
    )
    _write(
        sft,
        {
            "run_id": "sft",
            "kind": "stage5_curriculum_sft",
            "status": "validation_needs_review",
            "config": {"work_dir": "data/curriculum/run_001"},
            "dataset": {"rows": 24, "train_rows": 21, "val_rows": 3},
            "phase1_val": {"mean_expected_loops": 1.05, "expected_ce": 2.25},
            "phase1_checkpoint": "outputs/stage5/sft/phase1/phase1_step_150.pt",
            "validation_checks": {
                "status": "validation_needs_review",
                "issues": ["mean_expected_loops below threshold", "depth gradient not observed"],
                "depth_gradient": {"observed": False},
            },
        },
    )
    os.utime(trace_collection, (1000, 1000))
    os.utime(sft, (2000, 2000))

    payload = scan_progress(scan_root, run_id="ledger")
    sft_row = next(row for row in payload["curriculum_statuses"] if row["kind"] == "stage5_curriculum_sft")

    assert sft_row["validation_status"] == "validation_needs_review"
    assert sft_row["validation_issues"] == ["mean_expected_loops below threshold", "depth gradient not observed"]
    assert sft_row["depth_gradient_observed"] is False
    assert sft_row["next_action"].startswith("inspect curriculum SFT validation")
    assert payload["recommended_next_plan_source"] == str(trace_collection)


def test_progress_ledger_prefers_complete_curriculum_pipeline_over_older_benchmark_gate(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    benchmark = scan_root / "benchmark_gate" / "summary.json"
    pipeline = scan_root / "curriculum_pipeline" / "summary.json"
    _write(
        benchmark,
        {
            "run_id": "benchmark_gate",
            "gate": "stage5_broader_benchmark_suite",
            "status": "needs_benchmark_confirmation",
        },
    )
    _write(
        pipeline,
        {
            "run_id": "pipeline",
            "kind": "curriculum_pipeline_from_artifacts",
            "status": "complete",
            "work_dir": "data/curriculum/run_001",
            "counts": {"positive_sft_rows": 24},
        },
    )
    os.utime(benchmark, (1000, 1000))
    os.utime(pipeline, (2000, 2000))

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recommended_next_plan_source"] == str(pipeline)


def test_progress_ledger_prefers_complete_capability_ladder_over_older_benchmark_gate(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    benchmark = scan_root / "benchmark_gate" / "summary.json"
    capability = scan_root / "capability_ladder" / "summary.json"
    _write(
        benchmark,
        {
            "run_id": "benchmark_gate",
            "gate": "stage5_broader_benchmark_suite",
            "status": "needs_benchmark_confirmation",
        },
    )
    _write(
        capability,
        {
            "run_id": "capability",
            "kind": "capability_ladder_curriculum_pipeline",
            "status": "complete",
            "counts": {"positive_sft_rows": 30},
        },
    )
    os.utime(benchmark, (2000, 2000))
    os.utime(capability, (1000, 1000))

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recommended_next_plan_source"] == str(capability)


def test_progress_ledger_does_not_prefer_pending_curriculum_pipeline_over_benchmark_gate(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    benchmark = scan_root / "benchmark_gate" / "summary.json"
    pipeline = scan_root / "curriculum_pipeline" / "summary.json"
    _write(
        benchmark,
        {
            "run_id": "benchmark_gate",
            "gate": "stage5_broader_benchmark_suite",
            "status": "needs_benchmark_confirmation",
        },
    )
    _write(
        pipeline,
        {
            "run_id": "pipeline",
            "kind": "curriculum_pipeline_from_artifacts",
            "status": "pending_method_or_perturbation_responses",
        },
    )
    os.utime(benchmark, (1000, 1000))
    os.utime(pipeline, (2000, 2000))

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recommended_next_plan_source"] == str(benchmark)


def test_progress_ledger_skips_empty_and_malformed_eval_summaries(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    _write(scan_root / "empty" / "base_summary.json", {"summary": {}})
    _write(
        scan_root / "zero_evidence" / "base_summary.json",
        {
            "summary": {
                "first_exact": 0,
                "selected_exact": 0,
                "best_of_k_exact": 0,
                "examples_with_targets": 0,
                "valid_candidate_rate": 0.0,
            }
        },
    )
    malformed = scan_root / "broken" / "summary.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not json", encoding="utf-8")

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["scanned_files"] == 3
    assert payload["parsed_records"] == 0
    assert payload["skipped_files"] == [str(malformed)]
    assert payload["recommended_next_plan_source"] is None


def test_progress_ledger_writes_summary_markdown(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    _write(
        scan_root / "bench" / "summary.json",
        {
            "run_id": "bench",
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 3)},
            "recovered": {"summary": _summary(6, 7)},
            "deltas": {},
        },
    )
    payload = scan_progress(scan_root, run_id="ledger")
    output_dir = tmp_path / "ledger"

    write_report(payload, output_dir)

    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Stage 5 ARC-AGI Progress Ledger" in report
    assert "| `base` | 8 | 9 | 20 | `base` |" in report
    assert "selected delta `-2`" in report
    assert (output_dir / "summary.json").exists()


def test_progress_ledger_writes_candidate_gate_markdown(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    _write(
        scan_root / "candidate_gate" / "summary.json",
        {
            "run_id": "candidate_gate",
            "kind": "stage5_arc_agi_candidate_gate",
            "metadata": {"arc_version": "arc-agi-1", "arc_split": "eval", "limit": 6},
            "symbolic_coverage": {"examples_with_targets": 6, "exact_symbolic": 1},
            "rows": [
                {"variant": "phase1_model_only", "best": 0},
                {"variant": "phase1_hybrid_symbolic_first", "best": 1},
            ],
        },
    )
    payload = scan_progress(scan_root, run_id="ledger")
    output_dir = tmp_path / "ledger"

    write_report(payload, output_dir)

    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "## ARC-AGI Candidate Gates" in report
    assert "symbolic exact `1/6`" in report
    assert "phase1 hybrid best delta `1`" in report
