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
            "next_step": "recover more",
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


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
