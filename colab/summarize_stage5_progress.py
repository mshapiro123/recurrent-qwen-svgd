"""Summarize Stage 5 ARC-AGI progress across saved run summaries.

This is a no-GPU ledger. It scans ``outputs/stage5`` for run summaries and
individual eval summaries, normalizes exact-grid metrics, and writes a compact
progress report. The goal is to keep the research loop empirical: which arm is
best, where did recovered recurrent close the base gap, and which run summary
should feed the next planner invocation?
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_ROOT = "outputs/stage5"


def current_run_id() -> str:
    return os.environ.get("STAGE5_ARC_AGI_PROGRESS_RUN_ID") or time.strftime(
        "stage5_arc_agi_progress_%Y%m%d_%H%M%S"
    )


def current_scan_root() -> str:
    return os.environ.get("STAGE5_ARC_AGI_PROGRESS_SCAN_ROOT", DEFAULT_SCAN_ROOT)


def run_dir(run_id: str) -> Path:
    return ROOT / "outputs" / "stage5" / run_id


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def metric(summary: dict[str, Any] | None, key: str) -> int:
    if not summary:
        return 0
    return int(summary.get(key, 0) or 0)


def rate(summary: dict[str, Any] | None, key: str) -> float:
    if not summary:
        return 0.0
    return float(summary.get(key, 0.0) or 0.0)


def summary_metrics(payload_or_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload_or_summary:
        return None
    summary = payload_or_summary.get("summary")
    if isinstance(summary, dict):
        return summary
    if {"selected_exact", "best_of_k_exact"} & set(payload_or_summary):
        return payload_or_summary
    return None


def record_from_summary(
    *,
    path: Path,
    kind: str,
    arm: str,
    label: str,
    summary: dict[str, Any] | None,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    if not summary:
        return None
    examples = metric(summary, "examples_with_targets")
    selected_exact = metric(summary, "selected_exact")
    best_of_k_exact = metric(summary, "best_of_k_exact")
    first_exact = metric(summary, "first_exact")
    if examples <= 0 and selected_exact == 0 and best_of_k_exact == 0 and first_exact == 0:
        return None
    return {
        "path": path_for_cli(path),
        "run_id": run_id or path.parent.name,
        "kind": kind,
        "arm": arm,
        "label": label,
        "examples": examples,
        "selected_exact": selected_exact,
        "best_of_k_exact": best_of_k_exact,
        "first_exact": first_exact,
        "selected_accuracy": rate(summary, "selected_accuracy"),
        "best_of_k_accuracy": rate(summary, "best_of_k_accuracy"),
        "valid_candidate_rate": rate(summary, "valid_candidate_rate"),
    }


def looks_like_planner_source(payload: dict[str, Any]) -> bool:
    if any(
        key in payload
        for key in (
            "recovered_benchmark",
            "tta_sweep",
            "compact",
            "autopilot_compact",
            "best_by_label",
            "recovery_decision",
        )
    ):
        return True
    if isinstance(payload.get("rows"), list) and (
        "best_by_label" in payload or {"source_run_dir", "strategies"} <= set(payload)
    ):
        return True
    if payload.get("gate") == "stage5_gate1_selector_tta":
        return True
    if payload.get("gate") == "stage5_gate2_particle_mechanism":
        return True
    if payload.get("gate") == "stage5_selector_replication" or payload.get("kind") == "selector_replication":
        return True
    if payload.get("gate") == "stage5_same_recipe_selector_conversion" or payload.get("kind") == "recipe_selector_conversion":
        return True
    if payload.get("gate") == "stage5_same_recipe_architecture":
        return True
    if payload.get("gate") == "stage5_same_recipe_mcq_architecture":
        return True
    if payload.get("kind") in {
        "stage5_surface_alignment_repair",
        "stage5_surface_repair_assessment",
        "stage5_dense_mcq_trace_sft_control",
        "stage5_mcq_recipe_control_assessment",
    }:
        return True
    if payload.get("gate") == "stage5_release_benchmark_readiness":
        return True
    if payload.get("kind") == "stage5_benchmark_suite":
        return True
    if payload.get("kind") == "stage5_traced_sft_assessment":
        return True
    if payload.get("kind") == "stage5_direct_preservation_probe":
        return True
    if payload.get("gate") == "stage5_broader_benchmark_suite":
        return True
    if payload.get("kind") == "stage5_recovery_full_assessment":
        return True
    if payload.get("kind") == "stage5_balanced_mcq_checkpoint_assessment":
        return True
    if payload.get("kind") == "stage5_balanced_arc_mix_gate":
        return True
    if payload.get("gate") == "stage5_claim_readiness":
        return True
    if payload.get("gate") == "stage5_arc_agi_baseline_registry" or payload.get("kind") == "arc_agi_baseline_registry":
        return True
    if payload.get("gate") == "stage5_arc_agi_sota_comparison" or payload.get("kind") == "arc_agi_sota_comparison":
        return True
    if payload.get("kind") == "stage5_reasoning_dataset_audit":
        return True
    if payload.get("kind") in {
        "stage5_capability_ladder_mcq_probe",
        "stage5_capability_ladder_trace_jobs",
        "stage5_capability_ladder_trace_responses",
        "stage5_capability_ladder_trace_collection",
        "capability_ladder_curriculum_pipeline",
        "curriculum_pipeline_from_artifacts",
        "curriculum_sft_gate",
        "stage5_curriculum_sft",
    }:
        return True
    if payload.get("kind") in {
        "reentry_drift_diagnostic",
        "stage5_reentry_norm_eval_only",
        "stage5_reentry_repair_smoke",
        "stage5_reentry_recovery_training",
    }:
        return True
    if payload.get("kind") == "stage4_opus_finetune" or {"phase1_checkpoint", "phase2_checkpoint", "arc_ladder"} <= set(payload):
        return True
    if payload.get("gate") == "stage5_arc_agi_candidate_gate" or payload.get("kind") == "stage5_arc_agi_candidate_gate":
        return True
    if payload.get("gate") == "stage5_arc_agi_trace_sft_gate" or payload.get("kind") == "trace_sft_gate":
        return True
    if payload.get("gate") == "stage5_arc_agi_distill_sft_gate" or payload.get("kind") == "distill_sft_gate":
        return True
    comparison = payload.get("comparison")
    if isinstance(comparison, dict) and "grid_only" in comparison and isinstance(payload.get("arms"), list):
        return True
    if isinstance(comparison, dict) and {"distill_off", "distill_on"} <= set(comparison):
        return True
    if payload.get("kind") == "dense_sft_control":
        return True
    if "phase1_arc_agi_tuned" in payload and payload.get("tuned_checkpoint"):
        return True
    if {"base", "phase1_start", "recovered", "deltas"} <= set(payload):
        return True
    return False


def records_from_recovered_benchmark(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = payload.get("recovered_benchmark") or payload
    rows: list[dict[str, Any]] = []
    for arm in ("base", "phase1_start", "recovered"):
        record = record_from_summary(
            path=path,
            kind="recovered_benchmark",
            arm=arm,
            label=arm,
            summary=summary_metrics(benchmark.get(arm)),
            run_id=str(benchmark.get("run_id") or payload.get("run_id") or path.parent.name),
        )
        if record:
            rows.append(record)
    return rows


def records_from_selector_rescore(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        summary = {
            "examples_with_targets": row.get("examples"),
            "selected_exact": row.get("selected_exact"),
            "best_of_k_exact": row.get("best_of_k_exact"),
            "valid_candidate_rate": row.get("valid_candidate_rate"),
        }
        record = record_from_summary(
            path=path,
            kind="selector_rescore",
            arm=str(row.get("label", "unknown")),
            label=str(row.get("selection_strategy", "unknown")),
            summary=summary,
            run_id=str(payload.get("run_id") or path.parent.name),
        )
        if record:
            rows.append(record)
    return rows


def records_from_recovery_particle(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recovered = payload.get("recovered_checkpoint") or {}
    recovered_summary = summary_metrics(recovered.get("summary") if isinstance(recovered, dict) else None)
    record = record_from_summary(
        path=path,
        kind="recovery_particle_gate",
        arm="recovered",
        label="deterministic_recovery",
        summary=recovered_summary,
        run_id=str(payload.get("run_id") or path.parent.name),
    )
    if record:
        rows.append(record)
    particle = payload.get("particle_decision") or {}
    for name, variant in ((particle.get("evidence") or {}).get("variants") or {}).items():
        if not isinstance(variant, dict):
            continue
        mean_delta = variant.get("mean_delta_vs_tuned") or variant.get("delta_vs_tuned") or {}
        rows.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "kind": "recovery_particle_gate",
                "arm": "particle",
                "label": str(name),
                "examples": 0,
                "selected_exact": 0,
                "best_of_k_exact": 0,
                "first_exact": 0,
                "selected_accuracy": 0.0,
                "best_of_k_accuracy": 0.0,
                "valid_candidate_rate": 0.0,
                "selected_delta_vs_recovered": float(mean_delta.get("selected_delta", 0.0) or 0.0),
                "best_of_k_delta_vs_recovered": float(mean_delta.get("best_of_k_delta", 0.0) or 0.0),
                "passed": bool(variant.get("passed", False)),
            }
        )
    return rows


def records_from_dense_sft(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ("base", "dense_tuned", "phase1_start"):
        record = record_from_summary(
            path=path,
            kind="dense_sft_control",
            arm=arm,
            label=arm,
            summary=summary_metrics(payload.get(arm)),
            run_id=str(payload.get("run_id") or path.parent.name),
        )
        if record:
            rows.append(record)
    return rows


def records_from_recurrent_sft(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, label in (
        ("base", "base"),
        ("phase1_start", "phase1_start"),
        ("recurrent_tuned", "phase1_arc_agi_tuned"),
    ):
        record = record_from_summary(
            path=path,
            kind="recurrent_sft",
            arm=arm,
            label=label,
            summary=payload.get(label),
            run_id=str(payload.get("run_id") or path.parent.name),
        )
        if record:
            rows.append(record)
    return rows


def records_from_eval_summary(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = summary_metrics(payload)
    if not summary:
        return []
    stem = path.stem
    label = stem.removesuffix("_summary")
    arm = label.split("__", 1)[0]
    if arm not in {"base", "phase1_start", "recovered", "phase1", "phase2"}:
        if label.startswith("base"):
            arm = "base"
        elif label.startswith("recovered"):
            arm = "recovered"
        elif "phase1" in label:
            arm = "phase1_start"
        elif "phase2" in label or "particle" in label:
            arm = "particle"
        else:
            arm = "unknown"
    record = record_from_summary(
        path=path,
        kind="eval_summary",
        arm=arm,
        label=label,
        summary=summary,
        run_id=str(payload.get("run_id") or path.parent.name),
    )
    return [record] if record else []


def records_from_payload(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("recovered_benchmark") or {"base", "phase1_start", "recovered", "deltas"} <= set(payload):
        return records_from_recovered_benchmark(path, payload)
    if isinstance(payload.get("rows"), list) and ("best_by_label" in payload or "strategies" in payload):
        return records_from_selector_rescore(path, payload)
    if payload.get("kind") == "dense_sft_control":
        return records_from_dense_sft(path, payload)
    if "phase1_arc_agi_tuned" in payload and payload.get("tuned_checkpoint"):
        return records_from_recurrent_sft(path, payload)
    if isinstance(payload.get("recovery_decision"), dict) and isinstance(payload.get("particle_decision"), dict):
        return records_from_recovery_particle(path, payload)
    return records_from_eval_summary(path, payload)


def gate1_assessments(summary_files: list[Path]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("gate") != "stage5_gate1_selector_tta":
            continue
        assessments.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "source_summary": payload.get("source_summary"),
                "source_kind": payload.get("source_kind"),
                "reason": payload.get("reason"),
                "next_step": payload.get("next_step"),
                "passing_comparisons": payload.get("passing_comparisons") or [],
                "tradeoff_comparisons": payload.get("tradeoff_comparisons") or [],
                "num_comparisons": int(payload.get("num_comparisons", 0) or 0),
            }
        )
    return sorted(assessments, key=lambda item: str(item["path"]))


def gate2_assessments(summary_files: list[Path]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("gate") != "stage5_gate2_particle_mechanism":
            continue
        best = payload.get("best_variant") or {}
        assessments.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "source_summary": payload.get("source_summary"),
                "source_kind": payload.get("source_kind"),
                "reason": payload.get("reason"),
                "next_step": payload.get("next_step"),
                "best_variant": best.get("variant"),
                "selected_delta": float(best.get("selected_delta", 0.0) or 0.0),
                "best_of_k_delta": float(best.get("best_of_k_delta", 0.0) or 0.0),
            }
        )
    return sorted(assessments, key=lambda item: str(item["path"]))


def selector_replications(summary_files: list[Path]) -> list[dict[str, Any]]:
    replications: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        if payload.get("gate") != "stage5_selector_replication" and payload.get("kind") != "selector_replication":
            continue
        replications.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "replicated_comparisons": payload.get("replicated_comparisons") or [],
                "next_step": payload.get("next_step"),
            }
        )
    return sorted(replications, key=lambda item: str(item["path"]))


def recipe_selector_conversions(summary_files: list[Path]) -> list[dict[str, Any]]:
    conversions: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        if payload.get("gate") != "stage5_same_recipe_selector_conversion" and payload.get("kind") != "recipe_selector_conversion":
            continue
        best = payload.get("best_selector") or {}
        best_row = None
        for row in payload.get("selector_evidence") or []:
            if row.get("label") == best.get("label") and row.get("selection_strategy") == best.get("selection_strategy"):
                best_row = row
                break
        best_row = best_row or {}
        conversions.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "passing_selectors": payload.get("passing_selectors") or [],
                "best_selector": best,
                "claim_level_selector": bool(best_row.get("claim_level_selector", False)),
                "selector_generated_selected_exact": int(best_row.get("selector_generated_selected_exact", 0) or 0),
                "selected_exceeds_best_of_k": int(best_row.get("selected_exceeds_best_of_k", 0) or 0),
                "next_step": payload.get("next_step"),
            }
        )
    return sorted(conversions, key=lambda item: str(item["path"]))


def recipe_control_assessments(summary_files: list[Path]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        gate = payload.get("gate") if payload else None
        if not payload or not isinstance(gate, str) or gate not in {
            "stage5_same_recipe_architecture",
            "stage5_same_recipe_mcq_architecture",
        }:
            continue
        decision = payload.get("decision_evidence") or {}
        aggregate = decision.get("aggregate") or {}
        hard = decision.get("hard") or {}
        aggregate_best = decision.get("aggregate_best_of_k") or {}
        hard_best = decision.get("hard_best_of_k") or {}
        primary = decision.get("primary") or {}
        challenge_content = decision.get("arc_challenge_content") or {}
        challenge_cyclic = decision.get("arc_challenge_cyclic") or {}
        easy_content = decision.get("arc_easy_content") or {}
        easy_cyclic = decision.get("arc_easy_cyclic") or {}
        assessments.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "gate": gate,
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "dense_summary": payload.get("dense_summary"),
                "recurrent_summary": payload.get("recurrent_summary"),
                "reason": payload.get("reason"),
                "next_step": payload.get("next_step"),
                "aggregate_selected_delta": int(aggregate.get("delta_exact", 0) or 0),
                "hard_selected_delta": int(hard.get("delta_exact", 0) or 0),
                "aggregate_best_of_k_delta": int(aggregate_best.get("delta_exact", 0) or 0),
                "hard_best_of_k_delta": int(hard_best.get("delta_exact", 0) or 0),
                "primary_delta_recurrent_vs_dense": int(
                    primary.get("correct_delta_recurrent_vs_dense", 0) or 0
                ),
                "arc_challenge_content_delta_recurrent_vs_dense": int(
                    challenge_content.get("correct_delta_recurrent_vs_dense", 0) or 0
                ),
                "arc_challenge_cyclic_delta_recurrent_vs_dense": int(
                    challenge_cyclic.get("correct_delta_recurrent_vs_dense", 0) or 0
                ),
                "arc_easy_content_delta_recurrent_vs_dense": int(
                    easy_content.get("correct_delta_recurrent_vs_dense", 0) or 0
                ),
                "arc_easy_cyclic_delta_recurrent_vs_dense": int(
                    easy_cyclic.get("correct_delta_recurrent_vs_dense", 0) or 0
                ),
            }
        )
    return sorted(assessments, key=lambda item: str(item["path"]))


def surface_alignment_statuses(summary_files: list[Path]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("kind") != "stage5_surface_alignment_repair":
            continue
        statuses.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "source_summary": payload.get("source_summary"),
                "benchmark_summary": payload.get("benchmark_summary"),
                "checkpoint": payload.get("checkpoint"),
                "surface_alignment_rows": int(payload.get("surface_alignment_rows", 0) or 0),
                "surface_repair_assessment_status": payload.get("surface_repair_assessment_status"),
                "assessment_status": payload.get("assessment_status"),
                "next_step": payload.get("next_step"),
            }
        )
    return sorted(statuses, key=lambda item: str(item["path"]))


def dense_mcq_control_statuses(summary_files: list[Path]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("kind") != "stage5_dense_mcq_trace_sft_control":
            continue
        dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
        assessment = payload.get("recipe_control_assessment")
        if not isinstance(assessment, dict):
            assessment = {}
        statuses.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "source_summary": payload.get("source_summary"),
                "recurrent_benchmark_summary": payload.get("recurrent_benchmark_summary"),
                "train_rows": int(dataset.get("train_rows", 0) or 0),
                "extra_train_rows": int(dataset.get("extra_train_rows", 0) or 0),
                "dense_checkpoint": payload.get("dense_checkpoint"),
                "assessment_ran": bool(assessment.get("ran", False)),
                "assessment_status": assessment.get("status"),
                "assessment_passed": bool(assessment.get("passed", False)),
                "assessment_summary": assessment.get("summary_json"),
                "next_step": assessment.get("next_step"),
            }
        )
    return sorted(statuses, key=lambda item: str(item["path"]))


def release_gate_assessments(summary_files: list[Path]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("gate") != "stage5_release_benchmark_readiness":
            continue
        assessments.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "next_step": payload.get("next_step"),
                "min_arc_examples": int(payload.get("min_arc_examples", 0) or 0),
                "criteria": payload.get("criteria") or [],
            }
        )
    return sorted(assessments, key=lambda item: str(item["path"]))


def benchmark_suite_assessments(summary_files: list[Path]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("kind") != "stage5_benchmark_suite":
            continue
        deltas: list[dict[str, Any]] = []
        comparisons = payload.get("comparisons") or {}
        paired_comparisons = payload.get("paired_comparisons") or {}
        for benchmark, score_targets in sorted(comparisons.items()):
            if not isinstance(score_targets, dict):
                continue
            for score_target, aggregates in sorted(score_targets.items()):
                if not isinstance(aggregates, dict):
                    continue
                for aggregate, row in sorted(aggregates.items()):
                    if not isinstance(row, dict):
                        continue
                    paired = (
                        (paired_comparisons.get(benchmark) or {})
                        .get(score_target, {})
                        .get(aggregate, {})
                    )
                    deltas.append(
                        {
                            "benchmark": benchmark,
                            "score_target": score_target,
                            "aggregate": aggregate,
                            "correct_delta_recurrent_vs_base": int(
                                row.get("correct_delta_recurrent_vs_base", 0) or 0
                            ),
                            "accuracy_delta_recurrent_vs_base": float(
                                row.get("accuracy_delta_recurrent_vs_base", 0.0) or 0.0
                            ),
                            "paired_examples": int((paired or {}).get("paired_examples", 0) or 0),
                            "wins": int((paired or {}).get("wins", 0) or 0),
                            "losses": int((paired or {}).get("losses", 0) or 0),
                            "ties": int((paired or {}).get("ties", 0) or 0),
                            "sign_test_p_value": (paired or {}).get("sign_test_p_value"),
                        }
                    )
        assessments.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "checkpoint": payload.get("checkpoint"),
                "benchmarks": payload.get("benchmarks") or [],
                "deltas": deltas,
            }
        )
    return sorted(assessments, key=lambda item: str(item["path"]))


def broader_benchmark_gate_assessments(summary_files: list[Path]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("gate") != "stage5_broader_benchmark_suite":
            continue
        assessments.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "source_summary": payload.get("source_summary"),
                "next_step": payload.get("next_step"),
                "benchmarks": payload.get("benchmarks") or [],
            }
        )
    return sorted(assessments, key=lambda item: str(item["path"]))


def claim_readiness_packets(summary_files: list[Path]) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("gate") != "stage5_claim_readiness":
            continue
        linkage = (payload.get("artifacts") or {}).get("sota_export_linkage") or {}
        packets.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "claim_level": payload.get("claim_level"),
                "sota_export_linkage_passed": bool(linkage.get("passed", False)),
                "sota_export_linkage_verified": bool(linkage.get("verified", False)),
                "sota_export_linkage_matched_on": linkage.get("matched_on"),
                "sota_export_linkage_reason": linkage.get("reason"),
                "next_step": payload.get("next_step"),
            }
        )
    return sorted(packets, key=lambda item: str(item["path"]))


def arc_agi_sota_comparisons(summary_files: list[Path]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        if payload.get("gate") != "stage5_arc_agi_sota_comparison" and payload.get("kind") != "arc_agi_sota_comparison":
            continue
        comparisons.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "metric": payload.get("metric"),
                "candidate_accuracy": (payload.get("candidate") or {}).get("accuracy"),
                "best_baseline": (payload.get("best_baseline") or {}).get("name"),
                "best_baseline_accuracy": (payload.get("best_baseline") or {}).get("accuracy"),
                "delta_accuracy_vs_best_baseline": payload.get("delta_accuracy_vs_best_baseline"),
                "next_step": payload.get("next_step"),
            }
        )
    return sorted(comparisons, key=lambda item: str(item["path"]))


def arc_agi_baseline_registries(summary_files: list[Path]) -> list[dict[str, Any]]:
    registries: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        if payload.get("gate") != "stage5_arc_agi_baseline_registry" and payload.get("kind") != "arc_agi_baseline_registry":
            continue
        registries.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "metric": payload.get("metric"),
                "valid_baseline_count": int(payload.get("valid_baseline_count", 0) or 0),
                "best_baseline": (payload.get("best_baseline") or {}).get("name"),
                "best_baseline_accuracy": (payload.get("best_baseline") or {}).get("accuracy"),
                "next_step": payload.get("next_step"),
            }
        )
    return sorted(registries, key=lambda item: str(item["path"]))


def row_metric_by_variant(payload: dict[str, Any], variant: str, metric_name: str) -> int:
    for row in payload.get("rows") or []:
        if isinstance(row, dict) and row.get("variant") == variant:
            if metric_name in row:
                return int(row.get(metric_name, 0) or 0)
            if metric_name == "best":
                return int(row.get("best_exact", 0) or 0)
    return 0


def arc_agi_candidate_gates(summary_files: list[Path]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        if payload.get("gate") != "stage5_arc_agi_candidate_gate" and payload.get("kind") != "stage5_arc_agi_candidate_gate":
            continue
        metadata = payload.get("metadata") or {}
        coverage = payload.get("symbolic_coverage") or {}
        phase1_model_best = row_metric_by_variant(payload, "phase1_model_only", "best")
        phase1_hybrid_best = row_metric_by_variant(payload, "phase1_hybrid_symbolic_first", "best")
        base_model_best = row_metric_by_variant(payload, "base_model_only", "best")
        base_hybrid_best = row_metric_by_variant(payload, "base_hybrid_symbolic_first", "best")
        gates.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "arc_version": metadata.get("arc_version"),
                "arc_split": metadata.get("arc_split"),
                "limit": metadata.get("limit"),
                "grid_format": metadata.get("grid_format"),
                "selection_strategy": metadata.get("selection_strategy"),
                "examples": int(coverage.get("examples_with_targets", 0) or 0),
                "symbolic_exact": int(coverage.get("exact_symbolic", 0) or 0),
                "phase1_model_best": phase1_model_best,
                "phase1_hybrid_best": phase1_hybrid_best,
                "phase1_hybrid_best_delta": phase1_hybrid_best - phase1_model_best,
                "base_model_best": base_model_best,
                "base_hybrid_best": base_hybrid_best,
                "base_hybrid_best_delta": base_hybrid_best - base_model_best,
            }
        )
    return sorted(gates, key=lambda item: str(item["path"]))


def sft_recipe_score(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("best_best", row.get("tuned_best", 0)) or 0),
        int(row.get("best_selected", row.get("tuned_selected", 0)) or 0),
        int(row.get("tuned_best", 0) or 0),
        int(row.get("tuned_selected", 0) or 0),
    )


def trace_gate_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    comparison = payload.get("comparison") or {}
    grid = comparison.get("grid_only") or {}
    grid_score = sft_recipe_score(grid)
    trace_rows = [
        (sft_recipe_score(row), str(label), row)
        for label, row in comparison.items()
        if label != "grid_only" and isinstance(row, dict)
    ]
    best_score, best_arm, _best_row = max(trace_rows, default=((-grid_score[0], -grid_score[1], 0, 0), None, {}))
    return {
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "kind": "trace_sft_gate",
        "best_arm": best_arm,
        "best_delta": int(best_score[0]) - int(grid_score[0]),
        "selected_delta": int(best_score[1]) - int(grid_score[1]),
    }


def distill_gate_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    comparison = payload.get("comparison") or {}
    off = comparison.get("distill_off") or {}
    on = comparison.get("distill_on") or {}
    off_score = sft_recipe_score(off)
    on_score = sft_recipe_score(on)
    selected_arm = "distill_on" if on_score >= off_score else "distill_off"
    return {
        "path": path_for_cli(path),
        "run_id": str(payload.get("run_id") or path.parent.name),
        "kind": "distill_sft_gate",
        "best_arm": selected_arm,
        "best_delta": int(on_score[0]) - int(off_score[0]),
        "selected_delta": int(on_score[1]) - int(off_score[1]),
    }


def arc_agi_sft_recipe_gates(summary_files: list[Path]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        comparison = payload.get("comparison")
        if not isinstance(comparison, dict):
            continue
        if payload.get("gate") == "stage5_arc_agi_trace_sft_gate" or payload.get("kind") == "trace_sft_gate":
            gates.append(trace_gate_summary(path, payload))
        elif payload.get("gate") == "stage5_arc_agi_distill_sft_gate" or payload.get("kind") == "distill_sft_gate":
            gates.append(distill_gate_summary(path, payload))
        elif "grid_only" in comparison and isinstance(payload.get("arms"), list):
            gates.append(trace_gate_summary(path, payload))
        elif {"distill_off", "distill_on"} <= set(comparison):
            gates.append(distill_gate_summary(path, payload))
    return sorted(gates, key=lambda item: str(item["path"]))


def curriculum_sft_validation_checks(payload: dict[str, Any]) -> dict[str, Any]:
    checks = payload.get("validation_checks")
    return checks if isinstance(checks, dict) else {}


def curriculum_sft_validation_status(payload: dict[str, Any]) -> str:
    checks = curriculum_sft_validation_checks(payload)
    if checks:
        return str(checks.get("status") or payload.get("status") or "unknown")
    return str(payload.get("status") or "legacy")


def curriculum_sft_validation_issues(payload: dict[str, Any]) -> list[str]:
    checks = curriculum_sft_validation_checks(payload)
    issues = checks.get("issues")
    if isinstance(issues, list):
        return [str(issue) for issue in issues]
    return []


def curriculum_sft_depth_gradient_observed(payload: dict[str, Any]) -> bool | None:
    checks = curriculum_sft_validation_checks(payload)
    gradient = checks.get("depth_gradient")
    if not isinstance(gradient, dict):
        return None
    observed = gradient.get("observed")
    return bool(observed) if observed is not None else None


def curriculum_statuses(summary_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        kind = str(payload.get("kind") or "")
        if kind == "curriculum_pipeline_from_artifacts":
            counts = payload.get("counts") or {}
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status"),
                    "go": None,
                    "work_dir": payload.get("work_dir"),
                    "positive_rows": int(counts.get("positive_sft_rows", 0) or 0),
                    "train_rows": None,
                    "val_rows": None,
                    "mean_expected_loops": None,
                    "expected_ce": None,
                    "checkpoint": None,
                    "next_action": payload.get("next_action"),
                }
            )
        elif kind == "capability_ladder_curriculum_pipeline":
            counts = payload.get("counts") or {}
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status"),
                    "go": None,
                    "work_dir": payload.get("work_dir") or path_for_cli(path.parent),
                    "positive_rows": int(counts.get("positive_sft_rows", 0) or 0),
                    "train_rows": None,
                    "val_rows": None,
                    "mean_expected_loops": None,
                    "expected_ce": None,
                    "checkpoint": None,
                    "next_action": payload.get("next_action") or "run capability-ladder SFT gate",
                }
            )
        elif kind == "stage5_capability_ladder_mcq_probe":
            curriculum = payload.get("curriculum") or {}
            counts = curriculum.get("counts") or {}
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status"),
                    "go": None,
                    "work_dir": curriculum.get("work_dir"),
                    "positive_rows": int(counts.get("positive_sft_rows", 0) or 0),
                    "train_rows": None,
                    "val_rows": None,
                    "mean_expected_loops": None,
                    "expected_ce": None,
                    "checkpoint": None,
                    "next_action": payload.get("next_action") or "build capability-ladder strong-trace jobs",
                }
            )
        elif kind == "stage5_capability_ladder_trace_jobs":
            trace_jobs = payload.get("trace_jobs") or {}
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status"),
                    "go": None,
                    "work_dir": payload.get("work_dir") or path_for_cli(path.parent),
                    "positive_rows": int(trace_jobs.get("selected_rows", 0) or 0),
                    "train_rows": None,
                    "val_rows": None,
                    "mean_expected_loops": None,
                    "expected_ce": None,
                    "checkpoint": None,
                    "next_action": payload.get("next_action") or "run provider responses then collect traced rows",
                }
            )
        elif kind == "stage5_capability_ladder_trace_responses":
            report = payload.get("response_report") or {}
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status"),
                    "go": None,
                    "work_dir": payload.get("work_dir") or path_for_cli(path.parent),
                    "positive_rows": int(report.get("written", 0) or 0) + int(report.get("skipped", 0) or 0),
                    "train_rows": None,
                    "val_rows": None,
                    "mean_expected_loops": None,
                    "expected_ce": None,
                    "checkpoint": None,
                    "next_action": payload.get("next_action") or "collect capability-ladder trace rows",
                }
            )
        elif kind == "stage5_capability_ladder_trace_collection":
            curriculum = payload.get("curriculum") or {}
            counts = curriculum.get("counts") or {}
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status"),
                    "go": (payload.get("gate") or {}).get("go"),
                    "work_dir": curriculum.get("work_dir"),
                    "positive_rows": int(counts.get("positive_sft_rows", 0) or 0),
                    "train_rows": None,
                    "val_rows": None,
                    "mean_expected_loops": None,
                    "expected_ce": None,
                    "checkpoint": None,
                    "next_action": payload.get("next_action") or "run traced capability-ladder SFT gate",
                }
            )
        elif kind == "curriculum_sft_gate":
            positive = ((payload.get("checks") or {}).get("positive_sft") or {})
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status"),
                    "go": bool(payload.get("go", False)),
                    "work_dir": payload.get("work_dir"),
                    "positive_rows": int(positive.get("rows", 0) or 0),
                    "train_rows": None,
                    "val_rows": None,
                    "mean_expected_loops": None,
                    "expected_ce": None,
                    "checkpoint": None,
                    "next_action": "run_stage5_curriculum_sft.py" if payload.get("go") else "fix curriculum gate issues",
                }
            )
        elif kind in {"stage5_curriculum_sft", "stage5_reentry_recovery_training"}:
            dataset = payload.get("dataset") or {}
            phase1_val = payload.get("phase1_val") or {}
            validation_status = curriculum_sft_validation_status(payload)
            validation_issues = curriculum_sft_validation_issues(payload)
            has_validation_checks = bool(curriculum_sft_validation_checks(payload))
            if kind == "stage5_reentry_recovery_training":
                next_action = (
                    "inspect re-entry recovery validation before benchmarking"
                    if has_validation_checks and validation_status != "validation_sane"
                    else "run debiased_benchmark_suite before dense control or breadth diagnostics"
                )
                checkpoint = payload.get("phase1_checkpoint") or payload.get("checkpoint")
                work_dir = ((payload.get("config") or {}).get("work_dir"))
            else:
                next_action = (
                    "inspect curriculum SFT validation before GPU diagnostics"
                    if has_validation_checks and validation_status != "validation_sane"
                    else "run bounded routing diagnostic before broader benchmark suite"
                )
                checkpoint = payload.get("phase1_checkpoint")
                work_dir = ((payload.get("config") or {}).get("work_dir"))
            rows.append(
                {
                    "path": path_for_cli(path),
                    "run_id": str(payload.get("run_id") or path.parent.name),
                    "kind": kind,
                    "status": payload.get("status") or "completed",
                    "go": None,
                    "work_dir": work_dir,
                    "positive_rows": int(dataset.get("rows", 0) or 0),
                    "train_rows": int(dataset.get("train_rows", 0) or 0),
                    "val_rows": int(dataset.get("val_rows", 0) or 0),
                    "mean_expected_loops": phase1_val.get("mean_expected_loops"),
                    "expected_ce": phase1_val.get("expected_ce"),
                    "checkpoint": checkpoint,
                    "validation_status": validation_status,
                    "validation_issues": validation_issues,
                    "depth_gradient_observed": curriculum_sft_depth_gradient_observed(payload),
                    "next_action": next_action,
                }
            )
    return sorted(rows, key=lambda item: str(item["path"]))


def balanced_assessment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("balanced_assessment")
    return nested if isinstance(nested, dict) else payload


def balanced_assessment_rows(summary_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload:
            continue
        if payload.get("kind") not in {"stage5_recovery_full_assessment", "stage5_balanced_mcq_checkpoint_assessment"}:
            continue
        assessment = balanced_assessment_payload(payload)
        best = assessment.get("best_checkpoint") if isinstance(assessment.get("best_checkpoint"), dict) else {}
        rows.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or assessment.get("run_id") or path.parent.name),
                "kind": str(payload.get("kind")),
                "status": assessment.get("status") or payload.get("status"),
                "passed": bool(assessment.get("passed", payload.get("passed", False))),
                "selected_checkpoint": payload.get("selected_checkpoint") or best.get("checkpoint"),
                "label": best.get("label"),
                "micro_correct_delta": best.get("micro_correct_delta"),
                "macro_accuracy_delta": best.get("required_macro_accuracy_delta"),
                "base_correct": best.get("base_correct"),
                "recurrent_correct": best.get("recurrent_correct"),
                "total": best.get("total"),
                "combined_wins": best.get("combined_wins"),
                "combined_losses": best.get("combined_losses"),
                "combined_ties": best.get("combined_ties"),
                "child_returncode": payload.get("child_returncode"),
                "child_summary_path": payload.get("child_summary_path"),
                "child_stdout_tail": payload.get("child_stdout_tail"),
                "next_step": assessment.get("next_step") or payload.get("next_step"),
            }
        )
    return sorted(rows, key=lambda item: str(item["path"]))


def direct_preservation_statuses(summary_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("kind") != "stage5_direct_preservation_probe":
            continue
        best = payload.get("best_checkpoint") if isinstance(payload.get("best_checkpoint"), dict) else {}
        direct_sft = ((payload.get("data") or {}).get("direct_sft") or {})
        loop1_eval = best.get("loop1_eval") if isinstance(best.get("loop1_eval"), dict) else {}
        loop4_eval = best.get("loop4_eval") if isinstance(best.get("loop4_eval"), dict) else {}
        comparison = best.get("comparison_to_base") if isinstance(best.get("comparison_to_base"), dict) else {}
        rows.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "source_summary": payload.get("source_summary"),
                "resume_checkpoint": payload.get("resume_checkpoint"),
                "checkpoint": best.get("checkpoint"),
                "trained": best.get("trained"),
                "selected_rows": int(direct_sft.get("selected_rows", 0) or 0),
                "base_correct": (payload.get("base_eval") or {}).get("correct"),
                "start_loop1_correct": (payload.get("start_loop1_eval") or {}).get("correct"),
                "best_loop1_correct": loop1_eval.get("correct"),
                "best_loop4_correct": loop4_eval.get("correct"),
                "total": loop1_eval.get("total") or (payload.get("base_eval") or {}).get("total"),
                "helped": comparison.get("helped"),
                "hurt": comparison.get("hurt"),
                "mean_margin_delta": comparison.get("mean_margin_delta"),
                "max_abs_prediction_count_delta": comparison.get("max_abs_prediction_count_delta"),
                "calibration_ok": comparison.get("calibration_ok"),
                "next_step": payload.get("next_step"),
            }
        )
    return sorted(rows, key=lambda item: str(item["path"]))


def reentry_statuses(summary_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if not payload or payload.get("kind") not in {
            "reentry_drift_diagnostic",
            "stage5_reentry_norm_eval_only",
            "stage5_reentry_repair_smoke",
        }:
            continue
        assessment_path = path.parent / "reentry_assessment.json"
        assessment = safe_read_json(assessment_path) if assessment_path.exists() else None
        metrics = assessment.get("metrics") if isinstance(assessment, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "kind": payload.get("kind"),
                "stage": assessment.get("stage") if isinstance(assessment, dict) else None,
                "status": assessment.get("status") if isinstance(assessment, dict) else payload.get("status"),
                "recommendation": assessment.get("recommendation") if isinstance(assessment, dict) else None,
                "reason": assessment.get("reason") if isinstance(assessment, dict) else None,
                "assessment_path": path_for_cli(assessment_path) if assessment_path.exists() else None,
                "bridge_gate": metrics.get("bridge_gate") if "bridge_gate" in metrics else metrics.get("post_bridge_gate"),
                "bridge_live": metrics.get("bridge_live"),
                "bridge_moved": metrics.get("bridge_moved"),
                "adapter_live": metrics.get("adapter_live"),
                "adapter_moved": metrics.get("adapter_moved"),
                "loop1_best_hits_delta": metrics.get("loop1_best_hits_delta"),
                "loop1_candidate_hits_delta": metrics.get("loop1_candidate_hits_delta"),
                "candidate_hits_delta_entry_minus_none": metrics.get("candidate_hits_delta_entry_minus_none"),
                "best_hits_delta_entry_minus_none": metrics.get("best_hits_delta_entry_minus_none"),
                "loop8_output_over_entry_delta_entry_minus_none": metrics.get(
                    "loop8_output_over_entry_delta_entry_minus_none"
                ),
            }
        )
    return sorted(rows, key=lambda item: str(item["path"]))


def iter_summary_files(scan_root: Path) -> list[Path]:
    if not scan_root.exists():
        return []
    patterns = ["summary.json", "*_summary.json"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(scan_root.glob(f"**/{pattern}"))
    return sorted(set(paths), key=lambda path: str(path))


def best_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        arm = str(record.get("arm", "unknown"))
        if arm == "particle" and record.get("examples", 0) == 0:
            continue
        current = best.get(arm)
        if current is None or (
            int(record.get("examples", 0)),
            int(record.get("selected_exact", 0)),
            int(record.get("best_of_k_exact", 0)),
            float(record.get("valid_candidate_rate", 0.0)),
        ) > (
            int(current.get("examples", 0)),
            int(current.get("selected_exact", 0)),
            int(current.get("best_of_k_exact", 0)),
            float(current.get("valid_candidate_rate", 0.0)),
        ):
            best[arm] = record
    return best


def latest_planner_source(summary_files: list[Path]) -> str | None:
    current = current_pointer_summary(summary_files)
    if current is not None:
        payload = safe_read_json(current)
        if payload and looks_like_planner_source(payload):
            return path_for_cli(current)

    candidates: list[tuple[int, float, str, Path]] = []
    for path in summary_files:
        if path.name != "summary.json":
            continue
        payload = safe_read_json(path)
        if payload and looks_like_planner_source(payload):
            candidates.append((planner_source_priority(payload), path.stat().st_mtime, str(path), path))
    if not candidates:
        return None
    return path_for_cli(sorted(candidates, reverse=True)[0][3])


def current_pointer_summary(summary_files: list[Path]) -> Path | None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if not pointer.exists():
        return None
    raw = ""
    for line in pointer.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            raw = stripped
            break
    if not raw:
        return None
    candidate = resolve_path(raw)
    try:
        candidate_resolved = candidate.resolve()
        scanned = {path.resolve() for path in summary_files}
    except OSError:
        return None
    return candidate if candidate_resolved in scanned else None


def planner_source_priority(payload: dict[str, Any]) -> int:
    """Prefer the newest actionable recovery gate over generic readiness checks."""

    if payload.get("kind") == "stage5_curriculum_sft":
        checks = curriculum_sft_validation_checks(payload)
        if checks and checks.get("status") != "validation_sane":
            return 40
        if payload.get("status") == "validation_needs_review":
            return 40
        return 95
    if payload.get("kind") == "reentry_drift_diagnostic":
        return 106
    if payload.get("kind") == "stage5_reentry_norm_eval_only":
        return 107
    if payload.get("kind") == "stage5_reentry_repair_smoke":
        return 109
    if payload.get("kind") == "stage5_reentry_recovery_training":
        checks = curriculum_sft_validation_checks(payload)
        if checks and checks.get("status") != "validation_sane":
            return 40
        if payload.get("status") == "validation_needs_review":
            return 40
        return 111
    if payload.get("kind") == "curriculum_sft_gate":
        return 85 if payload.get("go") else 35
    if payload.get("kind") == "stage5_capability_ladder_mcq_probe":
        return 83 if str(payload.get("status") or "").endswith(("gate_ready", "needs_review")) else 25
    if payload.get("kind") == "stage5_capability_ladder_trace_jobs":
        return 82 if payload.get("status") == "ready" else 25
    if payload.get("kind") == "stage5_capability_ladder_trace_responses":
        return 83 if payload.get("status") == "responses_ready" else 28
    if payload.get("kind") == "stage5_capability_ladder_trace_collection":
        return 84 if payload.get("status") == "trace_curriculum_gate_ready" else 30
    if payload.get("kind") == "capability_ladder_curriculum_pipeline":
        return 84 if payload.get("status") == "complete" else 25
    if payload.get("kind") == "curriculum_pipeline_from_artifacts":
        return 84 if payload.get("status") == "complete" else 25
    if payload.get("kind") == "stage5_balanced_arc_mix_gate":
        if payload.get("status") in {"proxy_lift", "proxy_matches_base"}:
            return 100
        return 90
    if payload.get("kind") == "stage5_direct_preservation_probe":
        return 104 if payload.get("passed") else 94
    if payload.get("kind") == "stage5_surface_alignment_repair":
        if payload.get("status") in {"surface_alignment_passed", "surface_alignment_partial"}:
            return 108
        if payload.get("status") == "surface_alignment_tradeoff":
            return 70
        return 96
    if payload.get("kind") == "stage5_surface_repair_assessment":
        if payload.get("status") in {"surface_repair_passed", "surface_repair_partial"}:
            return 108
        if payload.get("status") == "surface_repair_tradeoff":
            return 70
        return 96
    if payload.get("kind") == "stage5_dense_mcq_trace_sft_control":
        assessment = payload.get("recipe_control_assessment")
        if isinstance(assessment, dict) and assessment.get("ran"):
            return 112
        return 100
    if payload.get("gate") == "stage5_same_recipe_mcq_architecture" or payload.get("kind") == "stage5_mcq_recipe_control_assessment":
        return 114
    if payload.get("kind") == "stage5_traced_sft_assessment":
        return 103 if payload.get("status") == "needs_direct_preservation_repair" else 93
    if payload.get("kind") == "stage5_recovery_full_assessment":
        return 110
    if payload.get("kind") == "stage5_balanced_mcq_checkpoint_assessment":
        return 105
    if payload.get("kind") == "stage5_broader_benchmark_suite_assessment":
        return 80
    if payload.get("gate") == "stage5_release_benchmark_readiness":
        return 20
    return 50


def recovered_base_gaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("kind") != "recovered_benchmark":
            continue
        by_run.setdefault(str(record["run_id"]), {})[str(record["arm"])] = record
    gaps: list[dict[str, Any]] = []
    for run_id, arms in by_run.items():
        base = arms.get("base")
        start = arms.get("phase1_start")
        recovered = arms.get("recovered")
        if not base or not start or not recovered:
            continue
        selected_initial_gap = int(base["selected_exact"]) - int(start["selected_exact"])
        selected_gain = int(recovered["selected_exact"]) - int(start["selected_exact"])
        best_initial_gap = int(base["best_of_k_exact"]) - int(start["best_of_k_exact"])
        best_gain = int(recovered["best_of_k_exact"]) - int(start["best_of_k_exact"])
        gaps.append(
            {
                "run_id": run_id,
                "examples": min(int(base["examples"]), int(start["examples"]), int(recovered["examples"])),
                "selected_initial_gap_to_base": selected_initial_gap,
                "selected_gain_from_start": selected_gain,
                "selected_delta_recovered_vs_base": int(recovered["selected_exact"]) - int(base["selected_exact"]),
                "selected_gap_closure_fraction": (
                    selected_gain / selected_initial_gap if selected_initial_gap > 0 else None
                ),
                "best_of_k_initial_gap_to_base": best_initial_gap,
                "best_of_k_gain_from_start": best_gain,
                "best_of_k_delta_recovered_vs_base": int(recovered["best_of_k_exact"]) - int(base["best_of_k_exact"]),
                "best_of_k_gap_closure_fraction": best_gain / best_initial_gap if best_initial_gap > 0 else None,
                "path": recovered["path"],
            }
        )
    return sorted(
        gaps,
        key=lambda row: (
            int(row["examples"]),
            int(row["selected_delta_recovered_vs_base"]),
            int(row["best_of_k_delta_recovered_vs_base"]),
        ),
        reverse=True,
    )


def scan_progress(scan_root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    summary_files = iter_summary_files(scan_root)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in summary_files:
        payload = safe_read_json(path)
        if payload is None:
            skipped.append(path_for_cli(path))
            continue
        records.extend(records_from_payload(path, payload))
    return {
        "run_id": run_id or current_run_id(),
        "scan_root": path_for_cli(scan_root),
        "scanned_files": len(summary_files),
        "parsed_records": len(records),
        "skipped_files": skipped,
        "records": records,
        "best_by_arm": best_records(records),
        "recovered_vs_base_gaps": recovered_base_gaps(records),
        "gate1_assessments": gate1_assessments(summary_files),
        "gate2_assessments": gate2_assessments(summary_files),
        "selector_replications": selector_replications(summary_files),
        "recipe_selector_conversions": recipe_selector_conversions(summary_files),
        "recipe_control_assessments": recipe_control_assessments(summary_files),
        "surface_alignment_statuses": surface_alignment_statuses(summary_files),
        "dense_mcq_control_statuses": dense_mcq_control_statuses(summary_files),
        "release_gate_assessments": release_gate_assessments(summary_files),
        "benchmark_suite_assessments": benchmark_suite_assessments(summary_files),
        "broader_benchmark_gate_assessments": broader_benchmark_gate_assessments(summary_files),
        "balanced_full_assessments": balanced_assessment_rows(summary_files),
        "reentry_statuses": reentry_statuses(summary_files),
        "direct_preservation_statuses": direct_preservation_statuses(summary_files),
        "claim_readiness_packets": claim_readiness_packets(summary_files),
        "arc_agi_baseline_registries": arc_agi_baseline_registries(summary_files),
        "arc_agi_sota_comparisons": arc_agi_sota_comparisons(summary_files),
        "arc_agi_candidate_gates": arc_agi_candidate_gates(summary_files),
        "arc_agi_sft_recipe_gates": arc_agi_sft_recipe_gates(summary_files),
        "curriculum_statuses": curriculum_statuses(summary_files),
        "recommended_next_plan_source": latest_planner_source(summary_files),
    }


def write_report(payload: dict[str, Any], output_dir: Path | None = None) -> None:
    output_dir = output_dir or run_dir(str(payload["run_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 ARC-AGI Progress Ledger - {payload['run_id']}",
        "",
        f"- Scan root: `{payload['scan_root']}`",
        f"- Scanned files: `{payload['scanned_files']}`",
        f"- Parsed records: `{payload['parsed_records']}`",
        f"- Recommended next-plan source: `{payload['recommended_next_plan_source']}`",
        "",
        "## Best By Arm",
        "",
        "| Arm | Selected | Best-of-K | Examples | Label | Source |",
        "|---|---:|---:|---:|---|---|",
    ]
    for arm, record in sorted(payload["best_by_arm"].items()):
        lines.append(
            f"| `{arm}` | {record['selected_exact']} | {record['best_of_k_exact']} | "
            f"{record['examples']} | `{record['label']}` | `{record['path']}` |"
        )
    lines.extend(["", "## Recovered vs Base Gaps", ""])
    for row in payload["recovered_vs_base_gaps"][:10]:
        selected_closure = row.get("selected_gap_closure_fraction")
        best_closure = row.get("best_of_k_gap_closure_fraction")
        selected_text = "n/a" if selected_closure is None else f"{float(selected_closure):.2%}"
        best_text = "n/a" if best_closure is None else f"{float(best_closure):.2%}"
        lines.append(
            f"- `{row['run_id']}` examples `{row['examples']}` selected delta "
            f"`{row['selected_delta_recovered_vs_base']}` closure `{selected_text}`, best delta "
            f"`{row['best_of_k_delta_recovered_vs_base']}` closure `{best_text}`"
        )
    if not payload["recovered_vs_base_gaps"]:
        lines.append("- No complete recovered-vs-base benchmark summaries found.")
    lines.extend(["", "## Gate 1 Assessments", ""])
    if payload["gate1_assessments"]:
        for assessment in payload["gate1_assessments"][-10:]:
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` source `{assessment['source_summary']}`: "
                f"{assessment['reason']}"
            )
    else:
        lines.append("- No Gate 1 assessment summaries found.")
    lines.extend(["", "## Gate 2 Assessments", ""])
    if payload["gate2_assessments"]:
        for assessment in payload["gate2_assessments"][-10:]:
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` variant `{assessment['best_variant']}` "
                f"selected delta `{assessment['selected_delta']}`, best-of-K delta "
                f"`{assessment['best_of_k_delta']}`: {assessment['reason']}"
            )
    else:
        lines.append("- No Gate 2 assessment summaries found.")
    lines.extend(["", "## Selector Replication Gates", ""])
    if payload["selector_replications"]:
        for assessment in payload["selector_replications"][-10:]:
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` replicated `{assessment['replicated_comparisons']}`: "
                f"{assessment['next_step']}"
            )
    else:
        lines.append("- No selector replication gates found.")
    lines.extend(["", "## Same-Recipe Selector Conversion Gates", ""])
    if payload["recipe_selector_conversions"]:
        for assessment in payload["recipe_selector_conversions"][-10:]:
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` passing `{assessment['passing_selectors']}` "
                f"best `{assessment['best_selector']}` claim-level `{assessment['claim_level_selector']}` "
                f"selector exact `{assessment['selector_generated_selected_exact']}` "
                f"beyond best-of-K `{assessment['selected_exceeds_best_of_k']}`: {assessment['next_step']}"
            )
    else:
        lines.append("- No same-recipe selector conversion gates found.")
    lines.extend(["", "## Same-Recipe Architecture Assessments", ""])
    if payload["recipe_control_assessments"]:
        for assessment in payload["recipe_control_assessments"][-10:]:
            mcq_fragment = ""
            if assessment.get("gate") == "stage5_same_recipe_mcq_architecture":
                mcq_fragment = (
                    f" MCQ primary `{assessment['primary_delta_recurrent_vs_dense']}`, "
                    f"challenge content/cyclic "
                    f"`{assessment['arc_challenge_content_delta_recurrent_vs_dense']}/"
                    f"{assessment['arc_challenge_cyclic_delta_recurrent_vs_dense']}`, "
                    f"easy content/cyclic "
                    f"`{assessment['arc_easy_content_delta_recurrent_vs_dense']}/"
                    f"{assessment['arc_easy_cyclic_delta_recurrent_vs_dense']}`"
                )
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` aggregate selected delta "
                f"`{assessment['aggregate_selected_delta']}`, hard selected delta "
                f"`{assessment['hard_selected_delta']}`, aggregate best-of-K delta "
                f"`{assessment['aggregate_best_of_k_delta']}`, hard best-of-K delta "
                f"`{assessment['hard_best_of_k_delta']}`{mcq_fragment}: {assessment['reason']}"
            )
    else:
        lines.append("- No same-recipe architecture assessment summaries found.")
    lines.extend(["", "## Surface Alignment Repairs", ""])
    if payload["surface_alignment_statuses"]:
        for row in payload["surface_alignment_statuses"][-10:]:
            lines.append(
                f"- `{row['run_id']}` status `{row['status']}` passed `{row['passed']}` "
                f"rows `{row['surface_alignment_rows']}` surface assessment "
                f"`{row['surface_repair_assessment_status']}` benchmark assessment "
                f"`{row['assessment_status']}` checkpoint `{row.get('checkpoint') or ''}`: "
                f"{row.get('next_step') or ''}"
            )
    else:
        lines.append("- No surface-alignment repair summaries found.")
    lines.extend(["", "## Dense MCQ Trace-SFT Controls", ""])
    if payload["dense_mcq_control_statuses"]:
        for row in payload["dense_mcq_control_statuses"][-10:]:
            lines.append(
                f"- `{row['run_id']}` train rows `{row['train_rows']}` extra "
                f"`{row['extra_train_rows']}` assessment ran `{row['assessment_ran']}` "
                f"status `{row['assessment_status']}` passed `{row['assessment_passed']}` "
                f"checkpoint `{row.get('dense_checkpoint') or ''}`: {row.get('next_step') or ''}"
            )
    else:
        lines.append("- No dense MCQ trace-SFT control summaries found.")
    lines.extend(["", "## Release / Benchmark Gates", ""])
    if payload["release_gate_assessments"]:
        for assessment in payload["release_gate_assessments"][-10:]:
            failed = [
                str(row.get("name"))
                for row in assessment.get("criteria", [])
                if isinstance(row, dict) and not row.get("passed")
            ]
            failed_text = ", ".join(failed) if failed else "none"
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` min ARC examples `{assessment['min_arc_examples']}` "
                f"failed criteria `{failed_text}`: {assessment['next_step']}"
            )
    else:
        lines.append("- No release / benchmark gate summaries found.")
    lines.extend(["", "## Broader Benchmark Suites", ""])
    if payload["benchmark_suite_assessments"]:
        for assessment in payload["benchmark_suite_assessments"][-10:]:
            delta_text = "; ".join(
                f"{row['benchmark']}/{row['score_target']}/{row['aggregate']}: "
                f"{row['correct_delta_recurrent_vs_base']:+d} "
                f"(W/L/T {row['wins']}/{row['losses']}/{row['ties']}, p {row['sign_test_p_value']})"
                for row in assessment.get("deltas", [])
            )
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` checkpoint "
                f"`{assessment['checkpoint']}` deltas: {delta_text or 'none'}"
            )
    else:
        lines.append("- No broader benchmark suite summaries found.")
    lines.extend(["", "## Broader Benchmark Gates", ""])
    if payload["broader_benchmark_gate_assessments"]:
        for assessment in payload["broader_benchmark_gate_assessments"][-10:]:
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` source `{assessment['source_summary']}`: "
                f"{assessment['next_step']}"
            )
    else:
        lines.append("- No broader benchmark gate summaries found.")
    lines.extend(["", "## Full Balanced ARC Assessments", ""])
    if payload["balanced_full_assessments"]:
        for assessment in payload["balanced_full_assessments"][-10:]:
            child_fragment = ""
            if assessment.get("child_returncode") is not None or assessment.get("child_summary_path"):
                child_fragment = (
                    f" child return `{assessment.get('child_returncode')}` child summary "
                    f"`{assessment.get('child_summary_path')}`"
                )
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` label `{assessment['label']}` micro delta "
                f"`{assessment['micro_correct_delta']}` recurrent/base "
                f"`{assessment['recurrent_correct']}/{assessment['base_correct']}` "
                f"W/L/T `{assessment['combined_wins']}/{assessment['combined_losses']}/"
                f"{assessment['combined_ties']}` checkpoint `{assessment['selected_checkpoint']}`: "
                f"{assessment['next_step']}{child_fragment}"
            )
    else:
        lines.append("- No full balanced ARC assessment summaries found.")
    lines.extend(["", "## Re-entry Phase 0", ""])
    if payload["reentry_statuses"]:
        lines.extend(
            [
                "| Run | Kind | Stage | Status | Recommendation | Bridge | Adapter | Loop-1 Delta | Entry-RMS Delta | Source |",
                "|---|---|---|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in payload["reentry_statuses"][-12:]:
            bridge = f"live={row.get('bridge_live')} moved={row.get('bridge_moved')}"
            adapter = f"live={row.get('adapter_live')} moved={row.get('adapter_moved')}"
            loop1 = row.get("loop1_best_hits_delta")
            entry_delta = row.get("candidate_hits_delta_entry_minus_none")
            lines.append(
                f"| `{row['run_id']}` | `{row['kind']}` | `{row.get('stage') or ''}` | "
                f"`{row.get('status')}` | `{row.get('recommendation')}` | {bridge} | {adapter} | "
                f"{'n/a' if loop1 is None else loop1} | {'n/a' if entry_delta is None else entry_delta} | "
                f"`{row['path']}` |"
            )
    else:
        lines.append("- No re-entry Phase 0 summaries found.")
    lines.extend(["", "## Direct Preservation Repairs", ""])
    if payload["direct_preservation_statuses"]:
        lines.extend(
            [
                "| Run | Status | Passed | Trained | Rows | Base | Start L1 | Best L1 | Best L4 | H/H | Shift | Margin | Checkpoint | Next |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in payload["direct_preservation_statuses"][-12:]:
            total = row.get("total") or "?"
            margin = "n/a" if row.get("mean_margin_delta") is None else f"{float(row['mean_margin_delta']):.3f}"
            shift = "n/a" if row.get("max_abs_prediction_count_delta") is None else str(row["max_abs_prediction_count_delta"])
            helped_hurt = (
                "n/a"
                if row.get("helped") is None or row.get("hurt") is None
                else f"{row.get('helped')}/{row.get('hurt')}"
            )
            lines.append(
                f"| `{row['run_id']}` | `{row['status']}` | `{row['passed']}` | `{row.get('trained')}` | "
                f"{row['selected_rows']} | {row.get('base_correct')}/{total} | "
                f"{row.get('start_loop1_correct')}/{total} | {row.get('best_loop1_correct')}/{total} | "
                f"{row.get('best_loop4_correct')}/{total} | {helped_hurt} | {shift} | {margin} | "
                f"`{row.get('checkpoint') or ''}` | {row.get('next_step') or ''} |"
            )
    else:
        lines.append("- No direct-preservation repair summaries found.")
    lines.extend(["", "## Claim Readiness Packets", ""])
    if payload["claim_readiness_packets"]:
        for packet in payload["claim_readiness_packets"][-10:]:
            lines.append(
                f"- `{packet['run_id']}` status `{packet['status']}` claim level "
                f"`{packet['claim_level']}` passed `{packet['passed']}` SOTA/export linkage "
                f"`{packet['sota_export_linkage_passed']}` matched on "
                f"`{packet['sota_export_linkage_matched_on']}`: {packet['next_step']}"
            )
    else:
        lines.append("- No claim-readiness packets found.")
    lines.extend(["", "## ARC-AGI Baseline Registries", ""])
    if payload["arc_agi_baseline_registries"]:
        for registry in payload["arc_agi_baseline_registries"][-10:]:
            lines.append(
                f"- `{registry['run_id']}` status `{registry['status']}` passed "
                f"`{registry['passed']}` metric `{registry['metric']}` valid baselines "
                f"`{registry['valid_baseline_count']}` best `{registry['best_baseline']}` "
                f"`{registry['best_baseline_accuracy']}`: {registry['next_step']}"
            )
    else:
        lines.append("- No ARC-AGI baseline registry validation artifacts found.")
    lines.extend(["", "## ARC-AGI SOTA Comparisons", ""])
    if payload["arc_agi_sota_comparisons"]:
        for comparison in payload["arc_agi_sota_comparisons"][-10:]:
            lines.append(
                f"- `{comparison['run_id']}` status `{comparison['status']}` passed "
                f"`{comparison['passed']}` metric `{comparison['metric']}` candidate "
                f"`{comparison['candidate_accuracy']}` vs best `{comparison['best_baseline']}` "
                f"`{comparison['best_baseline_accuracy']}`, delta "
                f"`{comparison['delta_accuracy_vs_best_baseline']}`: {comparison['next_step']}"
            )
    else:
        lines.append("- No ARC-AGI SOTA comparison artifacts found.")
    lines.extend(["", "## ARC-AGI Candidate Gates", ""])
    if payload["arc_agi_candidate_gates"]:
        for gate in payload["arc_agi_candidate_gates"][-10:]:
            lines.append(
                f"- `{gate['run_id']}` ARC `{gate['arc_version']}` split `{gate['arc_split']}` "
                f"limit `{gate['limit']}` symbolic exact `{gate['symbolic_exact']}/{gate['examples']}`, "
                f"phase1 hybrid best delta `{gate['phase1_hybrid_best_delta']}`, "
                f"base hybrid best delta `{gate['base_hybrid_best_delta']}`"
            )
    else:
        lines.append("- No ARC-AGI candidate-gate artifacts found.")
    lines.extend(["", "## ARC-AGI SFT Recipe Gates", ""])
    if payload["arc_agi_sft_recipe_gates"]:
        for gate in payload["arc_agi_sft_recipe_gates"][-10:]:
            lines.append(
                f"- `{gate['run_id']}` kind `{gate['kind']}` best arm `{gate['best_arm']}` "
                f"best delta `{gate['best_delta']}`, selected delta `{gate['selected_delta']}`"
            )
    else:
        lines.append("- No ARC-AGI SFT recipe gate artifacts found.")
    lines.extend(["", "## Generated Curriculum Pipeline", ""])
    if payload["curriculum_statuses"]:
        lines.extend(
            [
                "| Run | Kind | Status | Validation | Depth Grad | Go | Positive | Train/Val | Loops | CE | Checkpoint | Next |",
                "|---|---|---|---|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in payload["curriculum_statuses"][-12:]:
            train_val = "n/a" if row.get("train_rows") is None else f"{row.get('train_rows')}/{row.get('val_rows')}"
            loops = "n/a" if row.get("mean_expected_loops") is None else f"{float(row['mean_expected_loops']):.3f}"
            ce = "n/a" if row.get("expected_ce") is None else f"{float(row['expected_ce']):.3f}"
            validation = str(row.get("validation_status") or "n/a")
            depth_gradient = row.get("depth_gradient_observed")
            depth_gradient_text = "n/a" if depth_gradient is None else str(bool(depth_gradient))
            lines.append(
                f"| `{row['run_id']}` | `{row['kind']}` | `{row['status']}` | "
                f"`{validation}` | `{depth_gradient_text}` | `{row['go']}` | "
                f"{row['positive_rows']} | {train_val} | {loops} | {ce} | "
                f"`{row.get('checkpoint') or ''}` | {row.get('next_action') or ''} |"
            )
    else:
        lines.append("- No generated-curriculum pipeline/SFT summaries found.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


def backup_to_drive(output_dir: Path, *, run_id: str) -> None:
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - Colab only
            print(f"Drive mount skipped/failed: {exc}")
            return
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / run_id
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, backup / "run_dir", dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Scan outputs/stage5 and write a compact ARC-AGI progress ledger.")
        return 0
    run_id = current_run_id()
    output_dir = run_dir(run_id)
    payload = scan_progress(resolve_path(current_scan_root()), run_id=run_id)
    write_report(payload, output_dir)
    backup_to_drive(output_dir, run_id=run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
