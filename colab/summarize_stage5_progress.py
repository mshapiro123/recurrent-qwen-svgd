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
    if payload.get("gate") == "stage5_release_benchmark_readiness":
        return True
    if payload.get("kind") == "stage5_benchmark_suite":
        return True
    if payload.get("gate") == "stage5_broader_benchmark_suite":
        return True
    if payload.get("gate") == "stage5_claim_readiness":
        return True
    if payload.get("gate") == "stage5_arc_agi_baseline_registry" or payload.get("kind") == "arc_agi_baseline_registry":
        return True
    if payload.get("gate") == "stage5_arc_agi_sota_comparison" or payload.get("kind") == "arc_agi_sota_comparison":
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
        if not payload or payload.get("gate") != "stage5_same_recipe_architecture":
            continue
        decision = payload.get("decision_evidence") or {}
        aggregate = decision.get("aggregate") or {}
        hard = decision.get("hard") or {}
        aggregate_best = decision.get("aggregate_best_of_k") or {}
        hard_best = decision.get("hard_best_of_k") or {}
        assessments.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
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
            }
        )
    return sorted(assessments, key=lambda item: str(item["path"]))


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
        packets.append(
            {
                "path": path_for_cli(path),
                "run_id": str(payload.get("run_id") or path.parent.name),
                "status": payload.get("status"),
                "passed": bool(payload.get("passed", False)),
                "claim_level": payload.get("claim_level"),
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
    candidates: list[Path] = []
    for path in summary_files:
        if path.name != "summary.json":
            continue
        payload = safe_read_json(path)
        if payload and looks_like_planner_source(payload):
            candidates.append(path)
    if not candidates:
        return None
    return path_for_cli(sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0])


def recovered_base_gaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("kind") != "recovered_benchmark":
            continue
        by_run.setdefault(str(record["run_id"]), {})[str(record["arm"])] = record
    gaps: list[dict[str, Any]] = []
    for run_id, arms in by_run.items():
        base = arms.get("base")
        recovered = arms.get("recovered")
        if not base or not recovered:
            continue
        gaps.append(
            {
                "run_id": run_id,
                "examples": min(int(base["examples"]), int(recovered["examples"])),
                "selected_delta_recovered_vs_base": int(recovered["selected_exact"]) - int(base["selected_exact"]),
                "best_of_k_delta_recovered_vs_base": int(recovered["best_of_k_exact"]) - int(base["best_of_k_exact"]),
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
        "release_gate_assessments": release_gate_assessments(summary_files),
        "benchmark_suite_assessments": benchmark_suite_assessments(summary_files),
        "broader_benchmark_gate_assessments": broader_benchmark_gate_assessments(summary_files),
        "claim_readiness_packets": claim_readiness_packets(summary_files),
        "arc_agi_baseline_registries": arc_agi_baseline_registries(summary_files),
        "arc_agi_sota_comparisons": arc_agi_sota_comparisons(summary_files),
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
        lines.append(
            f"- `{row['run_id']}` examples `{row['examples']}` selected delta "
            f"`{row['selected_delta_recovered_vs_base']}`, best delta "
            f"`{row['best_of_k_delta_recovered_vs_base']}`"
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
            lines.append(
                f"- `{assessment['run_id']}` status `{assessment['status']}` passed "
                f"`{assessment['passed']}` aggregate selected delta "
                f"`{assessment['aggregate_selected_delta']}`, hard selected delta "
                f"`{assessment['hard_selected_delta']}`, aggregate best-of-K delta "
                f"`{assessment['aggregate_best_of_k_delta']}`, hard best-of-K delta "
                f"`{assessment['hard_best_of_k_delta']}`: {assessment['reason']}"
            )
    else:
        lines.append("- No same-recipe architecture assessment summaries found.")
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
    lines.extend(["", "## Claim Readiness Packets", ""])
    if payload["claim_readiness_packets"]:
        for packet in payload["claim_readiness_packets"][-10:]:
            lines.append(
                f"- `{packet['run_id']}` status `{packet['status']}` claim level "
                f"`{packet['claim_level']}` passed `{packet['passed']}`: {packet['next_step']}"
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
