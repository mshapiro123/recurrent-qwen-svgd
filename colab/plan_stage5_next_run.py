"""Plan the next Stage 5 ARC-AGI experiment from finished run summaries.

This is a no-GPU planner. It reads the latest autopilot/follow-up summary and
turns it into concrete next Colab commands. The intent is to keep expensive
A100 runs evidence-led:

* if recovered recurrent beats its start checkpoint but trails base, scale the
  deterministic curriculum;
* if it matches or beats base on a smoke limit, increase the held-out ARC limit;
* if TTA helps recovered recurrent, run a larger TTA benchmark;
* if candidate distillation failed, branch to a baseline curriculum rather than
  silently carrying a bad training signal.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

try:
    from colab.stage5_limits import limit_label, parse_optional_limit
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    from stage5_limits import limit_label, parse_optional_limit


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_NEXT_PLAN_RUN_ID") or time.strftime("stage5_arc_agi_next_plan_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_SUMMARY = os.environ.get("STAGE5_ARC_AGI_NEXT_PLAN_SOURCE_SUMMARY", "")
NEXT_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_NEXT_PLAN_NEXT_LIMIT", "100"))
CONFIRM_LIMIT = parse_optional_limit(os.environ.get("STAGE5_ARC_AGI_NEXT_PLAN_CONFIRM_LIMIT", "400"))
FULL_SPLIT_AFTER_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_NEXT_PLAN_FULL_SPLIT_AFTER_LIMIT", "400"))
MIN_RECOVERED_VS_START_DELTA = int(os.environ.get("STAGE5_ARC_AGI_NEXT_PLAN_MIN_RECOVERED_VS_START_DELTA", "0"))
DEFAULT_TRACE_SFT_GATE_ARMS = os.environ.get(
    "STAGE5_ARC_AGI_NEXT_PLAN_TRACE_SFT_GATE_ARMS",
    "grid_only,symbolic_program_trace_covered,symbolic_state_trace_covered",
)


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


def looks_like_stage5_result(payload: dict[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "recovered_benchmark",
            "tta_sweep",
            "compact",
            "autopilot_compact",
            "best_by_label",
            "recovery_decision",
        )
    ) or (
        selector_rescore_payload(payload) is not None
        or dense_sft_payload(payload) is not None
        or recurrent_sft_payload(payload) is not None
        or gate1_assessment_payload(payload) is not None
        or gate2_assessment_payload(payload) is not None
        or selector_replication_payload(payload) is not None
        or recipe_selector_conversion_payload(payload) is not None
        or recipe_control_assessment_payload(payload) is not None
        or release_gate_payload(payload) is not None
        or benchmark_suite_payload(payload) is not None
        or benchmark_suite_assessment_payload(payload) is not None
        or claim_readiness_payload(payload) is not None
        or arc_agi_baseline_registry_payload(payload) is not None
        or arc_agi_sota_comparison_payload(payload) is not None
    )


def latest_summary() -> Path:
    candidates: list[Path] = []
    for path in ROOT.glob("outputs/stage5/*/summary.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if looks_like_stage5_result(payload):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No Stage 5 result summary found. Set STAGE5_ARC_AGI_NEXT_PLAN_SOURCE_SUMMARY.")
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def resolve_source_summary() -> Path:
    return resolve_path(SOURCE_SUMMARY) if SOURCE_SUMMARY else latest_summary()


def metric(summary: dict[str, Any] | None, key: str) -> int:
    if not summary:
        return 0
    return int(summary.get(key, 0))


def rate(summary: dict[str, Any] | None, key: str) -> float:
    if not summary:
        return 0.0
    return float(summary.get(key, 0.0))


def benchmark_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("recovered_benchmark"):
        return payload["recovered_benchmark"]
    if {"base", "phase1_start", "recovered", "deltas"} <= set(payload):
        return payload
    return None


def tta_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload.get("tta_sweep")


def recovery_analysis_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    benchmark = benchmark_payload(payload)
    if benchmark and benchmark.get("recovery_analysis"):
        return benchmark["recovery_analysis"]
    analysis = payload.get("recovery_analysis")
    return analysis if isinstance(analysis, dict) else None


def selector_rescore_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    if "best_by_label" in payload or {"source_run_dir", "strategies"} <= set(payload):
        return payload
    return None


def recovery_particle_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(payload.get("recovery_decision"), dict) and isinstance(payload.get("particle_decision"), dict):
        return payload
    return None


def dense_sft_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("kind") == "dense_sft_control" else None


def gate1_assessment_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("gate") == "stage5_gate1_selector_tta" else None


def gate2_assessment_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("gate") == "stage5_gate2_particle_mechanism" else None


def selector_replication_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("gate") == "stage5_selector_replication" or payload.get("kind") == "selector_replication":
        return payload
    return None


def recipe_selector_conversion_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("gate") == "stage5_same_recipe_selector_conversion" or payload.get("kind") == "recipe_selector_conversion":
        return payload
    return None


def recipe_control_assessment_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("gate") == "stage5_same_recipe_architecture" else None


def release_gate_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("gate") == "stage5_release_benchmark_readiness" else None


def benchmark_suite_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("kind") == "stage5_benchmark_suite" else None


def benchmark_suite_assessment_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("gate") == "stage5_broader_benchmark_suite" else None


def claim_readiness_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if payload.get("gate") == "stage5_claim_readiness" else None


def arc_agi_baseline_registry_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("gate") == "stage5_arc_agi_baseline_registry" or payload.get("kind") == "arc_agi_baseline_registry":
        return payload
    return None


def arc_agi_sota_comparison_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("gate") == "stage5_arc_agi_sota_comparison" or payload.get("kind") == "arc_agi_sota_comparison":
        return payload
    return None


def recurrent_sft_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "phase1_arc_agi_tuned" in payload and payload.get("tuned_checkpoint"):
        return payload
    return None


def direct_tta_sweep_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(payload.get("rows"), list) and isinstance(payload.get("paired_comparisons"), dict):
        if "deltas" in payload and "best_by_label" not in payload:
            return payload
    return None


def needs_gate1_assessment(payload: dict[str, Any]) -> bool:
    if gate1_assessment_payload(payload):
        return False
    if selector_rescore_payload(payload) and bool(payload.get("paired_comparisons")):
        return True
    nested_tta = tta_payload(payload)
    if isinstance(nested_tta, dict) and bool(nested_tta.get("paired_comparisons")):
        return True
    return direct_tta_sweep_payload(payload) is not None and bool(payload.get("paired_comparisons"))


def needs_gate2_assessment(payload: dict[str, Any]) -> bool:
    recovery_particle = recovery_particle_payload(payload)
    if not recovery_particle:
        return False
    if gate2_assessment_payload(payload):
        return False
    recovery = recovery_particle.get("recovery_decision") or {}
    particle = recovery_particle.get("particle_decision") or {}
    return bool(recovery.get("passed")) and bool(particle.get("passed"))


def needs_recipe_control_assessment(payload: dict[str, Any]) -> bool:
    return recurrent_sft_payload(payload) is not None


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("autopilot_compact") or payload.get("compact") or {}


def delta_value(deltas: dict[str, Any] | None, group: str, key: str) -> int:
    if not deltas:
        return 0
    return int((deltas.get(group) or {}).get(key, 0))


def paired_metric(payload: dict[str, Any] | None, comparison: str, metric_name: str) -> dict[str, Any] | None:
    if not payload:
        return None
    comparisons = payload.get("paired_comparisons") or {}
    metrics = (comparisons.get(comparison) or {}).get("metrics") or {}
    metric_payload = metrics.get(metric_name)
    return metric_payload if isinstance(metric_payload, dict) else None


def paired_delta_or_aggregate(
    payload: dict[str, Any] | None,
    *,
    comparison: str,
    metric_name: str,
    aggregate_group: str,
    aggregate_key: str,
) -> int:
    paired = paired_metric(payload, comparison, metric_name)
    if paired is not None:
        return int(paired.get("delta_exact", 0))
    return delta_value((payload or {}).get("deltas") or {}, aggregate_group, aggregate_key)


def paired_supports_nonnegative(stats: dict[str, Any] | None, fallback_delta: int) -> bool:
    if stats is None:
        return fallback_delta >= 0
    return int(stats.get("delta_exact", 0)) >= 0 and int(stats.get("wins", 0)) >= int(stats.get("losses", 0))


def paired_supports_positive(stats: dict[str, Any] | None, fallback_delta: int) -> bool:
    if stats is None:
        return fallback_delta > 0
    return int(stats.get("delta_exact", 0)) > 0 and int(stats.get("wins", 0)) > int(stats.get("losses", 0))


def evidence_fragment(stats: dict[str, Any] | None, fallback_delta: int) -> str:
    if stats is None:
        return f"aggregate delta {fallback_delta}"
    ci = stats.get("bootstrap_delta_accuracy_ci95") or {}
    return (
        f"paired delta {stats.get('delta_exact', 0)} "
        f"({stats.get('wins', 0)}/{stats.get('losses', 0)}/{stats.get('ties', 0)} W/L/T, "
        f"CI95 [{ci.get('low')}, {ci.get('high')}])"
    )


def gap_closure_fraction(benchmark: dict[str, Any] | None, metric_name: str) -> float | None:
    if not benchmark:
        return None
    closure = ((benchmark.get("gap_closure") or {}).get(metric_name) or {}).get("closure_fraction")
    if closure is not None:
        return float(closure)
    base = (benchmark.get("base") or {}).get("summary") or {}
    start = (benchmark.get("phase1_start") or {}).get("summary") or {}
    recovered = (benchmark.get("recovered") or {}).get("summary") or {}
    initial_gap = metric(base, metric_name) - metric(start, metric_name)
    if initial_gap <= 0:
        return None
    return (metric(recovered, metric_name) - metric(start, metric_name)) / initial_gap


def gap_closure_fragment(benchmark: dict[str, Any] | None) -> str:
    selected = gap_closure_fraction(benchmark, "selected_exact")
    best = gap_closure_fraction(benchmark, "best_of_k_exact")
    selected_text = "n/a" if selected is None else f"{selected:.2%}"
    best_text = "n/a" if best is None else f"{best:.2%}"
    return f"Gap closure: selected {selected_text}, best-of-K {best_text}."


def best_recovered_tta_row(tta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tta:
        return None
    rows = [row for row in tta.get("rows", []) if row.get("arm") == "recovered"]
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            metric(row, "best_of_k_exact"),
            metric(row, "selected_exact"),
            rate(row, "valid_candidate_rate"),
            metric(row, "model_exact_count"),
        ),
    )


def command_env(assignments: dict[str, str], command: str) -> str:
    prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in assignments.items())
    return f"{prefix} {command}" if prefix else command


def env_bool(value: Any, *, default: bool = False) -> str:
    if value is None:
        return "1" if default else "0"
    if isinstance(value, str):
        return "1" if value.strip().lower() in {"1", "true", "yes", "y"} else "0"
    return "1" if bool(value) else "0"


def env_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).replace("\\", "/") for item in value if str(item))
    return str(value).replace("\\", "/")


def add_common_arc_recipe_env(assignments: dict[str, str], metadata: dict[str, Any]) -> None:
    for env_key, metadata_key in {
        "MODEL_NAME": "model_name",
        "STAGE5_ARC_AGI_VERSION": "arc_version",
        "STAGE5_ARC_AGI_TRAIN_SPLIT": "train_split",
        "STAGE5_ARC_AGI_EVAL_SPLIT": "eval_split",
        "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT": "train_task_limit",
        "STAGE5_ARC_AGI_EVAL_TASK_LIMIT": "eval_task_limit",
        "STAGE5_ARC_AGI_COLOR_AUGS": "color_augmentations",
        "STAGE5_ARC_AGI_GEOMETRY_AUGS": "geometry_augmentations",
        "STAGE5_ARC_AGI_TRACE_MODE": "trace_mode",
        "STAGE5_ARC_AGI_TRACE_FILTER": "trace_filter",
        "STAGE5_ARC_AGI_GRID_FORMAT": "grid_format",
        "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE": "program_parse_mode",
        "STAGE5_ARC_AGI_SELECTION_STRATEGY": "selection_strategy",
        "STAGE5_ARC_AGI_TRAIN_STEPS": "train_steps",
        "STAGE5_ARC_AGI_LR": "learning_rate",
        "STAGE5_ARC_AGI_SYNTHETIC_TASKS": "synthetic_tasks",
    }.items():
        value = metadata.get(metadata_key)
        if value is not None:
            assignments[env_key] = str(value).replace("\\", "/")

    assignments["STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER"] = env_bool(
        metadata.get("eval_checkpoint_ladder"),
        default=False,
    )
    assignments["STAGE5_ARC_AGI_INCLUDE_SYMBOLIC"] = env_bool(
        metadata.get("include_symbolic_candidates"),
        default=False,
    )

    distillation = metadata.get("distillation")
    if isinstance(distillation, dict):
        assignments["STAGE5_ARC_AGI_DISTILL"] = env_bool(distillation.get("enabled"), default=False)
        for env_key, metadata_key in {
            "STAGE5_ARC_AGI_DISTILL_WEIGHT": "weight",
            "STAGE5_ARC_AGI_DISTILL_TEMPERATURE": "temperature",
            "STAGE5_ARC_AGI_DISTILL_ON": "on",
        }.items():
            value = distillation.get(metadata_key)
            if value is not None:
                assignments[env_key] = str(value)

    candidates = env_csv(metadata.get("candidate_distill_jsonls"))
    if candidates:
        assignments["STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS"] = candidates
        for env_key, metadata_key in {
            "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE": "candidate_distill_choice",
            "STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE": "candidate_distill_completion_source",
        }.items():
            value = metadata.get(metadata_key)
            if value is not None:
                assignments[env_key] = str(value)

    if metadata.get("include_symbolic_candidates"):
        for env_key, metadata_key in {
            "STAGE5_ARC_AGI_SYMBOLIC_POSITION": "symbolic_position",
            "STAGE5_ARC_AGI_SYMBOLIC_CANDIDATE_FORMAT": "symbolic_candidate_format",
        }.items():
            value = metadata.get(metadata_key)
            if value is not None:
                assignments[env_key] = str(value)


def summary_metadata_from_path(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    try:
        payload = read_json(resolve_path(path_value))
    except Exception:
        return {}
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def next_validation_limit(current_examples: int, *, first_limit: int = NEXT_LIMIT) -> int | None:
    """Choose the next ARC eval cap.

    The planner graduates evidence from a small smoke sample to a larger
    confirmation sample, then to the full split once the current run is already
    at or beyond ``FULL_SPLIT_AFTER_LIMIT``.
    """

    if current_examples >= FULL_SPLIT_AFTER_LIMIT:
        return None
    if current_examples >= first_limit:
        return CONFIRM_LIMIT
    return first_limit


def make_action(name: str, reason: str, command: str, priority: int) -> dict[str, Any]:
    return {"name": name, "reason": reason, "command": command, "priority": priority}


def recommendation_areas(analysis: dict[str, Any] | None) -> set[str]:
    return {
        str(item.get("area"))
        for item in (analysis or {}).get("recommendations", [])
        if item.get("area")
    }


def recommendation_reason(analysis: dict[str, Any] | None, area: str) -> str:
    for item in (analysis or {}).get("recommendations", []):
        if item.get("area") == area:
            return str(item.get("reason", ""))
    return ""


def worst_recovery_families(analysis: dict[str, Any] | None, *, limit: int = 3) -> list[str]:
    rows = ((analysis or {}).get("family_gaps") or {}).get("recovered_vs_base") or []
    families: list[str] = []
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("selected_delta", 0)),
            int(item.get("best_of_k_delta", 0)),
            -int(item.get("paired_examples", 0)),
            str(item.get("family", "")),
        ),
    ):
        family = str(row.get("family", ""))
        if not family or family == "arc":
            continue
        if int(row.get("selected_delta", 0)) >= 0 and int(row.get("best_of_k_delta", 0)) >= 0:
            continue
        families.append(family)
        if len(families) >= limit:
            break
    return families


def recovery_stage_spec(analysis: dict[str, Any] | None) -> str:
    families = worst_recovery_families(analysis)
    if families:
        focus = ",".join(families)
        return f"focus:{focus}:260:320;mixed:all:340:420"
    return (
        "warmup:constant_output,geometry_color:180:200;"
        "crop:crop_non_background,crop_recolor,crop_transform_recolor:240:300;"
        "object:move_recolor,frame_object:240:300;"
        "mixed:all:320:400"
    )


def benchmark_run_dir(benchmark: dict[str, Any] | None, source_summary: Path) -> Path:
    run_id = (benchmark or {}).get("run_id")
    if run_id:
        return ROOT / "outputs" / "stage5" / str(run_id)
    return source_summary.parent


def command_path(path: Path) -> str:
    return path_for_cli(path).replace("\\", "/")


def selector_rescore_command(benchmark: dict[str, Any] | None, source_summary: Path) -> str:
    run_dir = benchmark_run_dir(benchmark, source_summary)
    return command_env(
        {
            "STAGE5_ARC_AGI_RESCORE_SOURCE_RUN_DIR": command_path(run_dir),
            "STAGE5_ARC_AGI_RESCORE_SOURCE_GLOB": "recovered_candidates.jsonl",
            "STAGE5_ARC_AGI_RESCORE_STRATEGIES": "self_consistency,reliability_vote,symbolic_priority,cell_vote",
            "STAGE5_ARC_AGI_RESCORE_WRITE_JSONL": "1",
            "STAGE5_ARC_AGI_RESCORE_RUN_ID": f"{RUN_ID}_selector_rescore",
        },
        "python colab/run_stage5_arc_agi_rescore_selectors.py",
    )


def selector_exact_candidate_distill_gate_action(compact: dict[str, Any]) -> dict[str, Any] | None:
    evidence = compact.get("candidate_distillation_evidence") or {}
    rows = int(evidence.get("candidate_distill_rows", 0) or 0)
    selector_rows = int(evidence.get("candidate_distill_selector_generated_rows", 0) or 0)
    if not compact.get("candidate_distillation_passed") or rows <= 0 or selector_rows > 0:
        return None
    return make_action(
        "Run selector-exact candidate-distillation gate",
        "Candidate distillation passed on generic exact rows, but the gate did not train on selector-generated rows. "
        "Run the selector-exact variant so candidate distillation can be interpreted as claim-level selector evidence rather than ordinary exact-candidate SFT.",
        command_env(
            {
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_GATE_RUN_ID": f"{RUN_ID}_selector_exact_candidate_distill_gate",
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_SELECTION_STRATEGY": "cell_vote",
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE": "selector_exact",
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE": "canonical_grid",
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_INCLUDE_SYMBOLIC": "1",
                "STAGE5_ARC_AGI_CANDIDATE_DISTILL_GEOMETRY_TTA": "all",
            },
            "python colab/run_stage5_arc_agi_candidate_distill_gate.py",
        ),
        8,
    )


def selector_source_summary_path(payload: dict[str, Any]) -> Path | None:
    source_run_dir = payload.get("source_run_dir")
    if not source_run_dir:
        return None
    summary_path = resolve_path(str(source_run_dir)) / "summary.json"
    return summary_path if summary_path.exists() else None


def source_summary_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    summary_path = selector_source_summary_path(payload)
    if summary_path is None:
        return {}
    try:
        source = read_json(summary_path)
    except Exception:
        return {}
    metadata = source.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def selector_rescore_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in payload.get("rows", []) if isinstance(row, dict)]


def selector_rescore_candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in selector_rescore_rows(payload)
        if not str(row.get("selection_strategy", "")).startswith("original:")
    ]
    recovered_rows = [row for row in rows if str(row.get("label", "")) == "recovered"]
    if not recovered_rows:
        recovered_rows = [row for row in rows if str(row.get("label", "")).startswith("recovered")]
    if not recovered_rows:
        recovered_rows = rows
    return recovered_rows


def best_selector_rescore_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    recovered_rows = selector_rescore_candidate_rows(payload)
    if not recovered_rows:
        return None
    return max(
        recovered_rows,
        key=lambda row: (
            metric(row, "selected_exact"),
            metric(row, "best_of_k_exact"),
            rate(row, "valid_candidate_rate"),
            int(row.get("selected_delta_vs_source") or 0),
        ),
    )


def selector_rescore_comparison(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    label = str(row.get("label", ""))
    strategy = str(row.get("selection_strategy", ""))
    comparisons = payload.get("paired_comparisons") or {}
    comparison = comparisons.get(f"{label}__selector_{strategy}_vs_source")
    return comparison if isinstance(comparison, dict) else None


def selector_rescore_paired_stats(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    comparison = selector_rescore_comparison(payload, row)
    metrics = (comparison or {}).get("metrics") or {}
    metric_payload = metrics.get("selected_exact")
    return metric_payload if isinstance(metric_payload, dict) else None


def selector_rescore_difficulty_stats(
    payload: dict[str, Any],
    row: dict[str, Any],
    *,
    bucket: str = "hard",
) -> dict[str, Any] | None:
    comparison = selector_rescore_comparison(payload, row)
    difficulty_metrics = (comparison or {}).get("difficulty_metrics") or {}
    selected = difficulty_metrics.get("selected_exact") or {}
    metric_payload = selected.get(bucket)
    return metric_payload if isinstance(metric_payload, dict) else None


def best_hard_tail_selector_rescore_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        row
        for row in selector_rescore_candidate_rows(payload)
        if selector_rescore_difficulty_stats(payload, row, bucket="hard") is not None
    ]
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int((selector_rescore_difficulty_stats(payload, row, bucket="hard") or {}).get("delta_exact", 0)),
            int((selector_rescore_difficulty_stats(payload, row, bucket="hard") or {}).get("wins", 0))
            - int((selector_rescore_difficulty_stats(payload, row, bucket="hard") or {}).get("losses", 0)),
            int((selector_rescore_difficulty_stats(payload, row, bucket="hard") or {}).get("candidate_exact", 0)),
            int((selector_rescore_difficulty_stats(payload, row, bucket="hard") or {}).get("paired_examples", 0)),
            int(row.get("selected_delta_vs_source") or 0),
        ),
    )


def selector_rerun_command(payload: dict[str, Any], row: dict[str, Any]) -> str | None:
    metadata = source_summary_metadata(payload)
    if not metadata:
        return None
    examples = metric(row, "examples")
    limit = limit_label(next_validation_limit(examples))
    strategy = str(row.get("selection_strategy", ""))
    assignments = {
        "STAGE5_ARC_AGI_SELECTION_STRATEGY": strategy,
        "STAGE5_ARC_AGI_LIMIT": limit,
        "STAGE5_ARC_AGI_RECOVERED_BENCHMARK_RUN_ID": f"{RUN_ID}_selector_{strategy}_limit{limit}",
    }
    optional_metadata_env = {
        "STAGE5_ARC_AGI_CURRICULUM_SUMMARY": "curriculum_summary",
        "STAGE5_PHASE1_CKPT": "phase1_start_checkpoint",
        "STAGE5_ARC_AGI_RECOVERED_CKPT": "recovered_checkpoint",
        "STAGE5_ARC_AGI_VERSION": "arc_version",
        "STAGE5_ARC_AGI_SPLIT": "arc_split",
        "STAGE5_ARC_AGI_GRID_FORMAT": "grid_format",
        "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE": "program_parse_mode",
        "STAGE5_ARC_AGI_DIFFICULTY_BUCKETS": "difficulty_buckets",
        "STAGE5_ARC_AGI_EXAMPLES_PER_DIFFICULTY": "examples_per_difficulty",
    }
    for env_key, metadata_key in optional_metadata_env.items():
        value = metadata.get(metadata_key)
        if value:
            assignments[env_key] = str(value).replace("\\", "/")
    return command_env(assignments, "python colab/run_stage5_arc_agi_recovered_benchmark.py")


def source_replan_command(payload: dict[str, Any], source_summary: Path) -> str:
    summary_path = selector_source_summary_path(payload) or source_summary
    return command_env(
        {"STAGE5_ARC_AGI_NEXT_PLAN_SOURCE_SUMMARY": command_path(summary_path)},
        "python colab/plan_stage5_next_run.py",
    )


def recipe_control_assessment_action(source_summary: Path) -> dict[str, Any]:
    return make_action(
        "Assess same-recipe recurrent-vs-dense control",
        "A recurrent ARC-AGI SFT summary exists; compare it against the latest dense SFT control with paired hard-tail evidence before claiming architecture lift.",
        command_env(
            {
                "STAGE5_RECIPE_CONTROL_ASSESSMENT_RUN_ID": f"{RUN_ID}_recipe_control",
            },
            f"python colab/assess_stage5_recipe_control.py --recurrent_summary_json {shlex.quote(path_for_cli(source_summary))}",
        ),
        10,
    )


def recipe_control_assessment_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    metadata = ((payload.get("evidence") or {}).get("recurrent_vs_dense") or {}).get("candidate_summary", {})
    examples = int(metadata.get("examples_with_targets", 0) or 0)
    next_limit = next_validation_limit(examples)
    next_label = limit_label(next_limit)
    dense_metadata = summary_metadata_from_path(payload.get("dense_summary"))
    dense_assignments = {
        "STAGE5_ARC_AGI_DENSE_SFT_RUN_ID": f"{RUN_ID}_dense_limit{next_label}",
        "STAGE5_ARC_AGI_EVAL_TASK_LIMIT": next_label,
    }
    add_common_arc_recipe_env(dense_assignments, dense_metadata)
    dense_assignments["STAGE5_ARC_AGI_EVAL_TASK_LIMIT"] = next_label
    recurrent_summary = payload.get("recurrent_summary")
    recurrent_summary_path = resolve_path(str(recurrent_summary)) if recurrent_summary else None
    recurrent_run_dir = recurrent_summary_path.parent if recurrent_summary_path is not None else source_summary.parent
    if status in {"passed", "needs_more_evidence"}:
        return [
            make_action(
                f"Replicate dense control at ARC limit {next_label}",
                "Same-recipe architecture evidence is promising or underpowered; rerun the dense control at a larger matched slice, then the planner will route to the recurrent arm.",
                command_env(dense_assignments, "python colab/run_stage5_arc_agi_dense_sft.py"),
                10,
            )
        ]
    if status == "needs_selector_conversion":
        return [
            make_action(
                "Rescore recurrent candidates with selectors",
                "The recurrent same-recipe arm improved candidate coverage but not selected accuracy; run the no-GPU selector rescoring pass on the recurrent candidate files.",
                command_env(
                    {
                        "STAGE5_ARC_AGI_RESCORE_RUN_ID": f"{RUN_ID}_recipe_selector",
                        "STAGE5_ARC_AGI_RESCORE_SOURCE_RUN_DIR": path_for_cli(recurrent_run_dir),
                        "STAGE5_ARC_AGI_RESCORE_RECIPE_CONTROL_SUMMARY": path_for_cli(source_summary),
                        "STAGE5_ARC_AGI_RESCORE_WRITE_JSONL": "1",
                    },
                    "python colab/run_stage5_arc_agi_rescore_selectors.py",
                ),
                10,
            )
        ]
    return [
        make_action(
            f"Inspect same-recipe assessment `{status}`",
            "The recurrent-vs-dense same-recipe gate did not clear cleanly; inspect the assessment before scaling recurrent-specific training.",
            f"cat {shlex.quote(path_for_cli(source_summary.with_suffix('.md')))}",
            10,
        )
    ]


def release_gate_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    if status == "ready_for_broader_benchmarks":
        return [
            make_action(
                "Run broader Stage 5 benchmark suite",
                "The release gate cleared; compare the recurrent artifact against base Qwen on ARC-Challenge and GPQA-lite before any broader claims.",
                command_env(
                    {
                        "STAGE5_BENCHMARK_SUITE_RUN_ID": f"{RUN_ID}_benchmark_suite",
                    },
                    "python colab/run_stage5_benchmark_suite.py",
                ),
                10,
            )
        ]
    if status == "needs_hf_export":
        return [
            make_action(
                "Export recurrent adapter with release-gate evidence",
                "The release gate found enough benchmark and architecture evidence, but no HF export artifact with checkpoint metadata.",
                command_env(
                    {
                        "STAGE5_HF_EXPORT_RUN_ID": f"{RUN_ID}_release_hf_export",
                    },
                    "python colab/run_stage5_publish_hf_adapter.py",
                ),
                10,
            )
        ]
    return [
        make_action(
            f"Inspect release gate `{status}`",
            "The release/benchmark readiness gate is the current summary; inspect its failed criteria before spending more GPU.",
            f"cat {shlex.quote(path_for_cli(source_summary.with_suffix('.md')))}",
            10,
        )
    ]


def benchmark_suite_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    return [
        make_action(
            f"Assess broader benchmark suite `{status}`",
            "The broader base-vs-recurrent benchmark suite finished; run the no-GPU paired-evidence gate before deciding whether to recover recurrent, expand benchmarks, or write up the result.",
            command_env(
                {
                    "STAGE5_BENCHMARK_ASSESS_RUN_ID": f"{RUN_ID}_benchmark_assessment",
                },
                f"python colab/assess_stage5_benchmark_suite.py --summary_json {shlex.quote(path_for_cli(source_summary))}",
            ),
            10,
        )
    ]


def benchmark_suite_assessment_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    if status == "passed":
        return [
            make_action(
                "Build Stage 5 claim readiness packet",
                "The broader paired benchmark gate passed; synthesize release, architecture, benchmark, HF export, and ARC-AGI claim-readiness evidence before writing up results.",
                command_env(
                    {
                        "STAGE5_CLAIM_PACKET_RUN_ID": f"{RUN_ID}_claim_packet",
                    },
                    "python colab/build_stage5_claim_packet.py",
                ),
                10,
            )
        ]
    if status == "needs_benchmark_confirmation":
        return [
            make_action(
                "Expand broader benchmark suite confirmation",
                "The benchmark-suite gate found too few paired examples; rerun ARC-Challenge/GPQA-lite with larger limits before interpreting deltas.",
                command_env(
                    {
                        "STAGE5_BENCHMARK_SUITE_RUN_ID": f"{RUN_ID}_expanded_benchmark_suite",
                        "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": "256",
                        "STAGE5_BENCHMARK_GPQA_LIMIT": "32",
                    },
                    "python colab/run_stage5_benchmark_suite.py",
                ),
                10,
            )
        ]
    if status == "needs_recurrent_recovery":
        return [
            make_action(
                "Run deterministic recurrent recovery ladder",
                "The broader benchmark gate says recurrent still trails base; improve deterministic recurrent competence before GPQA Diamond or release claims.",
                command_env(
                    {
                        "STAGE5_RUN_ID": f"{RUN_ID}_phase1_recovery",
                        "STAGE5_PHASE1_EXTRA_STEPS": "500",
                        "STAGE5_ARC_LIMIT": "256",
                    },
                    "python colab/run_stage5_phase1_recovery_ladder.py",
                ),
                10,
            )
        ]
    return [
        make_action(
            f"Inspect broader benchmark assessment `{status}`",
            "The broader benchmark gate is the current summary; inspect it before spending more GPU or making benchmark claims.",
            f"cat {shlex.quote(path_for_cli(source_summary.with_suffix('.md')))}",
            10,
        )
    ]


def claim_readiness_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    if status == "needs_selector_replication":
        return [
            make_action(
                "Assess selector replication for claim packet",
                "The claim packet is missing replicated Gate 1 selector/TTA evidence; run the no-GPU replication assessor before release or SOTA-facing claims.",
                command_env(
                    {
                        "STAGE5_SELECTOR_REPLICATION_RUN_ID": f"{RUN_ID}_claim_selector_replication",
                    },
                    "python colab/assess_stage5_selector_replication.py",
                ),
                10,
            )
        ]
    if status == "needs_particle_mechanism_gate":
        return [
            make_action(
                "Assess Gate 2 particle mechanism for claim packet",
                "The claim packet is missing passed Gate 2 recurrent-particle mechanism evidence; run the no-GPU Gate 2 assessor before architecture-facing claims.",
                command_env(
                    {
                        "STAGE5_GATE2_ASSESSMENT_RUN_ID": f"{RUN_ID}_claim_gate2_assessment",
                    },
                    "python colab/assess_stage5_gate2.py",
                ),
                10,
            )
        ]
    if status == "needs_hf_export":
        return [
            make_action(
                "Export recurrent adapter for claim packet",
                "The claim packet is missing HF export metadata; package the recurrent adapter before release-candidate writeup.",
                command_env(
                    {
                        "STAGE5_HF_EXPORT_RUN_ID": f"{RUN_ID}_claim_hf_export",
                    },
                    "python colab/run_stage5_publish_hf_adapter.py",
                ),
                10,
            )
        ]
    if status == "ready_for_release_candidate_not_sota":
        return [
            make_action(
                "Build ARC-AGI same-size comparison artifact",
                "The claim packet is release-candidate ready but not SOTA-ready; compare the recurrent ARC-AGI result against a sourced same-size baseline registry.",
                command_env(
                    {
                        "STAGE5_ARC_AGI_SOTA_COMPARISON_RUN_ID": f"{RUN_ID}_arc_agi_sota_comparison",
                    },
                    "python colab/build_stage5_arc_agi_sota_comparison.py",
                ),
                10,
            )
        ]
    return [
        make_action(
            f"Inspect claim readiness `{status}`",
            "The claim-readiness packet is the current summary; inspect it before writing release notes or any SOTA-facing claim.",
            f"cat {shlex.quote(path_for_cli(source_summary.with_suffix('.md')))}",
            10,
        )
    ]


def arc_agi_sota_comparison_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    if status in {"passed", "failed"}:
        return [
            make_action(
                "Rebuild claim packet with ARC-AGI comparison",
                "An ARC-AGI same-size comparison artifact exists; rebuild the claim packet so release/SOTA readiness reflects it.",
                command_env(
                    {
                        "STAGE5_CLAIM_PACKET_RUN_ID": f"{RUN_ID}_claim_packet_with_arc_agi",
                    },
                    "python colab/build_stage5_claim_packet.py",
                ),
                10,
            )
        ]
    return [
        make_action(
            f"Inspect ARC-AGI SOTA comparison `{status}`",
            "The ARC-AGI same-size comparison artifact needs human input, usually a sourced baseline registry or a larger candidate ARC-AGI eval.",
            f"cat {shlex.quote(path_for_cli(source_summary.with_suffix('.md')))}",
            10,
        )
    ]


def arc_agi_baseline_registry_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    if status == "passed":
        return [
            make_action(
                "Build ARC-AGI same-size comparison artifact",
                "The same-size baseline registry passed validation; compare the recurrent ARC-AGI candidate against the sourced baseline set.",
                command_env(
                    {
                        "STAGE5_ARC_AGI_SOTA_COMPARISON_RUN_ID": f"{RUN_ID}_arc_agi_sota_comparison",
                    },
                    "python colab/build_stage5_arc_agi_sota_comparison.py",
                ),
                10,
            )
        ]
    return [
        make_action(
            f"Inspect ARC-AGI baseline registry `{status}`",
            "The ARC-AGI same-size registry is missing sourced values or contains placeholders; fix it before any SOTA comparison.",
            f"cat {shlex.quote(path_for_cli(source_summary.with_suffix('.md')))}",
            10,
        )
    ]


def dense_sft_matched_recurrent_command(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    assignments = {
        "STAGE5_ARC_AGI_SFT_RUN_ID": f"{RUN_ID}_matched_recurrent_sft",
    }
    add_common_arc_recipe_env(assignments, metadata)
    assignments.setdefault("STAGE5_ARC_AGI_TRAIN_TASK_LIMIT", "100")
    assignments.setdefault("STAGE5_ARC_AGI_EVAL_TASK_LIMIT", "10")
    assignments.setdefault("STAGE5_ARC_AGI_TRACE_MODE", "symbolic_program")
    assignments.setdefault("STAGE5_ARC_AGI_TRACE_FILTER", "covered")
    assignments.setdefault("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
    return command_env(assignments, "python colab/run_stage5_arc_agi_sft.py")


def dense_sft_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    deltas = payload.get("deltas") or {}
    dense_vs_base = deltas.get("dense_tuned_vs_base") or {}
    phase1_vs_base = deltas.get("phase1_start_vs_base") or {}
    dense_selected_delta = int(dense_vs_base.get("selected_exact_delta", 0) or 0)
    phase1_selected_delta = int(phase1_vs_base.get("selected_exact_delta", 0) or 0)
    dense_vs_base_stats = paired_metric(payload, "dense_tuned_vs_base", "selected_exact")
    phase1_vs_base_stats = paired_metric(payload, "phase1_start_vs_base", "selected_exact")
    reason = (
        "A dense standard-control SFT run exists. Run the matched recurrent SFT arm under the same ARC row recipe so the project can compare recipe lift against architecture lift. "
        f"Dense-vs-base evidence: {evidence_fragment(dense_vs_base_stats, dense_selected_delta)}. "
        f"Phase1-start-vs-base evidence: {evidence_fragment(phase1_vs_base_stats, phase1_selected_delta)}."
    )
    if dense_selected_delta > phase1_selected_delta:
        reason += (
            f" Dense selected delta vs base `{dense_selected_delta}` exceeds Phase1-start delta `{phase1_selected_delta}`, so this is now a necessary control before attributing gains to recurrence."
        )
    return [
        make_action(
            "Run matched recurrent ARC-AGI SFT control",
            reason,
            dense_sft_matched_recurrent_command(payload),
            10,
        )
    ]


def selector_rescore_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    best_row = best_selector_rescore_row(payload)
    if best_row is None:
        return [
            make_action(
                "Inspect selector-rescore summary",
                "Selector-rescore output had no rescored rows; inspect the source run before spending GPU.",
                source_replan_command(payload, source_summary),
                10,
            )
        ]
    fallback_delta = int(best_row.get("selected_delta_vs_source") or 0)
    stats = selector_rescore_paired_stats(payload, best_row)
    strategy = str(best_row.get("selection_strategy", ""))
    label = str(best_row.get("label", ""))
    if paired_supports_positive(stats, fallback_delta):
        rerun = selector_rerun_command(payload, best_row)
        if rerun:
            return [
                make_action(
                    f"Promote selector `{strategy}` for `{label}` benchmark",
                    "Selector rescoring improved selected exact accuracy; rerun the recovered-vs-base benchmark with that selector before training further. "
                    f"Evidence: {evidence_fragment(stats, fallback_delta)}.",
                    rerun,
                    10,
                )
            ]
        return [
            make_action(
                f"Promote selector `{strategy}` after restoring source benchmark metadata",
                "Selector rescoring improved selected exact accuracy, but the source benchmark metadata was not available in this checkout. "
                f"Evidence: {evidence_fragment(stats, fallback_delta)}.",
                source_replan_command(payload, source_summary),
                10,
            )
        ]
    hard_row = best_hard_tail_selector_rescore_row(payload)
    hard_stats = selector_rescore_difficulty_stats(payload, hard_row, bucket="hard") if hard_row else None
    if hard_row is not None and paired_supports_positive(hard_stats, 0):
        hard_strategy = str(hard_row.get("selection_strategy", ""))
        hard_label = str(hard_row.get("label", ""))
        hard_fallback_delta = int(hard_row.get("selected_delta_vs_source") or 0)
        hard_aggregate_stats = selector_rescore_paired_stats(payload, hard_row)
        rerun = selector_rerun_command(payload, hard_row)
        if paired_supports_nonnegative(hard_aggregate_stats, hard_fallback_delta):
            if rerun:
                return [
                    make_action(
                        f"Validate hard-tail selector `{hard_strategy}` for `{hard_label}` benchmark",
                        "Selector rescoring improved selected exact accuracy on the hard difficulty bucket while aggregate selected accuracy was non-negative; rerun before treating this as a Gate 1 signal. "
                        f"Hard-bucket evidence: {evidence_fragment(hard_stats, 0)}. "
                        f"Aggregate evidence: {evidence_fragment(hard_aggregate_stats, hard_fallback_delta)}.",
                        rerun,
                        10,
                    )
                ]
            return [
                make_action(
                    f"Validate hard-tail selector `{hard_strategy}` after restoring source metadata",
                    "Selector rescoring improved selected exact accuracy on the hard difficulty bucket, but source benchmark metadata was not available in this checkout. "
                    f"Hard-bucket evidence: {evidence_fragment(hard_stats, 0)}. "
                    f"Aggregate evidence: {evidence_fragment(hard_aggregate_stats, hard_fallback_delta)}.",
                    source_replan_command(payload, source_summary),
                    10,
                )
            ]
        return [
            make_action(
                f"Inspect hard-tail selector tradeoff `{hard_strategy}`",
                "Selector rescoring improved the hard difficulty bucket but harmed aggregate selected accuracy; inspect the tradeoff before adopting it as default. "
                f"Hard-bucket evidence: {evidence_fragment(hard_stats, 0)}. "
                f"Aggregate evidence: {evidence_fragment(hard_aggregate_stats, hard_fallback_delta)}.",
                source_replan_command(payload, source_summary),
                9,
            )
        ]
    return [
        make_action(
            "Defer selector changes and continue recovery plan",
            "Selector rescoring did not show paired selected-accuracy lift; return to the source benchmark plan rather than adding selector complexity. "
            f"Best selector `{strategy}` evidence: {evidence_fragment(stats, fallback_delta)}.",
            source_replan_command(payload, source_summary),
            10,
        )
    ]


def recipe_selector_conversion_action(payload: dict[str, Any], *, source_summary: Path) -> dict[str, Any] | None:
    recipe_control = payload.get("recipe_control_summary")
    if not recipe_control:
        return None
    return make_action(
        "Assess same-recipe selector conversion",
        "Selector rescoring came from a same-recipe architecture gate that needed selector conversion; compare recurrent selector-selected answers directly against the dense control.",
        command_env(
            {
                "STAGE5_RECIPE_SELECTOR_CONVERSION_RUN_ID": f"{RUN_ID}_recipe_selector_conversion",
            },
            (
                "python colab/assess_stage5_recipe_selector_conversion.py "
                f"--recipe_control_summary {shlex.quote(str(recipe_control))} "
                f"--selector_rescore_summary {command_path(source_summary)}"
            ),
        ),
        10,
    )


def gate1_source_summary_path(payload: dict[str, Any], fallback: Path) -> Path:
    source = payload.get("source_summary")
    return resolve_path(str(source)) if source else fallback


def gate1_inspect_action(payload: dict[str, Any], source_summary: Path, *, priority: int = 10) -> dict[str, Any]:
    summary_md = source_summary.with_name("summary.md")
    inspect_path = summary_md if summary_md.exists() else source_summary
    status = str(payload.get("status", "unknown"))
    return make_action(
        f"Inspect Gate 1 assessment `{status}`",
        f"Gate 1 status `{status}` needs human review before spending more GPU. Reason: {payload.get('reason')}",
        f"cat {command_path(inspect_path)}",
        priority,
    )


def gate1_assessment_action(source_summary: Path) -> dict[str, Any]:
    return make_action(
        "Assess Gate 1 selector/TTA evidence",
        "Selector or TTA paired evidence exists; write the explicit Gate 1 assessment before promoting selector, TTA, or particle settings.",
        command_env(
            {"STAGE5_GATE1_ASSESSMENT_RUN_ID": f"{RUN_ID}_gate1_assessment"},
            f"python colab/assess_stage5_gate1.py --summary_json {command_path(source_summary)}",
        ),
        10,
    )


def safe_gate1_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if gate1_assessment_payload(payload) is not None else None


def previous_gate1_summary(current_summary: Path) -> Path | None:
    scan_root = current_summary.parent.parent if current_summary.name == "summary.json" else ROOT / "outputs" / "stage5"
    if not scan_root.exists():
        return None
    candidates: list[Path] = []
    for path in scan_root.glob("*/summary.json"):
        if path == current_summary:
            continue
        if safe_gate1_summary(path):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def selector_replication_assessment_action(*, discovery_summary: Path, confirmation_summary: Path) -> dict[str, Any]:
    return make_action(
        "Assess selector replication across Gate 1 slices",
        "A Gate 1 selector/TTA setting passed and an earlier Gate 1 assessment exists; verify that the same comparison passed both slices before promoting the selector.",
        command_env(
            {"STAGE5_SELECTOR_REPLICATION_RUN_ID": f"{RUN_ID}_selector_replication"},
            (
                "python colab/assess_stage5_selector_replication.py "
                f"--discovery_gate1_json {command_path(discovery_summary)} "
                f"--confirmation_gate1_json {command_path(confirmation_summary)}"
            ),
        ),
        10,
    )


def gate1_assessment_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    source_path = gate1_source_summary_path(payload, source_summary)
    source_actions: list[dict[str, Any]] = []
    if source_path != source_summary and source_path.exists():
        source_payload = read_json(source_path)
        if gate1_assessment_payload(source_payload) is None:
            source_actions = plan_next_actions(
                source_payload,
                source_summary=source_path,
                require_gate1_assessment=False,
            )

    if status != "passed":
        return [gate1_inspect_action(payload, source_summary, priority=10)]
    previous = previous_gate1_summary(source_summary)
    if previous is not None:
        return [
            selector_replication_assessment_action(
                discovery_summary=previous,
                confirmation_summary=source_summary,
            )
        ]
    if not source_actions:
        return [gate1_inspect_action(payload, source_summary, priority=10)]

    top = dict(source_actions[0])
    name = str(top.get("name", ""))
    if name.startswith("Promote selector"):
        name = name.replace("Promote selector", "Confirm selector", 1)
    if name.startswith("Validate hard-tail selector"):
        name = name.replace("Validate hard-tail selector", "Confirm hard-tail selector", 1)
    prefix = "Gate 1 discovery passed"
    priority = 10
    top["name"] = f"{prefix}: {name}"
    top["reason"] = (
        f"{prefix}; this is a confirmation run, not a final selector promotion. "
        f"Assessment reason: {payload.get('reason')} "
        f"Assessment next step: {payload.get('next_step')} "
        f"Source action reason: {top.get('reason')}"
    )
    top["priority"] = priority
    return [top]


def gate2_source_summary_path(payload: dict[str, Any], fallback: Path) -> Path:
    source = payload.get("source_summary")
    return resolve_path(str(source)) if source else fallback


def gate2_inspect_action(payload: dict[str, Any], source_summary: Path, *, priority: int = 10) -> dict[str, Any]:
    summary_md = source_summary.with_name("summary.md")
    inspect_path = summary_md if summary_md.exists() else source_summary
    status = str(payload.get("status", "unknown"))
    return make_action(
        f"Inspect Gate 2 assessment `{status}`",
        f"Gate 2 status `{status}` needs review before scaling particles. Reason: {payload.get('reason')}",
        f"cat {command_path(inspect_path)}",
        priority,
    )


def gate2_assessment_action(source_summary: Path) -> dict[str, Any]:
    return make_action(
        "Assess Gate 2 particle mechanism evidence",
        "Recovery-particle evidence passed its embedded particle decision; write the explicit Gate 2 assessment before treating particle behavior as a scaling signal.",
        command_env(
            {"STAGE5_GATE2_ASSESSMENT_RUN_ID": f"{RUN_ID}_gate2_assessment"},
            f"python colab/assess_stage5_gate2.py --summary_json {command_path(source_summary)}",
        ),
        10,
    )


def particle_replicate_action(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for action in actions:
        if str(action.get("name", "")).startswith("Replicate particle value"):
            return action
    return None


def gate2_assessment_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    source_path = gate2_source_summary_path(payload, source_summary)
    source_actions: list[dict[str, Any]] = []
    if source_path != source_summary and source_path.exists():
        source_payload = read_json(source_path)
        if gate2_assessment_payload(source_payload) is None:
            source_actions = plan_next_actions(
                source_payload,
                source_summary=source_path,
                require_gate1_assessment=False,
                require_gate2_assessment=False,
            )

    if status not in {"passed", "needs_more_evidence"}:
        return [gate2_inspect_action(payload, source_summary, priority=10)]
    if not source_actions:
        return [gate2_inspect_action(payload, source_summary, priority=10)]

    top = dict(particle_replicate_action(source_actions) or source_actions[0])
    prefix = "Gate 2 passed" if status == "passed" else "Gate 2 needs more evidence"
    priority = 10 if status == "passed" else max(9, int(top.get("priority", 0)))
    top["name"] = f"{prefix}: {top.get('name')}"
    top["reason"] = (
        f"{prefix}. Assessment reason: {payload.get('reason')} "
        f"Assessment next step: {payload.get('next_step')} "
        f"Source action reason: {top.get('reason')}"
    )
    top["priority"] = priority
    return [top]


def selector_replication_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    summary_md = source_summary.with_name("summary.md")
    inspect_path = summary_md if summary_md.exists() else source_summary
    if status == "passed":
        return [
            make_action(
                "Inspect replicated selector evidence",
                "A selector/TTA comparison passed Gate 1 on two saved slices. Inspect the replicated comparison before using it in release or broader benchmark evidence.",
                f"cat {command_path(inspect_path)}",
                10,
            )
        ]
    return [
        make_action(
            f"Inspect selector replication `{status}`",
            "Selector/TTA evidence has not replicated yet; either run a confirmation Gate 1 slice or return to selector design.",
            f"cat {command_path(inspect_path)}",
            10,
        )
    ]


def recipe_selector_conversion_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    status = str(payload.get("status", "unknown"))
    if status == "passed":
        actions = [
            make_action(
                "Run release gate with selector-conversion evidence",
                "A recurrent selector converted same-recipe candidate coverage into selected-answer lift versus the dense control; run the release readiness audit so this evidence can be combined with ARC confirmation and HF export metadata.",
                command_env(
                    {
                        "STAGE5_RELEASE_GATE_RUN_ID": f"{RUN_ID}_release_gate_from_selector_conversion",
                    },
                    "python colab/assess_stage5_release_gate.py",
                ),
                10,
            )
        ]
        best = payload.get("best_selector") or {}
        best_row = None
        for row in payload.get("selector_evidence") or []:
            if row.get("label") == best.get("label") and row.get("selection_strategy") == best.get("selection_strategy"):
                best_row = row
                break
        candidates_jsonl = str((best_row or {}).get("selector_candidates_jsonl") or "")
        if candidates_jsonl:
            actions.append(
                make_action(
                    "Run selector-exact candidate-distillation SFT",
                    "The passing selector-conversion gate includes a rescored candidate JSONL. Distill selector-generated exact ARC grids back into the recurrent model to test whether the model can internalize claim-level particle/selector wins.",
                    command_env(
                        {
                            "STAGE5_ARC_AGI_SFT_RUN_ID": f"{RUN_ID}_selector_exact_distill_sft",
                            "STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS": candidates_jsonl,
                            "STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE": "selector_exact",
                            "STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE": "canonical_grid",
                            "STAGE5_ARC_AGI_SELECTION_STRATEGY": "cell_vote",
                            "STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER": "1",
                        },
                        "python colab/run_stage5_arc_agi_sft.py",
                    ),
                    9,
                )
            )
        return actions
    return [
        make_action(
            f"Inspect same-recipe selector conversion `{status}`",
            "Selector conversion did not clear cleanly against the dense control; inspect this before further recurrent-specific scaling.",
            f"cat {command_path(source_summary.with_suffix('.md'))}",
            10,
        )
    ]


def recovery_particle_examples(payload: dict[str, Any]) -> int:
    settings = payload.get("settings") or {}
    if settings.get("eval_task_limit") is not None:
        return int(settings.get("eval_task_limit") or 0)
    evidence = ((payload.get("recovery_decision") or {}).get("evidence") or {})
    tuned = evidence.get("phase1_tuned") or {}
    return int(tuned.get("examples_with_targets") or tuned.get("examples") or 0)


def recovery_particle_paths(payload: dict[str, Any]) -> dict[str, str]:
    evidence = ((payload.get("recovery_decision") or {}).get("evidence") or {})
    recovered = payload.get("recovered_checkpoint") or evidence.get("phase1_recovered") or {}
    sft_metadata = ((payload.get("sft_summary") or {}).get("metadata") or {})
    paths: dict[str, str] = {}
    recovered_checkpoint = recovered.get("checkpoint")
    phase1_checkpoint = sft_metadata.get("phase1_checkpoint")
    if recovered_checkpoint:
        paths["STAGE5_ARC_AGI_RECOVERED_CKPT"] = str(recovered_checkpoint).replace("\\", "/")
    if phase1_checkpoint:
        paths["STAGE5_PHASE1_CKPT"] = str(phase1_checkpoint).replace("\\", "/")
    return paths


def recovery_particle_recovered_benchmark_command(payload: dict[str, Any], limit: int | None) -> str | None:
    paths = recovery_particle_paths(payload)
    if "STAGE5_ARC_AGI_RECOVERED_CKPT" not in paths:
        return None
    settings = payload.get("settings") or {}
    label = limit_label(limit)
    assignments = {
        **paths,
        "STAGE5_ARC_AGI_LIMIT": label,
        "STAGE5_ARC_AGI_RECOVERED_BENCHMARK_RUN_ID": f"{RUN_ID}_recovery_particle_benchmark_limit{label}",
    }
    optional_settings = {
        "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE": "program_parse_mode",
        "STAGE5_ARC_AGI_SELECTION_STRATEGY": "selection_strategy",
    }
    for env_key, settings_key in optional_settings.items():
        value = settings.get(settings_key)
        if value:
            assignments[env_key] = str(value)
    return command_env(assignments, "python colab/run_stage5_arc_agi_recovered_benchmark.py")


def recovery_particle_replicate_command(payload: dict[str, Any], limit: int | None) -> str:
    settings = payload.get("settings") or {}
    label = limit_label(limit)
    assignments = {
        "STAGE5_ARC_AGI_RECOVERY_PARTICLE_RUN_ID": f"{RUN_ID}_recovery_particle_replicate_limit{label}",
        "STAGE5_ARC_AGI_EVAL_TASK_LIMIT": str(limit if limit is not None else FULL_SPLIT_AFTER_LIMIT),
        "STAGE5_ARC_AGI_PARTICLE_SEEDS": "0,1,2,3,4",
    }
    for env_key, settings_key in {
        "STAGE5_ARC_AGI_SYNTHETIC_TASKS": "synthetic_tasks",
        "STAGE5_ARC_AGI_SYNTHETIC_MODES": "synthetic_modes",
        "STAGE5_ARC_AGI_TRAIN_STEPS": "train_steps",
        "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT": "train_task_limit",
        "STAGE5_ARC_AGI_TRACE_MODE": "trace_mode",
        "STAGE5_ARC_AGI_TRACE_FILTER": "trace_filter",
        "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE": "program_parse_mode",
        "STAGE5_ARC_AGI_SELECTION_STRATEGY": "selection_strategy",
        "STAGE5_ARC_AGI_PARTICLE_VARIANTS": "particle_variants",
    }.items():
        value = settings.get(settings_key)
        if not value:
            continue
        if settings_key == "particle_variants" and isinstance(value, list):
            value = ",".join(
                f"{item['name']}:{item['noise']}:{item['repulsion']}"
                for item in value
                if isinstance(item, dict) and {"name", "noise", "repulsion"} <= set(item)
            )
        assignments[env_key] = str(value)
    return command_env(assignments, "python colab/run_stage5_arc_agi_recovery_particle_gate.py")


def recovery_particle_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    recovery = payload.get("recovery_decision") or {}
    particle = payload.get("particle_decision") or {}
    recovery_evidence = recovery.get("evidence") or {}
    particle_evidence = particle.get("evidence") or {}
    examples = recovery_particle_examples(payload)
    next_limit = next_validation_limit(examples)
    next_label = limit_label(next_limit)

    if not bool(recovery.get("passed")):
        start_delta = recovery_evidence.get("phase1_tuned_vs_start") or {}
        return [
            make_action(
                "Compare ARC trace-training targets",
                "Deterministic recurrent recovery did not clear the non-negative gate; compare grid-only, symbolic-program trace, and symbolic-state trace SFT before scaling particles. "
                f"Recovered-vs-start evidence: `{start_delta}`.",
                command_env(
                    {
                        "STAGE5_ARC_AGI_TRACE_SFT_GATE_RUN_ID": f"{RUN_ID}_trace_sft_gate",
                        "STAGE5_ARC_AGI_TRACE_SFT_GATE_ARMS": DEFAULT_TRACE_SFT_GATE_ARMS,
                    },
                    "python colab/run_stage5_arc_agi_trace_sft_gate.py",
                ),
                10,
            )
        ]

    actions: list[dict[str, Any]] = []
    benchmark_command = recovery_particle_recovered_benchmark_command(payload, next_limit)
    if benchmark_command:
        actions.append(
            make_action(
                f"Benchmark recovered recurrent against base at ARC limit {next_label}",
                "Deterministic recurrent recovery cleared its start-checkpoint gate; now measure how much of the original Qwen gap remains before attributing lift to particles.",
                benchmark_command,
                10,
            )
        )
    else:
        actions.append(
            make_action(
                "Inspect recovery-particle checkpoint metadata",
                "Deterministic recovery passed, but the summary did not include a recovered checkpoint path for benchmark rerun.",
                source_replan_command(payload, source_summary),
                10,
            )
        )

    best_particle = particle_evidence.get("best_replicated_variant")
    if bool(particle.get("passed")):
        actions.append(
            make_action(
                f"Replicate particle value `{best_particle}` at ARC limit {next_label}",
                "Particle/SVGD cleared the replicated post-recovery gate; rerun with more tasks and five seeds before using particle behavior as a training signal.",
                recovery_particle_replicate_command(payload, next_limit),
                9,
            )
        )
    else:
        actions.append(
            make_action(
                "Defer particle/SVGD training pressure",
                "Particle/SVGD did not clear the post-recovery gate; prioritize deterministic recurrent recovery and base-gap measurement.",
                benchmark_command or source_replan_command(payload, source_summary),
                8,
            )
        )
    return actions


def plan_next_actions(
    payload: dict[str, Any],
    *,
    source_summary: Path,
    require_gate1_assessment: bool = True,
    require_gate2_assessment: bool = True,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    gate1 = gate1_assessment_payload(payload)
    if gate1:
        return gate1_assessment_actions(gate1, source_summary=source_summary)
    gate2 = gate2_assessment_payload(payload)
    if gate2:
        return gate2_assessment_actions(gate2, source_summary=source_summary)
    selector_replication = selector_replication_payload(payload)
    if selector_replication:
        return selector_replication_actions(selector_replication, source_summary=source_summary)
    recipe_selector_conversion = recipe_selector_conversion_payload(payload)
    if recipe_selector_conversion:
        return recipe_selector_conversion_actions(recipe_selector_conversion, source_summary=source_summary)
    recipe_control = recipe_control_assessment_payload(payload)
    if recipe_control:
        return recipe_control_assessment_actions(recipe_control, source_summary=source_summary)
    release_gate = release_gate_payload(payload)
    if release_gate:
        return release_gate_actions(release_gate, source_summary=source_summary)
    benchmark_suite_assessment = benchmark_suite_assessment_payload(payload)
    if benchmark_suite_assessment:
        return benchmark_suite_assessment_actions(benchmark_suite_assessment, source_summary=source_summary)
    claim_readiness = claim_readiness_payload(payload)
    if claim_readiness:
        return claim_readiness_actions(claim_readiness, source_summary=source_summary)
    arc_agi_baseline_registry = arc_agi_baseline_registry_payload(payload)
    if arc_agi_baseline_registry:
        return arc_agi_baseline_registry_actions(arc_agi_baseline_registry, source_summary=source_summary)
    arc_agi_sota = arc_agi_sota_comparison_payload(payload)
    if arc_agi_sota:
        return arc_agi_sota_comparison_actions(arc_agi_sota, source_summary=source_summary)
    benchmark_suite = benchmark_suite_payload(payload)
    if benchmark_suite:
        return benchmark_suite_actions(benchmark_suite, source_summary=source_summary)
    recipe_selector_action = recipe_selector_conversion_action(payload, source_summary=source_summary)
    if recipe_selector_action is not None:
        return [recipe_selector_action]
    if require_gate1_assessment and needs_gate1_assessment(payload):
        return [gate1_assessment_action(source_summary)]
    if require_gate2_assessment and needs_gate2_assessment(payload):
        return [gate2_assessment_action(source_summary)]
    if needs_recipe_control_assessment(payload):
        return [recipe_control_assessment_action(source_summary)]
    selector_rescore = selector_rescore_payload(payload)
    if selector_rescore:
        return selector_rescore_actions(selector_rescore, source_summary=source_summary)
    dense_sft = dense_sft_payload(payload)
    if dense_sft:
        return dense_sft_actions(dense_sft)
    recovery_particle = recovery_particle_payload(payload)
    if recovery_particle:
        return sorted(
            recovery_particle_actions(recovery_particle, source_summary=source_summary),
            key=lambda action: (-int(action["priority"]), action["name"]),
        )

    benchmark = benchmark_payload(payload)
    tta = tta_payload(payload)
    analysis = recovery_analysis_payload(payload)
    analysis_areas = recommendation_areas(analysis)
    compact = compact_payload(payload)
    source_summary_cli = path_for_cli(source_summary)

    candidate_passed = compact.get("candidate_distillation_passed")
    final_checkpoint = compact.get("final_checkpoint")
    particle_passed = bool(compact.get("particle_passed"))

    if candidate_passed is False:
        actions.append(
            make_action(
                "Run baseline curriculum without candidate distillation",
                "Candidate distillation failed its gate; get a clean deterministic recovery baseline before using generated candidates as training data.",
                command_env(
                    {
                        "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_ID": f"{RUN_ID}_baseline_no_candidate_distill",
                        "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_CANDIDATE_DISTILL_GATE": "0",
                    },
                    "python colab/run_stage5_arc_agi_curriculum_particle_autopilot.py",
                ),
                10,
            )
        )
        return actions

    if benchmark:
        recovered_vs_base_selected = paired_delta_or_aggregate(
            benchmark,
            comparison="recovered_vs_base",
            metric_name="selected_exact",
            aggregate_group="recovered_vs_base",
            aggregate_key="selected_exact_delta",
        )
        recovered_vs_base_best = paired_delta_or_aggregate(
            benchmark,
            comparison="recovered_vs_base",
            metric_name="best_of_k_exact",
            aggregate_group="recovered_vs_base",
            aggregate_key="best_of_k_exact_delta",
        )
        recovered_vs_start_selected = paired_delta_or_aggregate(
            benchmark,
            comparison="recovered_vs_start",
            metric_name="selected_exact",
            aggregate_group="recovered_vs_start",
            aggregate_key="selected_exact_delta",
        )
        recovered_vs_start_best = paired_delta_or_aggregate(
            benchmark,
            comparison="recovered_vs_start",
            metric_name="best_of_k_exact",
            aggregate_group="recovered_vs_start",
            aggregate_key="best_of_k_exact_delta",
        )
        recovered_vs_base_selected_stats = paired_metric(benchmark, "recovered_vs_base", "selected_exact")
        recovered_vs_base_best_stats = paired_metric(benchmark, "recovered_vs_base", "best_of_k_exact")
        recovered_vs_start_selected_stats = paired_metric(benchmark, "recovered_vs_start", "selected_exact")
        recovered_vs_start_best_stats = paired_metric(benchmark, "recovered_vs_start", "best_of_k_exact")
        examples = metric((benchmark.get("base") or {}).get("summary"), "examples_with_targets")
        recovered_matches_base = paired_supports_nonnegative(
            recovered_vs_base_selected_stats,
            recovered_vs_base_selected,
        ) and paired_supports_nonnegative(recovered_vs_base_best_stats, recovered_vs_base_best)
        recovered_improved_start = recovered_vs_start_selected >= MIN_RECOVERED_VS_START_DELTA or paired_supports_positive(
            recovered_vs_start_best_stats,
            recovered_vs_start_best,
        )

        if recovered_matches_base:
            confirm_limit = next_validation_limit(examples, first_limit=CONFIRM_LIMIT or NEXT_LIMIT)
            confirm_label = limit_label(confirm_limit)
            confirm_reason = (
                "Recovered recurrent matched or beat base on a confirmed sample; run the full ARC split before claiming lift."
                if confirm_limit is None
                else "Recovered recurrent matched or beat base on the current smoke comparison; validate at a larger held-out limit before claiming lift."
            )
            actions.append(
                make_action(
                    f"Confirm recovered-vs-base at ARC limit {confirm_label}",
                    f"{confirm_reason} "
                    f"Selected evidence: {evidence_fragment(recovered_vs_base_selected_stats, recovered_vs_base_selected)}. "
                    f"Best-of-K evidence: {evidence_fragment(recovered_vs_base_best_stats, recovered_vs_base_best)}. "
                    f"{gap_closure_fragment(benchmark)}",
                    command_env(
                        {
                            "STAGE5_ARC_AGI_AUTOPILOT_SUMMARY": source_summary_cli,
                            "STAGE5_ARC_AGI_FOLLOWUP_RUN_ID": f"{RUN_ID}_confirm_limit{confirm_label}",
                            "STAGE5_ARC_AGI_FOLLOWUP_LIMIT": confirm_label,
                        },
                        "python colab/run_stage5_arc_agi_autopilot_followup.py",
                    ),
                    10,
                )
            )
            actions.append(
                make_action(
                    "Export recovered adapter to Hugging Face",
                    "A matched-or-better smoke result is enough to preserve the artifact and model card while larger confirmation runs proceed.",
                    command_env(
                        {
                            "STAGE5_HF_SOURCE_SUMMARY": source_summary_cli,
                            "STAGE5_HF_EXPORT_RUN_ID": f"{RUN_ID}_hf_export",
                        },
                        "python colab/run_stage5_publish_hf_adapter.py",
                    ),
                    8,
                )
            )
        elif recovered_improved_start:
            stage_spec = recovery_stage_spec(analysis)
            families = worst_recovery_families(analysis)
            focus_reason = (
                f" Worst recovered-vs-base families: {', '.join(families)}."
                if families
                else ""
            )
            actions.append(
                make_action(
                    "Scale deterministic curriculum",
                    "Recovered recurrent improved over its start checkpoint but still trails base; spend GPU on more deterministic recovery before particle/SVGD training. "
                    f"Selected evidence: {evidence_fragment(recovered_vs_start_selected_stats, recovered_vs_start_selected)}. "
                    f"Best-of-K evidence: {evidence_fragment(recovered_vs_start_best_stats, recovered_vs_start_best)}."
                    f" {gap_closure_fragment(benchmark)}"
                    f"{focus_reason}",
                    command_env(
                        {
                            "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_ID": f"{RUN_ID}_scaled_curriculum",
                            "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT": "160",
                            "STAGE5_ARC_AGI_EVAL_TASK_LIMIT": str(max(NEXT_LIMIT, examples or NEXT_LIMIT)),
                            "STAGE5_ARC_AGI_CURRICULUM_STAGES": stage_spec,
                        },
                        "python colab/run_stage5_arc_agi_curriculum_particle_autopilot.py",
                    ),
                    10,
                )
            )
        else:
            actions.append(
                make_action(
                    "Run trace/candidate-distillation diagnostics before more SFT",
                    "Recovered recurrent did not improve over the start checkpoint; diagnose training target quality instead of scaling the same recipe. "
                    f"Selected evidence: {evidence_fragment(recovered_vs_start_selected_stats, recovered_vs_start_selected)}. "
                    f"Best-of-K evidence: {evidence_fragment(recovered_vs_start_best_stats, recovered_vs_start_best)}. "
                    f"{gap_closure_fragment(benchmark)}",
                    command_env(
                        {
                            "STAGE5_ARC_AGI_CANDIDATE_DISTILL_GATE_RUN_ID": f"{RUN_ID}_candidate_distill_diagnostic",
                        },
                        "python colab/run_stage5_arc_agi_candidate_distill_gate.py",
                    ),
                    10,
                )
            )
        if "selector_or_tta" in analysis_areas:
            actions.append(
                make_action(
                    "Rescore recovered candidates with selector variants",
                    "Recovery analysis found selector misses; try no-GPU self-consistency and symbolic-priority rescoring before spending more training time. "
                    f"{recommendation_reason(analysis, 'selector_or_tta')}",
                    selector_rescore_command(benchmark, source_summary),
                    9,
                )
            )
        if "format_parse" in analysis_areas:
            actions.append(
                make_action(
                    "Run output-format recovery curriculum",
                    "Recovery analysis found no-valid-grid failures; tighten grid-format behavior with a short deterministic recovery branch. "
                    f"{recommendation_reason(analysis, 'format_parse')}",
                    command_env(
                        {
                            "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_ID": f"{RUN_ID}_format_recovery",
                            "STAGE5_ARC_AGI_CURRICULUM_STAGES": (
                                "format:constant_output,geometry_color,crop_non_background:240:260;"
                                "mixed:all:260:320"
                            ),
                        },
                        "python colab/run_stage5_arc_agi_curriculum_particle_autopilot.py",
                    ),
                    8,
                )
            )
    elif final_checkpoint:
        actions.append(
            make_action(
                f"Run recovered-vs-base benchmark at ARC limit {NEXT_LIMIT}",
                "Autopilot produced a checkpoint but no benchmark summary was found.",
                command_env(
                    {
                        "STAGE5_ARC_AGI_AUTOPILOT_SUMMARY": source_summary_cli,
                        "STAGE5_ARC_AGI_FOLLOWUP_RUN_ID": f"{RUN_ID}_followup_limit{NEXT_LIMIT}",
                        "STAGE5_ARC_AGI_FOLLOWUP_LIMIT": str(NEXT_LIMIT),
                    },
                    "python colab/run_stage5_arc_agi_autopilot_followup.py",
                ),
                10,
            )
        )
    else:
        actions.append(
            make_action(
                "Run candidate-distill curriculum autopilot",
                "No recovered checkpoint or benchmark was found in the source summary.",
                "python colab/run_stage5_arc_agi_curriculum_particle_autopilot.py",
                10,
            )
        )

    selector_distill_action = selector_exact_candidate_distill_gate_action(compact)
    if selector_distill_action is not None:
        actions.append(selector_distill_action)

    best_tta = best_recovered_tta_row(tta)
    if best_tta and best_tta.get("tta_variant") not in {None, "none"}:
        none_rows = [
            row
            for row in (tta or {}).get("rows", [])
            if row.get("arm") == "recovered" and row.get("tta_variant") == "none"
        ]
        none_best = metric(none_rows[0], "best_of_k_exact") if none_rows else 0
        best_tta_delta = metric(best_tta, "best_of_k_exact") - none_best
        tta_stats = paired_metric(tta, f"recovered__tta_{best_tta['tta_variant']}_vs_none", "best_of_k_exact")
        if paired_supports_positive(tta_stats, best_tta_delta):
            tta_limit = next_validation_limit(metric(best_tta, "examples_with_targets"))
            tta_limit_label = limit_label(tta_limit)
            actions.append(
                make_action(
                    f"Replicate recovered TTA variant `{best_tta['tta_variant']}`",
                    "TTA improved recovered best-of-K on the current sweep; replicate at a larger limit before baking it into default eval. "
                    f"Evidence: {evidence_fragment(tta_stats, best_tta_delta)}.",
                    command_env(
                        {
                            "STAGE5_ARC_AGI_AUTOPILOT_SUMMARY": source_summary_cli,
                            "STAGE5_ARC_AGI_FOLLOWUP_RUN_ID": f"{RUN_ID}_tta_{best_tta['tta_variant']}_limit{tta_limit_label}",
                            "STAGE5_ARC_AGI_FOLLOWUP_LIMIT": tta_limit_label,
                            "STAGE5_ARC_AGI_TTA_VARIANTS": f"none,{best_tta['tta_variant']}",
                        },
                        "python colab/run_stage5_arc_agi_autopilot_followup.py",
                    ),
                    7,
                )
            )

    if particle_passed:
        particle_limit = next_validation_limit(
            metric((benchmark or {}).get("base", {}).get("summary") if benchmark else None, "examples_with_targets")
        )
        particle_limit_label = limit_label(particle_limit)
        actions.append(
            make_action(
                "Replicate particle gate at larger limit",
                "Particle/SVGD passed the replicated gate; verify the lift survives a larger ARC sample.",
                command_env(
                    {
                        "STAGE5_ARC_AGI_AUTOPILOT_SUMMARY": source_summary_cli,
                        "STAGE5_ARC_AGI_FOLLOWUP_RUN_ID": f"{RUN_ID}_particle_replicate",
                        "STAGE5_ARC_AGI_FOLLOWUP_LIMIT": particle_limit_label,
                    },
                    "python colab/run_stage5_arc_agi_autopilot_followup.py",
                ),
                6,
            )
        )

    return sorted(actions, key=lambda action: (-int(action["priority"]), action["name"]))


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Next-Run Plan - {RUN_ID}",
        "",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Source kind: `{payload['source_kind']}`",
        "",
        "## Recommended Actions",
        "",
    ]
    for index, action in enumerate(payload["actions"], start=1):
        lines.extend(
            [
                f"{index}. **{action['name']}**",
                f"   - Priority: `{action['priority']}`",
                f"   - Reason: {action['reason']}",
                f"   - Command: `{action['command']}`",
            ]
        )
    if not payload["actions"]:
        lines.append("No next action could be inferred from this summary.")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def source_kind(payload: dict[str, Any]) -> str:
    if gate1_assessment_payload(payload):
        return "gate1_assessment"
    if gate2_assessment_payload(payload):
        return "gate2_assessment"
    if selector_replication_payload(payload):
        return "selector_replication"
    if recipe_selector_conversion_payload(payload):
        return "recipe_selector_conversion"
    if recipe_control_assessment_payload(payload):
        return "recipe_control_assessment"
    if release_gate_payload(payload):
        return "release_gate"
    if benchmark_suite_assessment_payload(payload):
        return "benchmark_suite_assessment"
    if claim_readiness_payload(payload):
        return "claim_readiness"
    if arc_agi_baseline_registry_payload(payload):
        return "arc_agi_baseline_registry"
    if arc_agi_sota_comparison_payload(payload):
        return "arc_agi_sota_comparison"
    if benchmark_suite_payload(payload):
        return "benchmark_suite"
    if selector_rescore_payload(payload):
        return "selector_rescore"
    if dense_sft_payload(payload):
        return "dense_sft_control"
    if recurrent_sft_payload(payload):
        return "recurrent_sft"
    if recovery_particle_payload(payload):
        return "recovery_particle_gate"
    if direct_tta_sweep_payload(payload):
        return "tta_sweep"
    if "recovered_benchmark" in payload or "tta_sweep" in payload:
        return "followup"
    if "compact" in payload:
        return "autopilot"
    if benchmark_payload(payload):
        return "benchmark"
    return "unknown"


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in os.sys.argv[1:]):
        print("Read latest Stage 5 run summary and write a ranked next-run plan.")
        return 0
    summary_path = resolve_source_summary()
    payload = read_json(summary_path)
    plan = {
        "run_id": RUN_ID,
        "source_summary": path_for_cli(summary_path),
        "source_kind": source_kind(payload),
        "actions": plan_next_actions(payload, source_summary=summary_path),
    }
    write_report(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
