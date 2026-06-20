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
    return any(key in payload for key in ("recovered_benchmark", "tta_sweep", "compact", "autopilot_compact"))


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
    prefix = " ".join(f"{key}={value}" for key, value in assignments.items())
    return f"{prefix} {command}" if prefix else command


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
            "STAGE5_ARC_AGI_RESCORE_STRATEGIES": "self_consistency,symbolic_priority",
            "STAGE5_ARC_AGI_RESCORE_RUN_ID": f"{RUN_ID}_selector_rescore",
        },
        "python colab/run_stage5_arc_agi_rescore_selectors.py",
    )


def plan_next_actions(payload: dict[str, Any], *, source_summary: Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
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
                    f"Best-of-K evidence: {evidence_fragment(recovered_vs_base_best_stats, recovered_vs_base_best)}.",
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
                    f"Best-of-K evidence: {evidence_fragment(recovered_vs_start_best_stats, recovered_vs_start_best)}.",
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
