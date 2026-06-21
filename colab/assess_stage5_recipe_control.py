"""Assess same-recipe dense-vs-recurrent ARC-AGI evidence.

This is the standard-control gate for the program track. It compares a dense
Qwen LoRA SFT control against a matched recurrent SFT run using paired ARC-AGI
examples. A pass means the recurrent recipe arm beats the dense recipe arm in a
selector-relevant way, preferably on the hard bucket, without aggregate harm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from eval.compare_arc_agi_runs import compare_payloads


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_RECIPE_CONTROL_ASSESSMENT_RUN_ID") or time.strftime(
    "stage5_recipe_control_assessment_%Y%m%d_%H%M%S"
)

METADATA_KEYS = (
    "model_name",
    "params_b",
    "arc_version",
    "train_split",
    "eval_split",
    "train_task_limit",
    "eval_task_limit",
    "color_augmentations",
    "geometry_augmentations",
    "trace_mode",
    "trace_filter",
    "synthetic_tasks",
    "candidate_distill_jsonls",
    "grid_format",
    "program_parse_mode",
    "selection_strategy",
    "train_steps",
    "learning_rate",
    "distillation",
    "include_symbolic_candidates",
    "eval_checkpoint_ladder",
)


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_dense_summary(payload: dict[str, Any]) -> bool:
    return payload.get("kind") == "dense_sft_control"


def is_recurrent_sft_summary(payload: dict[str, Any]) -> bool:
    return "phase1_arc_agi_tuned" in payload and bool(payload.get("tuned_checkpoint"))


def latest_summary(scan_root: Path, predicate) -> Path:
    candidates: list[Path] = []
    for path in scan_root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and predicate(payload):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No matching Stage 5 summary found under {scan_root}")
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def eval_payload_from_summary(summary_path: Path, summary: dict[str, Any], label: str) -> dict[str, Any]:
    inline = summary.get(label)
    if isinstance(inline, dict) and isinstance(inline.get("examples"), list):
        return inline
    candidate = summary_path.parent / f"{label}_summary.json"
    if candidate.exists():
        return read_json(candidate)
    raise FileNotFoundError(f"Could not find full eval payload for {label!r} near {summary_path}")


def recurrent_eval_payload(summary_path: Path, summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    best = summary.get("best_checkpoint")
    if isinstance(best, dict) and best.get("step") is not None:
        step_label = f"phase1_arc_agi_step_{int(best['step'])}"
        candidate = summary_path.parent / f"{step_label}_summary.json"
        if candidate.exists():
            return step_label, read_json(candidate)
    return "phase1_arc_agi_tuned", eval_payload_from_summary(summary_path, summary, "phase1_arc_agi_tuned")


def metadata_value_for_compare(value: Any) -> str:
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def normalized_metadata(summary: dict[str, Any]) -> dict[str, str]:
    metadata = summary.get("metadata") or {}
    return {key: metadata_value_for_compare(metadata.get(key)) for key in METADATA_KEYS if metadata.get(key) is not None}


def metadata_differences(dense: dict[str, Any], recurrent: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    dense_meta = normalized_metadata(dense)
    recurrent_meta = normalized_metadata(recurrent)
    differences: dict[str, dict[str, str | None]] = {}
    for key in METADATA_KEYS:
        dense_value = dense_meta.get(key)
        recurrent_value = recurrent_meta.get(key)
        if dense_value != recurrent_value:
            differences[key] = {"dense": dense_value, "recurrent": recurrent_value}
    return differences


def metric_stats(comparison: dict[str, Any], metric_name: str = "selected_exact") -> dict[str, Any]:
    return (comparison.get("metrics") or {}).get(metric_name) or {}


def difficulty_stats(
    comparison: dict[str, Any],
    *,
    metric_name: str = "selected_exact",
    bucket: str = "hard",
) -> dict[str, Any]:
    difficulty_metrics = comparison.get("difficulty_metrics") or {}
    return ((difficulty_metrics.get(metric_name) or {}).get(bucket)) or {}


def delta(stats: dict[str, Any]) -> int:
    return int(stats.get("delta_exact", 0) or 0)


def wins(stats: dict[str, Any]) -> int:
    return int(stats.get("wins", 0) or 0)


def losses(stats: dict[str, Any]) -> int:
    return int(stats.get("losses", 0) or 0)


def paired_examples(stats: dict[str, Any]) -> int:
    return int(stats.get("paired_examples", 0) or 0)


def supports_positive(stats: dict[str, Any]) -> bool:
    return delta(stats) > 0 and wins(stats) > losses(stats)


def supports_nonnegative(stats: dict[str, Any]) -> bool:
    return delta(stats) >= 0 and wins(stats) >= losses(stats)


def evidence_fragment(stats: dict[str, Any]) -> str:
    if not stats:
        return "missing"
    ci = stats.get("bootstrap_delta_accuracy_ci95") or {}
    return (
        f"delta {stats.get('delta_exact', 0)} "
        f"({stats.get('wins', 0)}/{stats.get('losses', 0)}/{stats.get('ties', 0)} W/L/T, "
        f"n={stats.get('paired_examples', 0)}, p={stats.get('sign_test_p_value')}, "
        f"CI95 [{ci.get('low')}, {ci.get('high')}])"
    )


def assess_recipe_control(
    *,
    dense_summary_path: Path,
    recurrent_summary_path: Path,
    hard_bucket: str = "hard",
    min_total_examples: int = 20,
    min_hard_examples: int = 5,
) -> dict[str, Any]:
    dense_summary = read_json(dense_summary_path)
    recurrent_summary = read_json(recurrent_summary_path)
    dense_payload = eval_payload_from_summary(dense_summary_path, dense_summary, "dense_tuned")
    recurrent_label, recurrent_payload = recurrent_eval_payload(recurrent_summary_path, recurrent_summary)
    dense_base = eval_payload_from_summary(dense_summary_path, dense_summary, "base")
    phase1_start = eval_payload_from_summary(recurrent_summary_path, recurrent_summary, "phase1_start")

    recurrent_vs_dense = compare_payloads(
        dense_payload,
        recurrent_payload,
        reference_label="dense_tuned",
        candidate_label=recurrent_label,
        bootstrap_samples=1000,
        seed=0,
    )
    recurrent_vs_base = compare_payloads(
        dense_base,
        recurrent_payload,
        reference_label="base",
        candidate_label=recurrent_label,
        bootstrap_samples=1000,
        seed=0,
    )
    recurrent_vs_start = compare_payloads(
        phase1_start,
        recurrent_payload,
        reference_label="phase1_start",
        candidate_label=recurrent_label,
        bootstrap_samples=1000,
        seed=0,
    )

    aggregate = metric_stats(recurrent_vs_dense, "selected_exact")
    hard = difficulty_stats(recurrent_vs_dense, metric_name="selected_exact", bucket=hard_bucket)
    aggregate_best = metric_stats(recurrent_vs_dense, "best_of_k_exact")
    hard_best = difficulty_stats(recurrent_vs_dense, metric_name="best_of_k_exact", bucket=hard_bucket)
    metadata_diff = metadata_differences(dense_summary, recurrent_summary)
    total_sufficient = paired_examples(aggregate) >= min_total_examples
    hard_sufficient = paired_examples(hard) >= min_hard_examples
    aggregate_positive = supports_positive(aggregate)
    aggregate_nonnegative = supports_nonnegative(aggregate)
    hard_positive = hard_sufficient and supports_positive(hard)
    hard_nonnegative = bool(hard) and hard_sufficient and supports_nonnegative(hard)
    aggregate_best_positive = supports_positive(aggregate_best)
    aggregate_best_nonnegative = supports_nonnegative(aggregate_best)
    hard_best_positive = hard_sufficient and supports_positive(hard_best)
    hard_best_nonnegative = bool(hard_best) and hard_sufficient and supports_nonnegative(hard_best)

    if metadata_diff:
        status = "needs_review"
        reason = "Dense and recurrent summaries are not matched on recipe metadata."
        next_step = "Rerun the missing or mismatched arm with the same ARC recipe settings before judging architecture lift."
    elif not total_sufficient or not hard_sufficient:
        status = "needs_more_evidence"
        reason = "Same-recipe comparison has too few paired examples or too few hard-bucket examples."
        next_step = "Replicate the dense and recurrent controls on a larger stratified ARC slice."
    elif hard_positive and aggregate_nonnegative:
        status = "passed"
        reason = "Recurrent same-recipe arm improves hard-bucket selected-answer accuracy versus dense control with no aggregate selected-answer harm."
        next_step = "Replicate at a larger ARC slice, then consider recurrent-specific training or particle mechanisms only if the lift survives."
    elif hard_best_positive and aggregate_best_nonnegative:
        status = "needs_selector_conversion"
        reason = "Recurrent same-recipe arm improves hard-bucket exact candidate coverage versus dense control, but selected-answer accuracy does not yet convert it."
        next_step = "Run selector or verifier work on the recurrent candidate set before judging this architecture signal as failed."
    elif hard_best_positive and not aggregate_best_nonnegative:
        status = "needs_review"
        reason = "Recurrent arm improves hard-bucket candidate coverage but harms aggregate best-of-K coverage versus dense control."
        next_step = "Inspect task-family tradeoffs and selector failures before scaling this recipe."
    elif hard_positive and not aggregate_nonnegative:
        status = "needs_review"
        reason = "Recurrent arm improves the hard bucket but harms aggregate selected-answer accuracy versus dense control."
        next_step = "Inspect task-family tradeoffs before scaling this recipe."
    elif aggregate_positive:
        status = "needs_more_evidence"
        reason = "Recurrent arm improves aggregate selected accuracy, but hard-bucket support is missing."
        next_step = "Run a difficulty-stratified comparison before treating this as architecture evidence."
    elif aggregate_best_positive:
        status = "needs_more_evidence"
        reason = "Recurrent arm improves aggregate best-of-K coverage, but hard-bucket candidate-coverage support is missing."
        next_step = "Run a larger difficulty-stratified comparison before spending selector work on this candidate set."
    else:
        status = "failed"
        reason = "Recurrent same-recipe arm does not improve selected-answer accuracy versus dense control."
        next_step = "Improve deterministic recurrent recovery or training targets before scaling architecture-specific work."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_same_recipe_architecture",
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "next_step": next_step,
        "dense_summary": path_for_cli(dense_summary_path),
        "recurrent_summary": path_for_cli(recurrent_summary_path),
        "recurrent_label": recurrent_label,
        "hard_bucket": hard_bucket,
        "min_total_examples": min_total_examples,
        "min_hard_examples": min_hard_examples,
        "metadata_differences": metadata_diff,
        "evidence": {
            "recurrent_vs_dense": recurrent_vs_dense,
            "recurrent_vs_base": recurrent_vs_base,
            "recurrent_vs_phase1_start": recurrent_vs_start,
        },
        "decision_evidence": {
            "aggregate": aggregate,
            "hard": hard,
            "aggregate_best_of_k": aggregate_best,
            "hard_best_of_k": hard_best,
            "aggregate_evidence": evidence_fragment(aggregate),
            "hard_evidence": evidence_fragment(hard),
            "aggregate_best_of_k_evidence": evidence_fragment(aggregate_best),
            "hard_best_of_k_evidence": evidence_fragment(hard_best),
            "total_sufficient": total_sufficient,
            "hard_sufficient": hard_sufficient,
        },
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dense_stats = payload["decision_evidence"]["aggregate_evidence"]
    hard_stats = payload["decision_evidence"]["hard_evidence"]
    dense_best_stats = payload["decision_evidence"]["aggregate_best_of_k_evidence"]
    hard_best_stats = payload["decision_evidence"]["hard_best_of_k_evidence"]
    lines = [
        f"# Stage 5 Same-Recipe Architecture Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Dense summary: `{payload['dense_summary']}`",
        f"- Recurrent summary: `{payload['recurrent_summary']}`",
        f"- Recurrent label: `{payload['recurrent_label']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        "",
        "## Decision Evidence",
        "",
        f"- Aggregate recurrent-vs-dense selected: {dense_stats}",
        f"- `{payload['hard_bucket']}` recurrent-vs-dense selected: {hard_stats}",
        f"- Aggregate recurrent-vs-dense best-of-K: {dense_best_stats}",
        f"- `{payload['hard_bucket']}` recurrent-vs-dense best-of-K: {hard_best_stats}",
        f"- Metadata differences: `{payload['metadata_differences']}`",
        "",
        "## Other Comparisons",
        "",
    ]
    for name, comparison in payload["evidence"].items():
        stats = metric_stats(comparison, "selected_exact")
        lines.append(f"- `{name}` selected: {evidence_fragment(stats)}")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense_summary_json")
    parser.add_argument("--recurrent_summary_json")
    parser.add_argument("--scan_root", default="outputs/stage5")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--hard_bucket", default="hard")
    parser.add_argument("--min_total_examples", type=int, default=20)
    parser.add_argument("--min_hard_examples", type=int, default=5)
    args = parser.parse_args()

    scan_root = resolve_path(args.scan_root)
    dense_summary = (
        resolve_path(args.dense_summary_json)
        if args.dense_summary_json
        else latest_summary(scan_root, is_dense_summary)
    )
    recurrent_summary = (
        resolve_path(args.recurrent_summary_json)
        if args.recurrent_summary_json
        else latest_summary(scan_root, is_recurrent_sft_summary)
    )
    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_dir / "summary.md"
    payload = assess_recipe_control(
        dense_summary_path=dense_summary,
        recurrent_summary_path=recurrent_summary,
        hard_bucket=args.hard_bucket,
        min_total_examples=args.min_total_examples,
        min_hard_examples=args.min_hard_examples,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
