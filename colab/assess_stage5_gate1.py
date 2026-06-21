"""Assess Stage 5 Gate 1 selector/TTA evidence.

Gate 1 is the measurement gate before more architecture or kernel work. This
script reads a saved Stage 5 summary with paired comparisons and decides whether
the current selector/TTA mechanism has useful hard-tail evidence:

* selected-answer accuracy must improve in paired evidence;
* the hard difficulty bucket must be present;
* hard-tail lift cannot come with aggregate selected-answer harm.

The output is a no-GPU JSON/Markdown artifact suitable for committing from
Colab alongside run outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_GATE1_ASSESSMENT_RUN_ID") or time.strftime(
    "stage5_gate1_assessment_%Y%m%d_%H%M%S"
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


def source_kind(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("rows"), list) and ("best_by_label" in payload or "strategies" in payload):
        return "selector_rescore"
    if isinstance(payload.get("rows"), list) and payload.get("paired_comparisons"):
        return "tta_sweep"
    if payload.get("recovered_benchmark"):
        return "followup"
    if {"base", "phase1_start", "recovered", "deltas"} <= set(payload):
        return "recovered_benchmark"
    return "unknown"


def paired_comparisons(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("paired_comparisons"), dict):
        return payload["paired_comparisons"]
    nested = payload.get("tta_sweep") or payload.get("recovered_benchmark")
    if isinstance(nested, dict) and isinstance(nested.get("paired_comparisons"), dict):
        return nested["paired_comparisons"]
    return {}


def metric_stats(comparison: dict[str, Any], metric_name: str = "selected_exact") -> dict[str, Any] | None:
    metrics = comparison.get("metrics") or {}
    stats = metrics.get(metric_name)
    return stats if isinstance(stats, dict) else None


def difficulty_stats(
    comparison: dict[str, Any],
    *,
    metric_name: str = "selected_exact",
    bucket: str = "hard",
) -> dict[str, Any] | None:
    difficulty_metrics = comparison.get("difficulty_metrics") or {}
    metric_rows = difficulty_metrics.get(metric_name) or {}
    stats = metric_rows.get(bucket)
    return stats if isinstance(stats, dict) else None


def delta(stats: dict[str, Any] | None) -> int:
    return int((stats or {}).get("delta_exact", 0) or 0)


def wins(stats: dict[str, Any] | None) -> int:
    return int((stats or {}).get("wins", 0) or 0)


def losses(stats: dict[str, Any] | None) -> int:
    return int((stats or {}).get("losses", 0) or 0)


def paired_examples(stats: dict[str, Any] | None) -> int:
    return int((stats or {}).get("paired_examples", 0) or 0)


def supports_positive(stats: dict[str, Any] | None) -> bool:
    return delta(stats) > 0 and wins(stats) > losses(stats)


def supports_nonnegative(stats: dict[str, Any] | None) -> bool:
    return delta(stats) >= 0 and wins(stats) >= losses(stats)


def evidence_fragment(stats: dict[str, Any] | None) -> str:
    if stats is None:
        return "missing"
    ci = stats.get("bootstrap_delta_accuracy_ci95") or {}
    return (
        f"delta {stats.get('delta_exact', 0)} "
        f"({stats.get('wins', 0)}/{stats.get('losses', 0)}/{stats.get('ties', 0)} W/L/T, "
        f"n={stats.get('paired_examples', 0)}, "
        f"CI95 [{ci.get('low')}, {ci.get('high')}])"
    )


def comparison_evidence(
    name: str,
    comparison: dict[str, Any],
    *,
    hard_bucket: str,
    min_total_examples: int,
    min_hard_examples: int,
) -> dict[str, Any]:
    aggregate = metric_stats(comparison, "selected_exact")
    hard = difficulty_stats(comparison, metric_name="selected_exact", bucket=hard_bucket)
    total_examples = paired_examples(aggregate)
    hard_examples = paired_examples(hard)
    total_sufficient = total_examples >= min_total_examples
    hard_sufficient = hard_examples >= min_hard_examples
    aggregate_positive = supports_positive(aggregate)
    aggregate_nonnegative = supports_nonnegative(aggregate)
    hard_positive = hard_sufficient and supports_positive(hard)
    hard_nonnegative = hard is not None and hard_sufficient and supports_nonnegative(hard)

    passed = total_sufficient and (
        (aggregate_positive and hard_nonnegative)
        or (hard_positive and aggregate_nonnegative)
    )
    tradeoff = hard_positive and not aggregate_nonnegative
    aggregate_only = aggregate_positive and (hard is None or not hard_sufficient)

    return {
        "comparison": name,
        "total_sufficient": total_sufficient,
        "hard_sufficient": hard_sufficient,
        "aggregate_positive": aggregate_positive,
        "aggregate_nonnegative": aggregate_nonnegative,
        "hard_positive": hard_positive,
        "hard_nonnegative": hard_nonnegative,
        "passed": passed,
        "tradeoff": tradeoff,
        "aggregate_only": aggregate_only,
        "aggregate": aggregate,
        "hard": hard,
        "aggregate_evidence": evidence_fragment(aggregate),
        "hard_evidence": evidence_fragment(hard),
    }


def assess_gate1(
    payload: dict[str, Any],
    *,
    source_summary: str,
    hard_bucket: str = "hard",
    min_total_examples: int = 20,
    min_hard_examples: int = 5,
) -> dict[str, Any]:
    comparisons = paired_comparisons(payload)
    evidence = [
        comparison_evidence(
            name,
            comparison,
            hard_bucket=hard_bucket,
            min_total_examples=min_total_examples,
            min_hard_examples=min_hard_examples,
        )
        for name, comparison in sorted(comparisons.items())
        if isinstance(comparison, dict)
    ]
    passed = [row for row in evidence if row["passed"]]
    tradeoffs = [row for row in evidence if row["tradeoff"]]
    aggregate_only = [row for row in evidence if row["aggregate_only"]]
    insufficient = [
        row
        for row in evidence
        if not row["total_sufficient"] or not row["hard_sufficient"]
    ]

    if passed:
        status = "passed"
        reason = "At least one paired comparison shows selected-answer lift with hard-bucket support and no aggregate harm."
        next_step = "Replicate this setting on a larger stratified ARC slice before treating it as Gate 2 evidence."
    elif tradeoffs:
        status = "needs_review"
        reason = "Hard-bucket selected-answer lift exists, but aggregate selected accuracy is harmed."
        next_step = "Inspect tradeoff rows; do not promote the selector/TTA setting as default yet."
    elif aggregate_only or insufficient:
        status = "needs_more_evidence"
        reason = "Some evidence is promising or incomplete, but hard-bucket paired evidence is missing or too small."
        next_step = "Run a stratified Gate 1 slice with STAGE5_ARC_AGI_EXAMPLES_PER_DIFFICULTY set."
    else:
        status = "failed"
        reason = "No paired selected-answer lift was found on aggregate or hard-bucket evidence."
        next_step = "Return to deterministic recurrent recovery or selector design before adding mechanism complexity."

    return {
        "run_id": RUN_ID,
        "gate": "stage5_gate1_selector_tta",
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "next_step": next_step,
        "source_summary": source_summary,
        "source_kind": source_kind(payload),
        "hard_bucket": hard_bucket,
        "min_total_examples": min_total_examples,
        "min_hard_examples": min_hard_examples,
        "num_comparisons": len(evidence),
        "passing_comparisons": [row["comparison"] for row in passed],
        "tradeoff_comparisons": [row["comparison"] for row in tradeoffs],
        "aggregate_only_comparisons": [row["comparison"] for row in aggregate_only],
        "insufficient_comparisons": [row["comparison"] for row in insufficient],
        "evidence": evidence,
    }


def latest_summary(scan_root: Path) -> Path:
    candidates: list[Path] = []
    for path in scan_root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and paired_comparisons(payload):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No Stage 5 summary with paired comparisons found.")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Gate 1 Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source: `{payload['source_summary']}`",
        f"- Source kind: `{payload['source_kind']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        "",
        "## Evidence",
        "",
        "| Comparison | Aggregate | Hard Bucket | Decision |",
        "|---|---|---|---|",
    ]
    for row in payload["evidence"]:
        if row["passed"]:
            decision = "pass"
        elif row["tradeoff"]:
            decision = "tradeoff"
        elif row["aggregate_only"]:
            decision = "aggregate-only"
        elif not row["total_sufficient"] or not row["hard_sufficient"]:
            decision = "insufficient"
        else:
            decision = "fail"
        lines.append(
            f"| `{row['comparison']}` | {row['aggregate_evidence']} | "
            f"{row['hard_evidence']} | `{decision}` |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", help="Stage 5 summary to assess. Defaults to latest paired-comparison summary.")
    parser.add_argument("--scan_root", default="outputs/stage5")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--hard_bucket", default="hard")
    parser.add_argument("--min_total_examples", type=int, default=20)
    parser.add_argument("--min_hard_examples", type=int, default=5)
    args = parser.parse_args()

    source = resolve_path(args.summary_json) if args.summary_json else latest_summary(resolve_path(args.scan_root))
    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_dir / "summary.md"
    payload = assess_gate1(
        read_json(source),
        source_summary=path_for_cli(source),
        hard_bucket=args.hard_bucket,
        min_total_examples=args.min_total_examples,
        min_hard_examples=args.min_hard_examples,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
