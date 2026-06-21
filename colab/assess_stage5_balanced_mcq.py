"""Select Stage 5 deterministic checkpoints across paired MCQ benchmarks.

This is a no-GPU assessment over existing Stage 5 artifacts. It combines
full ARC-Easy and full ARC-Challenge evidence so checkpoint selection does
not overfit to a single benchmark slice.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_BALANCED_MCQ_RUN_ID") or time.strftime(
    "stage5_balanced_mcq_%Y%m%d_%H%M%S"
)
REQUIRED_BENCHMARKS = tuple(
    item.strip()
    for item in os.environ.get("STAGE5_BALANCED_MCQ_REQUIRED", "arc_easy,arc_challenge").split(",")
    if item.strip()
)
SCORE_TARGET = os.environ.get("STAGE5_BALANCED_MCQ_SCORE_TARGET", "label")
AGGREGATE = os.environ.get("STAGE5_BALANCED_MCQ_AGGREGATE", "mean")


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_summary(predicate) -> Path | None:
    root = ROOT / "outputs" / "stage5"
    if not root.exists():
        return None
    candidates: list[Path] = []
    for path in root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and predicate(payload):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0] if candidates else None


def all_summaries(predicate) -> list[Path]:
    root = ROOT / "outputs" / "stage5"
    if not root.exists():
        return []
    candidates: list[Path] = []
    for path in root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and predicate(payload):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.stat().st_mtime)


def is_arceasy_sweep(payload: dict[str, Any]) -> bool:
    return payload.get("kind") == "stage5_arceasy_checkpoint_sweep"


def has_full_arc_challenge(payload: dict[str, Any]) -> bool:
    if payload.get("kind") == "stage5_benchmark_suite":
        return "arc_challenge" in {str(item) for item in payload.get("benchmarks", [])}
    return isinstance(payload.get("full_arc_final"), dict)


def checkpoint_label(checkpoint: str | None, *, fallback: str = "unknown") -> str:
    if not checkpoint:
        return fallback
    match = re.search(r"phase1_step_(\d+)\.pt$", checkpoint)
    if match:
        return f"step_{int(match.group(1)):03d}"
    return fallback


def two_sided_sign_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(observed + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def metric_from_summary(
    *,
    benchmark: str,
    base: dict[str, Any],
    recurrent: dict[str, Any],
    paired: dict[str, Any] | None = None,
    source_summary: Path,
) -> dict[str, Any]:
    total = int(recurrent.get("total", base.get("total", 0)) or 0)
    base_correct = int(base.get("correct", 0) or 0)
    recurrent_correct = int(recurrent.get("correct", 0) or 0)
    correct_delta = recurrent_correct - base_correct
    base_accuracy = float(base.get("accuracy", base_correct / max(total, 1)) or 0.0)
    recurrent_accuracy = float(recurrent.get("accuracy", recurrent_correct / max(total, 1)) or 0.0)
    row = {
        "benchmark": benchmark,
        "source_summary": path_for_cli(source_summary),
        "base_correct": base_correct,
        "recurrent_correct": recurrent_correct,
        "total": total,
        "correct_delta_recurrent_vs_base": correct_delta,
        "base_accuracy": base_accuracy,
        "recurrent_accuracy": recurrent_accuracy,
        "accuracy_delta_recurrent_vs_base": recurrent_accuracy - base_accuracy,
        "wins": None,
        "losses": None,
        "ties": None,
        "sign_test_p_value": None,
    }
    if paired:
        wins = paired.get("wins", paired.get("helped"))
        losses = paired.get("losses", paired.get("hurt"))
        ties = paired.get("ties", paired.get("tied"))
        row.update(
            {
                "wins": int(wins or 0),
                "losses": int(losses or 0),
                "ties": int(ties or 0),
                "sign_test_p_value": paired.get("sign_test_p_value")
                or two_sided_sign_p_value(int(wins or 0), int(losses or 0)),
            }
        )
    return row


def add_metric(
    records: dict[str, dict[str, Any]],
    *,
    label: str,
    checkpoint: str | None,
    metric: dict[str, Any],
) -> None:
    row = records.setdefault(label, {"label": label, "checkpoint": checkpoint, "benchmarks": {}})
    if checkpoint and not row.get("checkpoint"):
        row["checkpoint"] = checkpoint
    row["benchmarks"][metric["benchmark"]] = metric


def ingest_arceasy_sweep(records: dict[str, dict[str, Any]], path: Path, payload: dict[str, Any]) -> None:
    base = payload["base"]["summary"]
    for arm in payload.get("arms", []):
        paired = arm.get("paired_vs_base") or {}
        label = str(arm.get("label") or checkpoint_label(arm.get("checkpoint")))
        add_metric(
            records,
            label=label,
            checkpoint=arm.get("checkpoint"),
            metric=metric_from_summary(
                benchmark="arc_easy",
                base=base,
                recurrent=arm["summary"],
                paired={
                    "wins": paired.get("wins"),
                    "losses": paired.get("losses"),
                    "ties": paired.get("ties"),
                    "sign_test_p_value": paired.get("sign_test_p_value"),
                },
                source_summary=path,
            ),
        )


def benchmark_suite_metric(path: Path, payload: dict[str, Any], benchmark: str) -> dict[str, Any] | None:
    result = (((payload.get("results") or {}).get(benchmark) or {}).get(SCORE_TARGET) or {})
    comparison = (
        (((payload.get("paired_comparisons") or {}).get(benchmark) or {}).get(SCORE_TARGET) or {}).get(AGGREGATE)
    )
    base = (result.get("base") or {}).get(AGGREGATE)
    recurrent = (result.get("recurrent") or {}).get(AGGREGATE)
    if not isinstance(base, dict) or not isinstance(recurrent, dict):
        return None
    return metric_from_summary(
        benchmark=benchmark,
        base=base,
        recurrent=recurrent,
        paired=comparison if isinstance(comparison, dict) else None,
        source_summary=path,
    )


def ingest_benchmark_suite(records: dict[str, dict[str, Any]], path: Path, payload: dict[str, Any]) -> None:
    checkpoint = str(payload.get("checkpoint") or "")
    label = checkpoint_label(checkpoint, fallback=str(payload.get("run_id") or path.parent.name))
    for benchmark in payload.get("benchmarks", []):
        metric = benchmark_suite_metric(path, payload, str(benchmark))
        if metric is not None:
            add_metric(records, label=label, checkpoint=checkpoint, metric=metric)


def ingest_recovery_ladder(records: dict[str, dict[str, Any]], path: Path, payload: dict[str, Any]) -> None:
    full = payload.get("full_arc_final")
    if not isinstance(full, dict):
        return
    base = (full.get("base") or {}).get(AGGREGATE)
    if not isinstance(base, dict):
        return

    start = (full.get("phase1_start") or {}).get(AGGREGATE)
    start_checkpoint = ((payload.get("phase1_start") or {}).get("checkpoint") or "")
    if isinstance(start, dict):
        add_metric(
            records,
            label="parent",
            checkpoint=start_checkpoint,
            metric=metric_from_summary(
                benchmark="arc_challenge",
                base=base,
                recurrent=start,
                paired=None,
                source_summary=path,
            ),
        )

    best = (full.get("phase1_best") or {}).get(AGGREGATE)
    best_checkpoint = str(full.get("best_checkpoint") or (payload.get("best_checkpoint") or {}).get("checkpoint") or "")
    if isinstance(best, dict):
        pair = full.get("best_vs_base") if isinstance(full.get("best_vs_base"), dict) else None
        add_metric(
            records,
            label=checkpoint_label(best_checkpoint, fallback="phase1_best"),
            checkpoint=best_checkpoint,
            metric=metric_from_summary(
                benchmark="arc_challenge",
                base=base,
                recurrent=best,
                paired=pair,
                source_summary=path,
            ),
        )


def aggregate_row(row: dict[str, Any], required_benchmarks: tuple[str, ...]) -> dict[str, Any]:
    metrics = row["benchmarks"]
    present = sorted(metrics)
    required_present = [item for item in required_benchmarks if item in metrics]
    deltas = [float(metrics[item]["accuracy_delta_recurrent_vs_base"]) for item in present]
    required_deltas = [float(metrics[item]["accuracy_delta_recurrent_vs_base"]) for item in required_present]
    paired_metrics = [
        metric
        for metric in metrics.values()
        if metric.get("wins") is not None and metric.get("losses") is not None and metric.get("ties") is not None
    ]
    wins = sum(int(metric["wins"]) for metric in paired_metrics)
    losses = sum(int(metric["losses"]) for metric in paired_metrics)
    ties = sum(int(metric["ties"]) for metric in paired_metrics)
    base_correct = sum(int(metric["base_correct"]) for metric in metrics.values())
    recurrent_correct = sum(int(metric["recurrent_correct"]) for metric in metrics.values())
    total = sum(int(metric["total"]) for metric in metrics.values())
    row.update(
        {
            "present_benchmarks": present,
            "missing_required_benchmarks": [item for item in required_benchmarks if item not in metrics],
            "full_required_coverage": all(item in metrics for item in required_benchmarks),
            "macro_accuracy_delta": sum(deltas) / max(len(deltas), 1),
            "required_macro_accuracy_delta": sum(required_deltas) / max(len(required_deltas), 1),
            "micro_correct_delta": recurrent_correct - base_correct,
            "base_correct": base_correct,
            "recurrent_correct": recurrent_correct,
            "total": total,
            "combined_wins": wins if paired_metrics else None,
            "combined_losses": losses if paired_metrics else None,
            "combined_ties": ties if paired_metrics else None,
            "combined_sign_test_p_value": two_sided_sign_p_value(wins, losses) if paired_metrics else None,
        }
    )
    return row


def build_assessment(
    *,
    arc_easy_sweep: Path,
    arc_challenge_summaries: list[Path],
    required_benchmarks: tuple[str, ...] = REQUIRED_BENCHMARKS,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}

    easy_payload = read_json(arc_easy_sweep)
    if not is_arceasy_sweep(easy_payload):
        raise ValueError(f"Not an ARC-Easy checkpoint sweep: {arc_easy_sweep}")
    ingest_arceasy_sweep(records, arc_easy_sweep, easy_payload)

    for path in arc_challenge_summaries:
        payload = read_json(path)
        if payload.get("kind") == "stage5_benchmark_suite":
            ingest_benchmark_suite(records, path, payload)
        elif isinstance(payload.get("full_arc_final"), dict):
            ingest_recovery_ladder(records, path, payload)

    rows = [aggregate_row(row, required_benchmarks) for row in records.values()]
    rows.sort(
        key=lambda row: (
            bool(row["full_required_coverage"]),
            float(row["required_macro_accuracy_delta"]),
            int(row["micro_correct_delta"]),
            str(row["label"]),
        ),
        reverse=True,
    )
    full_rows = [row for row in rows if row["full_required_coverage"]]
    best = full_rows[0] if full_rows else (rows[0] if rows else None)

    if best is None:
        status = "no_evidence"
        next_step = "Run ARC-Easy and ARC-Challenge paired benchmark evidence before selecting a checkpoint."
    elif not best["full_required_coverage"]:
        status = "partial_evidence"
        next_step = "Run missing full benchmark slices for candidate recurrent checkpoints."
    elif best["required_macro_accuracy_delta"] >= 0.0 and best["micro_correct_delta"] >= 0:
        status = "balanced_nonnegative"
        next_step = "Use the selected checkpoint for broader held-out benchmarks and release packaging."
    else:
        status = "needs_competence_recovery"
        next_step = (
            "Use the selected checkpoint as the current balanced baseline, then train with a competence-preserving "
            "mixed objective before returning to particles/SVGD."
        )

    return {
        "run_id": RUN_ID,
        "kind": "stage5_balanced_mcq_checkpoint_assessment",
        "arc_easy_sweep": path_for_cli(arc_easy_sweep),
        "arc_challenge_summaries": [path_for_cli(path) for path in arc_challenge_summaries],
        "required_benchmarks": list(required_benchmarks),
        "score_target": SCORE_TARGET,
        "aggregate": AGGREGATE,
        "status": status,
        "passed": status == "balanced_nonnegative",
        "next_step": next_step,
        "best_checkpoint": best,
        "ranked_checkpoints": rows,
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Balanced MCQ Checkpoint Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- ARC-Easy sweep: `{payload['arc_easy_sweep']}`",
        f"- ARC-Challenge summaries: `{payload['arc_challenge_summaries']}`",
        f"- Required benchmarks: `{payload['required_benchmarks']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Ranked Checkpoints",
        "",
        "| rank | label | full coverage | macro delta | micro delta | recurrent/base | W/L/T | checkpoint |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for idx, row in enumerate(payload["ranked_checkpoints"], start=1):
        wlt = (
            f"{row['combined_wins']}/{row['combined_losses']}/{row['combined_ties']}"
            if row.get("combined_wins") is not None
            else "n/a"
        )
        lines.append(
            f"| {idx} | `{row['label']}` | `{row['full_required_coverage']}` | "
            f"{row['required_macro_accuracy_delta']:.6f} | {row['micro_correct_delta']} | "
            f"{row['recurrent_correct']}/{row['base_correct']} | {wlt} | `{row.get('checkpoint')}` |"
        )
    lines.extend(["", "## Benchmark Details", ""])
    for row in payload["ranked_checkpoints"]:
        lines.append(f"### `{row['label']}`")
        lines.append("")
        for benchmark in payload["required_benchmarks"]:
            metric = row["benchmarks"].get(benchmark)
            if not metric:
                lines.append(f"- `{benchmark}`: missing")
                continue
            lines.append(
                f"- `{benchmark}`: recurrent `{metric['recurrent_correct']}/{metric['total']}`, "
                f"base `{metric['base_correct']}/{metric['total']}`, "
                f"delta `{metric['correct_delta_recurrent_vs_base']}`, "
                f"accuracy delta `{metric['accuracy_delta_recurrent_vs_base']:.6f}`, "
                f"W/L/T `{metric['wins']}/{metric['losses']}/{metric['ties']}`, "
                f"p `{metric['sign_test_p_value']}`"
            )
        lines.append("")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc_easy_sweep")
    parser.add_argument("--arc_challenge_summary", action="append", default=[])
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "stage5" / RUN_ID))
    args = parser.parse_args()

    easy_path = resolve_path(args.arc_easy_sweep) if args.arc_easy_sweep else latest_summary(is_arceasy_sweep)
    if easy_path is None:
        raise FileNotFoundError("No ARC-Easy checkpoint sweep summary found.")
    challenge_paths = [resolve_path(item) for item in args.arc_challenge_summary]
    if not challenge_paths:
        challenge_paths = all_summaries(has_full_arc_challenge)
    challenge_paths = [path for path in challenge_paths if path != easy_path]
    if not challenge_paths:
        raise FileNotFoundError("No ARC-Challenge benchmark or recovery summaries found.")

    payload = build_assessment(arc_easy_sweep=easy_path, arc_challenge_summaries=challenge_paths)
    output_dir = resolve_path(args.output_dir)
    write_report(payload, output_json=output_dir / "summary.json", output_md=output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
