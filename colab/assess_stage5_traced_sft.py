"""Assess traced capability-ladder SFT benchmark evidence.

This is a no-GPU steering gate for the local-HF trace -> recurrent SFT path.
It is deliberately narrower than the release benchmark gate: it checks whether
the surgically recurrent model is still near base performance, whether learned
depth routing is real, and whether the next spend should be more trace rows,
calibration repair, broader benchmarks, or Phase 2 particles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_TRACED_SFT_ASSESS_RUN_ID") or time.strftime(
    "stage5_traced_sft_assessment_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID

MIN_PAIRED_EXAMPLES = int(os.environ.get("STAGE5_TRACED_SFT_ASSESS_MIN_PAIRED_EXAMPLES", "96"))
TARGET_TRACE_ROWS_FOR_PHASE2 = int(os.environ.get("STAGE5_TRACED_SFT_ASSESS_TARGET_TRACE_ROWS", "96"))
ALLOWED_CONTENT_NEGATIVE_DELTA = int(
    os.environ.get("STAGE5_TRACED_SFT_ASSESS_ALLOWED_CONTENT_NEGATIVE_DELTA", "1")
)
ALLOWED_CYCLIC_NEGATIVE_DELTA = int(
    os.environ.get("STAGE5_TRACED_SFT_ASSESS_ALLOWED_CYCLIC_NEGATIVE_DELTA", "0")
)
MIN_DEPTH_MARGIN = float(os.environ.get("STAGE5_TRACED_SFT_ASSESS_MIN_DEPTH_MARGIN", "0.5"))
MAX_DIRECT_MEAN_LOOPS = float(os.environ.get("STAGE5_TRACED_SFT_ASSESS_MAX_DIRECT_MEAN_LOOPS", "1.35"))
MIN_DEEP_MEAN_LOOPS = float(os.environ.get("STAGE5_TRACED_SFT_ASSESS_MIN_DEEP_MEAN_LOOPS", "1.5"))
LOOP_TOLERANCE = float(os.environ.get("STAGE5_TRACED_SFT_ASSESS_LOOP_TOLERANCE", "0.02"))
PUSH_RESULTS = os.environ.get("STAGE5_TRACED_SFT_ASSESS_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def paired_row(payload: dict[str, Any], benchmark: str, score_target: str, aggregate: str) -> dict[str, Any] | None:
    row = (
        (payload.get("paired_comparisons") or {})
        .get(benchmark, {})
        .get(score_target, {})
        .get(aggregate)
    )
    return row if isinstance(row, dict) else None


def benchmark_evidence(payload: dict[str, Any], benchmark: str) -> dict[str, Any]:
    content = paired_row(payload, benchmark, "content_question_only", "mean")
    cyclic = paired_row(payload, benchmark, "cyclic_label_aggregated", "permutation_mean")
    return {
        "benchmark": benchmark,
        "content": summarize_pair(content),
        "cyclic": summarize_pair(cyclic),
    }


def summarize_pair(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "present": False,
            "paired_examples": 0,
            "base_correct": None,
            "recurrent_correct": None,
            "delta": None,
            "wins": None,
            "losses": None,
            "ties": None,
            "sign_test_p_value": None,
        }
    return {
        "present": True,
        "paired_examples": int(row.get("paired_examples", 0) or 0),
        "base_correct": int(row.get("base_correct", 0) or 0),
        "recurrent_correct": int(row.get("recurrent_correct", 0) or 0),
        "delta": int(row.get("correct_delta_recurrent_vs_base", 0) or 0),
        "wins": int(row.get("wins", 0) or 0),
        "losses": int(row.get("losses", 0) or 0),
        "ties": int(row.get("ties", 0) or 0),
        "sign_test_p_value": row.get("sign_test_p_value"),
    }


def mode_metric(sft_summary: dict[str, Any], mode: str, metric: str) -> float | None:
    by_mode = sft_summary.get("phase1_val_by_mode")
    if not isinstance(by_mode, dict):
        return None
    row = by_mode.get(mode)
    if not isinstance(row, dict):
        return None
    return finite_float(row.get(metric))


def depth_evidence(sft_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not sft_summary:
        return {
            "available": False,
            "direct_mean_expected_loops": None,
            "deep_narrow_mean_expected_loops": None,
            "margin": None,
            "direct_under_max": False,
            "deep_over_min": False,
            "margin_passed": False,
            "passed": False,
        }
    direct = mode_metric(sft_summary, "direct", "mean_expected_loops")
    deep = mode_metric(sft_summary, "deep_narrow", "mean_expected_loops")
    margin = None if direct is None or deep is None else deep - direct
    direct_under_max = direct is not None and direct <= MAX_DIRECT_MEAN_LOOPS + LOOP_TOLERANCE
    deep_over_min = deep is not None and deep >= MIN_DEEP_MEAN_LOOPS - LOOP_TOLERANCE
    margin_passed = margin is not None and margin >= MIN_DEPTH_MARGIN - LOOP_TOLERANCE
    return {
        "available": direct is not None and deep is not None,
        "direct_mean_expected_loops": direct,
        "deep_narrow_mean_expected_loops": deep,
        "margin": margin,
        "max_direct_mean_loops": MAX_DIRECT_MEAN_LOOPS,
        "min_deep_mean_loops": MIN_DEEP_MEAN_LOOPS,
        "min_depth_margin": MIN_DEPTH_MARGIN,
        "loop_tolerance": LOOP_TOLERANCE,
        "direct_under_max": direct_under_max,
        "deep_over_min": deep_over_min,
        "margin_passed": margin_passed,
        "passed": direct_under_max and deep_over_min and margin_passed,
    }


def trace_rows(sft_summary: dict[str, Any] | None) -> int:
    if not sft_summary:
        return 0
    dataset = sft_summary.get("dataset")
    if isinstance(dataset, dict):
        try:
            return int(dataset.get("rows") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def source_sft_summary(benchmark_summary: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    raw = str(benchmark_summary.get("source_summary") or "").strip()
    if not raw:
        return None, None
    path = resolve_path(raw)
    if not path.exists():
        return path, None
    try:
        return path, read_json(path)
    except Exception:
        return path, None


def criterion(name: str, passed: bool, reason: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "reason": reason, "evidence": evidence}


def benchmark_ok(rows: list[dict[str, Any]], key: str, allowed_negative: int) -> bool:
    seen = False
    for row in rows:
        item = row[key]
        if not item["present"]:
            continue
        seen = True
        if item["paired_examples"] < MIN_PAIRED_EXAMPLES:
            return False
        if item["delta"] is None or int(item["delta"]) < -allowed_negative:
            return False
    return seen


def assess(*, benchmark_summary_path: Path, benchmark_summary: dict[str, Any]) -> dict[str, Any]:
    sft_path, sft_summary = source_sft_summary(benchmark_summary)
    benchmarks = [str(item) for item in benchmark_summary.get("benchmarks", [])]
    rows = [benchmark_evidence(benchmark_summary, benchmark) for benchmark in benchmarks]
    failures = benchmark_summary.get("failures") or []
    suite_completed = benchmark_summary.get("status") == "completed" and not failures

    any_content = any(row["content"]["present"] for row in rows)
    any_cyclic = any(row["cyclic"]["present"] for row in rows)
    content_ok = benchmark_ok(rows, "content", ALLOWED_CONTENT_NEGATIVE_DELTA)
    cyclic_ok = benchmark_ok(rows, "cyclic", ALLOWED_CYCLIC_NEGATIVE_DELTA)
    enough_benchmark_coverage = any_content and any_cyclic and all(
        (not row["content"]["present"] or row["content"]["paired_examples"] >= MIN_PAIRED_EXAMPLES)
        and (not row["cyclic"]["present"] or row["cyclic"]["paired_examples"] >= MIN_PAIRED_EXAMPLES)
        for row in rows
    )
    depth = depth_evidence(sft_summary)
    rows_seen = trace_rows(sft_summary)

    criteria = [
        criterion(
            "suite_completed",
            suite_completed,
            "Benchmark suite completed without failures." if suite_completed else "Benchmark suite failed or is incomplete.",
            {"status": benchmark_summary.get("status"), "failures": failures},
        ),
        criterion(
            "benchmark_coverage",
            enough_benchmark_coverage,
            "Content and cyclic paired comparisons have enough examples."
            if enough_benchmark_coverage
            else "Need broader paired benchmark coverage before interpreting small deltas.",
            rows,
        ),
        criterion(
            "content_near_base",
            content_ok,
            "Content-only scoring is near base."
            if content_ok
            else "Content-only scoring regressed beyond the allowed tolerance.",
            {"allowed_negative_delta": ALLOWED_CONTENT_NEGATIVE_DELTA, "benchmarks": rows},
        ),
        criterion(
            "cyclic_nonnegative",
            cyclic_ok,
            "Cyclic label-aggregated scoring is nonnegative versus base."
            if cyclic_ok
            else "Cyclic label-aggregated scoring regressed versus base.",
            {"allowed_negative_delta": ALLOWED_CYCLIC_NEGATIVE_DELTA, "benchmarks": rows},
        ),
        criterion(
            "depth_gradient",
            depth["passed"],
            "Learned loop controller separates direct and deep-narrow rows."
            if depth["passed"]
            else "Depth routing is missing, weak, or overthinking direct rows.",
            depth,
        ),
    ]

    if not suite_completed:
        status = "needs_benchmark_rerun"
        next_step = "Inspect benchmark logs and rerun failed slices."
    elif not enough_benchmark_coverage:
        status = "needs_broader_benchmark"
        next_step = "Run ARC-Easy and ARC-Challenge with larger paired slices before Phase 2."
    elif not cyclic_ok:
        status = "needs_calibration_repair"
        next_step = "Fix answer calibration before adding more trace data or Phase 2 particles."
    elif not content_ok:
        status = "needs_direct_preservation_repair"
        next_step = "Add direct/base-preservation rows or distillation before scaling hard traces."
    elif not depth["passed"]:
        status = "needs_depth_routing_repair"
        next_step = "Repair learned-loop supervision before scaling data or adding particles."
    elif rows_seen < TARGET_TRACE_ROWS_FOR_PHASE2:
        status = "scale_trace_curriculum"
        next_step = (
            f"Scale local-HF trace curriculum toward {TARGET_TRACE_ROWS_FOR_PHASE2} verified rows, "
            "then rerun deterministic recurrent SFT and benchmark."
        )
    else:
        status = "ready_for_phase2_probe"
        next_step = "Run a bounded Phase 2 particle/SVGD probe; keep deterministic checkpoint as the baseline."

    return {
        "run_id": RUN_ID,
        "kind": "stage5_traced_sft_assessment",
        "source_summary": path_for_cli(benchmark_summary_path),
        "sft_summary": None if sft_path is None else path_for_cli(sft_path),
        "checkpoint": benchmark_summary.get("checkpoint"),
        "status": status,
        "passed": status == "ready_for_phase2_probe",
        "next_step": next_step,
        "thresholds": {
            "min_paired_examples": MIN_PAIRED_EXAMPLES,
            "target_trace_rows_for_phase2": TARGET_TRACE_ROWS_FOR_PHASE2,
            "allowed_content_negative_delta": ALLOWED_CONTENT_NEGATIVE_DELTA,
            "allowed_cyclic_negative_delta": ALLOWED_CYCLIC_NEGATIVE_DELTA,
        },
        "trace_rows": rows_seen,
        "benchmarks": rows,
        "depth": depth,
        "criteria": criteria,
    }


def write_report(payload: dict[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Traced SFT Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- SFT summary: `{payload.get('sft_summary')}`",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Trace rows: `{payload.get('trace_rows')}`",
        f"- Depth margin: `{payload['depth'].get('margin')}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Benchmarks",
    ]
    for row in payload["benchmarks"]:
        lines.append(
            "- "
            f"{row['benchmark']}: content delta `{row['content'].get('delta')}` "
            f"({row['content'].get('recurrent_correct')}/{row['content'].get('paired_examples')}), "
            f"cyclic delta `{row['cyclic'].get('delta')}` "
            f"({row['cyclic'].get('recurrent_correct')}/{row['cyclic'].get('paired_examples')})"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


def commit_results(output_dir: Path) -> None:
    if not PUSH_RESULTS:
        return
    update_current_source_summary(output_dir / "summary.json")
    run(["git", "add", "-f", path_for_cli(output_dir)])
    run(["git", "add", "config/stage5_current_source_summary.txt"])
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No traced SFT assessment changes to commit.")
        return
    run(["git", "commit", "-m", f"Record traced SFT assessment {RUN_ID} [skip ci]"])
    run(["git", "pull", "--rebase", "origin", "main"])
    run(["git", "push", "origin", "main"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args(argv)

    summary_path = resolve_path(args.summary_json)
    output_dir = resolve_path(args.output_dir) if args.output_dir else RUN_DIR
    payload = assess(benchmark_summary_path=summary_path, benchmark_summary=read_json(summary_path))
    write_report(payload, output_dir=output_dir)
    commit_results(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
