"""Assess broader Stage 5 base-vs-recurrent benchmark evidence.

This is a no-GPU gate that reads ``colab/run_stage5_benchmark_suite.py``
outputs. It decides whether the recurrent artifact is ready for larger public
claims, needs more benchmark coverage, or should return to recurrent recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_BENCHMARK_ASSESS_RUN_ID") or time.strftime(
    "stage5_benchmark_assessment_%Y%m%d_%H%M%S"
)

MIN_ARC_EXAMPLES = int(os.environ.get("STAGE5_BENCHMARK_ASSESS_MIN_ARC_EXAMPLES", "128"))
MIN_GPQA_EXAMPLES = int(os.environ.get("STAGE5_BENCHMARK_ASSESS_MIN_GPQA_EXAMPLES", "16"))
ARC_CHALLENGE_VALIDATION_EXAMPLES = int(
    os.environ.get("STAGE5_BENCHMARK_ASSESS_ARC_CHALLENGE_VALIDATION_EXAMPLES", "299")
)
ARC_EASY_VALIDATION_EXAMPLES = int(os.environ.get("STAGE5_BENCHMARK_ASSESS_ARC_EASY_VALIDATION_EXAMPLES", "570"))
REQUIRED_SCORE_TARGET = os.environ.get("STAGE5_BENCHMARK_ASSESS_SCORE_TARGET", "label")
REQUIRED_AGGREGATE = os.environ.get("STAGE5_BENCHMARK_ASSESS_AGGREGATE", "mean")
ALLOWED_NEGATIVE_DELTA = int(os.environ.get("STAGE5_BENCHMARK_ASSESS_ALLOWED_NEGATIVE_DELTA", "0"))
REQUIRED_BENCHMARKS = [
    item.strip()
    for item in os.environ.get("STAGE5_BENCHMARK_ASSESS_REQUIRED_BENCHMARKS", "arc_challenge,gpqa_lite").split(",")
    if item.strip()
]
NEGATIVE_EVIDENCE_MIN_ABS_DELTA = int(
    os.environ.get("STAGE5_BENCHMARK_ASSESS_NEGATIVE_EVIDENCE_MIN_ABS_DELTA", "2")
)
NEGATIVE_EVIDENCE_SIGN_TEST_P_THRESHOLD = float(
    os.environ.get("STAGE5_BENCHMARK_ASSESS_NEGATIVE_EVIDENCE_SIGN_TEST_P_THRESHOLD", "0.10")
)
PUSH_RESULTS = os.environ.get("STAGE5_BENCHMARK_ASSESS_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.stdout:
        print(result.stdout)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return result


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_benchmark_suite(payload: dict[str, Any]) -> bool:
    return payload.get("kind") == "stage5_benchmark_suite"


def latest_benchmark_suite() -> Path | None:
    candidates: list[Path] = []
    root = ROOT / "outputs" / "stage5"
    if not root.exists():
        return None
    for path in root.glob("**/summary.json"):
        payload = safe_read_json(path)
        if payload and is_benchmark_suite(payload):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def benchmark_min_examples(benchmark: str) -> int:
    if benchmark == "arc_challenge":
        return min(MIN_ARC_EXAMPLES, ARC_CHALLENGE_VALIDATION_EXAMPLES)
    if benchmark == "arc_easy":
        return min(MIN_ARC_EXAMPLES, ARC_EASY_VALIDATION_EXAMPLES)
    if benchmark == "gpqa_lite":
        return MIN_GPQA_EXAMPLES
    if benchmark == "open_hard_arc_challenge":
        return min(MIN_ARC_EXAMPLES, ARC_CHALLENGE_VALIDATION_EXAMPLES)
    return 1


def paired_row(payload: dict[str, Any], benchmark: str) -> dict[str, Any] | None:
    row = (
        (payload.get("paired_comparisons") or {})
        .get(benchmark, {})
        .get(REQUIRED_SCORE_TARGET, {})
        .get(REQUIRED_AGGREGATE)
    )
    return row if isinstance(row, dict) else None


def benchmark_evidence(payload: dict[str, Any], benchmark: str) -> dict[str, Any]:
    row = paired_row(payload, benchmark)
    if not row:
        return {
            "benchmark": benchmark,
            "present": False,
            "required_examples": benchmark_min_examples(benchmark),
            "paired_examples": 0,
            "correct_delta_recurrent_vs_base": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "sign_test_p_value": None,
            "negative_evidence": False,
        }
    delta = int(row.get("correct_delta_recurrent_vs_base", 0) or 0)
    p_value = row.get("sign_test_p_value")
    try:
        p_float = float(p_value) if p_value is not None else None
    except (TypeError, ValueError):
        p_float = None
    flagged_regression = delta < -ALLOWED_NEGATIVE_DELTA and abs(delta) >= NEGATIVE_EVIDENCE_MIN_ABS_DELTA
    statistically_supported_regression = (
        p_float is not None and p_float <= NEGATIVE_EVIDENCE_SIGN_TEST_P_THRESHOLD
    )
    negative_evidence = flagged_regression and statistically_supported_regression
    return {
        "benchmark": benchmark,
        "present": True,
        "required_examples": benchmark_min_examples(benchmark),
        "paired_examples": int(row.get("paired_examples", 0) or 0),
        "base_correct": int(row.get("base_correct", 0) or 0),
        "recurrent_correct": int(row.get("recurrent_correct", 0) or 0),
        "correct_delta_recurrent_vs_base": delta,
        "wins": int(row.get("wins", 0) or 0),
        "losses": int(row.get("losses", 0) or 0),
        "ties": int(row.get("ties", 0) or 0),
        "sign_test_p_value": row.get("sign_test_p_value"),
        "flagged_regression": flagged_regression,
        "statistically_supported_regression": statistically_supported_regression,
        "negative_evidence": negative_evidence,
    }


def criterion(name: str, passed: bool, reason: str, evidence: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "reason": reason, "evidence": evidence}


def assess_benchmark_suite(*, summary_json: Path, payload: dict[str, Any]) -> dict[str, Any]:
    benchmarks = [str(item) for item in payload.get("benchmarks", [])]
    for required in REQUIRED_BENCHMARKS:
        if required not in benchmarks:
            benchmarks.append(required)
    failures = payload.get("failures") or []
    status = str(payload.get("status") or "unknown")
    benchmark_rows = [benchmark_evidence(payload, benchmark) for benchmark in benchmarks]

    suite_completed = status == "completed" and not failures
    enough_examples = all(
        row["present"] and int(row["paired_examples"]) >= int(row["required_examples"])
        for row in benchmark_rows
    )
    nonnegative = all(
        int(row["correct_delta_recurrent_vs_base"]) >= -ALLOWED_NEGATIVE_DELTA
        for row in benchmark_rows
        if row["present"]
    )
    no_missing = bool(benchmark_rows) and all(row["present"] for row in benchmark_rows)
    instrument_complete = suite_completed and no_missing and enough_examples
    model_negative_evidence = any(bool(row.get("negative_evidence")) for row in benchmark_rows)

    criteria = [
        criterion(
            "suite_completed",
            suite_completed,
            "Benchmark suite completed without failures." if suite_completed else "Benchmark suite had failures or did not complete.",
            {"status": status, "failures": failures},
        ),
        criterion(
            "paired_coverage",
            no_missing and enough_examples,
            (
                "All benchmark slices have enough paired base-vs-recurrent examples."
                if no_missing and enough_examples
                else "Need more paired examples before interpreting broader benchmark deltas."
            ),
            benchmark_rows,
        ),
        criterion(
            "recurrent_nonnegative_vs_base",
            nonnegative and no_missing,
            (
                "Recurrent is non-negative versus base on required paired benchmark slices."
                if nonnegative and no_missing
                else (
                    "Recurrent has statistically supported negative evidence versus base."
                    if model_negative_evidence
                    else "Recurrent has only noise-level negative deltas or missing slices."
                )
            ),
            benchmark_rows,
        ),
    ]

    if not suite_completed or not no_missing:
        gate_status = "inconclusive"
        next_step = (
            "Complete the benchmark instrument by rerunning failed or missing slices before making a model-quality call."
        )
    elif not enough_examples:
        gate_status = "needs_benchmark_confirmation"
        next_step = "Rerun the broader benchmark suite with enough paired examples before making a claim."
    elif model_negative_evidence:
        gate_status = "needs_recurrent_recovery"
        next_step = "Return to deterministic recurrent recovery before GPQA Diamond or release claims."
    elif not nonnegative:
        gate_status = "inconclusive"
        next_step = (
            "Negative deltas are noise-level under the paired test; rerun with more paired examples before changing the model."
        )
    else:
        gate_status = "passed"
        next_step = "Proceed to release writeup or larger held-out benchmark confirmation."

    result = {
        "run_id": RUN_ID,
        "gate": "stage5_broader_benchmark_suite",
        "source_summary": path_for_cli(summary_json),
        "checkpoint": payload.get("checkpoint"),
        "status": gate_status,
        "passed": gate_status == "passed",
        "instrument_complete": instrument_complete,
        "model_negative_evidence": model_negative_evidence,
        "next_step": next_step,
        "required_score_target": REQUIRED_SCORE_TARGET,
        "required_aggregate": REQUIRED_AGGREGATE,
        "required_benchmarks": REQUIRED_BENCHMARKS,
        "allowed_negative_delta": ALLOWED_NEGATIVE_DELTA,
        "negative_evidence_min_abs_delta": NEGATIVE_EVIDENCE_MIN_ABS_DELTA,
        "negative_evidence_sign_test_p_threshold": NEGATIVE_EVIDENCE_SIGN_TEST_P_THRESHOLD,
        "benchmarks": benchmark_rows,
        "criteria": criteria,
    }
    after_dense = payload.get("after_confirmation_dense_control")
    if isinstance(after_dense, dict):
        result["after_confirmation_dense_control"] = after_dense
    return result


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Broader Benchmark Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Instrument complete: `{payload['instrument_complete']}`",
        f"- Model negative evidence: `{payload['model_negative_evidence']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Score target / aggregate: `{payload['required_score_target']}` / `{payload['required_aggregate']}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Criteria",
        "",
    ]
    for row in payload["criteria"]:
        lines.append(f"- `{row['name']}` passed `{row['passed']}`: {row['reason']}")
    lines.extend(["", "## Benchmark Evidence", ""])
    for row in payload["benchmarks"]:
        lines.append(
            f"- `{row['benchmark']}` paired `{row['paired_examples']}` / required "
            f"`{row['required_examples']}`; delta `{row['correct_delta_recurrent_vs_base']}`; "
            f"W/L/T `{row['wins']}/{row['losses']}/{row['ties']}`; p `{row['sign_test_p_value']}`"
            f"; flagged regression `{row.get('flagged_regression')}`"
            f"; negative evidence `{row.get('negative_evidence')}`"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


def commit_results(run_dir: Path) -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "status", "-sb"], check=False)
    run(["git", "add", "-f", path_for_cli(run_dir)], check=False)
    pointer = current_source_summary_file()
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], check=False)
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No benchmark assessment outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 benchmark assessment {RUN_ID} [skip ci]"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", help="Benchmark suite summary. Defaults to latest Stage 5 benchmark suite.")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    summary = resolve_path(args.summary_json) if args.summary_json else latest_benchmark_suite()
    if not summary:
        raise FileNotFoundError("No Stage 5 benchmark suite summary found.")
    payload = read_json(summary)
    if not is_benchmark_suite(payload):
        raise ValueError(f"Not a Stage 5 benchmark suite summary: {summary}")

    output_json = resolve_path(args.output_json) if args.output_json else ROOT / "outputs" / "stage5" / RUN_ID / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    assessment = assess_benchmark_suite(summary_json=summary, payload=payload)
    write_report(assessment, output_json=output_json, output_md=output_md)
    if not args.output_json:
        update_current_source_summary(output_json)
        commit_results(output_json.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
