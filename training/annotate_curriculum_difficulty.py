"""Annotate verified curriculum candidates with measured reference pass rates.

Difficulty in this project is measured by a fixed weak/reference model, not by
generator self-report. This script consumes per-sample attempt records from any
runner and writes candidates with a ``difficulty`` block suitable for curriculum
assembly and SFT export.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ID_FIELDS = ("record_id", "id", "problem_id", "curriculum_id")
CORRECT_FIELDS = ("correct", "is_correct", "matched", "success")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    for field in ID_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def bool_value(row: dict[str, Any]) -> bool | None:
    for field in CORRECT_FIELDS:
        value = row.get(field)
        if isinstance(value, bool):
            return value
    return None


def attempts_by_record(attempts: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[str] = []
    for index, attempt in enumerate(attempts, start=1):
        record_id = row_id(attempt)
        correct = bool_value(attempt)
        if not record_id:
            issues.append(f"attempt {index}: missing record id")
            continue
        if correct is None:
            issues.append(f"attempt {index}: missing boolean correctness field")
            continue
        grouped[record_id].append(attempt)
    return grouped, issues


def annotate_difficulty(
    candidates: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    *,
    reference_model: str,
    min_samples: int = 1,
    drop_unmeasured: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped_attempts, issues = attempts_by_record(attempts)
    annotated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pass_rates: list[float] = []

    for index, candidate in enumerate(candidates, start=1):
        candidate_id = row_id(candidate) or f"row-{index:06d}"
        candidate_attempts = grouped_attempts.get(candidate_id, [])
        if len(candidate_attempts) < min_samples:
            rejected_record = {
                "id": candidate_id,
                "reason": "insufficient_reference_samples",
                "samples": len(candidate_attempts),
                "min_samples": min_samples,
            }
            rejected.append(rejected_record)
            if drop_unmeasured:
                continue
            row = dict(candidate)
            row["difficulty"] = {
                "pass_rate": None,
                "reference_model": reference_model,
                "samples": len(candidate_attempts),
                "measured": False,
            }
            annotated.append(row)
            continue

        correct_count = sum(1 for attempt in candidate_attempts if bool_value(attempt) is True)
        sample_count = len(candidate_attempts)
        pass_rate = correct_count / sample_count
        pass_rates.append(pass_rate)
        row = dict(candidate)
        row["difficulty"] = {
            "pass_rate": pass_rate,
            "reference_model": reference_model,
            "samples": sample_count,
            "correct": correct_count,
            "measured": True,
        }
        row["difficulty_pass_rate"] = pass_rate
        annotated.append(row)

    measured = len(pass_rates)
    report = {
        "mode": "curriculum_difficulty_annotation",
        "candidates": len(candidates),
        "attempts": len(attempts),
        "annotated": len(annotated),
        "measured": measured,
        "rejected": len(rejected),
        "reference_model": reference_model,
        "min_samples": min_samples,
        "drop_unmeasured": drop_unmeasured,
        "mean_pass_rate": sum(pass_rates) / measured if measured else None,
        "pass_rate_counts": dict(sorted(Counter(pass_rates).items())),
        "issues": issues,
        "rejected_records": rejected,
    }
    return annotated, rejected, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--attempts_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--rejected_jsonl")
    parser.add_argument("--report_json")
    parser.add_argument("--reference_model", required=True)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--drop_unmeasured", action="store_true")
    args = parser.parse_args(argv)

    if args.min_samples < 1:
        raise ValueError("--min_samples must be positive.")

    annotated, rejected, report = annotate_difficulty(
        read_jsonl(args.candidates_jsonl),
        read_jsonl(args.attempts_jsonl),
        reference_model=args.reference_model,
        min_samples=args.min_samples,
        drop_unmeasured=args.drop_unmeasured,
    )
    write_jsonl(args.output_jsonl, annotated)
    if args.rejected_jsonl:
        write_jsonl(args.rejected_jsonl, rejected)
    write_report(args.report_json, report)

    print(f"candidates={report['candidates']}")
    print(f"annotated={report['annotated']}")
    print(f"measured={report['measured']}")
    print(f"rejected={report['rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
