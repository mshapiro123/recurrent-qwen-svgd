"""Annotate verified curriculum candidates with plausible false answers.

Perturbation prompts need a false answer, but that falsehood should be created
after answer verification and kept out of positive data routing. This script
adds deterministic, auditable false answers for numeric candidates and reuses a
seed model's wrong claimed answer when it disagrees with verified ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from training.collect_curriculum_job_outputs import normalize_answer


NUMBER_RE = re.compile(r"^(?P<prefix>[^0-9+\-.]*)(?P<number>[+-]?\d+(?:,\d{3})*(?:\.\d+)?)(?P<suffix>.*)$")


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


def candidate_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("record_id") or row.get("problem_id") or f"row-{index:06d}")


def verified_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, dict):
        value = str(answer.get("value") or "").strip()
        if value:
            return value
    return str(row.get("verified_answer") or row.get("answer") or "").strip()


def first_existing_false_answer(row: dict[str, Any], verified: str) -> str | None:
    candidates: list[Any] = []
    if isinstance(row.get("plausible_false_answers"), list):
        candidates.extend(row["plausible_false_answers"])
    for key in ("false_answer", "claimed_answer"):
        if row.get(key) is not None:
            candidates.append(row[key])
    normalized_verified = normalize_answer(verified)
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and normalize_answer(text) != normalized_verified:
            return text
    return None


def format_decimal(value: Decimal, original: str) -> str:
    if "." in original:
        places = len(original.split(".", 1)[1])
        quantized = value.quantize(Decimal(1).scaleb(-places))
        return f"{quantized:.{places}f}"
    if value == value.to_integral_value():
        return str(int(value))
    return str(value.normalize())


def numeric_false_answer(answer: str) -> str | None:
    match = NUMBER_RE.match(answer.strip())
    if not match:
        return None
    number_text = match.group("number").replace(",", "")
    try:
        value = Decimal(number_text)
    except InvalidOperation:
        return None

    deltas = [Decimal(1), Decimal(-1), Decimal(2), Decimal(-2), Decimal(10), Decimal(-10)]
    if value == 0:
        deltas = [Decimal(1), Decimal(-1), Decimal(2)]

    normalized_true = normalize_answer(answer)
    for delta in deltas:
        false_value = value + delta
        rendered_number = format_decimal(false_value, number_text)
        rendered = f"{match.group('prefix')}{rendered_number}{match.group('suffix')}".strip()
        if rendered and normalize_answer(rendered) != normalized_true:
            return rendered
    return None


def annotate_false_answers(
    candidates: list[dict[str, Any]],
    *,
    drop_unannotated: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    for index, candidate in enumerate(candidates, start=1):
        row = dict(candidate)
        record_id = candidate_id(candidate, index)
        answer = verified_answer(candidate)
        false_answer = first_existing_false_answer(candidate, answer) if answer else None
        source = "existing_wrong_answer" if false_answer else ""
        if not false_answer and answer:
            false_answer = numeric_false_answer(answer)
            source = "numeric_near_miss" if false_answer else ""

        if false_answer:
            row["false_answer"] = false_answer
            row["false_answer_metadata"] = {
                "source": source,
                "verified_answer": answer,
                "normalized_verified_answer": normalize_answer(answer),
                "normalized_false_answer": normalize_answer(false_answer),
            }
            source_counts[source] = source_counts.get(source, 0) + 1
            annotated.append(row)
            continue

        rejected_record = {"id": record_id, "reason": "could_not_construct_false_answer", "answer": answer}
        rejected.append(rejected_record)
        if not drop_unannotated:
            row["false_answer_metadata"] = {
                "source": None,
                "verified_answer": answer,
                "reason": rejected_record["reason"],
            }
            annotated.append(row)

    report = {
        "mode": "curriculum_false_answer_annotation",
        "candidates": len(candidates),
        "annotated": len(annotated),
        "with_false_answer": sum(1 for row in annotated if str(row.get("false_answer") or "").strip()),
        "rejected": len(rejected),
        "drop_unannotated": drop_unannotated,
        "source_counts": dict(sorted(source_counts.items())),
        "rejected_records": rejected,
    }
    return annotated, rejected, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--rejected_jsonl")
    parser.add_argument("--report_json")
    parser.add_argument("--drop_unannotated", action="store_true")
    args = parser.parse_args(argv)

    annotated, rejected, report = annotate_false_answers(
        read_jsonl(args.candidates_jsonl),
        drop_unannotated=args.drop_unannotated,
    )
    write_jsonl(args.output_jsonl, annotated)
    if args.rejected_jsonl:
        write_jsonl(args.rejected_jsonl, rejected)
    write_report(args.report_json, report)

    print(f"candidates={report['candidates']}")
    print(f"annotated={report['annotated']}")
    print(f"with_false_answer={report['with_false_answer']}")
    print(f"rejected={report['rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
