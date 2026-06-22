"""Decontaminate generated curriculum candidates against evaluation text.

This is a cheap deterministic first-pass guardrail. It intentionally uses
token n-gram overlap instead of embeddings so it can run locally before any
paid API or GPU work is spent on downstream verification.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TEXT_FIELDS = (
    "statement",
    "prompt",
    "question",
    "problem",
    "input",
    "text",
)
ID_FIELDS = ("id", "name", "task_id", "record_id", "question_id")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


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


def row_id(row: dict[str, Any], *, fallback: str) -> str:
    for field in ID_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def extract_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        value = row.get(field)
        text = stringify_value(value).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def token_ngrams(tokens: list[str], ngram_size: int, *, min_ngram_size: int = 3) -> set[tuple[str, ...]]:
    if not tokens:
        return set()
    lower = max(1, min(min_ngram_size, ngram_size))
    upper = min(ngram_size, len(tokens))
    if upper < lower:
        return {tuple(tokens)}
    ngrams: set[tuple[str, ...]] = set()
    for size in range(lower, upper + 1):
        ngrams.update(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))
    return ngrams


def overlap_scores(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> dict[str, float]:
    if not left or not right:
        return {
            "jaccard": 0.0,
            "candidate_containment": 0.0,
            "reference_containment": 0.0,
            "score": 0.0,
        }
    shared = left & right
    union = left | right
    jaccard = len(shared) / len(union)
    candidate_containment = len(shared) / len(left)
    reference_containment = len(shared) / len(right)
    return {
        "jaccard": jaccard,
        "candidate_containment": candidate_containment,
        "reference_containment": reference_containment,
        "score": max(jaccard, candidate_containment, reference_containment),
    }


def build_reference_index(
    reference_paths: list[str | Path],
    *,
    text_fields: tuple[str, ...],
    ngram_size: int,
    min_ngram_size: int = 3,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for path in reference_paths:
        rows = read_jsonl(path)
        for index, row in enumerate(rows, start=1):
            text = extract_text(row, text_fields)
            references.append(
                {
                    "id": row_id(row, fallback=f"{Path(path).name}:{index}"),
                    "path": str(path),
                    "line": index,
                    "text": text,
                    "ngrams": token_ngrams(tokenize(text), ngram_size, min_ngram_size=min_ngram_size),
                }
            )
    return references


def best_reference_match(
    candidate_ngrams: set[tuple[str, ...]],
    references: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for reference in references:
        scores = overlap_scores(candidate_ngrams, reference["ngrams"])
        if best is None or scores["score"] > best["score"]:
            best = {
                "reference_id": reference["id"],
                "reference_path": reference["path"],
                "reference_line": reference["line"],
                **scores,
            }
    return best


def annotate_candidates(
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    text_fields: tuple[str, ...],
    ngram_size: int,
    min_ngram_size: int = 3,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    annotated_rows: list[dict[str, Any]] = []
    all_scores: list[float] = []

    for index, candidate in enumerate(candidates, start=1):
        text = extract_text(candidate, text_fields)
        candidate_ngrams = token_ngrams(tokenize(text), ngram_size, min_ngram_size=min_ngram_size)
        match = best_reference_match(candidate_ngrams, references)
        if match is None:
            match = {
                "reference_id": None,
                "reference_path": None,
                "reference_line": None,
                "jaccard": 0.0,
                "candidate_containment": 0.0,
                "reference_containment": 0.0,
                "score": 0.0,
            }
        score = float(match["score"])
        all_scores.append(score)
        is_clean = score < threshold
        annotated = dict(candidate)
        annotated["decontaminated"] = is_clean
        annotated["decontamination"] = {
            "method": "token_ngram_overlap",
            "ngram_size": ngram_size,
            "min_ngram_size": min_ngram_size,
            "threshold": threshold,
            "score": score,
            "jaccard": float(match["jaccard"]),
            "candidate_containment": float(match["candidate_containment"]),
            "reference_containment": float(match["reference_containment"]),
            "matched_reference_id": match["reference_id"],
            "matched_reference_path": match["reference_path"],
            "matched_reference_line": match["reference_line"],
        }
        annotated_rows.append(annotated)
        if is_clean:
            accepted.append(annotated)
        else:
            rejected.append(annotated)

    report = {
        "mode": "curriculum_candidate_decontamination",
        "candidates": len(candidates),
        "references": len(references),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "ngram_size": ngram_size,
        "min_ngram_size": min_ngram_size,
        "threshold": threshold,
        "max_score": max(all_scores) if all_scores else 0.0,
        "annotated_rows": annotated_rows,
    }
    return accepted, rejected, report


def parse_fields(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_TEXT_FIELDS
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    if not fields:
        raise ValueError("--text_fields must include at least one field.")
    return fields


def parse_reference_paths(values: list[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        paths.extend(part.strip() for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("At least one --references_jsonl path is required.")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument(
        "--references_jsonl",
        action="append",
        required=True,
        help="Reference/evaluation JSONL path. May be repeated or comma-separated.",
    )
    parser.add_argument("--output_jsonl", required=True, help="Clean candidate output JSONL.")
    parser.add_argument("--rejected_jsonl", help="Optional rejected candidate JSONL.")
    parser.add_argument("--annotated_jsonl", help="Optional all-candidates annotated JSONL.")
    parser.add_argument("--report_json")
    parser.add_argument("--text_fields", help="Comma-separated text fields to compare.")
    parser.add_argument("--ngram_size", type=int, default=5)
    parser.add_argument("--min_ngram_size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.ngram_size < 1:
        raise ValueError("--ngram_size must be positive.")
    if args.min_ngram_size < 1 or args.min_ngram_size > args.ngram_size:
        raise ValueError("--min_ngram_size must be positive and no larger than --ngram_size.")
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("--threshold must be in (0, 1].")

    text_fields = parse_fields(args.text_fields)
    candidates = read_jsonl(args.candidates_jsonl)
    references = build_reference_index(
        parse_reference_paths(args.references_jsonl),
        text_fields=text_fields,
        ngram_size=args.ngram_size,
        min_ngram_size=args.min_ngram_size,
    )
    accepted, rejected, report = annotate_candidates(
        candidates,
        references,
        text_fields=text_fields,
        ngram_size=args.ngram_size,
        min_ngram_size=args.min_ngram_size,
        threshold=args.threshold,
    )

    write_jsonl(args.output_jsonl, accepted)
    if args.rejected_jsonl:
        write_jsonl(args.rejected_jsonl, rejected)
    if args.annotated_jsonl:
        write_jsonl(args.annotated_jsonl, report["annotated_rows"])
    report.pop("annotated_rows", None)
    write_report(args.report_json, report)

    print(f"candidates={report['candidates']}")
    print(f"accepted={report['accepted']}")
    print(f"rejected={report['rejected']}")
    print(f"max_score={report['max_score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
