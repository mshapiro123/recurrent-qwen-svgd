"""Collect capability-ladder trace job responses into scored rows with traces.

The capability-ladder MCQ probe creates answer-only scored rows. The trace job
builder asks external strong models to produce richer reasoning traces for those
same rows. This collector accepts only responses whose final ``ANSWER:`` line
matches the verified benchmark answer, then emits scored rows that can be fed to
``build_capability_ladder_curriculum.py`` without ``--allow_answer_only``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.build_capability_ladder_curriculum import answer_payload, answer_value, read_jsonl  # noqa: E402
from training.collect_curriculum_job_outputs import extract_answer, normalize_answer, response_text  # noqa: E402


TRACE_STAGE = "capability_ladder_trace_solve"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("task_id") or row.get("uid") or f"row-{index:06d}")


def response_by_job_id(responses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in responses:
        job_id = str(row.get("job_id") or "")
        if job_id:
            indexed[job_id] = row
    return indexed


def verified_answer_aliases(row: dict[str, Any]) -> set[str]:
    payload = answer_payload(row) or {}
    aliases = {answer_value(row), str(payload.get("value") or ""), str(payload.get("choice_text") or "")}
    return {normalize_answer(alias) for alias in aliases if str(alias).strip()}


def answer_match(row: dict[str, Any], text: str) -> dict[str, Any] | None:
    parsed = extract_answer(text)
    if not parsed:
        return None
    parsed_normalized = normalize_answer(parsed)
    aliases = verified_answer_aliases(row)
    if parsed_normalized not in aliases:
        return None
    value = answer_value(row)
    return {
        "matched": True,
        "source": "capability_ladder_trace_answer_line",
        "parsed_answer": parsed,
        "parsed_answer_normalized": parsed_normalized,
        "verified_answer": value,
        "verified_answer_normalized": normalize_answer(value),
        "accepted_aliases": sorted(aliases),
    }


def resolved_model_name(job: dict[str, Any], response: dict[str, Any]) -> str:
    return str(response.get("resolved_model") or response.get("model") or job.get("resolved_model") or job.get("model") or "")


def collect_trace_rows(
    scored_rows: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_id = {row_id(row, index): row for index, row in enumerate(scored_rows)}
    responses_index = response_by_job_id(responses)
    output_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    loop_counts: Counter[str] = Counter()
    issues: list[str] = []

    for job in jobs:
        if job.get("stage") != TRACE_STAGE:
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        record_id = str(metadata.get("record_id") or "")
        source_row = rows_by_id.get(record_id)
        if source_row is None:
            status_counts["missing_source_row"] += 1
            issues.append(f"{job.get('job_id')}: no scored row for record_id={record_id!r}")
            continue
        response = responses_index.get(str(job.get("job_id") or ""))
        if response is None:
            status_counts["missing_response"] += 1
            continue
        text = response_text(response)
        if not text:
            status_counts["empty_response"] += 1
            continue
        match = answer_match(source_row, text)
        if match is None:
            status_counts["wrong_or_missing_answer"] += 1
            continue

        solver_key = str(metadata.get("solver_key") or "")
        if not solver_key:
            status_counts["missing_solver_key"] += 1
            issues.append(f"{job.get('job_id')}: missing solver_key metadata")
            continue

        row = copy.deepcopy(source_row)
        row["id"] = f"{record_id}__trace__{job.get('job_id')}"
        row["trace_source_record_id"] = record_id
        model_results = row.setdefault("model_results", {})
        if not isinstance(model_results, dict):
            model_results = {}
            row["model_results"] = model_results
        solver_result = model_results.setdefault(solver_key, {})
        if not isinstance(solver_result, dict):
            solver_result = {}
            model_results[solver_key] = solver_result

        target_loop = int(metadata.get("target_loop_count") or 1)
        solver_result.update(
            {
                "correct": True,
                "solution": text,
                "trace": text,
                "answer_match": match,
                "source_model": resolved_model_name(job, response),
                "logical_source_model": job.get("model"),
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
                "steps": {1: 1, 2: 3, 3: 6}.get(target_loop, max(1, target_loop * 2)),
                "natural": True,
            }
        )
        row["capability_trace_job"] = {
            "job_id": job.get("job_id"),
            "response_id": response.get("response_id"),
            "target_loop_count": target_loop,
            "capability_tier": metadata.get("capability_tier"),
            "solver_key": solver_key,
            "model": job.get("model"),
            "resolved_model": resolved_model_name(job, response),
        }
        output_rows.append(row)
        status_counts["accepted"] += 1
        tier_counts[str(metadata.get("capability_tier") or "unknown")] += 1
        loop_counts[str(target_loop)] += 1

    report = {
        "kind": "capability_ladder_trace_collection",
        "status": "ready" if output_rows else "empty",
        "scored_rows": len(scored_rows),
        "jobs": len(jobs),
        "responses": len(responses),
        "accepted_rows": len(output_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "target_loop_counts": dict(sorted(loop_counts.items())),
        "issues": issues,
        "next_action": (
            "Run training/build_capability_ladder_curriculum.py on output_jsonl without --allow_answer_only."
            if output_rows
            else "Inspect response answers before building SFT data."
        ),
    }
    return output_rows, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_jsonl", required=True)
    parser.add_argument("--jobs_jsonl", required=True)
    parser.add_argument("--responses_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    args = parser.parse_args(argv)

    rows, report = collect_trace_rows(
        read_jsonl(args.scored_jsonl),
        read_jsonl(args.jobs_jsonl),
        read_jsonl(args.responses_jsonl),
    )
    write_jsonl(args.output_jsonl, rows)
    if args.report_json:
        write_json(args.report_json, report)
    print(f"status={report['status']}")
    print(f"accepted_rows={report['accepted_rows']}")
    print(f"status_counts={report['status_counts']}")
    print(f"target_loop_counts={report['target_loop_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
