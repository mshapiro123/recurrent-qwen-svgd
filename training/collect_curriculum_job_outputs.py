"""Collect provider-neutral curriculum job outputs into verified candidates.

This script consumes JSONL jobs from ``build_curriculum_generation_jobs.py`` and
separate JSONL response records produced by an external API runner. It performs
cheap deterministic parsing and conservative cross-model agreement checks. It
does not trust any model's self-labels as training labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


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


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def response_text(row: dict[str, Any]) -> str:
    for key in ("response_text", "response", "output_text", "output", "text", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    message = row.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    return ""


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates: list[str] = []
    candidates.extend(match.group(1) for match in FENCE_RE.finditer(text))
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(".;")
    normalized = normalized.replace(",", "")
    return normalized


def index_jobs(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        if not job_id:
            raise ValueError("Job row missing job_id.")
        indexed[job_id] = job
    return indexed


def responses_with_jobs(
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], list[str]]:
    job_by_id = index_jobs(jobs)
    paired: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    issues: list[str] = []
    for index, response in enumerate(responses):
        job_id = str(response.get("job_id") or "")
        if job_id not in job_by_id:
            issues.append(f"response {index}: unknown job_id {job_id!r}")
            continue
        text = response_text(response)
        if not text:
            issues.append(f"response {index}: missing response text")
            continue
        paired.append((job_by_id[job_id], response, text))
    return paired, issues


def candidate_id(job: dict[str, Any]) -> str:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    parts = [
        "candidate",
        str(metadata.get("domain") or "unknown"),
        str(metadata.get("difficulty") or "unknown"),
        str(metadata.get("target_steps") or "steps"),
        str(job.get("job_id") or "job"),
    ]
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", "-".join(parts)).strip("-").lower()


def seed_outputs_to_candidates(
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    *,
    mark_decontaminated: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired, issues = responses_with_jobs(jobs, responses)
    candidates: list[dict[str, Any]] = []
    for job, response, text in paired:
        if job.get("stage") != "seed_generation":
            continue
        payload = extract_json_object(text)
        if payload is None:
            issues.append(f"{job.get('job_id')}: could not parse seed JSON")
            continue
        statement = str(payload.get("statement") or "").strip()
        claimed_answer = str(payload.get("claimed_answer") or "").strip()
        if not statement or not claimed_answer:
            issues.append(f"{job.get('job_id')}: seed JSON missing statement or claimed_answer")
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        methods = payload.get("candidate_methods") if isinstance(payload.get("candidate_methods"), list) else []
        candidates.append(
            {
                "id": candidate_id(job),
                "domain": str(payload.get("domain") or metadata.get("domain") or "unknown"),
                "statement": statement,
                "claimed_answer": claimed_answer,
                "candidate_methods": [str(method) for method in methods if str(method).strip()],
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
                "generator_model": job.get("model"),
                "target_steps_hint": metadata.get("target_steps"),
                "difficulty_hint": metadata.get("difficulty"),
                "decontaminated": bool(mark_decontaminated),
            }
        )
    return candidates, {
        "mode": "seed_candidates",
        "jobs": len(jobs),
        "responses": len(responses),
        "candidates": len(candidates),
        "issues": issues,
    }


def group_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(candidates):
        key = str(row.get("id") or row.get("problem_id") or f"row-{index:06d}")
        grouped[key] = row
    return grouped


def ground_truth_outputs_to_verified_candidates(
    candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    *,
    min_agree: int = 2,
    mark_decontaminated: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_by_id = group_candidates(candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    answers_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for job, response, text in paired:
        if job.get("stage") != "ground_truth_solve":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        record_id = str(metadata.get("record_id") or "")
        if not record_id:
            issues.append(f"{job.get('job_id')}: missing record_id")
            continue
        answer = extract_answer(text)
        if not answer:
            issues.append(f"{job.get('job_id')}: missing ANSWER line")
            continue
        answers_by_record[record_id].append(
            {
                "job_id": job.get("job_id"),
                "model": job.get("model"),
                "answer": answer,
                "normalized": normalize_answer(answer),
                "response_id": response.get("response_id"),
            }
        )

    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record_id, answers in sorted(answers_by_record.items()):
        candidate = candidate_by_id.get(record_id)
        if candidate is None:
            issues.append(f"{record_id}: no matching candidate")
            continue
        by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for answer in answers:
            by_answer[answer["normalized"]].append(answer)
        normalized, agreeing = max(by_answer.items(), key=lambda item: (len(item[1]), item[0]))
        agreeing_models = sorted({str(item["model"]) for item in agreeing})
        if len(agreeing_models) >= min_agree:
            accepted = dict(candidate)
            accepted["answer"] = {
                "value": agreeing[0]["answer"],
                "normalized": normalized,
                "verified_by": ["cross_model"],
                "confidence": "high",
                "agreeing_models": agreeing_models,
                "agreement_count": len(agreeing),
            }
            accepted["ground_truth_solutions"] = answers
            accepted["decontaminated"] = bool(candidate.get("decontaminated") or mark_decontaminated)
            verified.append(accepted)
        else:
            rejected.append(
                {
                    "id": record_id,
                    "reason": "insufficient_cross_model_agreement",
                    "answers": answers,
                    "answer_counts": dict(Counter(answer["normalized"] for answer in answers)),
                }
            )

    return verified, {
        "mode": "verified_candidates",
        "candidate_rows": len(candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "verified": len(verified),
        "rejected": len(rejected),
        "min_agree": min_agree,
        "issues": issues,
        "rejected_records": rejected,
    }


def write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("seed_candidates", "verified_candidates"), required=True)
    parser.add_argument("--jobs_jsonl", required=True)
    parser.add_argument("--responses_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--candidates_jsonl", help="Required for --mode verified_candidates.")
    parser.add_argument("--min_agree", type=int, default=2)
    parser.add_argument("--mark_decontaminated", action="store_true")
    args = parser.parse_args(argv)

    jobs = read_jsonl(args.jobs_jsonl)
    responses = read_jsonl(args.responses_jsonl)
    if args.mode == "seed_candidates":
        rows, report = seed_outputs_to_candidates(
            jobs,
            responses,
            mark_decontaminated=args.mark_decontaminated,
        )
    else:
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for verified_candidates mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = ground_truth_outputs_to_verified_candidates(
            candidates,
            jobs,
            responses,
            min_agree=args.min_agree,
            mark_decontaminated=args.mark_decontaminated,
        )

    write_jsonl(args.output_jsonl, rows)
    write_report(args.report_json, report)
    print(f"mode={report['mode']}")
    for key in ("candidates", "verified", "rejected"):
        if key in report:
            print(f"{key}={report[key]}")
    print(f"issues={len(report['issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

