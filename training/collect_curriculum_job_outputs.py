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
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
METHOD_DOES_NOT_APPLY_RE = re.compile(r"^\s*METHOD DOES NOT APPLY\b", re.IGNORECASE)
NUMERIC_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
)


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


def parse_numeric_answer(answer: str) -> Fraction | None:
    """Extract a simple numeric value from an answer string.

    This intentionally handles only cheap, auditable cases: integers, decimals,
    fractions, optional commas, and percentages. It is a consistency check, not
    a general theorem prover.
    """

    match = NUMERIC_RE.search(answer.strip())
    if match is None:
        return None
    token = match.group(0).replace(",", "").replace(" ", "")
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            value = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        else:
            value = Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return value / 100 if is_percent else value


def answer_consistency_check(accepted_answer: str, claimed_answer: Any) -> dict[str, Any]:
    claimed = str(claimed_answer or "").strip()
    accepted = str(accepted_answer or "").strip()
    if not claimed or not accepted:
        return {"checked": False, "matched": None, "method": None, "reason": "missing_answer"}

    accepted_normalized = normalize_answer(accepted)
    claimed_normalized = normalize_answer(claimed)
    if accepted_normalized == claimed_normalized:
        return {
            "checked": True,
            "matched": True,
            "method": "normalized_exact",
            "accepted": accepted,
            "claimed": claimed,
        }

    accepted_numeric = parse_numeric_answer(accepted)
    claimed_numeric = parse_numeric_answer(claimed)
    if accepted_numeric is None or claimed_numeric is None:
        return {
            "checked": False,
            "matched": None,
            "method": None,
            "reason": "not_simple_numeric",
            "accepted": accepted,
            "claimed": claimed,
        }

    return {
        "checked": True,
        "matched": accepted_numeric == claimed_numeric,
        "method": "simple_numeric",
        "accepted": accepted,
        "claimed": claimed,
        "accepted_numeric": str(accepted_numeric),
        "claimed_numeric": str(claimed_numeric),
    }


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
    require_claimed_answer_match: bool = False,
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
    programmatic_check_counts: Counter[str] = Counter()
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
            claim_check = answer_consistency_check(agreeing[0]["answer"], candidate.get("claimed_answer"))
            if claim_check["checked"] and claim_check["matched"] is True:
                programmatic_check_counts[f"{claim_check['method']}_matched"] += 1
            elif claim_check["checked"] and claim_check["matched"] is False:
                programmatic_check_counts[f"{claim_check['method']}_mismatched"] += 1
                if require_claimed_answer_match:
                    rejected.append(
                        {
                            "id": record_id,
                            "reason": "claimed_answer_programmatic_mismatch",
                            "answers": answers,
                            "claim_check": claim_check,
                        }
                    )
                    continue
            else:
                programmatic_check_counts[str(claim_check.get("reason") or "not_checked")] += 1
                if require_claimed_answer_match:
                    rejected.append(
                        {
                            "id": record_id,
                            "reason": "claimed_answer_programmatic_check_unavailable",
                            "answers": answers,
                            "claim_check": claim_check,
                        }
                    )
                    continue

            accepted = dict(candidate)
            accepted["answer"] = {
                "value": agreeing[0]["answer"],
                "normalized": normalized,
                "verified_by": ["cross_model"],
                "confidence": "high",
                "agreeing_models": agreeing_models,
                "agreement_count": len(agreeing),
                "programmatic_check": claim_check,
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
        "require_claimed_answer_match": require_claimed_answer_match,
        "programmatic_check_counts": dict(sorted(programmatic_check_counts.items())),
        "issues": issues,
        "rejected_records": rejected,
    }


def verified_answer_normalized(candidate: dict[str, Any]) -> str:
    answer = candidate.get("answer")
    if not isinstance(answer, dict):
        return ""
    normalized = str(answer.get("normalized") or "").strip()
    if normalized:
        return normalized
    value = str(answer.get("value") or "").strip()
    return normalize_answer(value) if value else ""


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value).strip("-").lower()


def candidate_statement(candidate: dict[str, Any]) -> str:
    return str(candidate.get("statement") or candidate.get("prompt") or "")


def method_outputs_to_solution_candidates(
    verified_candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect correct method-constrained solutions for naturalness/depth jobs.

    The output intentionally includes only responses that reached the verified
    answer. Inapplicable methods and wrong/missing answers are preserved in the
    report, not forwarded to the next positive-data stage.
    """

    candidate_by_id = group_candidates(verified_candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for job, response, text in paired:
        if job.get("stage") != "method_constrained_solve":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        record_id = str(metadata.get("record_id") or "")
        method = str(metadata.get("method") or "").strip()
        candidate = candidate_by_id.get(record_id)
        if not record_id or candidate is None:
            issues.append(f"{job.get('job_id')}: no matching verified candidate for {record_id!r}")
            status_counts["missing_candidate"] += 1
            continue
        if not method:
            issues.append(f"{job.get('job_id')}: missing method metadata")
            status_counts["missing_method"] += 1
            continue

        if METHOD_DOES_NOT_APPLY_RE.search(text):
            status = "method_does_not_apply"
            status_counts[status] += 1
            rejected.append(
                {
                    "job_id": job.get("job_id"),
                    "record_id": record_id,
                    "method": method,
                    "status": status,
                    "model": job.get("model"),
                }
            )
            continue

        parsed_answer = extract_answer(text)
        if not parsed_answer:
            status = "missing_answer"
            status_counts[status] += 1
            rejected.append(
                {
                    "job_id": job.get("job_id"),
                    "record_id": record_id,
                    "method": method,
                    "status": status,
                    "model": job.get("model"),
                }
            )
            continue

        normalized = normalize_answer(parsed_answer)
        verified = verified_answer_normalized(candidate)
        if not verified or normalized != verified:
            status = "wrong_answer"
            status_counts[status] += 1
            rejected.append(
                {
                    "job_id": job.get("job_id"),
                    "record_id": record_id,
                    "method": method,
                    "status": status,
                    "model": job.get("model"),
                    "parsed_answer": parsed_answer,
                    "parsed_answer_normalized": normalized,
                    "verified_answer_normalized": verified,
                }
            )
            continue

        status_counts["correct_answer"] += 1
        answer = candidate.get("answer") if isinstance(candidate.get("answer"), dict) else {}
        rows.append(
            {
                "id": sanitize_id(f"{record_id}:{method}:{job.get('job_id')}"),
                "record_id": record_id,
                "domain": candidate.get("domain"),
                "statement": candidate.get("statement"),
                "method": method,
                "source_model": job.get("model"),
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
                "text": text,
                "solution": text,
                "answer": answer,
                "parsed_answer": parsed_answer,
                "parsed_answer_normalized": normalized,
                "correct": True,
                "natural": None,
                "candidate_methods": candidate.get("candidate_methods"),
            }
        )

    return rows, {
        "mode": "method_solutions",
        "verified_candidate_rows": len(verified_candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "solution_candidates": len(rows),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
        "rejected_records": rejected,
    }


def reference_outputs_to_attempts(
    verified_candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_by_id = group_candidates(verified_candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for job, response, text in paired:
        if job.get("stage") != "reference_attempt":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        record_id = str(metadata.get("record_id") or "")
        candidate = candidate_by_id.get(record_id)
        if not record_id or candidate is None:
            issues.append(f"{job.get('job_id')}: no matching verified candidate for {record_id!r}")
            status_counts["missing_candidate"] += 1
            continue
        parsed_answer = extract_answer(text)
        parsed_normalized = normalize_answer(parsed_answer) if parsed_answer else ""
        verified = verified_answer_normalized(candidate)
        correct = bool(parsed_normalized and verified and parsed_normalized == verified)
        if correct:
            status_counts["correct"] += 1
        elif parsed_answer:
            status_counts["wrong_answer"] += 1
        else:
            status_counts["missing_answer"] += 1

        rows.append(
            {
                "record_id": record_id,
                "sample_id": metadata.get("sample_id"),
                "model": job.get("model"),
                "correct": correct,
                "parsed_answer": parsed_answer,
                "parsed_answer_normalized": parsed_normalized,
                "verified_answer_normalized": verified,
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
                "text": text,
            }
        )

    return rows, {
        "mode": "reference_attempts",
        "verified_candidate_rows": len(verified_candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "attempts": len(rows),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
    }


def perturbation_outputs_to_traces(
    verified_candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_by_id = group_candidates(verified_candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for job, response, text in paired:
        stage = str(job.get("stage") or "")
        if stage not in {"false_answer_neutral", "false_answer_pressure"}:
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        record_id = str(metadata.get("record_id") or "")
        candidate = candidate_by_id.get(record_id)
        if candidate is None:
            issues.append(f"{job.get('job_id')}: no matching verified candidate for {record_id!r}")
            status_counts["missing_candidate"] += 1
            continue

        verified = verified_answer_normalized(candidate)
        parsed_answer = extract_answer(text)
        parsed_normalized = normalize_answer(parsed_answer) if parsed_answer else ""
        false_answer = str(metadata.get("false_answer") or candidate.get("false_answer") or "").strip()
        false_normalized = normalize_answer(false_answer) if false_answer else ""

        if parsed_normalized and verified and parsed_normalized == verified:
            role = "verifier_detection"
            correct = True
            status = "detected_or_resisted_falsehood"
            detected = True
            error_type = None
        elif stage == "false_answer_pressure" or (false_normalized and parsed_normalized == false_normalized):
            role = "verifier_rationalization"
            correct = False
            status = "rationalized_false_answer"
            detected = False
            error_type = "rationalization"
        else:
            role = "negative_contrastive"
            correct = False
            status = "wrong_or_unverified_answer"
            detected = False
            error_type = "genuine_slip" if parsed_normalized else "missing_answer"

        status_counts[status] += 1
        row: dict[str, Any] = {
            "id": sanitize_id(f"{record_id}:{stage}:{job.get('job_id')}"),
            "record_id": record_id,
            "statement": candidate_statement(candidate),
            "role": role,
            "correct": correct,
            "detected": detected,
            "error_type": error_type,
            "injected": "false_answer",
            "false_answer": false_answer,
            "parsed_answer": parsed_answer,
            "parsed_answer_normalized": parsed_normalized,
            "verified_answer_normalized": verified,
            "source_model": job.get("model"),
            "source_job_id": job.get("job_id"),
            "source_response_id": response.get("response_id"),
            "text": text,
        }
        rows.append(row)

    return rows, {
        "mode": "perturbation_traces",
        "verified_candidate_rows": len(verified_candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "traces": len(rows),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
    }


def collect_error_detection_judgments(
    trace_candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_by_id = group_candidates(trace_candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for job, response, text in paired:
        if job.get("stage") != "error_detection":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        source_trace_id = str(metadata.get("record_id") or "")
        source = trace_by_id.get(source_trace_id)
        if source is None:
            issues.append(f"{job.get('job_id')}: no matching trace candidate for {source_trace_id!r}")
            status_counts["missing_trace"] += 1
            continue
        payload = extract_json_object(text)
        if payload is None:
            issues.append(f"{job.get('job_id')}: could not parse error-detection JSON")
            status_counts["parse_error"] += 1
            continue
        verdict = str(payload.get("verdict") or "").strip().lower()
        if verdict not in {"correct", "incorrect"}:
            issues.append(f"{job.get('job_id')}: invalid error-detection verdict {verdict!r}")
            status_counts["invalid_payload"] += 1
            continue

        status_counts[f"verdict_{verdict}"] += 1
        parent_record_id = str(source.get("record_id") or metadata.get("parent_record_id") or "")
        rows.append(
            {
                "id": sanitize_id(f"{source_trace_id}:error_detection:{job.get('job_id')}"),
                "record_id": parent_record_id,
                "source_trace_id": source_trace_id,
                "role": "verifier_detection",
                "detected": verdict == "incorrect",
                "verdict": verdict,
                "first_error_step": payload.get("first_error_step"),
                "explanation": str(payload.get("explanation") or "").strip(),
                "correct": verdict == "correct",
                "source_model": job.get("model"),
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
                "text": text,
            }
        )

    return rows, {
        "mode": "error_detection_judgments",
        "trace_candidate_rows": len(trace_candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "judgments": len(rows),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
    }


def collect_naturalness_judgments(
    solution_candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solution_by_id = group_candidates(solution_candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for job, response, text in paired:
        if job.get("stage") != "naturalness_judge":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        solution_id = str(metadata.get("record_id") or "")
        solution = solution_by_id.get(solution_id)
        if solution is None:
            issues.append(f"{job.get('job_id')}: no matching solution candidate for {solution_id!r}")
            status_counts["missing_solution"] += 1
            continue
        payload = extract_json_object(text)
        if payload is None:
            issues.append(f"{job.get('job_id')}: could not parse naturalness JSON")
            status_counts["parse_error"] += 1
            continue
        natural = payload.get("natural")
        if not isinstance(natural, bool):
            issues.append(f"{job.get('job_id')}: naturalness JSON missing boolean natural")
            status_counts["invalid_payload"] += 1
            continue
        status_counts["natural_true" if natural else "natural_false"] += 1
        rows.append(
            {
                "solution_id": solution_id,
                "record_id": solution.get("record_id"),
                "method": solution.get("method"),
                "judge_model": job.get("model"),
                "natural": natural,
                "actually_uses": str(payload.get("actually_uses") or "").strip(),
                "reason": str(payload.get("reason") or "").strip(),
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
            }
        )

    return rows, {
        "mode": "naturalness_judgments",
        "solution_candidate_rows": len(solution_candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "judgments": len(rows),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
    }


def collect_distinctness_judgments(
    solution_candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solution_by_id = group_candidates(solution_candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for job, response, text in paired:
        if job.get("stage") != "method_distinctness_judge":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        solution_a_id = str(metadata.get("solution_a_id") or "")
        solution_b_id = str(metadata.get("solution_b_id") or "")
        if solution_a_id not in solution_by_id or solution_b_id not in solution_by_id:
            issues.append(
                f"{job.get('job_id')}: no matching solution candidate pair "
                f"for {solution_a_id!r}, {solution_b_id!r}"
            )
            status_counts["missing_solution"] += 1
            continue
        payload = extract_json_object(text)
        if payload is None:
            issues.append(f"{job.get('job_id')}: could not parse distinctness JSON")
            status_counts["parse_error"] += 1
            continue
        distinct = payload.get("distinct")
        if not isinstance(distinct, bool):
            issues.append(f"{job.get('job_id')}: distinctness JSON missing boolean distinct")
            status_counts["invalid_payload"] += 1
            continue
        status_counts["distinct_true" if distinct else "distinct_false"] += 1
        rows.append(
            {
                "record_id": str(metadata.get("record_id") or ""),
                "solution_a_id": solution_a_id,
                "solution_b_id": solution_b_id,
                "method_a": str(metadata.get("method_a") or ""),
                "method_b": str(metadata.get("method_b") or ""),
                "judge_model": job.get("model"),
                "distinct": distinct,
                "reason": str(payload.get("reason") or "").strip(),
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
            }
        )

    return rows, {
        "mode": "distinctness_judgments",
        "solution_candidate_rows": len(solution_candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "judgments": len(rows),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
    }


def collect_depth_measurements(
    solution_candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solution_by_id = group_candidates(solution_candidates)
    paired, issues = responses_with_jobs(jobs, responses)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for job, response, text in paired:
        if job.get("stage") != "depth_decomposition":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        solution_id = str(metadata.get("record_id") or "")
        solution = solution_by_id.get(solution_id)
        if solution is None:
            issues.append(f"{job.get('job_id')}: no matching solution candidate for {solution_id!r}")
            status_counts["missing_solution"] += 1
            continue
        payload = extract_json_object(text)
        if payload is None:
            issues.append(f"{job.get('job_id')}: could not parse depth JSON")
            status_counts["parse_error"] += 1
            continue
        count = payload.get("count")
        steps = payload.get("steps")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            issues.append(f"{job.get('job_id')}: depth JSON missing positive integer count")
            status_counts["invalid_payload"] += 1
            continue
        if not isinstance(steps, list) or not all(str(step).strip() for step in steps):
            issues.append(f"{job.get('job_id')}: depth JSON missing non-empty steps list")
            status_counts["invalid_payload"] += 1
            continue
        status_counts["valid_depth"] += 1
        rows.append(
            {
                "solution_id": solution_id,
                "record_id": solution.get("record_id"),
                "method": solution.get("method"),
                "judge_model": job.get("model"),
                "steps": [str(step) for step in steps],
                "count": count,
                "source_job_id": job.get("job_id"),
                "source_response_id": response.get("response_id"),
            }
        )

    return rows, {
        "mode": "depth_measurements",
        "solution_candidate_rows": len(solution_candidates),
        "jobs": len(jobs),
        "responses": len(responses),
        "measurements": len(rows),
        "issues": issues,
        "status_counts": dict(sorted(status_counts.items())),
    }


def write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "seed_candidates",
            "verified_candidates",
            "method_solutions",
            "reference_attempts",
            "perturbation_traces",
            "naturalness_judgments",
            "distinctness_judgments",
            "depth_measurements",
            "error_detection_judgments",
        ),
        required=True,
    )
    parser.add_argument("--jobs_jsonl", required=True)
    parser.add_argument("--responses_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument(
        "--candidates_jsonl",
        help=(
            "Required for verified_candidates, method_solutions, reference_attempts, naturalness_judgments, "
            "distinctness_judgments, depth_measurements, perturbation_traces, "
            "and error_detection_judgments modes."
        ),
    )
    parser.add_argument("--min_agree", type=int, default=2)
    parser.add_argument("--mark_decontaminated", action="store_true")
    parser.add_argument(
        "--require_claimed_answer_match",
        action="store_true",
        help="Reject verified candidates when the accepted answer cannot be matched to the seed claimed_answer by a cheap exact/numeric check.",
    )
    args = parser.parse_args(argv)

    jobs = read_jsonl(args.jobs_jsonl)
    responses = read_jsonl(args.responses_jsonl)
    if args.mode == "seed_candidates":
        rows, report = seed_outputs_to_candidates(
            jobs,
            responses,
            mark_decontaminated=args.mark_decontaminated,
        )
    elif args.mode == "verified_candidates":
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for verified_candidates mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = ground_truth_outputs_to_verified_candidates(
            candidates,
            jobs,
            responses,
            min_agree=args.min_agree,
            mark_decontaminated=args.mark_decontaminated,
            require_claimed_answer_match=args.require_claimed_answer_match,
        )
    elif args.mode == "method_solutions":
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for method_solutions mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = method_outputs_to_solution_candidates(candidates, jobs, responses)
    elif args.mode == "reference_attempts":
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for reference_attempts mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = reference_outputs_to_attempts(candidates, jobs, responses)
    elif args.mode == "perturbation_traces":
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for perturbation_traces mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = perturbation_outputs_to_traces(candidates, jobs, responses)
    elif args.mode == "naturalness_judgments":
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for naturalness_judgments mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = collect_naturalness_judgments(candidates, jobs, responses)
    elif args.mode == "distinctness_judgments":
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for distinctness_judgments mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = collect_distinctness_judgments(candidates, jobs, responses)
    elif args.mode == "error_detection_judgments":
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for error_detection_judgments mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = collect_error_detection_judgments(candidates, jobs, responses)
    else:
        if not args.candidates_jsonl:
            raise ValueError("--candidates_jsonl is required for depth_measurements mode.")
        candidates = read_jsonl(args.candidates_jsonl)
        rows, report = collect_depth_measurements(candidates, jobs, responses)

    write_jsonl(args.output_jsonl, rows)
    write_report(args.report_json, report)
    print(f"mode={report['mode']}")
    for key in (
        "candidates",
        "verified",
        "solution_candidates",
        "attempts",
        "traces",
        "judgments",
        "measurements",
        "rejected",
    ):
        if key in report:
            print(f"{key}={report[key]}")
    print(f"issues={len(report['issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
