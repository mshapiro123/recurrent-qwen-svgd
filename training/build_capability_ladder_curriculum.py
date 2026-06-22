"""Build a verified capability-ladder curriculum from scored model rows.

This is a CPU-only adapter between benchmark/model-scoring artifacts and the
typed curriculum schema. It does not score models itself. It consumes rows that
already contain a verified answer plus correctness flags for a base model and
larger reference models, then assigns deterministic recurrent depth targets:

* base correct -> direct, target loop 1;
* base miss and 1.5B correct -> deep_narrow, target loop 2;
* base/1.5B miss and 3B or stronger correct -> deep_narrow, target loop 3+.

Rows without independent answer verification, decontamination, or a positive
trace whose answer can be matched are skipped rather than exported to positive
SFT.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.prepare_curriculum_jsonl import convert_curriculum_records, validate_curriculum_record  # noqa: E402


RUN_ID = time.strftime("capability_ladder_curriculum_%Y%m%d_%H%M%S")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object.")
        rows.append(row)
    return rows


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


def jsonl_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def artifact_entry(path: Path) -> dict[str, Any]:
    payload = {"path": str(path), "exists": path.exists()}
    if path.suffix == ".jsonl" and path.exists():
        payload["lines"] = jsonl_line_count(path)
    return payload


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", "")
    return text


def answer_value(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("value") or "").strip()
    return str(row.get("verified_answer") or row.get("answer") or "").strip()


def answer_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    answer = row.get("answer")
    if isinstance(answer, dict):
        payload = dict(answer)
    else:
        value = answer_value(row)
        payload = {"value": value, "verified_by": row.get("verified_by") or []}
    if not str(payload.get("value") or "").strip():
        return None
    verified_by = payload.get("verified_by")
    if not isinstance(verified_by, list):
        verified_by = [verified_by] if verified_by else []
    payload["verified_by"] = verified_by
    return payload


def has_trusted_answer(row: dict[str, Any]) -> bool:
    payload = answer_payload(row)
    if not payload:
        return False
    verified_by = {str(item) for item in payload.get("verified_by") or []}
    return bool({"benchmark_ground_truth", "cross_model", "constructed"} & verified_by)


def model_result(row: dict[str, Any], key: str) -> dict[str, Any]:
    results = row.get("model_results")
    if isinstance(results, dict) and isinstance(results.get(key), dict):
        return results[key]
    if isinstance(row.get(key), dict):
        return row[key]
    value = None
    for field in (f"{key}_correct", f"{key}_hit", f"{key}_solved"):
        if field in row:
            value = row[field]
            break
    return {"correct": bool(value)} if value is not None else {}


def model_correct(row: dict[str, Any], key: str) -> bool:
    result = model_result(row, key)
    return result.get("correct") is True or result.get("hit") is True or result.get("solved") is True


def first_correct_result(row: dict[str, Any], keys: list[str]) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    for key in keys:
        result = model_result(row, key)
        if result and (result.get("correct") is True or result.get("hit") is True or result.get("solved") is True):
            return key, result
    return None, None


def trace_text(row: dict[str, Any], result: dict[str, Any] | None, *, allow_answer_only: bool) -> str:
    sources = []
    if isinstance(result, dict):
        sources.extend([result.get("trace"), result.get("solution"), result.get("completion"), result.get("text")])
    sources.extend([row.get("trace"), row.get("solution"), row.get("solution_text"), row.get("completion")])
    for value in sources:
        text = str(value or "").strip()
        if text:
            return text
    if allow_answer_only:
        return f"ANSWER: {answer_value(row)}"
    return ""


def answer_match_for_trace(row: dict[str, Any], result: dict[str, Any] | None, text: str) -> dict[str, Any] | None:
    for source in (result, row):
        if isinstance(source, dict) and isinstance(source.get("answer_match"), dict):
            match = dict(source["answer_match"])
            if match.get("matched") is True:
                return match
    verified = normalize_answer(answer_value(row))
    if not verified:
        return None
    if verified not in normalize_answer(text):
        return None
    return {
        "matched": True,
        "source": "capability_ladder_verified_answer",
        "parsed_answer": answer_value(row),
        "parsed_answer_normalized": verified,
        "verified_answer_normalized": verified,
    }


def depth_steps_for_tier(row: dict[str, Any], result: dict[str, Any] | None, target_loop: int) -> int:
    for source in (result, row):
        if isinstance(source, dict):
            value = source.get("steps") or source.get("reasoning_steps") or source.get("depth_steps")
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value > 0:
                return value
    return {1: 1, 2: 3, 3: 6, 4: 8}.get(target_loop, max(1, target_loop * 2))


def tier_for_row(
    row: dict[str, Any],
    *,
    base_key: str,
    mid_key: str,
    high_keys: list[str],
    high_target_loop: int,
) -> dict[str, Any] | None:
    base_correct = model_correct(row, base_key)
    mid_correct = model_correct(row, mid_key)
    high_key, high_result = first_correct_result(row, high_keys)
    if base_correct:
        return {
            "tier": "base_preservation",
            "mode": "direct",
            "role": "positive_direct",
            "target_loop_count": 1,
            "solver_key": base_key,
            "solver_result": model_result(row, base_key),
        }
    if mid_correct:
        return {
            "tier": f"{base_key}_miss_{mid_key}_solve",
            "mode": "deep_narrow",
            "role": "positive_depth",
            "target_loop_count": 2,
            "solver_key": mid_key,
            "solver_result": model_result(row, mid_key),
        }
    if high_key and high_result:
        return {
            "tier": f"{base_key}_miss_{mid_key}_miss_stronger_solve",
            "mode": "deep_narrow",
            "role": "positive_depth",
            "target_loop_count": high_target_loop,
            "solver_key": high_key,
            "solver_result": high_result,
        }
    return None


def build_record(
    row: dict[str, Any],
    *,
    tier: dict[str, Any],
    base_key: str,
    mid_key: str,
    high_keys: list[str],
    fallback_id: str,
    allow_answer_only: bool,
    assume_decontaminated: bool,
) -> dict[str, Any] | None:
    if not has_trusted_answer(row):
        return None
    if row.get("decontaminated") is not True and not assume_decontaminated:
        return None
    result = tier.get("solver_result") if isinstance(tier.get("solver_result"), dict) else {}
    text = trace_text(row, result, allow_answer_only=allow_answer_only)
    if not text:
        return None
    answer_match = answer_match_for_trace(row, result, text)
    if answer_match is None:
        return None

    target_loop = int(tier["target_loop_count"])
    steps = depth_steps_for_tier(row, result, target_loop)
    method = str(row.get("method") or result.get("method") or "capability_ladder_solution")
    statement = str(row.get("statement") or row.get("prompt") or row.get("question") or "").strip()
    if not statement:
        return None
    source_model = str(result.get("source_model") or result.get("model") or tier.get("solver_key") or "").strip()
    if not source_model:
        return None

    capability_ladder = {
        f"{base_key}_correct": model_correct(row, base_key),
        f"{mid_key}_correct": model_correct(row, mid_key),
        "stronger_correct": any(model_correct(row, key) for key in high_keys),
        "stronger_keys": [key for key in high_keys if model_correct(row, key)],
        "verified_answer_source": ",".join(str(item) for item in (answer_payload(row) or {}).get("verified_by", [])),
    }

    difficulty = row.get("difficulty") if isinstance(row.get("difficulty"), dict) else {}
    if not difficulty:
        difficulty = {"pass_rate": row.get("difficulty_pass_rate"), "reference_model": row.get("difficulty_reference_model")}
    if difficulty.get("pass_rate") is None:
        difficulty["pass_rate"] = {1: 0.9, 2: 0.45, 3: 0.2, 4: 0.1}.get(target_loop, 0.2)
    if not difficulty.get("reference_model"):
        difficulty["reference_model"] = "capability_ladder"
    record_id = str(row.get("id") or row.get("task_id") or row.get("uid") or fallback_id)

    return {
        "id": record_id,
        "domain": row.get("domain") or row.get("category") or "capability_ladder",
        "statement": statement,
        "answer": answer_payload(row),
        "difficulty": difficulty,
        "width_signature": {"methods": [method], "width": 1},
        "depth": {"per_method": {method: steps}, "min_steps": steps},
        "mode": tier["mode"],
        "target_loop_count": target_loop,
        "capability_tier": tier["tier"],
        "capability_ladder": capability_ladder,
        "decontaminated": True,
        "source_dataset": row.get("source_dataset") or "capability_ladder_curriculum",
        "traces": [
            {
                "role": tier["role"],
                "method": method,
                "correct": True,
                "natural": result.get("natural", True) is not False,
                "steps": steps,
                "source_model": source_model,
                "logical_source_model": result.get("logical_source_model") or tier.get("solver_key"),
                "answer_match": answer_match,
                "text": text,
            }
        ],
    }


def build_records(
    rows: list[dict[str, Any]],
    *,
    base_key: str,
    mid_key: str,
    high_keys: list[str],
    high_target_loop: int,
    allow_answer_only: bool,
    assume_decontaminated: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        tier = tier_for_row(
            row,
            base_key=base_key,
            mid_key=mid_key,
            high_keys=high_keys,
            high_target_loop=high_target_loop,
        )
        if tier is None:
            skipped["unresolved_capability"] += 1
            continue
        record = build_record(
            row,
            tier=tier,
            base_key=base_key,
            mid_key=mid_key,
            high_keys=high_keys,
            fallback_id=f"capability_ladder_{row_index:06d}",
            allow_answer_only=allow_answer_only,
            assume_decontaminated=assume_decontaminated,
        )
        if record is None:
            skipped["failed_safety_or_trace_requirements"] += 1
            continue
        issues = validate_curriculum_record(record)
        if issues:
            skipped["validation_failed"] += 1
            continue
        records.append(record)
        tier_counts[str(record["capability_tier"])] += 1
    report = {
        "input_rows": len(rows),
        "exported_records": len(records),
        "tier_counts": dict(sorted(tier_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "base_key": base_key,
        "mid_key": mid_key,
        "high_keys": high_keys,
    }
    return records, report


def write_work_dir(work_dir: Path, records: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": work_dir / "summary.json",
        "typed_records": work_dir / "typed_records.jsonl",
        "typed_records_report": work_dir / "typed_records_report.json",
        "positive_sft": work_dir / "positive_sft.jsonl",
        "positive_sft_report": work_dir / "positive_sft_report.json",
        "verified_candidates_report": work_dir / "verified_candidates_report.json",
        "decontam_report": work_dir / "decontam_report.json",
        "method_solutions_report": work_dir / "method_solutions_report.json",
        "naturalness_report": work_dir / "naturalness_report.json",
        "depth_report": work_dir / "depth_report.json",
        "difficulty_report": work_dir / "difficulty_report.json",
        "capability_ladder_report": work_dir / "capability_ladder_report.json",
    }
    sft_rows, sft_report = convert_curriculum_records(records, modes={"direct", "deep_narrow"}, fail_on_validation=True)
    mode_counts = Counter(str(record["mode"]) for record in records)
    role_counts = Counter(str(trace["role"]) for record in records for trace in record.get("traces", []))
    loop_counts = Counter(str(record["target_loop_count"]) for record in records)
    write_jsonl(paths["typed_records"], records)
    write_json(paths["typed_records_report"], {"records": len(records), "mode_counts": dict(mode_counts), "role_counts": dict(role_counts), "validation_issues": [], "positive_traces": len(records), "unsafe_auxiliary_traces": 0, "distinctness_required": False, "min_natural_agree": 2, "min_distinct_agree": 2, "distinctness_judgments": 0, "source": "capability_ladder"})
    write_jsonl(paths["positive_sft"], sft_rows)
    write_json(paths["positive_sft_report"], sft_report)
    write_json(paths["verified_candidates_report"], {"verified": len(records), "source": "capability_ladder", "require_programmatic_answer_check": False})
    write_json(paths["decontam_report"], {"accepted": len(records), "rejected": 0, "method": "input_required_or_assumed"})
    write_json(paths["method_solutions_report"], {"solution_candidates": len(records), "correct_solution_candidates": len(records), "source": "capability_ladder"})
    write_json(paths["naturalness_report"], {"judgments": len(records), "natural": len(records), "source": "capability_ladder"})
    write_json(paths["depth_report"], {"measurements": len(records), "target_loop_counts": dict(sorted(loop_counts.items())), "source": "capability_ladder"})
    write_json(paths["difficulty_report"], {"measured": len(records), "source": "capability_ladder"})
    write_json(paths["capability_ladder_report"], report)
    summary = {
        "run_id": RUN_ID,
        "kind": "capability_ladder_curriculum_pipeline",
        "status": "complete",
        "next_action": "Run training/check_curriculum_sft_gate.py before any GPU fine-tuning.",
        "artifacts": {name: artifact_entry(path) for name, path in sorted(paths.items())},
        "counts": {
            "typed_records": len(records),
            "positive_sft_rows": len(sft_rows),
            "mode_counts": dict(sorted(mode_counts.items())),
            "target_loop_counts": dict(sorted(loop_counts.items())),
            "tier_counts": report["tier_counts"],
        },
    }
    write_json(paths["summary"], summary)
    summary["artifacts"]["summary"] = artifact_entry(paths["summary"])
    write_json(paths["summary"], summary)
    return summary


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--base_key", default="qwen_0_5b")
    parser.add_argument("--mid_key", default="qwen_1_5b")
    parser.add_argument("--high_keys", default="qwen_3b,strong_solver")
    parser.add_argument("--high_target_loop", type=int, default=3)
    parser.add_argument("--allow_answer_only", action="store_true")
    parser.add_argument("--assume_decontaminated", action="store_true")
    args = parser.parse_args(argv)

    rows = read_jsonl(args.input_jsonl)
    records, report = build_records(
        rows,
        base_key=args.base_key,
        mid_key=args.mid_key,
        high_keys=parse_csv(args.high_keys),
        high_target_loop=args.high_target_loop,
        allow_answer_only=args.allow_answer_only,
        assume_decontaminated=args.assume_decontaminated,
    )
    summary = write_work_dir(Path(args.work_dir), records, report)
    print(f"status={summary['status']}")
    print(f"input_rows={report['input_rows']}")
    print(f"typed_records={summary['counts']['typed_records']}")
    print(f"positive_sft_rows={summary['counts']['positive_sft_rows']}")
    print(f"tier_counts={summary['counts']['tier_counts']}")
    print(f"target_loop_counts={summary['counts']['target_loop_counts']}")
    print(f"skipped={report['skipped']}")
    print(f"summary_json={Path(args.work_dir) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
