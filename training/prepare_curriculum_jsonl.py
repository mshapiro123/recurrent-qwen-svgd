"""Validate typed curriculum records and export safe causal-SFT JSONL rows.

The curriculum schema intentionally mixes positive reasoning traces with
contrastive negatives and verifier data. This converter is the safety boundary:
positive SFT rows are emitted only from roles that begin with ``positive_``.
Verifier and negative roles are counted in the report but never exported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VALID_MODES = {"direct", "deep_narrow", "wide", "both"}
POSITIVE_PREFIX = "positive_"
NON_POSITIVE_PREFIXES = ("negative_", "verifier_")


def qwen_instruct_prompt(statement: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{statement.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} is not a JSON object.")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def trace_role(trace: dict[str, Any]) -> str:
    return str(trace.get("role") or "").strip()


def is_positive_trace(trace: dict[str, Any], *, role_prefix: str = POSITIVE_PREFIX) -> bool:
    return trace_role(trace).startswith(role_prefix)


def validate_curriculum_record(record: dict[str, Any], *, line_no: int | None = None) -> list[str]:
    label = f"line {line_no}: " if line_no is not None else ""
    issues: list[str] = []
    if not str(record.get("id") or "").strip():
        issues.append(f"{label}missing id")
    if not str(record.get("statement") or record.get("prompt") or "").strip():
        issues.append(f"{label}missing statement/prompt")

    mode = str(record.get("mode") or "").strip()
    if mode not in VALID_MODES:
        issues.append(f"{label}invalid mode {mode!r}")

    answer = record.get("answer")
    if not isinstance(answer, dict) or not str(answer.get("value") or "").strip():
        issues.append(f"{label}missing verified answer.value")

    if record.get("decontaminated") is not True:
        issues.append(f"{label}decontaminated must be true")

    traces = record.get("traces")
    if not isinstance(traces, list) or not traces:
        issues.append(f"{label}traces must be a non-empty list")
        return issues

    positive_count = 0
    for trace_index, trace in enumerate(traces):
        trace_label = f"{label}trace {trace_index}: "
        if not isinstance(trace, dict):
            issues.append(f"{trace_label}trace is not an object")
            continue

        role = trace_role(trace)
        if not role:
            issues.append(f"{trace_label}missing role")
        elif not (role.startswith(POSITIVE_PREFIX) or role.startswith(NON_POSITIVE_PREFIXES)):
            issues.append(f"{trace_label}unknown role {role!r}")

        if role.startswith(POSITIVE_PREFIX):
            positive_count += 1
            if trace.get("correct") is not True:
                issues.append(f"{trace_label}positive trace must have correct=true")
            if not str(trace.get("text") or "").strip():
                issues.append(f"{trace_label}positive trace missing text")
            if trace.get("natural") is False:
                issues.append(f"{trace_label}positive trace marked natural=false")
        elif role in {"negative_contrastive", "verifier_rationalization"} and trace.get("correct") is True:
            issues.append(f"{trace_label}{role} must not have correct=true")

    if positive_count == 0:
        issues.append(f"{label}record has no positive traces")
    return issues


def inherited_target_loop(record: dict[str, Any], trace: dict[str, Any], *, default_direct_target_loop: int | None) -> int | None:
    if "target_loop_count" in trace:
        return int(trace["target_loop_count"])
    if "target_loop_count" in record:
        return int(record["target_loop_count"])
    if record.get("mode") == "direct" and default_direct_target_loop is not None:
        return int(default_direct_target_loop)
    return None


def positive_trace_to_causal_example(
    record: dict[str, Any],
    trace: dict[str, Any],
    *,
    prompt_style: str = "qwen_instruct",
    default_direct_target_loop: int | None = 1,
) -> dict[str, Any]:
    statement = str(record.get("prompt") or record.get("statement") or "")
    if prompt_style == "qwen_instruct":
        prompt = qwen_instruct_prompt(statement)
    elif prompt_style == "plain":
        prompt = statement
    else:
        raise ValueError(f"Unknown prompt_style={prompt_style!r}")

    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    example: dict[str, Any] = {
        "prompt": prompt,
        "completion": str(trace["text"]),
        "cot": str(trace["text"]),
        "source_dataset": record.get("source_dataset") or "curriculum",
        "category": record.get("domain"),
        "difficulty": (record.get("difficulty") or {}).get("pass_rate") if isinstance(record.get("difficulty"), dict) else record.get("difficulty"),
        "curriculum_id": record.get("id"),
        "curriculum_mode": record.get("mode"),
        "routing_type": record.get("mode"),
        "trace_role": trace_role(trace),
        "method": trace.get("method"),
        "source_model": trace.get("source_model"),
        "verified_answer": answer.get("value"),
    }
    if isinstance(trace.get("steps"), int):
        example["reasoning_steps"] = int(trace["steps"])

    target_loop = inherited_target_loop(
        record,
        trace,
        default_direct_target_loop=default_direct_target_loop,
    )
    if target_loop is not None:
        example["target_loop_count"] = target_loop
    return example


def convert_curriculum_records(
    records: list[dict[str, Any]],
    *,
    modes: set[str] | None = None,
    prompt_style: str = "qwen_instruct",
    default_direct_target_loop: int | None = 1,
    fail_on_validation: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    issues: list[str] = []
    role_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    exported_role_counts: Counter[str] = Counter()

    for line_no, record in enumerate(records, start=1):
        record_issues = validate_curriculum_record(record, line_no=line_no)
        if record_issues:
            issues.extend(record_issues)
            if fail_on_validation:
                continue

        mode = str(record.get("mode") or "")
        if modes is not None and mode not in modes:
            continue
        mode_counts[mode] += 1

        traces = record.get("traces")
        if not isinstance(traces, list):
            continue
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            role = trace_role(trace)
            role_counts[role] += 1
            if not is_positive_trace(trace):
                continue
            if trace.get("correct") is not True or not str(trace.get("text") or "").strip():
                continue
            example = positive_trace_to_causal_example(
                record,
                trace,
                prompt_style=prompt_style,
                default_direct_target_loop=default_direct_target_loop,
            )
            exported.append(example)
            exported_role_counts[role] += 1

    report = {
        "records": len(records),
        "exported_examples": len(exported),
        "issues": issues,
        "role_counts": dict(sorted(role_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "exported_role_counts": dict(sorted(exported_role_counts.items())),
    }
    if fail_on_validation and issues:
        raise ValueError("Curriculum validation failed:\n" + "\n".join(issues[:50]))
    return exported, report


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--modes", help="Optional comma-separated mode filter, e.g. direct,deep_narrow")
    parser.add_argument("--prompt_style", choices=("qwen_instruct", "plain"), default="qwen_instruct")
    parser.add_argument("--default_direct_target_loop", type=int, default=1)
    parser.add_argument("--allow_validation_issues", action="store_true")
    args = parser.parse_args(argv)

    records = read_jsonl(args.input_jsonl)
    examples, report = convert_curriculum_records(
        records,
        modes=parse_csv_set(args.modes),
        prompt_style=args.prompt_style,
        default_direct_target_loop=args.default_direct_target_loop,
        fail_on_validation=not args.allow_validation_issues,
    )
    write_jsonl(args.output_jsonl, examples)
    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"records={report['records']}")
    print(f"exported_examples={report['exported_examples']}")
    print(f"issues={len(report['issues'])}")
    print(f"role_counts={report['role_counts']}")
    print(f"exported_role_counts={report['exported_role_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
