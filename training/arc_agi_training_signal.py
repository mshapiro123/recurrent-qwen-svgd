"""Profile ARC-AGI recurrent training JSONL signal.

This is a cheap audit layer for Stage 5 runs. It does not score the model.
Instead, it answers what the next SFT run actually trained on: grid-only rows,
symbolic traces, program-style traces, synthetic ARC families, and candidate
distillation examples.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SYNTHETIC_TASK_RE = re.compile(r"^synthetic_(.+)_\d{6}(?::.*)?$")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def task_family(task_id: Any) -> str:
    match = SYNTHETIC_TASK_RE.match(str(task_id or ""))
    return match.group(1) if match else "arc"


def count_by(rows: list[dict[str, Any]], key: str, *, missing: str = "unknown") -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        label = missing if value is None or value == "" else str(value)
        counter[label] += 1
    return dict(sorted(counter.items()))


def length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)

    def percentile(pct: float) -> int:
        if len(ordered) == 1:
            return ordered[0]
        idx = round((len(ordered) - 1) * pct)
        return ordered[idx]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def has_think_trace(row: dict[str, Any]) -> bool:
    completion = str(row.get("completion", ""))
    return bool(row.get("trace_source")) or "<think>" in completion.lower()


def has_program_trace(row: dict[str, Any]) -> bool:
    completion = str(row.get("completion", ""))
    return row.get("trace_mode") == "symbolic_program" or "program:" in completion


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = [task_family(row.get("task_id")) for row in rows]
    trace_rows = [row for row in rows if has_think_trace(row)]
    program_trace_rows = [row for row in rows if has_program_trace(row)]
    candidate_distill_rows = [
        row for row in rows if str(row.get("source_dataset", "")) == "arc-agi-candidate-distill"
    ]
    synthetic_rows = [row for row, family in zip(rows, families) if family != "arc"]
    public_arc_rows = [row for row, family in zip(rows, families) if family == "arc"]
    return {
        "rows": len(rows),
        "source_dataset_counts": count_by(rows, "source_dataset"),
        "category_counts": count_by(rows, "category"),
        "trace_mode_counts": count_by(rows, "trace_mode", missing="none"),
        "trace_source_counts": count_by(rows, "trace_source", missing="none"),
        "task_family_counts": dict(sorted(Counter(families).items())),
        "candidate_distill_rows": len(candidate_distill_rows),
        "synthetic_rows": len(synthetic_rows),
        "public_arc_rows": len(public_arc_rows),
        "trace_rows": len(trace_rows),
        "program_trace_rows": len(program_trace_rows),
        "grid_only_rows": len(rows) - len(trace_rows),
        "trace_row_fraction": len(trace_rows) / max(len(rows), 1),
        "program_trace_row_fraction": len(program_trace_rows) / max(len(rows), 1),
        "prompt_chars": length_stats([len(str(row.get("prompt", ""))) for row in rows]),
        "completion_chars": length_stats([len(str(row.get("completion", ""))) for row in rows]),
        "total_chars": length_stats(
            [len(str(row.get("prompt", ""))) + len(str(row.get("completion", ""))) for row in rows]
        ),
        "cot_tokens": length_stats([int(row.get("cot_tokens", 0) or 0) for row in rows]),
        "top_task_ids": dict(Counter(str(row.get("task_id", "unknown")) for row in rows).most_common(10)),
    }


def _metadata_bool(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return str(value or "").strip().lower() not in {"", "0", "false", "none", "[]"}


def warnings_for_summary(summary: dict[str, Any], metadata: dict[str, Any] | None = None) -> list[str]:
    metadata = metadata or {}
    warnings: list[str] = []
    if summary["rows"] == 0:
        warnings.append("No training rows were found.")
        return warnings
    expected_trace_mode = str(metadata.get("trace_mode", "none"))
    expected_trace_filter = str(metadata.get("trace_filter", "all"))
    if expected_trace_mode in {"symbolic", "symbolic_program"} and summary["trace_rows"] == 0:
        warnings.append(f"Trace mode `{expected_trace_mode}` was requested but no traced rows were found.")
    if expected_trace_filter == "covered" and summary["grid_only_rows"] > 0:
        warnings.append("Trace filter `covered` was requested but some rows have no trace.")
    if int(metadata.get("synthetic_tasks", 0) or 0) > 0 and summary["synthetic_rows"] == 0:
        warnings.append("Synthetic tasks were requested but no synthetic rows were found.")
    if _metadata_bool(metadata, "candidate_distill_jsonls") and summary["candidate_distill_rows"] == 0:
        warnings.append("Candidate distillation sources were configured but no candidate-distill rows were found.")
    if summary["trace_rows"] and summary["program_trace_rows"] == 0 and expected_trace_mode == "symbolic_program":
        warnings.append("Symbolic-program traces were requested but no program-style traces were detected.")
    return warnings


def summarize_training_signal(
    train_jsonl: str | Path,
    *,
    val_jsonl: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_rows = read_jsonl(train_jsonl)
    train_summary = summarize_rows(train_rows)
    payload: dict[str, Any] = {
        "train_jsonl": str(train_jsonl),
        "train": train_summary,
        "warnings": warnings_for_summary(train_summary, metadata),
    }
    if val_jsonl is not None and Path(val_jsonl).exists():
        val_rows = read_jsonl(val_jsonl)
        payload["val_jsonl"] = str(val_jsonl)
        payload["val"] = summarize_rows(val_rows)
    if metadata:
        payload["metadata_projection"] = {
            key: metadata.get(key)
            for key in (
                "trace_mode",
                "trace_filter",
                "synthetic_tasks",
                "synthetic_modes",
                "candidate_distill_jsonls",
                "candidate_distill_choice",
                "candidate_distill_completion_source",
                "grid_format",
                "train_task_limit",
            )
            if key in metadata
        }
    return payload


def markdown_table(counts: dict[str, int], *, limit: int = 12) -> list[str]:
    if not counts:
        return ["_None._"]
    lines = ["| Label | Rows |", "|---|---:|"]
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        lines.append(f"| `{label}` | {count} |")
    return lines


def training_signal_markdown(payload: dict[str, Any]) -> str:
    train = payload["train"]
    lines = [
        "# ARC-AGI Training Signal Audit",
        "",
        f"- Train JSONL: `{payload['train_jsonl']}`",
        f"- Rows: `{train['rows']}`",
        f"- Public ARC rows: `{train['public_arc_rows']}`",
        f"- Synthetic rows: `{train['synthetic_rows']}`",
        f"- Candidate-distill rows: `{train['candidate_distill_rows']}`",
        f"- Trace rows: `{train['trace_rows']}` ({train['trace_row_fraction']:.2%})",
        f"- Program-trace rows: `{train['program_trace_rows']}` ({train['program_trace_row_fraction']:.2%})",
        f"- Completion chars p90/max: `{train['completion_chars']['p90']}` / `{train['completion_chars']['max']}`",
        f"- Total chars p90/max: `{train['total_chars']['p90']}` / `{train['total_chars']['max']}`",
        "",
    ]
    if payload.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
        lines.append("")
    lines.extend(["## Families", ""])
    lines.extend(markdown_table(train["task_family_counts"]))
    lines.extend(["", "## Sources", ""])
    lines.extend(markdown_table(train["source_dataset_counts"]))
    lines.extend(["", "## Trace Sources", ""])
    lines.extend(markdown_table(train["trace_source_counts"]))
    if "val" in payload:
        val = payload["val"]
        lines.extend(
            [
                "",
                "## Validation Split",
                "",
                f"- Val JSONL: `{payload['val_jsonl']}`",
                f"- Rows: `{val['rows']}`",
                f"- Trace rows: `{val['trace_rows']}` ({val['trace_row_fraction']:.2%})",
                f"- Program-trace rows: `{val['program_trace_rows']}` ({val['program_trace_row_fraction']:.2%})",
            ]
        )
    return "\n".join(lines) + "\n"


def write_training_signal_report(payload: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_out = Path(json_path)
    md_out = Path(md_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_out.write_text(training_signal_markdown(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl")
    parser.add_argument("--metadata_json")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8")) if args.metadata_json else None
    payload = summarize_training_signal(args.train_jsonl, val_jsonl=args.val_jsonl, metadata=metadata)
    if args.output_json or args.output_md:
        write_training_signal_report(
            payload,
            args.output_json or Path(args.train_jsonl).with_suffix(".training_signal.json"),
            args.output_md or Path(args.train_jsonl).with_suffix(".training_signal.md"),
        )
    print(training_signal_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
