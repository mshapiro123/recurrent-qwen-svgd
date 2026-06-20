"""Build ARC-AGI SFT rows from exact saved candidate generations.

This script turns diagnostic/evaluation candidate JSONL files into supervised
rows for recurrent fine-tuning. Use it on training or synthetic candidate runs,
not on held-out benchmark outputs that you plan to report as final scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc_agi_utils import format_grid_completion  # noqa: E402
from eval.rescore_arc_agi_candidates import group_candidate_rows, score_exact  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_rank(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Rank exact candidates for distillation, lower is better."""

    source = str(row.get("candidate_source", ""))
    text = str(row.get("candidate_text", ""))
    return (
        0 if row.get("selected") else 1,
        0 if row.get("program_fits_train") else 1,
        0 if source.startswith("symbolic_") else 1,
        len(text),
        int(row.get("candidate_index", 0)),
    )


def choose_exact_candidates(
    group_rows: list[dict[str, Any]],
    *,
    choice: str,
) -> list[dict[str, Any]]:
    exact_rows = [row for row in group_rows if score_exact(row) and row.get("parsed_grid") is not None]
    if not exact_rows:
        return []
    if choice == "selected_exact":
        return [row for row in exact_rows if row.get("selected")]
    if choice == "best_exact":
        return [min(exact_rows, key=candidate_rank)]
    if choice == "all_exact":
        return sorted(exact_rows, key=candidate_rank)
    raise ValueError(f"Unknown choice={choice!r}")


def completion_from_candidate(
    row: dict[str, Any],
    *,
    completion_source: str,
    output_format: str,
) -> str:
    parsed = row.get("parsed_grid")
    if completion_source == "candidate_text":
        text = str(row.get("candidate_text", "")).strip()
        if text:
            return text
        completion_source = "canonical_grid"
    if completion_source == "canonical_grid":
        if parsed is None:
            raise ValueError("canonical_grid completion requires parsed_grid")
        return format_grid_completion(parsed, output_format=output_format)
    if completion_source == "trace_then_canonical_grid":
        if parsed is None:
            raise ValueError("trace_then_canonical_grid completion requires parsed_grid")
        trace = extract_think_trace(str(row.get("candidate_text", "")))
        canonical = format_grid_completion(parsed, output_format=output_format)
        return f"{trace}\n{canonical}" if trace else canonical
    raise ValueError(f"Unknown completion_source={completion_source!r}")


def extract_think_trace(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("<think>")
    end = stripped.find("</think>", start + len("<think>")) if start >= 0 else -1
    if start < 0 or end < 0:
        return ""
    return stripped[start : end + len("</think>")].strip()


def cot_from_completion(completion: str) -> str:
    trace = extract_think_trace(completion)
    return trace or completion.strip()


def candidate_to_jsonl_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    append_eos: bool,
    completion_source: str,
    output_format: str,
    choice: str,
    source_jsonl: str | Path,
) -> dict[str, object]:
    prompt = str(row.get("prompt", ""))
    if not prompt:
        raise ValueError("candidate row is missing prompt")
    completion = completion_from_candidate(row, completion_source=completion_source, output_format=output_format)
    if append_eos and getattr(tokenizer, "eos_token", None):
        completion += tokenizer.eos_token
    cot = cot_from_completion(completion)
    cot_tokens = max(1, len(tokenizer(cot, add_special_tokens=False)["input_ids"]))
    return {
        "prompt": prompt,
        "completion": completion,
        "cot": cot,
        "cot_tokens": cot_tokens,
        "source_dataset": "arc-agi-candidate-distill",
        "category": str(row.get("candidate_source", "unknown")),
        "difficulty": None,
        "task_id": row.get("task_id"),
        "test_index": row.get("test_index"),
        "candidate_index": row.get("candidate_index"),
        "candidate_source": row.get("candidate_source"),
        "parse_method": row.get("parse_method"),
        "program_fits_train": bool(row.get("program_fits_train")),
        "distill_choice": choice,
        "completion_source": completion_source,
        "source_jsonl": str(source_jsonl),
    }


def build_distill_rows(
    tokenizer: Any,
    candidate_rows: list[dict[str, Any]],
    *,
    choice: str,
    completion_source: str,
    output_format: str,
    append_eos: bool,
    source_jsonl: str | Path,
    max_rows: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _key, group_rows in sorted(group_candidate_rows(candidate_rows).items()):
        for candidate in choose_exact_candidates(group_rows, choice=choice):
            rows.append(
                candidate_to_jsonl_row(
                    tokenizer,
                    candidate,
                    append_eos=append_eos,
                    completion_source=completion_source,
                    output_format=output_format,
                    choice=choice,
                    source_jsonl=source_jsonl,
                )
            )
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--choice", default="best_exact", choices=("selected_exact", "best_exact", "all_exact"))
    parser.add_argument(
        "--completion_source",
        default="trace_then_canonical_grid",
        choices=("candidate_text", "canonical_grid", "trace_then_canonical_grid"),
    )
    parser.add_argument("--output_format", default="compact", choices=("json", "compact", "tagged"))
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_rows", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    candidate_rows = read_jsonl(args.candidates_jsonl)
    rows = build_distill_rows(
        tokenizer,
        candidate_rows,
        choice=args.choice,
        completion_source=args.completion_source,
        output_format=args.output_format,
        append_eos=args.append_eos,
        source_jsonl=args.candidates_jsonl,
        max_rows=args.max_rows,
    )
    write_jsonl(args.output_jsonl, rows)
    print(f"candidate_rows={len(candidate_rows)}")
    print(f"distill_rows={len(rows)}")
    print(f"output_jsonl={args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
