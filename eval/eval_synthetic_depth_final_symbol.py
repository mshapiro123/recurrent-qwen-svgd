"""Score synthetic-depth final answers with the same full-symbol reader.

The older forced-depth matrix scores MCQ option surfaces.  That is useful as a
reader stress test, but it is not the aligned final-answer metric for the
full-symbol chain objective.  This evaluator uses the same candidate space as
``eval_synthetic_depth_active_labels.py`` and reads the final answer at
``loop == depth`` for every row.  When MCQ choices are present it also maps the
predicted symbol back to the deterministic option label, so downstream reports
can show both the raw symbol hit and the mapped-option hit without invoking a
different reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from eval.eval_synthetic_depth_active_labels import (  # noqa: E402
    candidates_for_row,
    prompt_for_row,
    read_jsonl,
    row_symbol_names,
    score_candidates_all_loops,
    symbol,
)


def choice_label_for_symbol(row: dict[str, Any], predicted_symbol: str) -> str | None:
    choices = row.get("choices") or {}
    for label, value in choices.items():
        if str(value).strip() == str(predicted_symbol).strip():
            return str(label)
    return None


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.data_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        depth = int(row["depth"])
        if depth > int(args.max_loops):
            raise ValueError(f"Row depth {depth} exceeds --max_loops={args.max_loops}: {row.get('id')}")
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style=args.prompt_style)
        candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix=args.value_prefix)
        scores_by_loop = score_candidates_all_loops(
            wrapper,
            tokenizer,
            prompt,
            candidates,
            args,
            loop_counts=[depth],
        )
        scores = scores_by_loop[depth]
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        target = symbol(row["target"], prefix=args.value_prefix, row_symbols=row_symbol_names(row))
        mapped_label = choice_label_for_symbol(row, prediction)
        answer_label = str(row.get("answer", "")).strip() or None
        output_rows.append(
            {
                "id": row.get("id") or row.get("instance_id"),
                "depth": depth,
                "forced_loop_count": depth,
                "prediction": prediction,
                "target": target,
                "same_reader_final_hit": prediction == target,
                "mapped_choice_label": mapped_label,
                "answer": answer_label,
                "mapped_final_hit": bool(answer_label and mapped_label == answer_label),
                "prediction_space": "full_symbols",
                "prompt_style": args.prompt_style,
                "scores": scores,
            }
        )
    return output_rows


def summarize_final_symbol_rows(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    by_depth: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_depth.setdefault(str(int(row["depth"])), {"same": 0, "mapped": 0, "total": 0})
        bucket["same"] += int(bool(row.get("same_reader_final_hit")))
        bucket["mapped"] += int(bool(row.get("mapped_final_hit")))
        bucket["total"] += 1

    depth_summary: dict[str, dict[str, float | int]] = {}
    same_total = {"correct": 0, "total": 0}
    mapped_total = {"correct": 0, "total": 0}
    for depth, counts in sorted(by_depth.items(), key=lambda item: int(item[0])):
        total = int(counts["total"])
        same = int(counts["same"])
        mapped = int(counts["mapped"])
        same_total["correct"] += same
        same_total["total"] += total
        mapped_total["correct"] += mapped
        mapped_total["total"] += total
        depth_summary[depth] = {
            "total": total,
            "same_reader_correct": same,
            "same_reader_accuracy": same / total if total else 0.0,
            "mapped_final_correct": mapped,
            "mapped_final_accuracy": mapped / total if total else 0.0,
            "same_reader_clears_threshold": (same / total if total else 0.0) >= threshold,
        }

    same_accuracy = same_total["correct"] / same_total["total"] if same_total["total"] else 0.0
    mapped_accuracy = mapped_total["correct"] / mapped_total["total"] if mapped_total["total"] else 0.0
    return {
        "kind": "synthetic_depth_same_reader_final_symbol",
        "threshold": threshold,
        "depths": [int(depth) for depth in depth_summary],
        "by_depth": depth_summary,
        "same_reader_total": {**same_total, "accuracy": same_accuracy},
        "mapped_final_total": {**mapped_total, "accuracy": mapped_accuracy},
        "all_depths_clear_threshold": bool(depth_summary)
        and all(bool(item["same_reader_clears_threshold"]) for item in depth_summary.values()),
        "metric_policy": {
            "active": "full-symbol active-label diagonal",
            "raw_final": "same-reader full-symbol argmax at loop == depth",
            "mapped_final": "same-reader full-symbol argmax deterministically mapped to MCQ label",
            "suspended_reader": "option-text/choice MCQ final-answer matrices remain diagnostic only",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--max_loops", type=int, default=14)
    parser.add_argument("--threshold", type=float, default=0.71)
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="question_only")
    parser.add_argument("--value_prefix", default="")
    parser.add_argument("--normalize_candidate_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="concat")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    args = parser.parse_args()

    rows = evaluate(args)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary = summarize_final_symbol_rows(rows, threshold=args.threshold)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "prediction_space": "full_symbols",
            "prompt_style": args.prompt_style,
            "rows": len(rows),
        }
    )
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
