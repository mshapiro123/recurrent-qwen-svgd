"""Evaluate only the forced-loop diagonal of a synthetic chain dataset."""

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

from eval.eval_mcq import load_recurrent_wrapper
from eval.eval_synthetic_depth_active_labels import (
    active_target_for_loop,
    candidates_for_row,
    prompt_for_row,
    read_jsonl,
    row_symbol_names,
    score_candidates_all_loops,
    symbol,
)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_depth: dict[str, dict[str, Any]] = {}
    for depth in sorted({int(row["depth"]) for row in rows}):
        depth_rows = [row for row in rows if int(row["depth"]) == depth]
        correct = sum(int(row["hit"]) for row in depth_rows)
        by_depth[str(depth)] = {
            "correct": correct,
            "total": len(depth_rows),
            "accuracy": correct / len(depth_rows) if depth_rows else 0.0,
        }
    diagonal = {depth: float(value["accuracy"]) for depth, value in by_depth.items()}
    return {
        "kind": "synthetic_diagonal_guardrail",
        "rows": len(rows),
        "by_depth": by_depth,
        "active_diagonal": diagonal,
        "active_diagonal_min": min(diagonal.values()) if diagonal else 0.0,
        "correct": sum(int(row["hit"]) for row in rows),
        "accuracy": sum(int(row["hit"]) for row in rows) / len(rows) if rows else 0.0,
    }


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(args.data_jsonl) if int(row["depth"]) <= int(args.max_depth)]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        if args.progress_every and (row_index == 1 or row_index % args.progress_every == 0):
            print(f"diagonal_guardrail_progress row={row_index}/{len(rows)}", flush=True)
        depth = int(row["depth"])
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix=args.value_prefix)
        scores = score_candidates_all_loops(
            wrapper,
            tokenizer,
            prompt,
            candidates,
            args,
            loop_counts=[depth],
        )[depth]
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        target = active_target_for_loop(
            row,
            depth,
            prediction_space="full_symbols",
            value_prefix=args.value_prefix,
        )
        if target is None:
            target = symbol(
                row["target"],
                prefix=args.value_prefix,
                row_symbols=row_symbol_names(row),
            )
        output_rows.append(
            {
                "id": row.get("id") or row.get("instance_id"),
                "depth": depth,
                "prediction": prediction,
                "target": target,
                "hit": prediction == target,
            }
        )
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--value_prefix", default="letter:")
    parser.add_argument("--normalize_candidate_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_slow_candidate_score", action="store_true")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    rows = evaluate(args)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = summarize_rows(rows)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "max_depth": args.max_depth,
            "value_prefix": args.value_prefix,
        }
    )
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

