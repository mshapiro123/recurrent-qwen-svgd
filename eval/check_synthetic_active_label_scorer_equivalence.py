"""Check fast-vs-slow active-label scorer equivalence on a small batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
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
    score_candidates_all_loops,
)


def prediction(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0]


def compare(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.data_jsonl)[: args.max_rows]
    loop_counts = [int(item) for item in args.loop_counts.split(",") if item.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    records: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        prompt = prompt_for_row(row, prediction_space=args.prediction_space, prompt_style=args.prompt_style)
        candidates = candidates_for_row(row, prediction_space=args.prediction_space, value_prefix=args.value_prefix)
        base = vars(args).copy()
        fast_args = SimpleNamespace(**base, force_slow_candidate_score=False)
        slow_args = SimpleNamespace(**base, force_slow_candidate_score=True)
        with torch.no_grad():
            fast = score_candidates_all_loops(wrapper, tokenizer, prompt, candidates, fast_args, loop_counts=loop_counts)
            slow = score_candidates_all_loops(wrapper, tokenizer, prompt, candidates, slow_args, loop_counts=loop_counts)
        for loop in loop_counts:
            fast_pred = prediction(fast[loop])
            slow_pred = prediction(slow[loop])
            target = active_target_for_loop(
                row,
                loop,
                prediction_space=args.prediction_space,
                value_prefix=args.value_prefix,
            )
            record = {
                "id": row.get("id") or row.get("instance_id"),
                "depth": int(row["depth"]),
                "loop": int(loop),
                "target": target,
                "fast_prediction": fast_pred,
                "slow_prediction": slow_pred,
                "fast_hit": bool(target is not None and fast_pred == target),
                "slow_hit": bool(target is not None and slow_pred == target),
            }
            if record["fast_prediction"] != record["slow_prediction"] or record["fast_hit"] != record["slow_hit"]:
                mismatches.append(record)
            records.append(record)
    return {
        "kind": "synthetic_active_label_scorer_equivalence",
        "data_jsonl": args.data_jsonl,
        "checkpoint": args.checkpoint,
        "max_rows": args.max_rows,
        "loop_counts": loop_counts,
        "prediction_space": args.prediction_space,
        "prompt_style": args.prompt_style,
        "value_prefix": args.value_prefix,
        "records": len(records),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--max_rows", type=int, default=2)
    parser.add_argument("--loop_counts", default="1,2,3,4")
    parser.add_argument("--prediction_space", choices=("choice_labels", "full_symbols"), default="full_symbols")
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

    summary = compare(args)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
