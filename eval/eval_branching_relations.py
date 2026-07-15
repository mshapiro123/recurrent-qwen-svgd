"""Score branching-relation rows by exact reachable-set validity."""

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
    score_candidates_all_loops,
)
from training.branching_relations_task import assess_validity_gate  # noqa: E402


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate = assess_validity_gate(rows)
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({str(row["reachable_set_stratum"]) for row in rows}):
        selected = [row for row in rows if str(row["reachable_set_stratum"]) == stratum]
        correct = sum(bool(row["valid"]) for row in selected)
        by_stratum[stratum] = {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected) if selected else 0.0,
        }
    return {
        "kind": "branching_relations_zero_shot_validity",
        "gate": gate,
        "by_reachable_set_stratum": by_stratum,
        "rows": len(rows),
    }


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.data_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == len(rows)):
            print(f"branching_eval_progress row={index}/{len(rows)} depth={row['depth']}", flush=True)
        depth = int(row["depth"])
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="name:")
        scores = score_candidates_all_loops(
            wrapper,
            tokenizer,
            prompt,
            candidates,
            args,
            loop_counts=[depth],
        )[depth]
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        reachable = [str(value) for value in row["reachable_symbols"]]
        output.append(
            {
                "id": row["id"],
                "depth": depth,
                "rendering": row["rendering"],
                "reachable_set_stratum": row["reachable_set_stratum"],
                "reachable_set_size": int(row["reachable_set_size"]),
                "prediction": prediction,
                "reachable_symbols": reachable,
                "valid": prediction in reachable,
                "scores": scores,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--normalize_candidate_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_slow_candidate_score", action="store_true")
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    rows = evaluate(args)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = summarize_rows(rows)
    summary.update({"checkpoint": args.checkpoint, "data_jsonl": args.data_jsonl})
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
