"""Score forced-depth and learned Ponder depth on the same full-symbol rows."""

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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "kind": "ponder_depth_evaluation",
        "rows": total,
        "forced_depth_correct": sum(int(row["forced_hit"]) for row in rows),
        "forced_depth_accuracy": sum(int(row["forced_hit"]) for row in rows) / total if total else 0.0,
        "learned_depth_correct": sum(int(row["learned_hit"]) for row in rows),
        "learned_depth_accuracy": sum(int(row["learned_hit"]) for row in rows) / total if total else 0.0,
        "mean_selected_loops": (
            sum(int(row["selected_loop"]) for row in rows) / total if total else 0.0
        ),
        "mean_expected_loops": (
            sum(float(row["expected_loops"]) for row in rows) / total if total else 0.0
        ),
        "selected_depth_histogram": {
            str(depth): sum(int(row["selected_loop"]) == depth for row in rows)
            for depth in range(1, 5)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--value_prefix", default="letter:")
    args = parser.parse_args()

    load_args = SimpleNamespace(
        model_name=args.model_name,
        dtype=args.dtype,
        attn_implementation="default",
        device=args.device,
        split=args.split,
        bridge_projection_mode="split",
        adapter_dtype=args.adapter_dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(load_args, args.checkpoint)
    score_args = SimpleNamespace(
        device=args.device,
        force_slow_candidate_score=False,
        normalize_candidate_score=True,
    )
    results: list[dict[str, Any]] = []
    source_rows = read_jsonl(args.data_jsonl)
    for index, row in enumerate(source_rows, start=1):
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix=args.value_prefix)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
        with torch.no_grad():
            output = wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                max_loops=4,
                num_trajectories=1,
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
        weights = output.halting_weights[0, 0].detach().float()
        selected_loop = int(weights.argmax().item()) + 1
        expected_loops = float(output.expected_loops[0, 0].detach().float().item())
        depth = int(row["depth"])
        scores = score_candidates_all_loops(
            wrapper,
            tokenizer,
            prompt,
            candidates,
            score_args,
            loop_counts=sorted({depth, selected_loop}),
        )
        forced_prediction = max(scores[depth].items(), key=lambda item: item[1])[0]
        learned_prediction = max(scores[selected_loop].items(), key=lambda item: item[1])[0]
        forced_target = active_target_for_loop(
            row,
            depth,
            prediction_space="full_symbols",
            value_prefix=args.value_prefix,
        )
        learned_target = active_target_for_loop(
            row,
            selected_loop,
            prediction_space="full_symbols",
            value_prefix=args.value_prefix,
        )
        results.append(
            {
                "id": row.get("id") or row.get("instance_id"),
                "depth": depth,
                "selected_loop": selected_loop,
                "expected_loops": expected_loops,
                "halting_weights": weights.tolist(),
                "forced_prediction": forced_prediction,
                "forced_target": forced_target,
                "forced_hit": forced_prediction == forced_target,
                "learned_prediction": learned_prediction,
                "learned_target": learned_target,
                "learned_hit": learned_target is not None and learned_prediction == learned_target,
            }
        )
        if index == 1 or index % 32 == 0:
            print(f"ponder_eval_progress={index}/{len(source_rows)}", flush=True)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in results),
        encoding="utf-8",
    )
    payload = summarize(results)
    payload.update({"checkpoint": args.checkpoint, "data_jsonl": args.data_jsonl})
    Path(args.output_summary).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
