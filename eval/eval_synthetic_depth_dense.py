"""Evaluate a dense model on frozen synthetic-depth rows with one shared symbol reader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs
from eval.dense_response_reader import extract_first_completed_symbol


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_final_symbol(text: str, candidates: list[str]) -> str | None:
    return extract_first_completed_symbol(text, candidates)


def prompt_for_row(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt is not None:
        return str(prompt)
    return str(row["question"]).rstrip() + "\nAnswer:"


def candidates_for_row(row: dict[str, Any]) -> list[str]:
    mapping = row.get("mapping") or {}
    candidates = sorted({str(value).strip().upper() for value in mapping} | {str(value).strip().upper() for value in mapping.values()})
    if not candidates:
        candidates = sorted({str(value).strip().upper() for value in row.get("orbit") or []})
    return candidates


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_depth: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_depth.setdefault(str(int(row["depth"])), {"correct": 0, "total": 0, "parse_failures": 0})
        bucket["correct"] += int(bool(row.get("correct")))
        bucket["total"] += 1
        bucket["parse_failures"] += int(row.get("prediction") is None)
    output: dict[str, dict[str, float | int]] = {}
    for depth, counts in sorted(by_depth.items(), key=lambda item: int(item[0])):
        total = int(counts["total"])
        output[depth] = {
            **counts,
            "accuracy": int(counts["correct"]) / total if total else 0.0,
        }
    total = sum(int(row["total"]) for row in output.values())
    correct = sum(int(row["correct"]) for row in output.values())
    return {
        "kind": "stage5_synthetic_depth_dense_eval",
        "reader": "leading_symbol_else_first_answer_else_first_valid_full_symbol",
        "by_depth": output,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.data_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        **model_load_kwargs(args.dtype, args.attn_implementation, low_cpu_mem_usage=True),
    ).to(args.device)
    model.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [prompt_for_row(row) for row in batch_rows]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        continuations = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        for row, continuation in zip(batch_rows, continuations):
            candidates = candidates_for_row(row)
            prediction = extract_final_symbol(continuation, candidates)
            target = str(row["target"]).strip().upper()
            output.append(
                {
                    "id": row.get("id") or row.get("instance_id"),
                    "depth": int(row["depth"]),
                    "target": target,
                    "prediction": prediction,
                    "correct": prediction == target,
                    "continuation": continuation,
                }
            )
        completed = min(start + len(batch_rows), len(rows))
        if completed == len(rows) or completed % 128 == 0:
            print(f"dense_eval_progress row={completed}/{len(rows)}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = evaluate(args)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary = summarize_rows(rows)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
