"""Evaluate exact preimage validity and answer-head sampling coverage."""

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
    candidates_for_row,
    prompt_for_row,
    read_jsonl,
    score_candidates_all_loops,
)


def parse_sample_counts(text: str) -> list[int]:
    counts = sorted({int(item) for item in text.split(",") if item.strip()})
    if not counts or counts[0] < 1:
        raise ValueError("sample_counts must contain positive integers")
    return counts


def sample_names(
    scores: dict[str, float],
    *,
    count: int,
    temperature: float,
    generator: torch.Generator,
) -> list[str]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    names = list(scores)
    logits = torch.tensor([float(scores[name]) for name in names], dtype=torch.float64)
    probabilities = torch.softmax(logits / float(temperature), dim=0)
    indices = torch.multinomial(probabilities, int(count), replacement=True, generator=generator)
    return [names[int(index)] for index in indices.tolist()]


def score_sample_prefix(samples: list[str], valid_starts: set[str]) -> dict[str, Any]:
    valid_samples = [sample for sample in samples if sample in valid_starts]
    unique_valid = sorted(set(valid_samples))
    return {
        "samples": samples,
        "valid_samples": len(valid_samples),
        "invalid_samples": len(samples) - len(valid_samples),
        "unique_samples": len(set(samples)),
        "unique_valid": unique_valid,
        "unique_valid_count": len(unique_valid),
        "coverage": len(unique_valid) / len(valid_starts),
        "full_coverage": set(unique_valid) == valid_starts,
    }


def summarize_rows(rows: list[dict[str, Any]], sample_counts: list[int]) -> dict[str, Any]:
    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        result: dict[str, Any] = {
            "rows": total,
            "greedy_valid": sum(int(row["greedy_valid"]) for row in subset),
        }
        result["greedy_valid_rate"] = result["greedy_valid"] / total if total else 0.0
        result["sampling"] = {}
        for count in sample_counts:
            key = str(count)
            values = [row["sampling"][key] for row in subset]
            samples_total = total * count
            valid_samples = sum(int(value["valid_samples"]) for value in values)
            result["sampling"][key] = {
                "K": count,
                "valid_sample_rate": valid_samples / samples_total if samples_total else 0.0,
                "mean_unique_valid": (
                    sum(int(value["unique_valid_count"]) for value in values) / total if total else 0.0
                ),
                "mean_coverage": sum(float(value["coverage"]) for value in values) / total if total else 0.0,
                "full_coverage_rate": (
                    sum(int(value["full_coverage"]) for value in values) / total if total else 0.0
                ),
                "duplicate_rate": (
                    1.0
                    - sum(int(value["unique_samples"]) for value in values) / samples_total
                    if samples_total
                    else 0.0
                ),
            }
        return result

    by_depth = {
        str(depth): aggregate([row for row in rows if int(row["depth"]) == depth])
        for depth in sorted({int(row["depth"]) for row in rows})
    }
    by_solution_count = {
        str(count): aggregate([row for row in rows if int(row["coverage_denominator"]) == count])
        for count in sorted({int(row["coverage_denominator"]) for row in rows})
    }
    return {
        "kind": "abductive_exact_coverage",
        "rows": len(rows),
        "overall": aggregate(rows),
        "by_depth": by_depth,
        "by_solution_count": by_solution_count,
    }


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.data_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    sample_counts = parse_sample_counts(args.sample_counts)
    max_samples = max(sample_counts)
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if args.progress_every and (row_index == 0 or (row_index + 1) % args.progress_every == 0):
            print(f"abductive_eval_progress row={row_index + 1}/{len(rows)}", flush=True)
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="name:")
        depth = int(row["depth"])
        scores = score_candidates_all_loops(
            wrapper,
            tokenizer,
            prompt,
            candidates,
            args,
            loop_counts=[depth],
        )[depth]
        greedy = max(scores.items(), key=lambda item: item[1])[0]
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + row_index)
        samples = sample_names(
            scores,
            count=max_samples,
            temperature=args.temperature,
            generator=generator,
        )
        valid_starts = {str(value) for value in row["valid_starts"]}
        output_rows.append(
            {
                "id": row.get("id") or row.get("instance_id"),
                "depth": depth,
                "task_mode": row.get("task_mode"),
                "valid_starts": sorted(valid_starts),
                "coverage_denominator": len(valid_starts),
                "greedy_prediction": greedy,
                "greedy_valid": greedy in valid_starts,
                "scores": scores if args.include_scores else None,
                "sampling": {
                    str(count): score_sample_prefix(samples[:count], valid_starts)
                    for count in sample_counts
                },
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
    parser.add_argument("--sample_counts", default="1,2,4,8,20")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2_718_281)
    parser.add_argument("--include_scores", action="store_true")
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

    output_rows = evaluate(args)
    sample_counts = parse_sample_counts(args.sample_counts)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8"
    )
    summary = summarize_rows(output_rows, sample_counts)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "temperature": args.temperature,
            "sample_counts": sample_counts,
            "seed": args.seed,
            "sampling_mode": "answer_head_temperature",
        }
    )
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

