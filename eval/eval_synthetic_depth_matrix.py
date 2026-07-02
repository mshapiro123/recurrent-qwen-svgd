"""Evaluate forced-loop accuracy matrices for the synthetic depth task."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from eval.eval_mcq import (
    CompletionScore,
    MCQExample,
    extract_loop_diagnostics,
    format_completion,
    format_prompt,
    load_base_model,
    load_recurrent_wrapper,
    predict_from_scores,
    read_examples,
    sequence_logprobs,
    select_forced_loop_logits,
)


@dataclass(frozen=True)
class MatrixCell:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_accuracy_matrix(rows: list[dict[str, Any]]) -> dict[tuple[int, int], MatrixCell]:
    counts: dict[tuple[int, int], list[int]] = {}
    for row in rows:
        depth = int(row["depth"])
        loop = int(row["forced_loop_count"])
        correct, total = counts.setdefault((depth, loop), [0, 0])
        counts[(depth, loop)] = [correct + int(bool(row.get("hit"))), total + 1]
    return {key: MatrixCell(correct=value[0], total=value[1]) for key, value in counts.items()}


def largest_depth_at_threshold(
    matrix: dict[tuple[int, int], MatrixCell],
    *,
    loop: int,
    threshold: float,
) -> int:
    depths = sorted(depth for depth, cell_loop in matrix if cell_loop == loop)
    reached = 0
    for depth in depths:
        cell = matrix[(depth, loop)]
        if cell.total and cell.accuracy >= threshold:
            reached = max(reached, depth)
    return reached


def summarize_matrix(rows: list[dict[str, Any]], *, threshold: float = 0.75) -> dict[str, Any]:
    matrix = build_accuracy_matrix(rows)
    depths = sorted({depth for depth, _ in matrix})
    loops = sorted({loop for _, loop in matrix})
    frontier = {
        str(loop): largest_depth_at_threshold(matrix, loop=loop, threshold=threshold)
        for loop in loops
    }
    frontier_values = [frontier[str(loop)] for loop in loops]
    non_decreasing = all(left <= right for left, right in zip(frontier_values, frontier_values[1:]))
    strictly_expands = any(left < right for left, right in zip(frontier_values, frontier_values[1:]))
    serialized_matrix: dict[str, dict[str, Any]] = {}
    for depth in depths:
        serialized_matrix[str(depth)] = {}
        for loop in loops:
            cell = matrix.get((depth, loop), MatrixCell())
            serialized_matrix[str(depth)][str(loop)] = {
                "correct": cell.correct,
                "total": cell.total,
                "accuracy": cell.accuracy,
            }
    return {
        "kind": "synthetic_depth_accuracy_matrix",
        "threshold": threshold,
        "depths": depths,
        "loops": loops,
        "matrix": serialized_matrix,
        "frontier_by_loop": frontier,
        "frontier_is_non_decreasing": non_decreasing,
        "frontier_strictly_expands": strictly_expands,
    }


def _metadata_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def score_completion_all_loops(
    wrapper,
    tokenizer,
    prompt: str,
    completion: str,
    args: argparse.Namespace,
    *,
    loop_counts: list[int],
) -> dict[int, CompletionScore]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=True).to(args.device)
    labels = encoded["input_ids"].clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    max_loops = max(loop_counts)
    with torch.no_grad():
        output = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=max_loops,
            num_trajectories=args.num_trajectories,
            sample_latents=args.sample_latents,
            latent_injection_mode=args.latent_injection_mode,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
        )
    result: dict[int, CompletionScore] = {}
    for loop in loop_counts:
        logits = select_forced_loop_logits(output, loop)
        scores = sequence_logprobs(logits, labels, normalize=args.normalize_option_score)
        diagnostics = {"forced_loop_count": loop}
        if args.include_loop_diagnostics:
            diagnostics.update(extract_loop_diagnostics(output))
        result[loop] = CompletionScore(scores=scores, diagnostics=diagnostics)
    return result


def score_base_completion(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=True).to(args.device)
    labels = encoded["input_ids"].clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            use_cache=False,
            return_dict=True,
        )
    return sequence_logprobs(output.logits, labels, normalize=args.normalize_option_score)


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def evaluate_recurrent_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = read_examples(args.data_jsonl)
    raw_rows = _metadata_by_id(read_jsonl_rows(args.data_jsonl))
    loop_counts = [int(item) for item in args.loop_counts.split(",") if item.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = format_prompt(example, args.prompt_style)
        option_scores_by_loop = {loop: {} for loop in loop_counts}
        for label, text in example.choices:
            completion = format_completion(label, text, args.score_target)
            scored = score_completion_all_loops(
                wrapper,
                tokenizer,
                prompt,
                completion,
                args,
                loop_counts=loop_counts,
            )
            for loop, completion_score in scored.items():
                option_scores_by_loop[loop][label] = completion_score.scores
        metadata = raw_rows.get(example.id, {})
        for loop in loop_counts:
            prediction, scalar_scores = predict_from_scores(option_scores_by_loop[loop], args.aggregate)
            hit = prediction == example.answer
            row = {
                "id": example.id,
                "depth": int(metadata.get("depth", 0)),
                "forced_loop_count": loop,
                "prediction": prediction,
                "answer": example.answer,
                "hit": hit,
                "scores": scalar_scores,
                "target": metadata.get("target"),
                "synthetic_task": metadata.get("synthetic_task", "iterated_function"),
            }
            append_jsonl(args.output_jsonl, row)
            rows.append(row)
    return rows


def evaluate_base_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = read_examples(args.data_jsonl)
    raw_rows = _metadata_by_id(read_jsonl_rows(args.data_jsonl))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_base_model(args, load_dense_lora_checkpoint=False)
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = format_prompt(example, args.prompt_style)
        option_scores: dict[str, torch.Tensor] = {}
        for label, text in example.choices:
            completion = format_completion(label, text, args.score_target)
            option_scores[label] = score_base_completion(model, tokenizer, prompt, completion, args)
        prediction, scalar_scores = predict_from_scores(option_scores, args.aggregate)
        metadata = raw_rows.get(example.id, {})
        row = {
            "id": example.id,
            "depth": int(metadata.get("depth", 0)),
            "forced_loop_count": 0,
            "prediction": prediction,
            "answer": example.answer,
            "hit": prediction == example.answer,
            "scores": scalar_scores,
            "target": metadata.get("target"),
            "synthetic_task": metadata.get("synthetic_task", "iterated_function"),
        }
        append_jsonl(args.output_jsonl, row)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=("recurrent", "base"), default="recurrent")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--loop_counts", default="1,2,4,8")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="with_options")
    parser.add_argument("--score_target", choices=("label", "option_text", "label_and_text"), default="option_text")
    parser.add_argument("--aggregate", choices=("mean", "max", "vote"), default="mean")
    parser.add_argument("--normalize_option_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_loop_diagnostics", action="store_true")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="concat")
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--sample_latents", action="store_true")
    parser.add_argument("--latent_injection_mode", default="post", choices=("pre", "post", "both"))
    args = parser.parse_args()

    if args.mode == "recurrent" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for recurrent mode")
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).write_text("", encoding="utf-8")
    rows = evaluate_base_rows(args) if args.mode == "base" else evaluate_recurrent_matrix(args)
    summary = summarize_matrix(rows, threshold=args.threshold)
    summary.update(
        {
            "mode": args.mode,
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "score_target": args.score_target,
            "aggregate": args.aggregate,
            "rows": len(rows),
        }
    )
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
