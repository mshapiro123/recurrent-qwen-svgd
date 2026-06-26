"""Eval-only tail-damper sweep for forced recurrent depth.

The script calibrates a low-rank tail damper from the no-damper loop-1
entry/exit mismatch, then sweeps damping strengths while measuring both:

* loop-tail energy through a deeper horizon, and
* forced-loop MCQ correctness/churn for loops 1/2/3.

It is intentionally read-only: no checkpoint mutation and no optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import (  # noqa: E402
    MCQExample,
    format_completion,
    format_prompt,
    load_recurrent_wrapper,
    predict_from_scores,
    read_examples,
    sequence_logprobs,
    set_seed,
)
from eval.eval_reentry_drift import (  # noqa: E402
    masked_token_matrix,
    prepare_recurrent_inputs,
    rms,
    run_recurrent_block,
)
from eval.eval_reentry_tail_diagnostic import (  # noqa: E402
    arc_examples,
    centered_covariance,
    correction_class,
    finite_float,
    path_for_cli,
    tail_basis,
    tail_decomposition,
    tail_trace,
    write_damper_artifact,
)
from models.reentry_tail_damper import apply_tail_damper  # noqa: E402


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_examples(args: argparse.Namespace) -> list[MCQExample]:
    if args.mcq_jsonl:
        examples = read_examples(args.mcq_jsonl)
    else:
        examples = arc_examples(
            config=args.arc_config,
            split=args.arc_split,
            offset=args.arc_offset,
            limit=args.arc_limit,
            seed=args.arc_seed,
        )
    return examples[: args.limit] if args.limit and args.limit > 0 else examples


def collect_tokens(
    *,
    wrapper: Any,
    tokenizer: Any,
    examples: list[MCQExample],
    args: argparse.Namespace,
    loop_counts: list[int],
    damper: dict[str, torch.Tensor] | None = None,
    strength: float = 0.0,
) -> dict[str, torch.Tensor]:
    max_loop = max(loop_counts)
    tokens_by_stage: dict[str, list[torch.Tensor]] = {"entry": []}
    for loop in loop_counts:
        tokens_by_stage[f"loop{loop}"] = []

    for example in examples:
        prompt = format_prompt(example, args.prompt_style)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        with torch.no_grad():
            entry_state, mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
                wrapper,
                input_ids,
                attention_mask,
            )
            tokens_by_stage["entry"].append(masked_token_matrix(entry_state, mask).cpu())
            entry_rms = rms(entry_state, mask).clamp_min(1e-8)
            recurrent_state = entry_state
            for loop_idx in range(max_loop):
                loop_number = loop_idx + 1
                loop_input = recurrent_state if loop_idx == 0 else wrapper.bridge(recurrent_state)
                if loop_idx > 0 and args.reentry_rescale_mode == "entry_rms":
                    current_rms = rms(loop_input, mask).clamp_min(1e-8)
                    loop_input = loop_input * (entry_rms / current_rms).to(dtype=loop_input.dtype)
                if loop_idx > 0 and damper is not None and strength > 0.0:
                    loop_input = apply_tail_damper(
                        loop_input,
                        mean=damper["mean"],
                        basis=damper["basis"],
                        damper_scale=damper["damper_scale"],
                        strength=strength,
                    )
                loop_output = run_recurrent_block(
                    wrapper,
                    loop_input,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                )
                if loop_number in loop_counts:
                    tokens_by_stage[f"loop{loop_number}"].append(masked_token_matrix(loop_output, mask).cpu())
                recurrent_state = loop_output
    return {key: torch.cat(value, dim=0) for key, value in tokens_by_stage.items()}


def tail_trace_summary(
    tokens_by_stage: dict[str, torch.Tensor],
    *,
    mean_entry: torch.Tensor,
    basis: torch.Tensor,
    loop_counts: list[int],
) -> dict[str, dict[str, float]]:
    entry_tail = tail_trace(tokens_by_stage["entry"], mean_entry, basis)
    out = {"entry": {"tail_trace": entry_tail, "ratio_vs_entry": 1.0}}
    for loop in loop_counts:
        trace = tail_trace(tokens_by_stage[f"loop{loop}"], mean_entry, basis)
        out[f"loop{loop}"] = {
            "tail_trace": trace,
            "ratio_vs_entry": trace / max(entry_tail, 1e-12),
        }
    return out


def mask_completion_labels(tokenizer: Any, prompt: str, texts: list[str], encoded: Any) -> torch.Tensor:
    labels = encoded["input_ids"].clone()
    prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    labels[:, : min(prompt_len, labels.shape[1])] = -100
    if "attention_mask" in encoded:
        labels = labels.masked_fill(encoded["attention_mask"].eq(0), -100)
    return labels


def score_examples(
    *,
    wrapper: Any,
    tokenizer: Any,
    examples: list[MCQExample],
    args: argparse.Namespace,
    loops: list[int],
    damper_path: Path,
    strength: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_loop = max(loops)
    records: list[dict[str, Any]] = []
    loop_correct = {str(loop): 0 for loop in loops}
    pattern_counts: Counter[str] = Counter()

    for example in examples:
        prompt = format_prompt(example, args.prompt_style)
        labels = [label for label, _text in example.choices]
        completions = [
            format_completion(label, text, args.score_target)
            for label, text in example.choices
        ]
        texts = [prompt + completion for completion in completions]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        label_tensor = mask_completion_labels(tokenizer, prompt, texts, encoded)
        with torch.no_grad():
            output = wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                max_loops=max_loop,
                num_trajectories=1,
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
                reentry_rescale_mode=args.reentry_rescale_mode,
                reentry_tail_damper_path=path_for_cli(damper_path),
                reentry_tail_damper_strength=strength,
            )
        loop_logits = output.loop_logits[:, 0]
        hits: dict[str, bool] = {}
        predictions: dict[str, str] = {}
        loop_scores: dict[str, dict[str, float]] = {}
        for loop in loops:
            scores = sequence_logprobs(loop_logits[:, loop - 1], label_tensor, normalize=True).cpu()
            option_scores = {label: scores[idx : idx + 1] for idx, label in enumerate(labels)}
            prediction, scalar_scores = predict_from_scores(option_scores, "mean")
            hit = prediction == example.answer
            hits[str(loop)] = hit
            predictions[str(loop)] = prediction
            loop_scores[str(loop)] = scalar_scores
            loop_correct[str(loop)] += int(hit)
        pattern = "".join("1" if hits[str(loop)] else "0" for loop in loops)
        pattern_counts[pattern] += 1
        records.append(
            {
                "id": example.id,
                "answer": example.answer,
                "strength": strength,
                "loops": loops,
                "hits": hits,
                "pattern": pattern,
                "predictions": predictions,
                "scores": loop_scores,
            }
        )

    loop_results = {
        str(loop): {
            "correct": loop_correct[str(loop)],
            "total": len(examples),
            "accuracy": loop_correct[str(loop)] / max(len(examples), 1),
        }
        for loop in loops
    }
    loop1_key = str(loops[0])
    oracle_correct = sum(1 for row in records if any(row["hits"].values()))
    stable_correct = sum(1 for row in records if all(row["hits"].values()))
    stable_wrong = sum(1 for row in records if not any(row["hits"].values()))
    rescued = sum(1 for row in records if not row["hits"][loop1_key] and any(row["hits"].values()))
    harmed = sum(1 for row in records if row["hits"][loop1_key] and not all(row["hits"].values()))
    summary = {
        "strength": strength,
        "examples": len(examples),
        "loop_results": loop_results,
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / max(len(examples), 1),
        "oracle_gap_vs_loop1": oracle_correct - loop_results[loop1_key]["correct"],
        "rescued_vs_loop1": rescued,
        "harmed_vs_loop1": harmed,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "pattern_counts": dict(sorted(pattern_counts.items())),
    }
    return records, summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Tail-Damper Forced-Depth Sweep - {summary['run_id']}",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Examples: `{summary['examples']}`",
        f"- Loops scored: `{summary['score_loops']}`",
        f"- Tail loops: `{summary['tail_loop_counts']}`",
        f"- Re-entry rescale: `{summary['reentry_rescale_mode']}`",
        "",
        "## Calibration",
        "",
        f"- Correction class: `{summary['calibration']['correction_class']['action']}`",
        f"- Tail mismatch: `{summary['calibration']['tail_decomposition_loop1']['tail_mismatch']:.6f}`",
        f"- After damper: `{summary['calibration']['tail_decomposition_loop1']['after_damper']:.6f}`",
        f"- Damper scales: `{summary['calibration']['tail_decomposition_loop1']['damper_scale']}`",
        "",
        "## Strength Sweep",
        "",
        "| strength | loop8 tail ratio | loop1 | loop2 | loop3 | oracle | gap | rescued | harmed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["strength_summaries"]:
        loops = row["score_summary"]["loop_results"]
        tail = row["tail_trace"].get("loop8") or {}
        lines.append(
            "| "
            f"{row['strength']:.2f} | "
            f"{tail.get('ratio_vs_entry', 0.0):.3f} | "
            f"{loops.get('1', {}).get('correct', 0)}/{row['score_summary']['examples']} | "
            f"{loops.get('2', {}).get('correct', 0)}/{row['score_summary']['examples']} | "
            f"{loops.get('3', {}).get('correct', 0)}/{row['score_summary']['examples']} | "
            f"{row['score_summary']['oracle_correct']}/{row['score_summary']['examples']} | "
            f"{row['score_summary']['oracle_gap_vs_loop1']} | "
            f"{row['score_summary']['rescued_vs_loop1']} | "
            f"{row['score_summary']['harmed_vs_loop1']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mcq_jsonl", default="")
    parser.add_argument("--arc_config", default="ARC-Challenge")
    parser.add_argument("--arc_split", default="validation")
    parser.add_argument("--arc_offset", type=int, default=0)
    parser.add_argument("--arc_limit", type=int, default=256)
    parser.add_argument("--arc_seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt_style", default="question_only", choices=("question_only", "with_options"))
    parser.add_argument("--score_target", default="option_text", choices=("label", "option_text", "label_and_text"))
    parser.add_argument("--score_loops", default="1,2,3")
    parser.add_argument("--tail_loop_counts", default="1,2,3,4,8")
    parser.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--n_tail", type=int, default=7)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--reentry_rescale_mode", default="none", choices=("none", "entry_rms"))
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source_summary", default="")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args)
    if not examples:
        raise ValueError("No examples for tail-damper sweep")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper = load_recurrent_wrapper(args, args.checkpoint)

    score_loops = parse_csv_ints(args.score_loops)
    tail_loop_counts = parse_csv_ints(args.tail_loop_counts)
    strengths = parse_csv_floats(args.strengths)

    calibration_tokens = collect_tokens(
        wrapper=wrapper,
        tokenizer=tokenizer,
        examples=examples,
        args=args,
        loop_counts=tail_loop_counts,
    )
    sigma_entry, mean_entry, basis, entry_evals = tail_basis(calibration_tokens["entry"], args.n_tail)
    loop1_cov = centered_covariance(calibration_tokens["loop1"])[0]
    decomp = tail_decomposition(sigma_entry, loop1_cov, n_tail=args.n_tail)
    calibration = {
        "dominant_entry_eigenvalue": finite_float(entry_evals[0]),
        "tail_decomposition_loop1": decomp,
        "correction_class": correction_class(decomp),
    }
    damper_path = output_dir / "tail_damper.pt"
    damper_summary = {
        "kind": "stage5_tail_damper_depth_sweep",
        "run_id": output_dir.name,
        "checkpoint": args.checkpoint,
        "benchmark": args.arc_config,
        "score_target": args.score_target,
        "n_tail": args.n_tail,
        "hidden_dim": int(calibration_tokens["entry"].shape[1]),
    }
    write_damper_artifact(
        path=damper_path,
        mean_entry=mean_entry,
        basis=basis,
        decomp=decomp,
        summary=damper_summary,
    )

    damper_payload = {
        "mean": mean_entry.detach().float().cpu(),
        "basis": basis.detach().float().cpu(),
        "damper_scale": torch.tensor(decomp["damper_scale"], dtype=torch.float32),
    }
    strength_summaries: list[dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()
    for strength in strengths:
        print(f"\n===== tail_damper_strength={strength} =====", flush=True)
        tokens = (
            calibration_tokens
            if strength == 0.0
            else collect_tokens(
                wrapper=wrapper,
                tokenizer=tokenizer,
                examples=examples,
                args=args,
                loop_counts=tail_loop_counts,
                damper=damper_payload,
                strength=strength,
            )
        )
        tail = tail_trace_summary(tokens, mean_entry=mean_entry, basis=basis, loop_counts=tail_loop_counts)
        records, score_summary = score_examples(
            wrapper=wrapper,
            tokenizer=tokenizer,
            examples=examples,
            args=args,
            loops=score_loops,
            damper_path=damper_path,
            strength=strength,
        )
        with records_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        row = {"strength": strength, "tail_trace": tail, "score_summary": score_summary}
        strength_summaries.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary = {
        "kind": "stage5_tail_damper_depth_sweep",
        "run_id": output_dir.name,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "source_summary": args.source_summary or None,
        "examples": len(examples),
        "arc_config": args.arc_config,
        "arc_split": args.arc_split,
        "arc_offset": args.arc_offset,
        "arc_limit": args.arc_limit,
        "score_target": args.score_target,
        "score_loops": score_loops,
        "tail_loop_counts": tail_loop_counts,
        "strengths": strengths,
        "reentry_rescale_mode": args.reentry_rescale_mode,
        "damper_artifact": path_for_cli(damper_path),
        "records_jsonl": path_for_cli(records_path),
        "calibration": calibration,
        "strength_summaries": strength_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = output_dir / "summary.md"
    write_markdown(summary, md_path)
    print(md_path.read_text(encoding="utf-8"), flush=True)
    print(f"saved_summary={path_for_cli(summary_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
