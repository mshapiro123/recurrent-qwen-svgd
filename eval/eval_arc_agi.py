"""Evaluate base or recurrent Qwen on ARC-AGI JSON grid tasks.

This is a literal grid-generation harness. It is intentionally separate from
the ARC-Challenge multiple-choice proxy used for early recurrent recovery.
"""

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

from eval.arc_agi_utils import (  # noqa: E402
    ArcAgiExample,
    grid_to_json_text,
    load_arc_agi_examples,
    parse_grid_from_text,
    render_arc_prompt,
    score_grid_prediction,
)
from eval.eval_best_of_k_jsonl import generate_candidates, load_wrapper, parse_optional_float  # noqa: E402
from eval.eval_identity import model_load_kwargs  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    model.eval()
    return model


def trim_completion(text: str) -> str:
    cut = len(text)
    for marker in ("<|im_start|>", "<|im_end|>", "Training example", "Test input:"):
        idx = text.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def generate_base_candidates(
    model,
    tokenizer,
    prompt: str,
    *,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    seed: int,
) -> list[str]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    candidates: list[str] = []
    with torch.no_grad():
        for idx in range(num_candidates):
            set_seed(seed + idx)
            do_sample = temperature > 0
            kwargs: dict[str, Any] = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded.get("attention_mask"),
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": do_sample,
            }
            if do_sample:
                kwargs["temperature"] = temperature
            output = model.generate(**kwargs)
            completion = output[:, encoded["input_ids"].shape[-1] :]
            candidates.extend(trim_completion(text) for text in tokenizer.batch_decode(completion, skip_special_tokens=True))
    return candidates


def generate_recurrent_candidates(
    wrapper,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, float], int]:
    if args.num_trajectories > 1:
        result = generate_candidates(
            wrapper,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            max_loops=args.max_loops,
            num_trajectories=args.num_trajectories,
            sample_latents=args.sample_latents,
            latent_injection_mode=args.latent_injection_mode,
            particle_update_mode=args.particle_update_mode,
            particle_init_noise=args.particle_init_noise,
            svgd_eps=args.svgd_eps,
            svgd_repulsion_scale=args.svgd_repulsion_scale,
            svgd_bandwidth=args.svgd_bandwidth,
            svgd_bandwidth_floor=args.svgd_bandwidth_floor,
            svgd_repulsion_max_norm=args.svgd_repulsion_max_norm,
            svgd_kernel_projection_dim=args.svgd_kernel_projection_dim,
            svgd_kernel_projection_path=args.svgd_kernel_projection_path,
            svgd_kernel_geometry=args.svgd_kernel_geometry,
            svgd_projection_seed=args.svgd_projection_seed,
            particle_noise_every_step=args.particle_noise_every_step,
            particle_noise_steps=args.particle_noise_steps,
            stop_on_final_answer=False,
            temperature=args.temperature,
            device=args.device,
        )
        return result.candidates, result.diagnostics, result.generation_steps

    candidates: list[str] = []
    diagnostics: dict[str, float] = {}
    generation_steps = 0
    for idx in range(args.num_candidates):
        set_seed(args.seed + idx)
        result = generate_candidates(
            wrapper,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            max_loops=args.max_loops,
            num_trajectories=1,
            sample_latents=args.sample_latents,
            latent_injection_mode=args.latent_injection_mode,
            particle_update_mode="none",
            particle_init_noise=0.0,
            svgd_eps=args.svgd_eps,
            svgd_repulsion_scale=args.svgd_repulsion_scale,
            svgd_bandwidth=args.svgd_bandwidth,
            svgd_bandwidth_floor=args.svgd_bandwidth_floor,
            svgd_repulsion_max_norm=args.svgd_repulsion_max_norm,
            svgd_kernel_projection_dim=args.svgd_kernel_projection_dim,
            svgd_kernel_projection_path=args.svgd_kernel_projection_path,
            svgd_kernel_geometry=args.svgd_kernel_geometry,
            svgd_projection_seed=args.svgd_projection_seed,
            particle_noise_every_step=False,
            particle_noise_steps=0,
            stop_on_final_answer=False,
            temperature=args.temperature,
            device=args.device,
        )
        candidates.extend(result.candidates)
        diagnostics = result.diagnostics
        generation_steps = max(generation_steps, result.generation_steps)
    return candidates, diagnostics, generation_steps


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_example(
    example: ArcAgiExample,
    candidates: list[str],
    *,
    diagnostics: dict[str, float],
    generation_steps: int,
    output_format: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    any_exact = False
    first_exact = False
    valid_count = 0
    target = example.test_output
    for idx, text in enumerate(candidates):
        parsed = parse_grid_from_text(text, output_format=output_format)
        score = score_grid_prediction(parsed, target)
        valid_count += int(bool(score["valid"]))
        exact = bool(score.get("exact"))
        if idx == 0:
            first_exact = exact
        any_exact = any_exact or exact
        rows.append(
            {
                "task_id": example.task_id,
                "test_index": example.test_index,
                "candidate_index": idx,
                "candidate_text": text,
                "parsed_grid": parsed,
                "target_grid": target,
                "target_json": grid_to_json_text(target) if target is not None else None,
                "score": score,
                "diagnostics": diagnostics,
                "generation_steps": generation_steps,
            }
        )
    summary = {
        "task_id": example.task_id,
        "test_index": example.test_index,
        "has_target": target is not None,
        "first_exact": first_exact if target is not None else None,
        "best_of_k_exact": any_exact if target is not None else None,
        "valid_candidates": valid_count,
        "num_candidates": len(candidates),
    }
    return rows, summary


def summarize_examples(example_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in example_summaries if item["has_target"]]
    first = sum(1 for item in scored if item["first_exact"])
    best = sum(1 for item in scored if item["best_of_k_exact"])
    task_ids = sorted({item["task_id"] for item in scored})
    solved_tasks = 0
    for task_id in task_ids:
        task_items = [item for item in scored if item["task_id"] == task_id]
        if task_items and all(item["best_of_k_exact"] for item in task_items):
            solved_tasks += 1
    return {
        "examples_with_targets": len(scored),
        "first_exact": first,
        "best_of_k_exact": best,
        "first_accuracy": first / max(len(scored), 1),
        "best_of_k_accuracy": best / max(len(scored), 1),
        "tasks_with_targets": len(task_ids),
        "tasks_solved_best_of_k": solved_tasks,
        "task_solve_rate_best_of_k": solved_tasks / max(len(task_ids), 1),
        "valid_candidate_rate": sum(item["valid_candidates"] for item in example_summaries)
        / max(sum(item["num_candidates"] for item in example_summaries), 1),
    }


def write_summary(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_md(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    summary = payload["summary"]
    lines = [
        f"# ARC-AGI Evaluation - {payload['mode']}",
        "",
        f"- Tasks path: `{payload['tasks_path']}`",
        f"- Grid format: `{payload['grid_format']}`",
        f"- Examples with targets: `{summary['examples_with_targets']}`",
        f"- First-candidate exact: `{summary['first_exact']}` / `{summary['examples_with_targets']}` = `{summary['first_accuracy']}`",
        f"- Best-of-K exact: `{summary['best_of_k_exact']}` / `{summary['examples_with_targets']}` = `{summary['best_of_k_accuracy']}`",
        f"- Tasks solved best-of-K: `{summary['tasks_solved_best_of_k']}` / `{summary['tasks_with_targets']}` = `{summary['task_solve_rate_best_of_k']}`",
        f"- Valid candidate rate: `{summary['valid_candidate_rate']}`",
        "",
        "This is exact-grid scoring on ARC-AGI-format tasks, not ARC-Challenge multiple choice.",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks_path", required=True)
    parser.add_argument("--solutions_path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=("base", "phase1", "phase2"), default="base")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--summary_json")
    parser.add_argument("--summary_md")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--grid_format", default="json", choices=("json", "compact", "tagged"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--sample_latents", action="store_true")
    parser.add_argument("--latent_injection_mode", default="post", choices=("pre", "post", "both"))
    parser.add_argument("--particle_update_mode", default="none", choices=("none", "svgd"))
    parser.add_argument("--particle_init_noise", type=float, default=0.0)
    parser.add_argument("--svgd_eps", type=float, default=1.0)
    parser.add_argument("--svgd_repulsion_scale", type=float, default=0.5)
    parser.add_argument("--svgd_bandwidth", default="median")
    parser.add_argument("--svgd_bandwidth_floor", type=float, default=1e-6)
    parser.add_argument("--svgd_repulsion_max_norm", type=parse_optional_float)
    parser.add_argument("--svgd_kernel_projection_dim", type=int)
    parser.add_argument("--svgd_kernel_projection_path")
    parser.add_argument("--svgd_kernel_geometry", default="euclidean", choices=("euclidean", "spherical"))
    parser.add_argument("--svgd_projection_seed", type=int, default=0)
    parser.add_argument("--particle_noise_every_step", action="store_true")
    parser.add_argument("--particle_noise_steps", type=int, default=32)
    args = parser.parse_args()

    if args.mode != "base" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for phase1/phase2 modes")
    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_jsonl).write_text("", encoding="utf-8")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_arc_agi_examples(
        args.tasks_path,
        solutions_path=args.solutions_path,
        limit=args.limit,
    )
    print(f"loaded_examples={len(examples)}")

    if args.mode == "base":
        model_or_wrapper = load_base(args)
    else:
        model_or_wrapper = load_wrapper(args, args.checkpoint)

    example_summaries: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        prompt = render_arc_prompt(example, output_format=args.grid_format)
        if args.mode == "base":
            candidates = generate_base_candidates(
                model_or_wrapper,
                tokenizer,
                prompt,
                num_candidates=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=args.device,
                seed=args.seed + idx * max(args.num_candidates, 1),
            )
            diagnostics: dict[str, float] = {}
            generation_steps = args.max_new_tokens
        else:
            candidates, diagnostics, generation_steps = generate_recurrent_candidates(
                model_or_wrapper,
                tokenizer,
                prompt,
                args,
            )
        rows, example_summary = evaluate_example(
            example,
            candidates,
            diagnostics=diagnostics,
            generation_steps=generation_steps,
            output_format=args.grid_format,
        )
        example_summaries.append(example_summary)
        for row in rows:
            row.update({"mode": args.mode, "prompt": prompt})
            append_jsonl(args.output_jsonl, row)
        print(json.dumps(example_summary, ensure_ascii=False))

    payload = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "tasks_path": args.tasks_path,
        "solutions_path": args.solutions_path,
        "limit": args.limit,
        "num_candidates": args.num_candidates,
        "num_trajectories": args.num_trajectories,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "grid_format": args.grid_format,
        "summary": summarize_examples(example_summaries),
        "examples": example_summaries,
    }
    write_summary(args.summary_json, payload)
    write_summary_md(args.summary_md, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
