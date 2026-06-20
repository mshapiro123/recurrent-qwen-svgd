"""Evaluate base or recurrent Qwen on ARC-AGI JSON grid tasks.

This is a literal grid-generation harness. It is intentionally separate from
the ARC-Challenge multiple-choice proxy used for early recurrent recovery.
"""

from __future__ import annotations

import argparse
import json
import re
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
    Grid,
    format_grid_completion,
    grid_to_json_text,
    load_arc_agi_examples,
    parse_grid_from_text,
    render_arc_prompt,
    score_grid_prediction,
)
from eval.arc_agi_program import arc_program_training_match_count, parse_arc_program_from_text  # noqa: E402
from eval.arc_agi_symbolic import SymbolicCandidate, format_symbolic_program_trace, symbolic_candidates  # noqa: E402
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


def format_symbolic_candidate_texts(
    candidates: list[SymbolicCandidate],
    *,
    output_format: str,
    candidate_format: str,
) -> list[tuple[str, str]]:
    if candidate_format not in {"grid", "program", "both"}:
        raise ValueError(f"Unknown symbolic candidate format: {candidate_format!r}")

    rows: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate_format in {"grid", "both"}:
            rows.append((format_grid_completion(candidate.grid, output_format=output_format), "symbolic_grid"))
        if candidate_format in {"program", "both"}:
            rows.append((format_symbolic_program_trace(candidate), "symbolic_program"))
    return rows


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def inferred_output_shapes(example: ArcAgiExample) -> list[tuple[int, int]]:
    """Infer plausible output shapes from demonstrations without using target."""

    shapes: list[tuple[int, int]] = []
    train_pairs = [pair for pair in example.train if pair.output is not None]
    if not train_pairs:
        return shapes

    input_shapes = [grid_shape(pair.input) for pair in train_pairs]
    output_shapes = [grid_shape(pair.output) for pair in train_pairs if pair.output is not None]
    test_shape = grid_shape(example.test_input)

    if all(in_shape == out_shape for in_shape, out_shape in zip(input_shapes, output_shapes)):
        shapes.append(test_shape)

    if len(set(output_shapes)) == 1:
        shapes.append(output_shapes[0])

    deltas = [(out_h - in_h, out_w - in_w) for (in_h, in_w), (out_h, out_w) in zip(input_shapes, output_shapes)]
    if len(set(deltas)) == 1:
        delta_h, delta_w = deltas[0]
        candidate = (test_shape[0] + delta_h, test_shape[1] + delta_w)
        if 1 <= candidate[0] <= 30 and 1 <= candidate[1] <= 30:
            shapes.append(candidate)

    seen: set[tuple[int, int]] = set()
    deduped = []
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            deduped.append(shape)
    return deduped


def select_candidate_index(example: ArcAgiExample, candidate_rows: list[dict[str, Any]]) -> int:
    preferred_shapes = set(inferred_output_shapes(example))
    for idx, row in enumerate(candidate_rows):
        if (
            row["parsed_grid"] is not None
            and row.get("parse_method") == "program"
            and row.get("program_fits_train")
        ):
            return idx

    first_valid: int | None = None
    for idx, row in enumerate(candidate_rows):
        parsed = row["parsed_grid"]
        if parsed is None:
            continue
        if first_valid is None:
            first_valid = idx
        if preferred_shapes and grid_shape(parsed) in preferred_shapes:
            return idx
    if first_valid is not None:
        return first_valid
    return 0


def evaluate_example(
    example: ArcAgiExample,
    candidates: list[str],
    *,
    candidate_sources: list[str] | None = None,
    diagnostics: dict[str, float],
    generation_steps: int,
    output_format: str,
    program_parse_mode: str = "fallback",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if program_parse_mode not in {"off", "fallback", "prefer", "program_only"}:
        raise ValueError(f"Unknown program_parse_mode={program_parse_mode!r}")
    rows: list[dict[str, Any]] = []
    any_exact = False
    first_exact = False
    selected_exact = False
    valid_count = 0
    target = example.test_output
    sources = candidate_sources or ["model"] * len(candidates)
    for idx, text in enumerate(candidates):
        parsed, parse_method = parse_candidate_grid(
            example,
            text,
            output_format=output_format,
            program_parse_mode=program_parse_mode,
        )
        score = score_grid_prediction(parsed, target)
        program_train_matches, program_train_total = arc_program_training_match_count(example, text)
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
                "candidate_source": sources[idx] if idx < len(sources) else "unknown",
                "candidate_text": text,
                "parsed_grid": parsed,
                "parse_method": parse_method,
                "program_train_matches": program_train_matches,
                "program_train_total": program_train_total,
                "program_fits_train": program_train_total > 0 and program_train_matches == program_train_total,
                "target_grid": target,
                "target_json": grid_to_json_text(target) if target is not None else None,
                "score": score,
                "diagnostics": diagnostics,
                "generation_steps": generation_steps,
            }
        )
    selected_index = select_candidate_index(example, rows) if rows else 0
    if rows:
        rows[selected_index]["selected"] = True
        selected_exact = bool(rows[selected_index]["score"].get("exact"))
        for idx, row in enumerate(rows):
            row.setdefault("selected", idx == selected_index)
    summary = {
        "task_id": example.task_id,
        "test_index": example.test_index,
        "has_target": target is not None,
        "first_exact": first_exact if target is not None else None,
        "selected_exact": selected_exact if target is not None else None,
        "selected_index": selected_index,
        "inferred_shapes": [list(shape) for shape in inferred_output_shapes(example)],
        "best_of_k_exact": any_exact if target is not None else None,
        "valid_candidates": valid_count,
        "num_candidates": len(candidates),
    }
    return rows, summary


def parse_candidate_grid(
    example: ArcAgiExample,
    text: str,
    *,
    output_format: str,
    program_parse_mode: str,
) -> tuple[Grid | None, str]:
    if program_parse_mode == "program_only":
        program_grid = parse_arc_program_from_text(example, text)
        return program_grid, "program" if program_grid is not None else "none"

    if program_parse_mode == "prefer":
        program_grid = parse_arc_program_from_text(example, text)
        if program_grid is not None:
            return program_grid, "program"

    grid = parse_grid_from_text(text, output_format=output_format)
    if grid is not None:
        return grid, "grid"

    if program_parse_mode == "fallback":
        program_grid = parse_arc_program_from_text(example, text)
        if program_grid is not None:
            return program_grid, "program"

    return None, "none"


def summarize_examples(example_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in example_summaries if item["has_target"]]
    first = sum(1 for item in scored if item["first_exact"])
    selected = sum(1 for item in scored if item["selected_exact"])
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
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_accuracy": first / max(len(scored), 1),
        "selected_accuracy": selected / max(len(scored), 1),
        "best_of_k_accuracy": best / max(len(scored), 1),
        "tasks_with_targets": len(task_ids),
        "tasks_solved_best_of_k": solved_tasks,
        "task_solve_rate_best_of_k": solved_tasks / max(len(task_ids), 1),
        "valid_candidate_rate": sum(item["valid_candidates"] for item in example_summaries)
        / max(sum(item["num_candidates"] for item in example_summaries), 1),
    }


def task_family(task_id: str) -> str:
    synthetic_match = re.match(r"^synthetic_(.+)_\d{6}(?::.*)?$", task_id)
    if synthetic_match:
        return synthetic_match.group(1)
    return "arc"


def summarize_task_families(example_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in example_summaries if item["has_target"]]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in scored:
        by_family.setdefault(task_family(str(item["task_id"])), []).append(item)

    summary: dict[str, Any] = {}
    for family, items in sorted(by_family.items()):
        first = sum(1 for item in items if item["first_exact"])
        selected = sum(1 for item in items if item["selected_exact"])
        best = sum(1 for item in items if item["best_of_k_exact"])
        task_ids = sorted({item["task_id"] for item in items})
        solved_tasks = 0
        for task_id in task_ids:
            task_items = [item for item in items if item["task_id"] == task_id]
            if task_items and all(item["best_of_k_exact"] for item in task_items):
                solved_tasks += 1
        valid_candidates = sum(item["valid_candidates"] for item in items)
        total_candidates = sum(item["num_candidates"] for item in items)
        summary[family] = {
            "examples_with_targets": len(items),
            "first_exact": first,
            "selected_exact": selected,
            "best_of_k_exact": best,
            "first_accuracy": first / max(len(items), 1),
            "selected_accuracy": selected / max(len(items), 1),
            "best_of_k_accuracy": best / max(len(items), 1),
            "tasks_with_targets": len(task_ids),
            "tasks_solved_best_of_k": solved_tasks,
            "task_solve_rate_best_of_k": solved_tasks / max(len(task_ids), 1),
            "valid_candidate_rate": valid_candidates / max(total_candidates, 1),
        }
    return summary


def summarize_candidate_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row.get("candidate_source", "unknown"))
        stats = by_source.setdefault(source, {"count": 0, "valid": 0, "exact": 0, "selected": 0, "selected_exact": 0})
        stats["count"] += 1
        score = row.get("score", {})
        stats["valid"] += int(bool(score.get("valid")))
        stats["exact"] += int(bool(score.get("exact")))
        stats["selected"] += int(bool(row.get("selected")))
        stats["selected_exact"] += int(bool(row.get("selected")) and bool(score.get("exact")))
    return by_source


def summarize_parse_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, int]] = {}
    for row in rows:
        method = str(row.get("parse_method", "unknown"))
        stats = by_method.setdefault(method, {"count": 0, "exact": 0, "selected": 0, "selected_exact": 0})
        stats["count"] += 1
        score = row.get("score", {})
        stats["exact"] += int(bool(score.get("exact")))
        stats["selected"] += int(bool(row.get("selected")))
        stats["selected_exact"] += int(bool(row.get("selected")) and bool(score.get("exact")))
    return by_method


def summarize_program_verifier(rows: list[dict[str, Any]]) -> dict[str, int]:
    with_program = [row for row in rows if int(row.get("program_train_total", 0)) > 0]
    fits = [row for row in with_program if row.get("program_fits_train")]
    return {
        "candidates_with_program": len(with_program),
        "candidates_program_fits_train": len(fits),
        "program_fit_exact": sum(1 for row in fits if bool(row.get("score", {}).get("exact"))),
        "program_fit_selected": sum(1 for row in fits if bool(row.get("selected"))),
        "program_fit_selected_exact": sum(
            1 for row in fits if bool(row.get("selected")) and bool(row.get("score", {}).get("exact"))
        ),
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
    source_summary = payload.get("candidate_source_summary", {})
    parse_summary = payload.get("parse_method_summary", {})
    program_verifier = payload.get("program_verifier_summary", {})
    family_summary = payload.get("task_family_summary", {})
    lines = [
        f"# ARC-AGI Evaluation - {payload['mode']}",
        "",
        f"- Tasks path: `{payload['tasks_path']}`",
        f"- Grid format: `{payload['grid_format']}`",
        f"- Program parse mode: `{payload['program_parse_mode']}`",
        f"- Symbolic candidate format: `{payload.get('symbolic_candidate_format', 'grid')}`",
        f"- Examples with targets: `{summary['examples_with_targets']}`",
        f"- First-candidate exact: `{summary['first_exact']}` / `{summary['examples_with_targets']}` = `{summary['first_accuracy']}`",
        f"- Selected-candidate exact: `{summary['selected_exact']}` / `{summary['examples_with_targets']}` = `{summary['selected_accuracy']}`",
        f"- Best-of-K exact: `{summary['best_of_k_exact']}` / `{summary['examples_with_targets']}` = `{summary['best_of_k_accuracy']}`",
        f"- Tasks solved best-of-K: `{summary['tasks_solved_best_of_k']}` / `{summary['tasks_with_targets']}` = `{summary['task_solve_rate_best_of_k']}`",
        f"- Valid candidate rate: `{summary['valid_candidate_rate']}`",
        "",
        "This is exact-grid scoring on ARC-AGI-format tasks, not ARC-Challenge multiple choice.",
    ]
    if source_summary:
        lines += ["", "## Candidate Sources"]
        for source, stats in sorted(source_summary.items()):
            lines.append(
                f"- `{source}`: count `{stats['count']}`, valid `{stats['valid']}`, "
                f"exact `{stats['exact']}`, selected `{stats['selected']}`, "
                f"selected_exact `{stats['selected_exact']}`"
            )
    if family_summary:
        lines += ["", "## Task Families"]
        for family, stats in sorted(family_summary.items()):
            lines.append(
                f"- `{family}`: selected `{stats['selected_exact']}` / `{stats['examples_with_targets']}`, "
                f"best `{stats['best_of_k_exact']}` / `{stats['examples_with_targets']}`, "
                f"tasks `{stats['tasks_solved_best_of_k']}` / `{stats['tasks_with_targets']}`, "
                f"valid_rate `{stats['valid_candidate_rate']}`"
            )
    if parse_summary:
        lines += ["", "## Parse Methods"]
        for method, stats in sorted(parse_summary.items()):
            lines.append(
                f"- `{method}`: count `{stats['count']}`, exact `{stats['exact']}`, "
                f"selected `{stats['selected']}`, selected_exact `{stats['selected_exact']}`"
            )
    if program_verifier:
        lines += [
            "",
            "## Program Verifier",
            f"- Candidates with executable programs: `{program_verifier['candidates_with_program']}`",
            f"- Candidates fitting all demonstrations: `{program_verifier['candidates_program_fits_train']}`",
            f"- Program-fit exact candidates: `{program_verifier['program_fit_exact']}`",
            f"- Program-fit selected candidates: `{program_verifier['program_fit_selected']}`",
            f"- Program-fit selected exact: `{program_verifier['program_fit_selected_exact']}`",
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
    parser.add_argument(
        "--program_parse_mode",
        default="fallback",
        choices=("off", "fallback", "prefer", "program_only"),
        help=(
            "How to use tiny symbolic_program traces during scoring: off=grid only, "
            "fallback=execute only if no grid parses, prefer=execute before grid parsing, "
            "program_only=ignore literal grids."
        ),
    )
    parser.add_argument("--include_symbolic_candidates", action="store_true")
    parser.add_argument(
        "--symbolic_position",
        default="after_model",
        choices=("before_model", "after_model", "only"),
        help="Where to place deterministic symbolic ARC candidates relative to model candidates.",
    )
    parser.add_argument(
        "--symbolic_candidate_format",
        default="grid",
        choices=("grid", "program", "both"),
        help="How to render deterministic symbolic candidates when included.",
    )
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

    symbolic_only = args.include_symbolic_candidates and args.symbolic_position == "only"

    if not symbolic_only and args.mode != "base" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for phase1/phase2 modes")
    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_jsonl).write_text("", encoding="utf-8")

    set_seed(args.seed)
    tokenizer = None
    if not symbolic_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    examples = load_arc_agi_examples(
        args.tasks_path,
        solutions_path=args.solutions_path,
        limit=args.limit,
    )
    print(f"loaded_examples={len(examples)}")

    model_or_wrapper = None
    if symbolic_only:
        print("symbolic_only=True; skipping model load")
    elif args.mode == "base":
        model_or_wrapper = load_base(args)
    else:
        model_or_wrapper = load_wrapper(args, args.checkpoint)

    example_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        prompt = render_arc_prompt(example, output_format=args.grid_format)
        candidates: list[str] = []
        candidate_sources: list[str] = []
        diagnostics: dict[str, float] = {}
        generation_steps = 0
        symbolic_rows: list[tuple[str, str]] = []
        if args.include_symbolic_candidates:
            symbolic_rows = format_symbolic_candidate_texts(
                symbolic_candidates(example),
                output_format=args.grid_format,
                candidate_format=args.symbolic_candidate_format,
            )

        if args.include_symbolic_candidates and args.symbolic_position == "before_model":
            candidates.extend(text for text, _source in symbolic_rows)
            candidate_sources.extend(source for _text, source in symbolic_rows)

        if not (args.include_symbolic_candidates and args.symbolic_position == "only") and args.mode == "base":
            assert tokenizer is not None and model_or_wrapper is not None
            generated = generate_base_candidates(
                model_or_wrapper,
                tokenizer,
                prompt,
                num_candidates=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=args.device,
                seed=args.seed + idx * max(args.num_candidates, 1),
            )
            candidates.extend(generated)
            candidate_sources.extend(["model"] * len(generated))
            generation_steps = args.max_new_tokens
        elif not (args.include_symbolic_candidates and args.symbolic_position == "only"):
            assert tokenizer is not None and model_or_wrapper is not None
            generated, diagnostics, generation_steps = generate_recurrent_candidates(
                model_or_wrapper,
                tokenizer,
                prompt,
                args,
            )
            candidates.extend(generated)
            candidate_sources.extend(["model"] * len(generated))

        if args.include_symbolic_candidates and args.symbolic_position == "after_model":
            candidates.extend(text for text, _source in symbolic_rows)
            candidate_sources.extend(source for _text, source in symbolic_rows)
        elif args.include_symbolic_candidates and args.symbolic_position == "only":
            candidates.extend(text for text, _source in symbolic_rows)
            candidate_sources.extend(source for _text, source in symbolic_rows)

        rows, example_summary = evaluate_example(
            example,
            candidates,
            candidate_sources=candidate_sources,
            diagnostics=diagnostics,
            generation_steps=generation_steps,
            output_format=args.grid_format,
            program_parse_mode=args.program_parse_mode,
        )
        example_summaries.append(example_summary)
        for row in rows:
            row.update({"mode": args.mode, "prompt": prompt})
            append_jsonl(args.output_jsonl, row)
            all_rows.append(row)
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
        "program_parse_mode": args.program_parse_mode,
        "include_symbolic_candidates": args.include_symbolic_candidates,
        "symbolic_position": args.symbolic_position,
        "symbolic_candidate_format": args.symbolic_candidate_format,
        "summary": summarize_examples(example_summaries),
        "candidate_source_summary": summarize_candidate_sources(all_rows),
        "task_family_summary": summarize_task_families(example_summaries),
        "parse_method_summary": summarize_parse_methods(all_rows),
        "program_verifier_summary": summarize_program_verifier(all_rows),
        "examples": example_summaries,
    }
    write_summary(args.summary_json, payload)
    write_summary_md(args.summary_md, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
