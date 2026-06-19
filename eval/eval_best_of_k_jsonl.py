"""Compare single-trajectory and best-of-K recurrent generation on exact tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from models.lora import apply_lora_to_recurrent_block
from models.recurrent_wrapper import RecurrentQwenForCausalLM
from training.checkpointing import load_trainable_checkpoint


@dataclass(frozen=True)
class ExactTask:
    name: str
    prompt: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class GenerationResult:
    candidates: list[str]
    diagnostics: dict[str, float]
    generation_steps: int


def read_tasks(path: str | Path) -> list[ExactTask]:
    tasks: list[ExactTask] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        patterns = tuple(row.get("patterns", ()))
        if not patterns:
            raise ValueError(f"Task has no patterns: {row}")
        tasks.append(ExactTask(name=row["name"], prompt=row["prompt"], patterns=patterns))
    return tasks


def matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def trim_completion(text: str) -> str:
    """Remove obvious prompt-continuation spillover from generated candidates."""

    cut = len(text)
    for marker in ("Human:", "<|im_start|>", "<|im_end|>"):
        idx = text.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def has_final_answer_shape(text: str) -> bool:
    """Heuristic early-stop signal for smoke tests, not a general verifier."""

    text = trim_completion(text)
    final_patterns = (
        r"\bfinal answer\b",
        r"\btherefore\b",
        r"\bso,\s+the\b",
        r"\banswer is\b",
        r"\\boxed\{",
        r"\b\d+\s*(miles per hour|mph)\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in final_patterns)


def load_wrapper(args: argparse.Namespace, checkpoint: str | None) -> RecurrentQwenForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    adapter_dtype = resolve_dtype(args.adapter_dtype)
    replaced = apply_lora_to_recurrent_block(
        wrapper,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        adapter_dtype=adapter_dtype,
    )
    print(f"lora_recurrent_modules={replaced}")
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    if checkpoint:
        load_info = load_trainable_checkpoint(wrapper, checkpoint)
        print(f"loaded_checkpoint={checkpoint} loaded_keys={len(load_info['loaded_keys'])}")
        if load_info["skipped"]:
            print(f"skipped_keys={len(load_info['skipped'])}")
    wrapper.eval()
    return wrapper


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}"


def _metric_value(value: object) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _record_metric_history(history: dict[str, list[float]], metrics: dict[str, object]) -> None:
    for key, value in metrics.items():
        scalar = _metric_value(value)
        if scalar is not None:
            history.setdefault(key, []).append(scalar)


def _mean_metric_history(history: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: sum(values) / max(len(values), 1)
        for key, values in sorted(history.items())
        if values
    }


def phase2_run_descriptor(args: argparse.Namespace, *, seed: int | None = None, steps: int | None = None) -> str:
    parts = [
        f"mode={args.phase2_particle_update_mode}",
        f"temp={_format_float(args.temperature)}",
        f"noise={_format_float(args.particle_init_noise)}",
    ]
    if args.particle_noise_every_step or args.particle_noise_steps_sweep:
        parts.append(f"noise_steps={steps if steps is not None else args.particle_noise_steps}")
    if args.phase2_particle_update_mode == "svgd":
        parts.extend(
            [
                f"eps={_format_float(args.svgd_eps)}",
                f"repulsion={_format_float(args.svgd_repulsion_scale)}",
                f"max_norm={_format_float(args.svgd_repulsion_max_norm)}",
                f"proj_dim={args.svgd_kernel_projection_dim or 0}",
                f"geom={args.svgd_kernel_geometry}",
            ]
        )
        if args.svgd_kernel_projection_path:
            parts.append(f"proj_path={Path(args.svgd_kernel_projection_path).stem}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return " ".join(parts)


def _next_token(
    output: object,
    num_trajectories: int,
    temperature: float,
) -> torch.Tensor:
    if num_trajectories > 1:
        logits = output.trajectory_logits[:, :, -1, :]
        if temperature <= 0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(
            logits.shape[0],
            num_trajectories,
        )

    logits = output.logits[:, -1, :]
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_candidates(
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_loops: int,
    num_trajectories: int,
    sample_latents: bool,
    latent_injection_mode: str,
    particle_update_mode: str,
    particle_init_noise: float,
    svgd_eps: float,
    svgd_repulsion_scale: float,
    svgd_bandwidth: str,
    svgd_bandwidth_floor: float,
    svgd_repulsion_max_norm: float | None,
    svgd_kernel_projection_dim: int | None,
    svgd_kernel_projection_path: str | None,
    svgd_kernel_geometry: str,
    svgd_projection_seed: int,
    particle_noise_every_step: bool,
    particle_noise_steps: int,
    stop_on_final_answer: bool,
    temperature: float,
    device: str,
    early_stop_patterns: tuple[str, ...] = (),
) -> GenerationResult:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = encoded["input_ids"].shape[-1]
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    particle_noise = particle_init_noise
    metrics_history: dict[str, list[float]] = {}
    generation_steps = 0

    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            generation_steps = step_idx + 1
            output = wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_loops=max_loops,
                num_trajectories=num_trajectories,
                sample_latents=sample_latents,
                latent_injection_mode=latent_injection_mode,
                particle_update_mode=particle_update_mode,
                particle_init_noise=particle_noise,
                svgd_eps=svgd_eps,
                svgd_repulsion_scale=svgd_repulsion_scale,
                svgd_bandwidth=svgd_bandwidth,
                svgd_bandwidth_floor=svgd_bandwidth_floor,
                svgd_repulsion_max_norm=svgd_repulsion_max_norm,
                svgd_kernel_projection_dim=svgd_kernel_projection_dim,
                svgd_kernel_projection_path=svgd_kernel_projection_path,
                svgd_kernel_geometry=svgd_kernel_geometry,
                svgd_projection_seed=svgd_projection_seed,
                use_cache=False,
                return_dict=True,
            )
            _record_metric_history(metrics_history, output.metrics)
            next_token = _next_token(output, num_trajectories, temperature)
            if num_trajectories == 1:
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.ones_like(input_ids)
            elif input_ids.dim() == 2:
                input_ids = input_ids[:, None, :].expand(-1, num_trajectories, -1).contiguous()
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.ones_like(input_ids)
            if particle_noise_every_step and step_idx + 1 < particle_noise_steps:
                particle_noise = particle_init_noise
            else:
                particle_noise = 0.0

            current = (
                input_ids[:, prompt_len:]
                if num_trajectories == 1
                else input_ids[:, :, prompt_len:].reshape(num_trajectories, -1)
            )
            decoded_so_far = [
                trim_completion(text)
                for text in tokenizer.batch_decode(current, skip_special_tokens=True)
            ]
            if early_stop_patterns and all(matches_any(text, early_stop_patterns) for text in decoded_so_far):
                break
            if (
                not early_stop_patterns
                and stop_on_final_answer
                and all(has_final_answer_shape(text) for text in decoded_so_far)
            ):
                break

            if next_token.eq(tokenizer.eos_token_id).all():
                break

    if num_trajectories == 1:
        completions = input_ids[:, prompt_len:]
    else:
        completions = input_ids[:, :, prompt_len:].reshape(num_trajectories, -1)
    return GenerationResult(
        candidates=[trim_completion(text) for text in tokenizer.batch_decode(completions, skip_special_tokens=True)],
        diagnostics=_mean_metric_history(metrics_history),
        generation_steps=generation_steps,
    )


def append_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_suite(
    label: str,
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: AutoTokenizer,
    tasks: list[ExactTask],
    args: argparse.Namespace,
    *,
    num_trajectories: int,
    sample_latents: bool,
    latent_injection_mode: str,
    particle_update_mode: str,
    particle_init_noise: float,
) -> tuple[int, int, list[dict[str, int | str | bool]]]:
    best_hits = 0
    candidate_hits = 0
    total_candidates = 0
    task_summaries: list[dict[str, int | str | bool]] = []
    print(f"\n=== {label} ===")
    for task in tasks:
        result = generate_candidates(
            wrapper,
            tokenizer,
            task.prompt,
            max_new_tokens=args.max_new_tokens,
            max_loops=args.max_loops,
            num_trajectories=num_trajectories,
            sample_latents=sample_latents,
            latent_injection_mode=latent_injection_mode,
            particle_update_mode=particle_update_mode,
            particle_init_noise=particle_init_noise,
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
            stop_on_final_answer=args.stop_on_final_answer,
            early_stop_patterns=task.patterns,
            temperature=args.temperature,
            device=args.device,
        )
        outputs = result.candidates
        hits = [matches_any(text, task.patterns) for text in outputs]
        unique_count = len({text.strip() for text in outputs})
        best_hit = any(hits)
        best_hits += int(best_hit)
        candidate_hits += sum(hits)
        total_candidates += len(outputs)
        task_summaries.append(
            {
                "task": task.name,
                "best_hit": best_hit,
                "candidate_hits": sum(hits),
                "candidates": len(outputs),
                "unique": unique_count,
            }
        )
        print(f"\n{task.name} best_of_{len(outputs)}={best_hit} hits={sum(hits)}/{len(outputs)} unique={unique_count}")
        if not args.compact:
            for idx, (text, hit) in enumerate(zip(outputs, hits)):
                print(f"--- traj {idx} hit={hit} ---")
                print(text.strip())
        if args.output_jsonl:
            append_jsonl(
                args.output_jsonl,
                [
                    {
                        "label": label,
                        "task": task.name,
                        "prompt": task.prompt,
                        "patterns": list(task.patterns),
                        "seed": args.seed,
                        "candidate_index": idx,
                        "candidate": text,
                        "hit": hit,
                        "best_hit": best_hit,
                        "task_candidate_hits": sum(hits),
                        "task_candidate_count": len(outputs),
                        "unique_count": unique_count,
                        "generation_steps": result.generation_steps,
                        "mode": particle_update_mode,
                        "num_trajectories": num_trajectories,
                        "sample_latents": sample_latents,
                        "latent_injection_mode": latent_injection_mode,
                        "temperature": args.temperature,
                        "particle_init_noise": particle_init_noise,
                        "particle_noise_every_step": args.particle_noise_every_step,
                        "particle_noise_steps": args.particle_noise_steps,
                        "svgd_eps": args.svgd_eps,
                        "svgd_repulsion_scale": args.svgd_repulsion_scale,
                        "svgd_repulsion_max_norm": args.svgd_repulsion_max_norm,
                        "svgd_kernel_projection_dim": args.svgd_kernel_projection_dim,
                        "svgd_kernel_projection_path": args.svgd_kernel_projection_path,
                        "svgd_kernel_geometry": args.svgd_kernel_geometry,
                        "svgd_projection_seed": args.svgd_projection_seed,
                        "diagnostics": result.diagnostics,
                    }
                    for idx, (text, hit) in enumerate(zip(outputs, hits))
                ],
            )
    print(
        f"\n{label} summary: best_hits={best_hits}/{len(tasks)} "
        f"candidate_hits={candidate_hits}/{max(total_candidates, 1)}"
    )
    return best_hits, candidate_hits, task_summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tasks_jsonl", default="eval/smoke_exact_tasks.jsonl")
    parser.add_argument("--phase1_checkpoint", required=True)
    parser.add_argument("--phase2_checkpoint", required=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--phase2_num_trajectories", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", help="Comma-separated seeds for a multi-seed Phase 2 sweep.")
    parser.add_argument(
        "--particle_noise_steps_sweep",
        help="Comma-separated particle_noise_steps values for a one-load Phase 2 sweep.",
    )
    parser.add_argument(
        "--svgd_repulsion_scale_sweep",
        help="Comma-separated SVGD repulsion_scale values for a one-load Phase 2 sweep.",
    )
    parser.add_argument(
        "--svgd_kernel_projection_dim_sweep",
        help="Comma-separated SVGD kernel projection dims for a one-load sweep. Use 0 for raw hidden space.",
    )
    parser.add_argument("--compact", action="store_true", help="Print task summaries without candidate text.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--phase2_sample_latents", action="store_true")
    parser.add_argument("--skip_phase1", action="store_true", help="Only run the Phase 2 suite to save GPU time.")
    parser.add_argument("--phase2_latent_injection_mode", default="post", choices=("pre", "post", "both"))
    parser.add_argument("--phase2_particle_update_mode", default="none", choices=("none", "svgd"))
    parser.add_argument("--particle_init_noise", type=float, default=0.0)
    parser.add_argument("--svgd_eps", type=float, default=1.0)
    parser.add_argument("--svgd_repulsion_scale", type=float, default=0.5)
    parser.add_argument("--svgd_bandwidth", default="median")
    parser.add_argument("--svgd_bandwidth_floor", type=float, default=1e-6)
    parser.add_argument("--svgd_repulsion_max_norm", type=parse_optional_float)
    parser.add_argument(
        "--svgd_kernel_projection_dim",
        type=int,
        help="Compute the SVGD kernel in a projection of this hidden dimension. With a loaded projection path, slices to this width.",
    )
    parser.add_argument(
        "--svgd_kernel_projection_path",
        help="Optional .pt file containing a calibrated projection tensor or dict with key 'projection'.",
    )
    parser.add_argument(
        "--svgd_kernel_geometry",
        default="euclidean",
        choices=("euclidean", "spherical"),
        help="Kernel geometry. 'spherical' L2-normalizes projected pooled vectors before the RBF.",
    )
    parser.add_argument(
        "--svgd_projection_seed",
        type=int,
        default=0,
        help="Seed for the fixed random SVGD kernel projection.",
    )
    parser.add_argument(
        "--particle_noise_every_step",
        action="store_true",
        help="Reapply particle_init_noise on each no-cache generation step for deterministic SVGD diagnostics.",
    )
    parser.add_argument(
        "--particle_noise_steps",
        type=int,
        default=32,
        help="Number of generated tokens that receive repeated particle noise when enabled.",
    )
    parser.add_argument(
        "--stop_on_final_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop generation early when all candidates look like they reached an answer.",
    )
    parser.add_argument("--output_jsonl", help="Optional path for per-candidate structured eval rows.")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"seed={args.seed}")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = read_tasks(args.tasks_jsonl)
    sweep_seeds = parse_seeds(args.seeds)
    sweep_steps = parse_seeds(args.particle_noise_steps_sweep) or [args.particle_noise_steps]
    sweep_repulsions = parse_floats(args.svgd_repulsion_scale_sweep) or [args.svgd_repulsion_scale]
    sweep_projection_dims = (
        parse_seeds(args.svgd_kernel_projection_dim_sweep)
        if args.svgd_kernel_projection_dim_sweep
        else [args.svgd_kernel_projection_dim or 0]
    )

    if (
        sweep_seeds
        or args.particle_noise_steps_sweep
        or args.svgd_repulsion_scale_sweep
        or args.svgd_kernel_projection_dim_sweep
    ):
        if not sweep_seeds:
            sweep_seeds = [args.seed]
        if not args.skip_phase1:
            print("Multi-seed mode only runs Phase 2. Use --skip_phase1 to make that explicit.")
        print(f"phase2_config={phase2_run_descriptor(args)}")
        phase2 = load_wrapper(args, args.phase2_checkpoint)
        sweep_rows = []
        original_seed = args.seed
        original_steps = args.particle_noise_steps
        original_repulsion = args.svgd_repulsion_scale
        original_projection_dim = args.svgd_kernel_projection_dim
        for projection_dim in sweep_projection_dims:
            args.svgd_kernel_projection_dim = projection_dim or None
            for repulsion in sweep_repulsions:
                args.svgd_repulsion_scale = repulsion
                for steps in sweep_steps:
                    args.particle_noise_steps = steps
                    for seed in sweep_seeds:
                        args.seed = seed
                        set_seed(seed)
                        descriptor = phase2_run_descriptor(args, seed=seed, steps=steps)
                        print(f"\n\n### {descriptor} ###")
                        phase2_hits, phase2_candidate_hits, task_summaries = run_suite(
                            f"Phase 2 K={args.phase2_num_trajectories} {descriptor}",
                            phase2,
                            tokenizer,
                            tasks,
                            args,
                            num_trajectories=args.phase2_num_trajectories,
                            sample_latents=args.phase2_sample_latents,
                            latent_injection_mode=args.phase2_latent_injection_mode,
                            particle_update_mode=args.phase2_particle_update_mode,
                            particle_init_noise=args.particle_init_noise,
                        )
                        sweep_rows.append(
                            (
                                projection_dim,
                                repulsion,
                                steps,
                                seed,
                                phase2_hits,
                                phase2_candidate_hits,
                                task_summaries,
                            )
                        )
        args.seed = original_seed
        args.particle_noise_steps = original_steps
        args.svgd_repulsion_scale = original_repulsion
        args.svgd_kernel_projection_dim = original_projection_dim

        print("\n=== SWEEP SUMMARY ===")
        for projection_dim, repulsion, steps, seed, best_hits, candidate_hits, _ in sweep_rows:
            args.svgd_kernel_projection_dim = projection_dim or None
            args.svgd_repulsion_scale = repulsion
            descriptor = phase2_run_descriptor(args, seed=seed, steps=steps)
            print(
                f"{descriptor} best_hits={best_hits}/{len(tasks)} "
                f"candidate_hits={candidate_hits}/{len(tasks) * args.phase2_num_trajectories}"
            )
        args.svgd_repulsion_scale = original_repulsion
        args.svgd_kernel_projection_dim = original_projection_dim

        print("\n=== PROJECTION/REPULSION/STEPS SUMMARY ===")
        for projection_dim in sweep_projection_dims:
            for repulsion in sweep_repulsions:
                for steps in sweep_steps:
                    rows = [
                        row
                        for row in sweep_rows
                        if row[0] == projection_dim and row[1] == repulsion and row[2] == steps
                    ]
                    mean_best_for_rows = sum(row[4] for row in rows) / max(len(rows), 1)
                    mean_candidates_for_rows = sum(row[5] for row in rows) / max(len(rows), 1)
                    print(
                        f"proj_dim={projection_dim or 0} repulsion={_format_float(repulsion)} steps={steps} "
                        f"mean_best_hits={mean_best_for_rows:.3f}/{len(tasks)} "
                        f"mean_candidate_hits={mean_candidates_for_rows:.3f}/{len(tasks) * args.phase2_num_trajectories}"
                    )
        mean_best = sum(row[4] for row in sweep_rows) / max(len(sweep_rows), 1)
        mean_candidates = sum(row[5] for row in sweep_rows) / max(len(sweep_rows), 1)
        print(f"mean_best_hits={mean_best:.3f}/{len(tasks)}")
        print(f"mean_candidate_hits={mean_candidates:.3f}/{len(tasks) * args.phase2_num_trajectories}")
        return 0

    phase1_hits: int | None = None
    if not args.skip_phase1:
        phase1 = load_wrapper(args, args.phase1_checkpoint)
        phase1_hits, _, _ = run_suite(
            "Phase 1 K=1",
            phase1,
            tokenizer,
            tasks,
            args,
            num_trajectories=1,
            sample_latents=False,
            latent_injection_mode="pre",
            particle_update_mode="none",
            particle_init_noise=0.0,
        )
        del phase1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase2 = load_wrapper(args, args.phase2_checkpoint)
    phase2_hits, phase2_candidate_hits, _ = run_suite(
        f"Phase 2 K={args.phase2_num_trajectories}",
        phase2,
        tokenizer,
        tasks,
        args,
        num_trajectories=args.phase2_num_trajectories,
        sample_latents=args.phase2_sample_latents,
        latent_injection_mode=args.phase2_latent_injection_mode,
        particle_update_mode=args.phase2_particle_update_mode,
        particle_init_noise=args.particle_init_noise,
    )
    phase1_summary = f"{phase1_hits}/{len(tasks)}" if phase1_hits is not None else "skipped"
    print(
        "\n=== OVERALL ===\n"
        f"phase1_k1_hits={phase1_summary}\n"
        f"phase2_best_of_k_hits={phase2_hits}/{len(tasks)}\n"
        f"phase2_candidate_hits={phase2_candidate_hits}/{len(tasks) * args.phase2_num_trajectories}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
