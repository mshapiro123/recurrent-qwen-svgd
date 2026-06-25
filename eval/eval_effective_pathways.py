"""Effective-pathway diagnostic for recurrent latent dynamics.

The purpose is to test whether the recurrent map itself preserves multiple
initial-condition basins when particles are run without SVGD repulsion or latent
sampling. If the effective pathway count collapses near one, kernel tuning is
not the bottleneck; the dynamics are.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype  # noqa: E402
from eval.pathway_diversity import effective_pathways, pairwise_squared_distances  # noqa: E402
from models.halting import masked_mean  # noqa: E402
from models.lora import apply_lora_to_recurrent_block  # noqa: E402
from models.recurrent_wrapper import RecurrentQwenForCausalLM  # noqa: E402
from training.checkpointing import load_trainable_checkpoint  # noqa: E402


DEFAULT_PROMPTS = (
    "Solve: If a train travels 120 miles in 3 hours, what is its average speed?",
    "A rectangle is 7 units long and 5 units wide. What is its area?",
    "Name one strategy for solving a Sudoku puzzle.",
)


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_prompts(path: str | None, *, limit: int | None = None) -> list[str]:
    if not path:
        prompts = list(DEFAULT_PROMPTS)
    else:
        prompts = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("question") or row.get("text")
            if not prompt:
                raise ValueError(f"JSONL row has no prompt/question/text: {row}")
            prompts.append(str(prompt))
    if limit is not None and limit > 0:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError("No prompts available for effective-pathway diagnostic")
    return prompts


def load_wrapper(args: argparse.Namespace) -> RecurrentQwenForCausalLM:
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
    print(f"lora_recurrent_modules={replaced}", flush=True)
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    if args.checkpoint:
        load_info = load_trainable_checkpoint(wrapper, args.checkpoint)
        print(f"loaded_checkpoint={args.checkpoint} loaded_keys={len(load_info['loaded_keys'])}", flush=True)
        if load_info["skipped"]:
            print(f"skipped_keys={len(load_info['skipped'])}", flush=True)
    wrapper.eval()
    return wrapper


def repeat_noisy_inputs(
    wrapper: RecurrentQwenForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    num_particles: int,
    noise_std: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    embeds = wrapper.qwen.embed_tokens(input_ids)
    generator = torch.Generator(device=embeds.device)
    generator.manual_seed(int(seed))
    expanded = embeds.unsqueeze(1).expand(-1, num_particles, -1, -1).contiguous()
    if noise_std > 0:
        noise = torch.randn(
            expanded.shape,
            generator=generator,
            device=expanded.device,
            dtype=torch.float32,
        ).to(dtype=expanded.dtype)
        expanded = expanded + float(noise_std) * noise
    expanded_mask = attention_mask.unsqueeze(1).expand(-1, num_particles, -1).contiguous()
    initial_pooled = masked_mean(
        expanded.reshape(-1, expanded.shape[-2], expanded.shape[-1]),
        expanded_mask.reshape(-1, expanded_mask.shape[-1]),
    ).view(input_ids.shape[0], num_particles, -1)
    initial_spread = mean_pairwise_distance(initial_pooled[0])
    return expanded, expanded_mask, initial_pooled, initial_spread


def mean_pairwise_distance(states: torch.Tensor) -> float:
    if states.shape[0] <= 1:
        return 0.0
    d2 = pairwise_squared_distances(states)
    mask = ~torch.eye(states.shape[0], dtype=torch.bool, device=states.device)
    return float(d2[mask].clamp_min(0.0).sqrt().mean().detach().cpu())


def project_states(states: torch.Tensor, projection_path: str | None, projection_dim: int | None) -> torch.Tensor:
    if not projection_path:
        return states.float()
    payload = torch.load(projection_path, map_location="cpu")
    projection = payload.get("projection") if isinstance(payload, dict) else payload
    if not torch.is_tensor(projection):
        raise TypeError(f"Projection file must contain a tensor or dict['projection']: {projection_path}")
    projection = projection.float()
    if projection.shape[0] != states.shape[-1]:
        raise ValueError(
            "Projection input dimension must match hidden size. "
            f"Got projection={tuple(projection.shape)}, hidden={states.shape[-1]}."
        )
    if projection_dim and projection_dim > 0:
        projection = projection[:, : int(projection_dim)]
    return states.float() @ projection.to(device=states.device)


def q_values(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        out.append(math.inf if item in {"inf", "infinity"} else float(item))
    return out


def run_prompt(
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    prompt: str,
    *,
    prompt_index: int,
    args: argparse.Namespace,
    qs: list[float],
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    noisy_embeds, noisy_mask, _initial_pooled, initial_spread = repeat_noisy_inputs(
        wrapper,
        input_ids,
        attention_mask,
        num_particles=args.num_particles,
        noise_std=args.particle_init_noise,
        seed=args.seed + prompt_index,
    )
    with torch.no_grad():
        output = wrapper(
            inputs_embeds=noisy_embeds,
            attention_mask=noisy_mask,
            max_loops=args.max_loops,
            num_trajectories=args.num_particles,
            sample_latents=False,
            particle_update_mode="none",
            reentry_rescale_mode=args.reentry_rescale_mode,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    assert output.final_recurrent_hidden is not None
    pooled = masked_mean(
        output.final_recurrent_hidden[0],
        attention_mask.expand(args.num_particles, -1),
    )
    states = project_states(pooled, args.kernel_projection_path, args.kernel_projection_dim)
    if args.kernel_geometry == "spherical":
        states = torch.nn.functional.normalize(states.float(), p=2, dim=-1, eps=1e-8)
    diversity, diagnostics = effective_pathways(
        states,
        qs=qs,
        bw_factor=args.bw_factor,
        sigma_floor=args.sigma_floor,
    )
    final_spread = mean_pairwise_distance(pooled)
    ratio = final_spread / max(initial_spread, args.sigma_floor)
    lyapunov_proxy = math.log(max(ratio, args.sigma_floor)) / max(int(args.max_loops), 1)
    argmax_ids = output.trajectory_logits[0, :, -1].argmax(dim=-1).detach().cpu().tolist() if output.trajectory_logits is not None else []
    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "num_particles": args.num_particles,
        "max_loops": args.max_loops,
        "particle_init_noise": args.particle_init_noise,
        "reentry_rescale_mode": args.reentry_rescale_mode,
        "effective_pathways": diversity,
        "pathway_diagnostics": diagnostics,
        "initial_pairwise_distance": initial_spread,
        "final_pairwise_distance": final_spread,
        "spread_ratio_final_over_initial": ratio,
        "lyapunov_proxy_per_loop": lyapunov_proxy,
        "unique_next_token_argmax": len(set(argmax_ids)),
        "next_token_argmax": argmax_ids,
        "metrics": {
            key: float(value.detach().float().cpu())
            for key, value in output.metrics.items()
            if torch.is_tensor(value) and value.numel() == 1
        },
    }


def aggregate(records: list[dict[str, Any]], qs: list[float]) -> dict[str, Any]:
    def qkey(q: float) -> str:
        return "inf" if math.isinf(q) else (str(int(q)) if float(q).is_integer() else f"{q:g}")

    summary: dict[str, Any] = {
        "prompts": len(records),
        "mean_initial_pairwise_distance": sum(row["initial_pairwise_distance"] for row in records) / len(records),
        "mean_final_pairwise_distance": sum(row["final_pairwise_distance"] for row in records) / len(records),
        "mean_spread_ratio_final_over_initial": sum(row["spread_ratio_final_over_initial"] for row in records)
        / len(records),
        "mean_lyapunov_proxy_per_loop": sum(row["lyapunov_proxy_per_loop"] for row in records) / len(records),
        "mean_unique_next_token_argmax": sum(row["unique_next_token_argmax"] for row in records) / len(records),
        "mean_effective_pathways": {},
    }
    for q in qs:
        key = qkey(q)
        summary["mean_effective_pathways"][key] = sum(row["effective_pathways"][key] for row in records) / len(records)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompts_jsonl", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=8)
    parser.add_argument("--num_particles", type=int, default=16)
    parser.add_argument("--particle_init_noise", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--qs", default="0,1,2,inf")
    parser.add_argument("--bw_factor", type=float, default=2.0)
    parser.add_argument("--sigma_floor", type=float, default=1e-6)
    parser.add_argument("--kernel_projection_path", default="")
    parser.add_argument("--kernel_projection_dim", type=int, default=0)
    parser.add_argument("--kernel_geometry", default="euclidean", choices=("euclidean", "spherical"))
    parser.add_argument("--reentry_rescale_mode", default="none", choices=("none", "entry_rms"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_jsonl", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_wrapper(args)
    prompts = read_prompts(args.prompts_jsonl or None, limit=args.limit or None)
    qs = q_values(args.qs)
    records = [
        run_prompt(wrapper, tokenizer, prompt, prompt_index=idx, args=args, qs=qs)
        for idx, prompt in enumerate(prompts)
    ]
    summary = {
        "kind": "effective_pathway_diagnostic",
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_loops": args.max_loops,
        "num_particles": args.num_particles,
        "particle_init_noise": args.particle_init_noise,
        "kernel_geometry": args.kernel_geometry,
        "reentry_rescale_mode": args.reentry_rescale_mode,
        "kernel_projection_path": args.kernel_projection_path,
        "kernel_projection_dim": args.kernel_projection_dim,
        "aggregate": aggregate(records, qs),
        "records": records,
    }
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_summary={path_for_cli(path)}", flush=True)
    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records), encoding="utf-8")
        print(f"saved_records={path_for_cli(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
