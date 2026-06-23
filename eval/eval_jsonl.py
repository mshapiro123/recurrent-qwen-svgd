"""Evaluate recurrent-depth Qwen on this project's JSONL format."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from models.lora import apply_lora_to_recurrent_block
from models.recurrent_wrapper import RecurrentQwenForCausalLM
from training.checkpointing import load_trainable_checkpoint
from training.dataset import JsonlCausalDataset, collate_causal_batch
from training.stability import assert_finite_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--sample_latents", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--latent_injection_mode", default="pre", choices=("pre", "post", "both"))
    parser.add_argument("--particle_update_mode", default="none", choices=("none", "svgd"))
    parser.add_argument("--particle_init_noise", type=float, default=0.0)
    parser.add_argument("--svgd_eps", type=float, default=1.0)
    parser.add_argument("--svgd_repulsion_scale", type=float, default=1.0)
    parser.add_argument("--svgd_bandwidth", default="median")
    parser.add_argument("--svgd_bandwidth_floor", type=float, default=1e-6)
    parser.add_argument("--svgd_repulsion_max_norm", type=float)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument(
        "--use_target_loop_control",
        action="store_true",
        help="Pass each row's target_loop_count into the structural halt-control channel.",
    )
    parser.add_argument(
        "--use_learned_loop_control",
        action="store_true",
        help="Use the learned target-loop router to condition sequence halting.",
    )
    parser.add_argument("--loop_control_ce_weight", type=float, default=0.0)
    parser.add_argument(
        "--group_by_field",
        help=(
            "Optional JSONL row field for grouped metrics. Requires batch_size=1 "
            "so per-example metrics can be assigned without another model pass."
        ),
    )
    args = parser.parse_args()
    if args.group_by_field and args.batch_size != 1:
        raise SystemExit("--group_by_field currently requires --batch_size 1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    if args.checkpoint:
        load_info = load_trainable_checkpoint(wrapper, args.checkpoint)
        print(f"loaded_checkpoint={args.checkpoint} loaded_keys={len(load_info['loaded_keys'])}")
        if load_info["skipped"]:
            print(f"skipped_keys={len(load_info['skipped'])}")

    wrapper.eval()
    dataset = JsonlCausalDataset(
        args.data_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_train_loops=args.max_loops,
        train_on_prompt=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id),
    )

    totals: dict[str, float] = {}
    grouped_totals: dict[str, dict[str, float]] = {}
    grouped_counts: dict[str, int] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            if args.use_target_loop_control:
                batch["halt_control_loop_counts"] = batch["target_loop_counts"]
            output = wrapper(
                **batch,
                max_loops=args.max_loops,
                num_trajectories=args.num_trajectories,
                sample_latents=args.sample_latents,
                beta=args.beta,
                eta=args.eta,
                rho=args.rho,
                latent_injection_mode=args.latent_injection_mode,
                particle_update_mode=args.particle_update_mode,
                particle_init_noise=args.particle_init_noise,
                svgd_eps=args.svgd_eps,
                svgd_repulsion_scale=args.svgd_repulsion_scale,
                svgd_bandwidth=args.svgd_bandwidth,
                svgd_bandwidth_floor=args.svgd_bandwidth_floor,
                svgd_repulsion_max_norm=args.svgd_repulsion_max_norm,
                use_learned_loop_control=args.use_learned_loop_control,
                loop_control_ce_weight=args.loop_control_ce_weight,
                use_cache=False,
                return_dict=True,
            )
            assert_finite_metrics(output.metrics, count)
            batch_size = batch["input_ids"].shape[0]
            if args.group_by_field:
                row = dataset.rows[count]
                group_value = str(row.get(args.group_by_field) or "missing")
                grouped_counts[group_value] = grouped_counts.get(group_value, 0) + batch_size
                group_totals = grouped_totals.setdefault(group_value, {})
                for key, value in output.metrics.items():
                    group_totals[key] = group_totals.get(key, 0.0) + float(value) * batch_size
            count += batch_size
            for key, value in output.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size

    print(f"examples={count}")
    for key in sorted(totals):
        print(f"{key}={totals[key] / max(count, 1):.6f}")
    for group in sorted(grouped_totals):
        group_count = grouped_counts[group]
        print(f"group/{args.group_by_field}/{group}/examples={group_count}")
        for key in sorted(grouped_totals[group]):
            print(f"group/{args.group_by_field}/{group}/{key}={grouped_totals[group][key] / max(group_count, 1):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
