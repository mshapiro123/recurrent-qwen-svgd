"""Verify zero-init recurrent-block LoRA preserves max_loops=1 base logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from models.lora import apply_lora_to_recurrent_block
from models.recurrent_wrapper import RecurrentQwenForCausalLM
from training.checkpointing import save_trainable_checkpoint
from training.train_unfrozen_recurrent import hash_pretrained_base_parameters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--prompt", default="Solve: If x + 2 = 5, what is x?")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_checkpoint", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, "default"),
    ).to(args.device)
    model.eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    wrapper.bridge.convert_to_split_projection()
    replaced = apply_lora_to_recurrent_block(
        wrapper,
        rank=args.rank,
        alpha=args.alpha,
        dropout=0.0,
        adapter_dtype=resolve_dtype(args.adapter_dtype),
    )
    wrapper.eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        base_logits = model(**encoded, use_cache=False, return_dict=True).logits
        wrapped_logits = wrapper(
            **encoded,
            max_loops=1,
            num_trajectories=1,
            sample_latents=False,
            use_cache=False,
            return_dict=True,
        ).logits
    diff = (base_logits - wrapped_logits).abs()
    payload = {
        "kind": "peft_identity_gate",
        "rank": args.rank,
        "alpha": args.alpha,
        "seed": args.seed,
        "lora_modules": replaced,
        "bridge_projection_mode": "split",
        "max_loops": 1,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "threshold": args.threshold,
        "passed": float(diff.max().item()) < args.threshold,
        "pretrained_base_sha256": hash_pretrained_base_parameters(wrapper),
    }
    if args.output_checkpoint:
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
        for name, parameter in wrapper.named_parameters():
            active_bridge_parameter = (
                name.startswith("bridge.")
                and not name.startswith("bridge.proj.")
            )
            if ".lora_a." in name or ".lora_b." in name or active_bridge_parameter:
                parameter.requires_grad_(True)
        checkpoint = save_trainable_checkpoint(
            wrapper,
            Path(args.output_checkpoint).parent,
            "peft_identity",
            0,
            {
                "lora": {"rank": args.rank, "alpha": args.alpha},
                "bridge_projection_mode": "split",
            },
        )
        requested = Path(args.output_checkpoint)
        if checkpoint != requested:
            requested.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.replace(requested)
        payload["checkpoint"] = str(requested)
    destination = Path(args.output_summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
