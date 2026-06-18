"""Slow no-cache recurrent generation for quick local experiments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split
from models.recurrent_wrapper import RecurrentQwenForCausalLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--sample_latents", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    model.eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    wrapper.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    input_ids = encoded["input_ids"].repeat(args.num_trajectories, 1)
    attention_mask = encoded["attention_mask"].repeat(args.num_trajectories, 1)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generated = 0
    last_metrics = {}
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            output = wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_loops=args.max_loops,
                num_trajectories=1,
                sample_latents=args.sample_latents,
                use_cache=False,
                return_dict=True,
            )
            last_metrics = output.metrics
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            generated += next_token.numel()
            if tokenizer.eos_token_id is not None and bool((next_token == tokenizer.eos_token_id).all()):
                break

    elapsed = max(time.perf_counter() - started, 1e-6)
    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    for idx, text in enumerate(decoded):
        print(f"--- candidate {idx + 1} ---")
        print(text)
    for key, value in last_metrics.items():
        print(f"{key}={float(value):.6f}")
    print(f"tokens_per_sec={generated / elapsed:.2f}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"peak_vram_gb={peak_gb:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
