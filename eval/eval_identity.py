"""Phase 0 identity gate for the manual recurrent Qwen wrapper path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM


def parse_split(value: str | None) -> LayerSplit | None:
    if value is None or value.lower() == "auto":
        return None
    left, right = value.split(",", maxsplit=1)
    return LayerSplit(prelude_end=int(left), recurrent_end=int(right))


def resolve_dtype(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping[value]


def model_load_kwargs(dtype: str, attn_implementation: str | None = None, **extra):
    kwargs = {"dtype": resolve_dtype(dtype), **extra}
    if attn_implementation and attn_implementation != "default":
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Solve: If x + 2 = 5, what is x?")
    parser.add_argument("--split", default="6,18", help="'auto' or 'prelude_end,recurrent_end'")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--attn_implementation",
        default="default",
        help="Transformers attention implementation, e.g. default, eager, sdpa, flash_attention_2.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(
            args.dtype,
            args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        ),
    ).to(args.device)
    model.eval()

    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    wrapper.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)

    with torch.no_grad():
        original = model(**encoded, use_cache=False, return_dict=True).logits
        wrapped = wrapper(
            **encoded,
            max_loops=1,
            num_trajectories=1,
            sample_latents=False,
            use_cache=False,
            return_dict=True,
        ).logits

    diff = (original - wrapped).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    print(f"max_abs_diff={max_abs_diff:.8f}")
    print(f"mean_abs_diff={mean_abs_diff:.8f}")
    print(f"threshold={args.threshold:.8f}")
    if max_abs_diff >= args.threshold:
        print("FAIL: identity wrapper drift exceeds threshold")
        return 1
    print("PASS: identity wrapper drift is within threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
