"""Telemetry-only eval for deterministic PonderNet halting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split
from models.recurrent_wrapper import RecurrentQwenForCausalLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Give a concise solution: 17 * 23 = ?")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    model.eval()
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    wrapper.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        output = wrapper(
            **encoded,
            max_loops=args.max_loops,
            use_cache=False,
            return_dict=True,
        )

    print(f"expected_loops={output.expected_loops.squeeze(0).tolist()}")
    print(f"halting_weights={output.halting_weights.squeeze(0).tolist()}")
    for key, value in output.metrics.items():
        print(f"{key}={float(value):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
