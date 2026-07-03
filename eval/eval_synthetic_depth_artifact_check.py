"""Check that synthetic-depth extrapolation can mechanically run beyond depth 4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper
from eval.eval_synthetic_depth_active_labels import prompt_for_row, read_jsonl


def source_contains_loop_capacity_limit() -> dict[str, Any]:
    wrapper_text = (ROOT / "models" / "recurrent_wrapper.py").read_text(encoding="utf-8")
    halting_text = (ROOT / "models" / "halting.py").read_text(encoding="utf-8")
    return {
        "learned_loop_control_optional": "use_learned_loop_control" in wrapper_text,
        "learned_loop_control_default_false": "use_learned_loop_control: bool = False" in wrapper_text,
        "halting_loop_embedding_present": "loop_embedding" in halting_text,
        "halting_target_loop_router_present": "target_loop_router" in halting_text,
        "max_loop_embedding_capacity_64": "max_loop_embeddings: int = 64" in halting_text,
        "no_literal_four_loop_cap_in_wrapper": "range(4)" not in wrapper_text and "max_loops = 4" not in wrapper_text,
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.data_jsonl)
    if not rows:
        raise ValueError(f"No rows in {args.data_jsonl}")
    row = rows[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    calls: list[int] = []
    hook = wrapper.bridge.register_forward_hook(lambda _module, _inputs, _output: calls.append(1))
    prompt = prompt_for_row(row, prediction_space=args.prediction_space, prompt_style=args.prompt_style)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
    with torch.no_grad():
        output = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=args.max_loops,
            num_trajectories=1,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
        )
    hook.remove()
    loop_logits = output.loop_logits
    if loop_logits is None:
        raise RuntimeError("Expected loop_logits with return_loop_logits=True")
    loop_count = int(loop_logits.shape[2])
    expected_bridge_calls = max(0, int(args.max_loops) - 1)
    summary = {
        "kind": "synthetic_depth_extrapolation_artifact_check",
        "checkpoint": args.checkpoint,
        "data_jsonl": args.data_jsonl,
        "sample_id": row.get("id") or row.get("instance_id"),
        "requested_max_loops": int(args.max_loops),
        "loop_logits_shape": list(loop_logits.shape),
        "loop_states_observed": loop_count,
        "bridge_forward_calls": len(calls),
        "expected_bridge_forward_calls": expected_bridge_calls,
        "forced_loop_count_honored": loop_count == int(args.max_loops),
        "bridge_fires_on_reentry": len(calls) == expected_bridge_calls,
        "source_static_checks": source_contains_loop_capacity_limit(),
    }
    summary["pass"] = bool(
        summary["forced_loop_count_honored"]
        and summary["bridge_fires_on_reentry"]
        and summary["source_static_checks"]["no_literal_four_loop_cap_in_wrapper"]
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--max_loops", type=int, default=6)
    parser.add_argument("--prediction_space", choices=("choice_labels", "full_symbols"), default="full_symbols")
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="question_only")
    parser.add_argument("--value_prefix", default="")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    args = parser.parse_args()

    summary = run_check(args)
    out = Path(args.output_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
