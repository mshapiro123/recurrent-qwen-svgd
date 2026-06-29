"""Prelude re-injection ablation for the corrected recurrent bridge.

The corrected loop closure re-enters the recurrent block with a bridge over
``[prelude_hidden, recurrent_state]``. This diagnostic tests whether the learned
prelude half is actually used by comparing normal prelude re-entry to zeroed and
token-shuffled prelude variants at fixed loop counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_effective_pathways import read_prompts  # noqa: E402
from eval.eval_reentry_drift import load_wrapper, prepare_recurrent_inputs, run_recurrent_block  # noqa: E402


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def finite_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        value = float(value.detach().float().cpu())
    value = float(value)
    return value if math.isfinite(value) else 0.0


def parse_loop_counts(value: str, *, max_loops: int) -> list[int]:
    counts = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not counts:
        raise ValueError("loop_counts must contain at least one loop")
    if counts[0] < 1 or counts[-1] > max_loops:
        raise ValueError(f"loop_counts must be within [1, {max_loops}], got {counts}")
    return counts


def prelude_variant(entry_state: torch.Tensor, attention_mask: torch.Tensor, variant: str) -> torch.Tensor:
    """Return a prelude tensor variant with the same shape as ``entry_state``."""

    if variant == "normal":
        return entry_state
    if variant == "zero":
        return torch.zeros_like(entry_state)
    if variant != "shuffled":
        raise ValueError("variant must be one of: normal, zero, shuffled")

    shuffled = entry_state.clone()
    mask = attention_mask.to(device=entry_state.device, dtype=torch.bool)
    for batch_idx in range(entry_state.shape[0]):
        valid = mask[batch_idx].nonzero(as_tuple=False).flatten()
        if valid.numel() > 1:
            shuffled[batch_idx, valid] = entry_state[batch_idx, torch.roll(valid, shifts=1)]
    return shuffled


def selected_next_token_logits(
    wrapper: Any,
    recurrent_state: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
) -> torch.Tensor:
    coda_hidden, _ = wrapper._run_layer_range(  # noqa: SLF001
        start=wrapper.layer_split.recurrent_end,
        end=len(wrapper.qwen.layers),
        hidden_states=recurrent_state,
        causal_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        collect_hidden=False,
        hidden_history=None,
    )
    normed = wrapper.qwen.norm(coda_hidden)
    logits = wrapper.lm_head(normed).float()
    last_index = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0).to(device=logits.device)
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_index, last_index]


def run_variant_logits(
    wrapper: Any,
    entry_state: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
    *,
    variant: str,
    loop_counts: list[int],
) -> dict[int, torch.Tensor]:
    max_loop = max(loop_counts)
    requested = set(loop_counts)
    variant_prelude = prelude_variant(entry_state, attention_mask, variant)
    recurrent_state = entry_state
    out: dict[int, torch.Tensor] = {}
    for loop_idx in range(max_loop):
        loop_input = (
            recurrent_state
            if loop_idx == 0
            else wrapper.bridge(recurrent_state, prelude_hidden=variant_prelude)
        )
        recurrent_state = run_recurrent_block(
            wrapper,
            loop_input,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        loop_number = loop_idx + 1
        if loop_number in requested:
            out[loop_number] = selected_next_token_logits(
                wrapper,
                recurrent_state,
                attention_mask,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
            ).detach()
    return out


def logit_comparison_metrics(normal_logits: torch.Tensor, variant_logits: torch.Tensor) -> dict[str, Any]:
    delta = (variant_logits - normal_logits).float()
    normal_argmax = normal_logits.argmax(dim=-1)
    variant_argmax = variant_logits.argmax(dim=-1)
    return {
        "logit_mean_abs_delta": finite_float(delta.abs().mean()),
        "logit_max_abs_delta": finite_float(delta.abs().max()),
        "normal_argmax": int(normal_argmax[0].detach().cpu()),
        "variant_argmax": int(variant_argmax[0].detach().cpu()),
        "top1_changed": bool((normal_argmax != variant_argmax).any().detach().cpu()),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    for row in records:
        loop_key = str(row["loop"])
        variant = str(row["variant"])
        aggregate.setdefault(loop_key, {}).setdefault(
            variant,
            {
                "count": 0,
                "top1_changed": 0,
                "logit_mean_abs_delta_sum": 0.0,
                "logit_max_abs_delta_max": 0.0,
            },
        )
        cell = aggregate[loop_key][variant]
        cell["count"] += 1
        cell["top1_changed"] += int(bool(row["top1_changed"]))
        cell["logit_mean_abs_delta_sum"] += float(row["logit_mean_abs_delta"])
        cell["logit_max_abs_delta_max"] = max(
            float(cell["logit_max_abs_delta_max"]),
            float(row["logit_max_abs_delta"]),
        )
    for loop_data in aggregate.values():
        for cell in loop_data.values():
            count = max(1, int(cell["count"]))
            cell["top1_changed_fraction"] = float(cell["top1_changed"] / count)
            cell["logit_mean_abs_delta"] = float(cell.pop("logit_mean_abs_delta_sum") / count)
            cell["logit_max_abs_delta"] = float(cell.pop("logit_max_abs_delta_max"))
    return aggregate


@torch.no_grad()
def run_prompt(
    wrapper: Any,
    tokenizer: Any,
    prompt: str,
    *,
    prompt_index: int,
    args: argparse.Namespace,
    loop_counts: list[int],
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = encoded["input_ids"].to(wrapper.device)
    attention_mask = encoded["attention_mask"].to(wrapper.device)
    entry_state, mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
        wrapper,
        input_ids,
        attention_mask,
    )
    normal = run_variant_logits(
        wrapper,
        entry_state,
        mask,
        causal_mask,
        position_ids,
        cache_position,
        position_embeddings,
        variant="normal",
        loop_counts=loop_counts,
    )
    rows: list[dict[str, Any]] = []
    for variant in ("zero", "shuffled"):
        variant_logits = run_variant_logits(
            wrapper,
            entry_state,
            mask,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
            variant=variant,
            loop_counts=loop_counts,
        )
        for loop in loop_counts:
            metrics = logit_comparison_metrics(normal[loop], variant_logits[loop])
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "loop": loop,
                    "variant": variant,
                    **metrics,
                    "normal_argmax_text": tokenizer.decode([metrics["normal_argmax"]]),
                    "variant_argmax_text": tokenizer.decode([metrics["variant_argmax"]]),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts_jsonl", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=8)
    parser.add_argument("--loop_counts", default="1,2,4,8")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_jsonl", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loop_counts = parse_loop_counts(args.loop_counts, max_loops=args.max_loops)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_wrapper(args)
    prompts = read_prompts(args.prompts_jsonl or None, limit=args.limit or None)
    records: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        records.extend(run_prompt(wrapper, tokenizer, prompt, prompt_index=idx, args=args, loop_counts=loop_counts))
    summary = {
        "kind": "prelude_ablation",
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_loops": args.max_loops,
        "loop_counts": loop_counts,
        "max_length": args.max_length,
        "num_prompts": len(prompts),
        "aggregate": aggregate_records(records),
        "records": records,
    }
    print(json.dumps({"aggregate": summary["aggregate"]}, indent=2), flush=True)
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
