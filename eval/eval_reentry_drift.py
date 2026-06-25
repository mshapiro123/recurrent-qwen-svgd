"""Re-entry drift diagnostic for recurrent Qwen loop closure.

This script is read-only: it loads a recurrent checkpoint and measures whether
the recurrent-block output lives on the same distribution as the recurrent-block
input it is fed back into on later loops.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_effective_pathways import read_prompts  # noqa: E402
from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype  # noqa: E402
from models.bridge import IdentityGatedBridge  # noqa: E402
from models.halting import masked_mean  # noqa: E402
from models.lora import apply_lora_to_recurrent_block  # noqa: E402
from models.recurrent_wrapper import RecurrentQwenForCausalLM  # noqa: E402
from training.checkpointing import load_trainable_checkpoint  # noqa: E402


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def finite_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        value = float(value.detach().float().cpu())
    value = float(value)
    if math.isfinite(value):
        return value
    return 0.0


def rms(hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    values = hidden.float().square().mean(dim=-1)
    if attention_mask is None:
        return values.mean().sqrt()
    mask = attention_mask.to(device=hidden.device, dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(0)
    return (values * mask).sum().div(mask.sum().clamp_min(1.0)).sqrt()


def masked_token_matrix(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if hidden.shape[:2] != mask.shape:
        raise ValueError(f"hidden/mask shape mismatch: hidden={tuple(hidden.shape)}, mask={tuple(mask.shape)}")
    return hidden[mask].float()


def pooled_cosine(a: torch.Tensor, b: torch.Tensor, attention_mask: torch.Tensor) -> float:
    pooled_a = masked_mean(a, attention_mask).float()
    pooled_b = masked_mean(b, attention_mask).float()
    cosine = F.cosine_similarity(pooled_a, pooled_b, dim=-1)
    return finite_float(cosine.mean())


def pca_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    matrix = matrix.float()
    if matrix.dim() != 2:
        raise ValueError("matrix must be 2D")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    max_rank = min(rank, centered.shape[0], centered.shape[1])
    if max_rank <= 0:
        return centered.new_zeros((centered.shape[1], 0))
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:max_rank].T.contiguous()


def subspace_overlap(
    entry_tokens: torch.Tensor,
    exit_tokens: torch.Tensor,
    *,
    rank: int = 8,
) -> dict[str, Any]:
    rank = min(int(rank), entry_tokens.shape[0], exit_tokens.shape[0], entry_tokens.shape[1], exit_tokens.shape[1])
    if rank <= 0:
        return {
            "rank": 0,
            "overlap": 0.0,
            "principal_cosines": [],
            "aligned_dims_cos_ge_0p8": 0,
            "aligned_dims_cos_ge_0p9": 0,
        }
    entry_basis = pca_basis(entry_tokens, rank)
    exit_basis = pca_basis(exit_tokens, rank)
    if entry_basis.numel() == 0 or exit_basis.numel() == 0:
        return {
            "rank": 0,
            "overlap": 0.0,
            "principal_cosines": [],
            "aligned_dims_cos_ge_0p8": 0,
            "aligned_dims_cos_ge_0p9": 0,
        }
    cosines = torch.linalg.svdvals(entry_basis.T @ exit_basis).clamp(0.0, 1.0)
    overlap = cosines.square().mean()
    return {
        "rank": int(rank),
        "overlap": finite_float(overlap),
        "principal_cosines": [finite_float(value) for value in cosines],
        "aligned_dims_cos_ge_0p8": int(cosines.ge(0.8).sum().item()),
        "aligned_dims_cos_ge_0p9": int(cosines.ge(0.9).sum().item()),
    }


def bridge_stats(bridge: IdentityGatedBridge, sample_state: torch.Tensor) -> dict[str, Any]:
    weight = bridge.proj.weight.detach().float().cpu()
    bias = bridge.proj.bias.detach().float().cpu()
    eye = torch.eye(weight.shape[0], dtype=weight.dtype)
    with torch.no_grad():
        bridge_out = bridge(sample_state.detach())
    return {
        "bridge_gate": finite_float(bridge.bridge_gate),
        "proj_identity_max_abs_diff": finite_float((weight - eye).abs().max()),
        "proj_bias_max_abs": finite_float(bias.abs().max()),
        "sample_bridge_delta_rms": finite_float(rms(bridge_out - sample_state, None)),
        "sample_state_rms": finite_float(rms(sample_state, None)),
    }


def bridge_gradient_liveness(bridge: IdentityGatedBridge, sample_state: torch.Tensor) -> dict[str, Any]:
    was_training = bridge.training
    bridge.train()
    bridge.zero_grad(set_to_none=True)
    sample = sample_state.detach().float().requires_grad_(True)
    output = bridge(sample)
    loss = output.float().square().mean()
    loss.backward()
    gate_grad = bridge.bridge_gate.grad
    weight_grad = bridge.proj.weight.grad
    bias_grad = bridge.proj.bias.grad
    bridge.zero_grad(set_to_none=True)
    bridge.train(was_training)
    return {
        "loss": finite_float(loss),
        "gate_grad_abs": finite_float(gate_grad.abs() if gate_grad is not None else 0.0),
        "weight_grad_rms": finite_float(weight_grad.float().square().mean().sqrt() if weight_grad is not None else 0.0),
        "bias_grad_rms": finite_float(bias_grad.float().square().mean().sqrt() if bias_grad is not None else 0.0),
    }


def reference_bridge_liveness(hidden_size: int) -> dict[str, Any]:
    sample = torch.randn(2, 3, hidden_size)
    dead = IdentityGatedBridge(hidden_size=hidden_size, gate_init=0.0)
    live = IdentityGatedBridge(hidden_size=hidden_size, gate_init=1.0)
    return {
        "identity_gate0": bridge_gradient_liveness(dead, sample),
        "identity_gate1": bridge_gradient_liveness(live, sample),
    }


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


def prepare_recurrent_inputs(
    wrapper: RecurrentQwenForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, Any]:
    prepared = wrapper._prepare_inputs(  # noqa: SLF001
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
    )
    inputs_embeds = prepared["inputs_embeds"]
    position_ids = prepared["position_ids"]
    cache_position = prepared["cache_position"]
    causal_mask = wrapper._update_causal_mask(  # noqa: SLF001
        prepared["attention_mask"],
        inputs_embeds,
        cache_position,
        past_key_values=None,
        output_attentions=False,
    )
    position_embeddings = wrapper._rotary_embeddings(inputs_embeds, position_ids)  # noqa: SLF001
    hidden, _ = wrapper._run_layer_range(  # noqa: SLF001
        start=0,
        end=wrapper.layer_split.prelude_end,
        hidden_states=inputs_embeds,
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
    return hidden, attention_mask, causal_mask, position_ids, cache_position, position_embeddings


def run_recurrent_block(
    wrapper: RecurrentQwenForCausalLM,
    hidden: torch.Tensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
) -> torch.Tensor:
    out, _ = wrapper._run_layer_range(  # noqa: SLF001
        start=wrapper.layer_split.prelude_end,
        end=wrapper.layer_split.recurrent_end,
        hidden_states=hidden,
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
    return out


def loop_records_for_prompt(
    wrapper: RecurrentQwenForCausalLM,
    entry_state: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
    *,
    max_loops: int,
) -> list[dict[str, Any]]:
    entry_rms = rms(entry_state, attention_mask).clamp_min(1e-8)
    records: list[dict[str, Any]] = []
    recurrent_state = entry_state
    for loop_idx in range(max_loops):
        raw_loop_input = recurrent_state
        loop_input = raw_loop_input if loop_idx == 0 else wrapper.bridge(raw_loop_input)
        loop_output = run_recurrent_block(
            wrapper,
            loop_input,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        input_rms = rms(loop_input, attention_mask)
        output_rms = rms(loop_output, attention_mask)
        raw_input_rms = rms(raw_loop_input, attention_mask)
        bridge_delta_rms = rms(loop_input - raw_loop_input, attention_mask)
        records.append(
            {
                "loop": loop_idx + 1,
                "raw_input_rms": finite_float(raw_input_rms),
                "input_rms": finite_float(input_rms),
                "output_rms": finite_float(output_rms),
                "input_over_entry_rms": finite_float(input_rms / entry_rms),
                "output_over_entry_rms": finite_float(output_rms / entry_rms),
                "output_over_input_rms": finite_float(output_rms / input_rms.clamp_min(1e-8)),
                "bridge_delta_rms": finite_float(bridge_delta_rms),
                "pooled_input_output_cosine": pooled_cosine(loop_input, loop_output, attention_mask),
            }
        )
        recurrent_state = loop_output
    return records


def run_prompt(
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    prompt: str,
    *,
    prompt_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    with torch.no_grad():
        entry_state, mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
            wrapper,
            input_ids,
            attention_mask,
        )
        exit_state = run_recurrent_block(
            wrapper,
            entry_state,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        loop_records = loop_records_for_prompt(
            wrapper,
            entry_state,
            mask,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
            max_loops=args.max_loops,
        )
    entry_rms = rms(entry_state, mask)
    exit_rms = rms(exit_state, mask)
    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "tokens": int(mask.sum().item()),
        "entry_rms": finite_float(entry_rms),
        "exit_rms": finite_float(exit_rms),
        "exit_over_entry_rms": finite_float(exit_rms / entry_rms.clamp_min(1e-8)),
        "pooled_entry_exit_cosine": pooled_cosine(entry_state, exit_state, mask),
        "entry_tokens": masked_token_matrix(entry_state, mask).cpu(),
        "exit_tokens": masked_token_matrix(exit_state, mask).cpu(),
        "loop_records": loop_records,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_prompt_records(records: list[dict[str, Any]], *, subspace_rank: int) -> dict[str, Any]:
    entry_tokens = torch.cat([row["entry_tokens"] for row in records], dim=0)
    exit_tokens = torch.cat([row["exit_tokens"] for row in records], dim=0)
    loop_values: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        for loop_row in row["loop_records"]:
            loop_values.setdefault(int(loop_row["loop"]), []).append(loop_row)
    loop_summary = {}
    for loop, rows in sorted(loop_values.items()):
        metric_names = [
            "raw_input_rms",
            "input_rms",
            "output_rms",
            "input_over_entry_rms",
            "output_over_entry_rms",
            "output_over_input_rms",
            "bridge_delta_rms",
            "pooled_input_output_cosine",
        ]
        loop_summary[str(loop)] = {
            name: mean([float(row[name]) for row in rows])
            for name in metric_names
        }
    return {
        "prompts": len(records),
        "tokens": int(sum(int(row["tokens"]) for row in records)),
        "mean_entry_rms": mean([float(row["entry_rms"]) for row in records]),
        "mean_exit_rms": mean([float(row["exit_rms"]) for row in records]),
        "mean_exit_over_entry_rms": mean([float(row["exit_over_entry_rms"]) for row in records]),
        "mean_pooled_entry_exit_cosine": mean([float(row["pooled_entry_exit_cosine"]) for row in records]),
        "entry_exit_subspace": subspace_overlap(entry_tokens, exit_tokens, rank=subspace_rank),
        "loop_summary": loop_summary,
    }


def jsonable_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"entry_tokens", "exit_tokens"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompts_jsonl", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--subspace_rank", type=int, default=8)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_wrapper(args)
    prompts = read_prompts(args.prompts_jsonl or None, limit=args.limit or None)
    records = [
        run_prompt(wrapper, tokenizer, prompt, prompt_index=idx, args=args)
        for idx, prompt in enumerate(prompts)
    ]
    sample_state = records[0]["exit_tokens"][: min(16, records[0]["exit_tokens"].shape[0])].unsqueeze(0)
    sample_state = sample_state.to(device=wrapper.device, dtype=next(wrapper.bridge.parameters()).dtype)
    summary = {
        "kind": "reentry_drift_diagnostic",
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_loops": args.max_loops,
        "max_length": args.max_length,
        "subspace_rank": args.subspace_rank,
        "aggregate": aggregate_prompt_records(records, subspace_rank=args.subspace_rank),
        "bridge": bridge_stats(wrapper.bridge, sample_state),
        "bridge_gradient_liveness": bridge_gradient_liveness(wrapper.bridge, sample_state),
        "reference_bridge_gradient_liveness": reference_bridge_liveness(sample_state.shape[-1]),
        "records": [jsonable_record(row) for row in records],
    }
    print(json.dumps({k: summary[k] for k in ("aggregate", "bridge", "bridge_gradient_liveness")}, indent=2), flush=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_summary={path_for_cli(path)}", flush=True)
    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [jsonable_record(row) for row in records]
        path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
        print(f"saved_records={path_for_cli(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
