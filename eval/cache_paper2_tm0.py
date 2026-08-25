"""Cache prompt-only student and teacher trajectories for ratified TM-0.

The cache is deliberately score-blind. Correctness labels are produced by the
standing P3.1 readers in a separate pass so gold answers never enter the state
forward used for TM geometry.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import (
    _chat_prompt,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
)
from training.paper2_tm0 import atomic_json, load_lock, read_jsonl, sha256_file


def prompt_text(tokenizer: Any, row: Mapping[str, Any]) -> str:
    battery = str(row["battery"])
    if battery in {"arc_easy", "arc_challenge", "mmlu"}:
        question, choices, _answer = _mcq(row)
        return _mcq_prompt(question, choices)
    content, _cap = _generation_prompt(row)
    return _chat_prompt(tokenizer, content)


def active_pools(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return last-active and active-token mean pools as [batch, 2, width]."""

    mask = attention_mask.to(dtype=torch.bool)
    lengths = mask.sum(dim=1)
    if bool((lengths < 1).any()):
        raise RuntimeError("TM-0 prompt contains no active tokens")
    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
    mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
    return torch.stack((last, mean), dim=1)


def encode_prompts(
    tokenizer: Any, rows: Sequence[Mapping[str, Any]], device: torch.device
) -> dict[str, torch.Tensor]:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer(
        [prompt_text(tokenizer, row) for row in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.inference_mode()
def forward_pools(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    student: bool,
) -> dict[str, torch.Tensor]:
    encoded = encode_prompts(tokenizer, rows, device)
    output = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if student:
        # hidden_states[0] is the embedding output; index 6 is layer 5 output.
        h0 = active_pools(output.hidden_states[6].float(), encoded["attention_mask"])
        previous = active_pools(output.hidden_states[4].float(), encoded["attention_mask"])
        result = {"h0": h0.cpu(), "delta_h_p": (h0 - previous).cpu()}
    else:
        layers = [
            active_pools(hidden.float(), encoded["attention_mask"]).cpu()
            for hidden in output.hidden_states[1:]
        ]
        result = {"layers": torch.stack(layers, dim=1)}
    # The final active-token logits are an execution-schedule identity probe only.
    result["last_logits"] = active_pools(output.logits.float(), encoded["attention_mask"])[:, 0].cpu()
    return result


def batch_invariance_probe(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    student: bool,
) -> dict[str, Any]:
    batched = forward_pools(model, tokenizer, rows, device=device, student=student)
    sequential_parts: dict[str, list[torch.Tensor]] = {key: [] for key in batched}
    for row in rows:
        current = forward_pools(model, tokenizer, [row], device=device, student=student)
        for key, value in current.items():
            sequential_parts[key].append(value)
    sequential = {key: torch.cat(values, dim=0) for key, values in sequential_parts.items()}
    comparisons = {}
    exact = True
    for key in batched:
        left = batched[key]
        right = sequential[key]
        equal = bool(torch.equal(left, right))
        exact &= equal
        comparisons[key] = {
            "exact": equal,
            "maximum_absolute_difference": float((left - right).abs().max()),
        }
    return {
        "rows": len(rows),
        "exact": exact,
        "comparisons": comparisons,
        "selected_schedule": "batched" if exact else "sequential",
    }


def _write_shard(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def cache_model(
    *,
    model_key: str,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    model_cache: Path,
    shard_rows: int,
    probe_rows: Sequence[Mapping[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    lock = load_lock()
    spec = lock["models"][model_key]
    student = model_key == "student"
    if not torch.cuda.is_available():
        raise RuntimeError("TM-0 cache pass requires CUDA")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], cache_dir=model_cache
    )
    loaded_at = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        cache_dir=model_cache,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    load_seconds = time.perf_counter() - loaded_at
    probe = batch_invariance_probe(
        model, tokenizer, probe_rows, device=device, student=student
    )
    schedule = probe["selected_schedule"]
    batch_size = 1 if schedule == "sequential" else shard_rows
    selected = list(probe_rows) if dry_run else list(rows)
    started = time.perf_counter()
    shards = []
    buffered: dict[str, list[torch.Tensor]] = {}
    buffered_ids: list[str] = []
    shard_index = 0
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        values = forward_pools(model, tokenizer, batch, device=device, student=student)
        values.pop("last_logits")
        for key, tensor in values.items():
            buffered.setdefault(key, []).append(tensor.to(torch.float16))
        buffered_ids.extend(str(row["item_id"]) for row in batch)
        if len(buffered_ids) >= shard_rows or start + len(batch) == len(selected):
            payload = {
                "kind": "paper2_tm0_state_cache_shard_v1",
                "model_key": model_key,
                "model": spec["id"],
                "revision": spec["revision"],
                "prompt_contract": "prompt_only_no_gold_answer_v1",
                "pooling": ["last_active_token", "active_token_mean"],
                "item_ids": buffered_ids,
                **{key: torch.cat(parts, dim=0) for key, parts in buffered.items()},
            }
            shard_path = output_dir / "shards" / f"{model_key}_{shard_index:04d}.pt"
            shards.append(_write_shard(shard_path, payload))
            buffered, buffered_ids = {}, []
            shard_index += 1
        if start == 0 or (start + len(batch)) % 64 == 0:
            print(
                f"tm0_cache_progress model={model_key} rows={start + len(batch)}/{len(selected)}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    per_row = elapsed / max(len(selected), 1)
    receipt = {
        "kind": "paper2_tm0_state_cache_index_v1",
        "model_key": model_key,
        "model": spec["id"],
        "revision": spec["revision"],
        "rows": len(selected),
        "full_panel_rows": len(rows),
        "dry_run": dry_run,
        "dtype": "float16_storage_from_bfloat16_forward",
        "attention": "sdpa",
        "prompt_contract": "prompt_only_no_gold_answer_v1",
        "pooling": ["last_active_token", "active_token_mean"],
        "load_seconds": load_seconds,
        "forward_seconds": elapsed,
        "seconds_per_row": per_row,
        "projected_full_forward_seconds": per_row * len(rows),
        "batch_invariance": probe,
        "shards": shards,
        "battery_counts": dict(sorted(Counter(str(row["battery"]) for row in selected).items())),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "injection_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(output_dir / f"{model_key}_cache_index.json", receipt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--probe_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--model_key", choices=("student", "teacher_7b", "teacher_14b"), required=True)
    parser.add_argument("--shard_rows", type=int, default=64)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    lock = load_lock()
    rows = read_jsonl(args.panel)
    probes_by_battery: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(args.probe_manifest):
        probes_by_battery.setdefault(str(row["battery"]), []).append(row)
    probe_rows = []
    for battery in sorted(probes_by_battery):
        probe_rows.extend(
            probes_by_battery[battery][: int(lock["runtime"]["dry_run_rows_per_battery"])]
        )
    if len(rows) != int(lock["panels"]["total_rows"]):
        raise RuntimeError("TM-0 state cache panel cardinality changed")
    if not probe_rows:
        raise RuntimeError("TM-0 state cache has no batch-invariance probe rows")
    receipt = cache_model(
        model_key=args.model_key,
        rows=rows,
        output_dir=args.output_dir,
        model_cache=args.model_cache,
        shard_rows=args.shard_rows,
        probe_rows=probe_rows,
        dry_run=args.dry_run,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
