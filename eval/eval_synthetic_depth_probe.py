"""Probe synthetic-depth recurrent states for orbit position and loop index."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper
from eval.eval_reentry_drift import prepare_recurrent_inputs, run_recurrent_block
from eval.eval_synthetic_depth_active_labels import (
    continued_symbol_for_loop,
    parse_int_symbol,
    prompt_for_row,
    read_jsonl,
)
from models.halting import masked_mean


def parse_csv_ints(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item.strip()]


def deterministic_split(n: int, *, train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    cut = max(1, min(n - 1, int(round(n * train_frac)))) if n > 1 else n
    return indices[:cut], indices[cut:]


def standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def ridge_multiclass_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    n_classes: int,
    l2: float,
) -> float:
    train_x, test_x = standardize(train_x.float(), test_x.float())
    train_aug = torch.cat([train_x, torch.ones(train_x.shape[0], 1)], dim=1)
    test_aug = torch.cat([test_x, torch.ones(test_x.shape[0], 1)], dim=1)
    y_onehot = torch.nn.functional.one_hot(train_y.long(), num_classes=n_classes).float()
    eye = torch.eye(train_aug.shape[1], dtype=train_aug.dtype)
    eye[-1, -1] = 0.0
    lhs = train_aug.T @ train_aug + float(l2) * eye
    rhs = train_aug.T @ y_onehot
    try:
        weights = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        weights = torch.linalg.pinv(lhs) @ rhs
    pred = (test_aug @ weights).argmax(dim=-1)
    return float((pred == test_y.long()).float().mean().item())


def permutation_p95(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    n_classes: int,
    l2: float,
    permutations: int,
    seed: int,
) -> float:
    if permutations <= 0:
        return 0.0
    gen = torch.Generator().manual_seed(seed)
    values: list[float] = []
    for _ in range(permutations):
        order = torch.randperm(train_y.numel(), generator=gen)
        values.append(
            ridge_multiclass_accuracy(
                train_x,
                train_y[order],
                test_x,
                test_y,
                n_classes=n_classes,
                l2=l2,
            )
        )
    values.sort()
    idx = min(len(values) - 1, int(0.95 * (len(values) - 1)))
    return float(values[idx])


def target_for_step(row: dict[str, Any], step: int, *, value_prefix: str) -> int:
    text = continued_symbol_for_loop(row, step, value_prefix=value_prefix)
    if text is None:
        raise ValueError(f"Could not compute f^{step}(x) for row {row.get('id')}")
    return parse_int_symbol(text, prefix=value_prefix)


def collect_state_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    rows = read_jsonl(args.data_jsonl)
    max_rows = int(args.max_rows)
    if max_rows > 0:
        rows = rows[:max_rows]
    loops = parse_csv_ints(args.loop_counts)
    target_steps = parse_csv_ints(args.target_steps)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    records: list[dict[str, Any]] = []
    max_loop = max(loops)
    with torch.no_grad():
        for row_idx, row in enumerate(rows):
            prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style=args.prompt_style)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
            ).to(args.device)
            prelude, mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
                wrapper,
                encoded["input_ids"],
                encoded["attention_mask"],
            )
            recurrent_state = prelude
            for loop_idx in range(max_loop):
                loop_number = loop_idx + 1
                loop_input = (
                    recurrent_state
                    if loop_idx == 0
                    else wrapper.bridge(recurrent_state, prelude_hidden=prelude)
                )
                recurrent_state = run_recurrent_block(
                    wrapper,
                    loop_input,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                )
                if loop_number not in loops:
                    continue
                pooled = masked_mean(recurrent_state, mask).squeeze(0).detach().float().cpu()
                targets = {
                    str(step): target_for_step(row, step, value_prefix=args.value_prefix)
                    for step in target_steps
                }
                records.append(
                    {
                        "row_index": row_idx,
                        "id": row.get("id") or row.get("instance_id"),
                        "depth": int(row["depth"]),
                        "loop": loop_number,
                        "feature": pooled,
                        "targets": targets,
                    }
                )
    n_symbols = int(rows[0]["n_symbols"]) if rows else int(args.n_symbols)
    return records, n_symbols


def probe_grid(records: list[dict[str, Any]], args: argparse.Namespace, *, n_symbols: int) -> dict[str, Any]:
    loops = parse_csv_ints(args.loop_counts)
    target_steps = parse_csv_ints(args.target_steps)
    row_ids = sorted({int(record["row_index"]) for record in records})
    train_rows, test_rows = deterministic_split(len(row_ids), train_frac=args.train_frac, seed=args.seed)
    row_id_by_pos = {pos: row_id for pos, row_id in enumerate(row_ids)}
    train_set = {row_id_by_pos[pos] for pos in train_rows}
    test_set = {row_id_by_pos[pos] for pos in test_rows}
    by_loop: dict[int, list[dict[str, Any]]] = {loop: [] for loop in loops}
    for record in records:
        by_loop[int(record["loop"])].append(record)

    grid: dict[str, dict[str, Any]] = {}
    for loop in loops:
        loop_records = by_loop[loop]
        train_records = [record for record in loop_records if int(record["row_index"]) in train_set]
        test_records = [record for record in loop_records if int(record["row_index"]) in test_set]
        train_x = torch.stack([record["feature"] for record in train_records])
        test_x = torch.stack([record["feature"] for record in test_records])
        grid[str(loop)] = {}
        for target_step in target_steps:
            train_y = torch.tensor([record["targets"][str(target_step)] for record in train_records], dtype=torch.long)
            test_y = torch.tensor([record["targets"][str(target_step)] for record in test_records], dtype=torch.long)
            accuracy = ridge_multiclass_accuracy(
                train_x,
                train_y,
                test_x,
                test_y,
                n_classes=n_symbols,
                l2=args.ridge_l2,
            )
            null_p95 = permutation_p95(
                train_x,
                train_y,
                test_x,
                test_y,
                n_classes=n_symbols,
                l2=args.ridge_l2,
                permutations=args.permutations,
                seed=args.seed + 997 * loop + target_step,
            )
            grid[str(loop)][str(target_step)] = {
                "accuracy": accuracy,
                "permutation_p95": null_p95,
                "lift_over_p95": accuracy - null_p95,
                "train_rows": len(train_records),
                "test_rows": len(test_records),
            }

    all_records = [record for record in records if int(record["loop"]) in loops]
    train_records = [record for record in all_records if int(record["row_index"]) in train_set]
    test_records = [record for record in all_records if int(record["row_index"]) in test_set]
    train_x = torch.stack([record["feature"] for record in train_records])
    test_x = torch.stack([record["feature"] for record in test_records])
    train_y = torch.tensor([int(record["loop"]) - 1 for record in train_records], dtype=torch.long)
    test_y = torch.tensor([int(record["loop"]) - 1 for record in test_records], dtype=torch.long)
    loop_index = {
        "accuracy": ridge_multiclass_accuracy(
            train_x,
            train_y,
            test_x,
            test_y,
            n_classes=len(loops),
            l2=args.ridge_l2,
        ),
        "permutation_p95": permutation_p95(
            train_x,
            train_y,
            test_x,
            test_y,
            n_classes=len(loops),
            l2=args.ridge_l2,
            permutations=args.permutations,
            seed=args.seed + 12345,
        ),
        "train_rows": len(train_records),
        "test_rows": len(test_records),
    }
    loop_index["lift_over_p95"] = loop_index["accuracy"] - loop_index["permutation_p95"]
    return {
        "grid": grid,
        "loop_index_probe": loop_index,
        "split": {
            "train_row_indices": sorted(train_set),
            "test_row_indices": sorted(test_set),
            "train_frac": args.train_frac,
            "seed": args.seed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--loop_counts", default="1,2,3,4")
    parser.add_argument("--target_steps", default="0,1,2,3,4")
    parser.add_argument("--n_symbols", type=int, default=16)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="question_only")
    parser.add_argument("--value_prefix", default="letter:")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--ridge_l2", type=float, default=1e-2)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    args = parser.parse_args()

    records, n_symbols = collect_state_rows(args)
    payload = probe_grid(records, args, n_symbols=n_symbols)
    summary = {
        "kind": "synthetic_depth_state_probe",
        "checkpoint": args.checkpoint,
        "data_jsonl": args.data_jsonl,
        "rows": len({record["row_index"] for record in records}),
        "state_records": len(records),
        "n_symbols": n_symbols,
        "loop_counts": parse_csv_ints(args.loop_counts),
        "target_steps": parse_csv_ints(args.target_steps),
        **payload,
    }
    out = Path(args.output_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
