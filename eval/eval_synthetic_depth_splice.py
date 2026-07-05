"""Counterfactual hidden-state splice test for the synthetic depth task.

The active-label accuracy curve cannot distinguish a genuine state-driven
iterator from a prompt/position shortcut.  This diagnostic captures row B's
recurrent state after loop k0, injects that state into row A before loop k0+1,
and then scores whether later predictions follow A's table from B's symbol
(``lawful``) or stay on A's original orbit (``shortcut``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper
from eval.eval_synthetic_depth_active_labels import (
    apply_mapping,
    parse_int_symbol,
    prompt_for_row,
    row_mapping,
    symbol,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("instance_id") or "")


def prompt_for_splice(row: dict[str, Any]) -> str:
    return prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")


def token_ids_for_symbols(tokenizer: Any, n_symbols: int, *, value_prefix: str) -> dict[str, int]:
    ids: dict[str, int] = {}
    for idx in range(n_symbols):
        name = symbol(idx, prefix=value_prefix)
        encoded = tokenizer(f" {name}", add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(
                "Splice diagnostic requires one-token full symbols for prompt-only next-token scoring; "
                f"symbol={name!r} token_ids={encoded!r}"
            )
        ids[name] = int(encoded[0])
    return ids


def predict_symbols_by_loop(
    wrapper: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    token_ids: dict[str, int],
    max_loops: int,
    recurrent_state_overrides: dict[int, torch.Tensor] | None = None,
    return_states: bool = False,
) -> tuple[dict[int, str], torch.Tensor | None]:
    encoded = tokenizer(prompt_for_splice(row), return_tensors="pt", add_special_tokens=True).to(args.device)
    with torch.no_grad():
        output = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=max_loops,
            num_trajectories=1,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            return_loop_recurrent_states=return_states,
            recurrent_state_overrides=recurrent_state_overrides,
            bridge_projection_mode=args.bridge_projection_mode,
            logits_to_keep=1,
        )
    loop_logits = output.loop_logits
    if loop_logits is None:
        raise RuntimeError("Expected loop_logits from recurrent wrapper")
    predictions: dict[int, str] = {}
    for loop in range(1, max_loops + 1):
        logits = loop_logits[0, 0, loop - 1, -1]
        predictions[loop] = max(token_ids, key=lambda name: float(logits[token_ids[name]].item()))
    return predictions, output.loop_recurrent_states


def symbol_at_orbit(row: dict[str, Any], loop: int, *, value_prefix: str) -> str:
    orbit = list(row.get("orbit") or [])
    if len(orbit) > loop:
        return symbol(orbit[loop], prefix=value_prefix)
    mapping = row_mapping(row, value_prefix=value_prefix)
    if mapping is None:
        raise ValueError(f"Row {row_id(row)} missing mapping")
    start = parse_int_symbol(row["start"], prefix=value_prefix)
    return symbol(apply_mapping(mapping, start, loop), prefix=value_prefix)


def lawful_symbol(row_a: dict[str, Any], start_symbol: str, steps: int, *, value_prefix: str) -> str:
    mapping = row_mapping(row_a, value_prefix=value_prefix)
    if mapping is None:
        raise ValueError(f"Row {row_id(row_a)} missing mapping")
    current = parse_int_symbol(start_symbol, prefix=value_prefix)
    return symbol(apply_mapping(mapping, current, steps), prefix=value_prefix)


def compatible_rows(rows: list[dict[str, Any]], tokenizer: Any, *, target_depth: int) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("depth", 0)) != int(target_depth):
            continue
        length = len(tokenizer(prompt_for_splice(row), add_special_tokens=True)["input_ids"])
        buckets[length].append(row)
    return {length: bucket for length, bucket in buckets.items() if len(bucket) >= 2}


def paired_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    target_depth: int,
    n_pairs: int,
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    generator = torch.Generator().manual_seed(int(seed))
    buckets = compatible_rows(rows, tokenizer, target_depth=target_depth)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, bucket in sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True):
        order = torch.randperm(len(bucket), generator=generator).tolist()
        shuffled = [bucket[idx] for idx in order]
        for pair_idx in range(n_pairs - len(pairs)):
            left = shuffled[pair_idx % len(shuffled)]
            right = shuffled[(len(shuffled) - 1 - pair_idx) % len(shuffled)]
            if row_id(left) == row_id(right):
                right = shuffled[(pair_idx + 1) % len(shuffled)]
            pairs.append((left, right))
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


def classify_prediction(prediction: str, lawful: str, shortcut: str) -> str:
    if prediction == lawful:
        return "lawful"
    if prediction == shortcut:
        return "shortcut"
    return "other"


def summarize_records(records: list[dict[str, Any]], *, lawful_bar: float, shortcut_bar: float) -> dict[str, Any]:
    by_k0_j: dict[str, dict[str, int]] = {}
    verdict_rows: list[dict[str, Any]] = []
    overall = {"lawful": 0, "shortcut": 0, "other": 0, "n": 0}
    verdict_totals = {"lawful": 0, "shortcut": 0, "other": 0, "n": 0}
    for record in records:
        key = f"k0={record['k0']},j={record['j']}"
        counts = by_k0_j.setdefault(key, {"lawful": 0, "shortcut": 0, "other": 0, "n": 0})
        label = str(record["classification"])
        counts[label] += 1
        counts["n"] += 1
        overall[label] += 1
        overall["n"] += 1
        if int(record["j"]) <= 3:
            verdict_totals[label] += 1
            verdict_totals["n"] += 1

    for key, counts in by_k0_j.items():
        n = max(int(counts["n"]), 1)
        verdict_rows.append(
            {
                "condition": key,
                **counts,
                "lawful_fraction": counts["lawful"] / n,
                "shortcut_fraction": counts["shortcut"] / n,
                "other_fraction": counts["other"] / n,
            }
        )

    verdict_n = max(int(verdict_totals["n"]), 1)
    lawful_fraction = verdict_totals["lawful"] / verdict_n
    shortcut_fraction = verdict_totals["shortcut"] / verdict_n
    if lawful_fraction >= lawful_bar:
        verdict = "state_driven"
    elif shortcut_fraction >= shortcut_bar:
        verdict = "prompt_position_shortcut"
    else:
        verdict = "mixed"
    return {
        "kind": "synthetic_depth_splice_injection",
        "records": len(records),
        "by_k0_j": verdict_rows,
        "overall_counts": overall,
        "verdict_window": "j<=3",
        "verdict_counts": verdict_totals,
        "lawful_fraction_j1_to_j3": lawful_fraction,
        "shortcut_fraction_j1_to_j3": shortcut_fraction,
        "lawful_bar": lawful_bar,
        "shortcut_bar": shortcut_bar,
        "verdict": verdict,
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(args.data_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    token_ids = token_ids_for_symbols(tokenizer, args.n_symbols, value_prefix=args.value_prefix)
    pairs = paired_rows(
        rows,
        tokenizer,
        target_depth=args.target_depth,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )
    if not pairs:
        raise ValueError(f"No compatible row pairs found at depth {args.target_depth}")

    records: list[dict[str, Any]] = []
    for k0 in parse_int_list(args.splice_points):
        for pair_index, (row_a, row_b) in enumerate(pairs):
            _, states_b = predict_symbols_by_loop(
                wrapper,
                tokenizer,
                row_b,
                args,
                token_ids=token_ids,
                max_loops=k0,
                return_states=True,
            )
            if states_b is None:
                raise RuntimeError("Expected loop_recurrent_states while capturing row B")
            state_b = states_b[0, 0, k0 - 1].unsqueeze(0)
            predictions, _ = predict_symbols_by_loop(
                wrapper,
                tokenizer,
                row_a,
                args,
                token_ids=token_ids,
                max_loops=args.max_loops,
                recurrent_state_overrides={k0: state_b},
                return_states=False,
            )
            spliced = symbol_at_orbit(row_b, k0, value_prefix=args.value_prefix)
            for j in range(1, args.max_loops - k0 + 1):
                loop = k0 + j
                lawful = lawful_symbol(row_a, spliced, j, value_prefix=args.value_prefix)
                shortcut = symbol_at_orbit(row_a, loop, value_prefix=args.value_prefix)
                if lawful == shortcut:
                    continue
                prediction = predictions[loop]
                records.append(
                    {
                        "pair_index": pair_index,
                        "row_a_id": row_id(row_a),
                        "row_b_id": row_id(row_b),
                        "k0": k0,
                        "j": j,
                        "loop": loop,
                        "spliced_symbol": spliced,
                        "lawful_target": lawful,
                        "shortcut_target": shortcut,
                        "prediction": prediction,
                        "classification": classify_prediction(prediction, lawful, shortcut),
                    }
                )
    summary = summarize_records(
        records,
        lawful_bar=args.lawful_bar,
        shortcut_bar=args.shortcut_bar,
    )
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "target_depth": args.target_depth,
            "splice_points": parse_int_list(args.splice_points),
            "max_loops": args.max_loops,
            "n_pairs_requested": args.n_pairs,
            "n_pairs_used": len(pairs),
            "n_symbols": args.n_symbols,
            "value_prefix": args.value_prefix,
        }
    )
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--target_depth", type=int, default=8)
    parser.add_argument("--splice_points", default="2,4")
    parser.add_argument("--max_loops", type=int, default=8)
    parser.add_argument("--n_pairs", type=int, default=128)
    parser.add_argument("--n_symbols", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lawful_bar", type=float, default=0.75)
    parser.add_argument("--shortcut_bar", type=float, default=0.50)
    parser.add_argument("--value_prefix", default="letter:")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    args = parser.parse_args()

    records, summary = evaluate(args)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
