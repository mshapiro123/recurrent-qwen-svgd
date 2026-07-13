"""Read-only localization of the Phase G injective curriculum failure.

The evaluator scores every loop on identical train and held-out rows, reports
active reverse-chain accuracy, diagonal confusion, above-diagonal dynamics,
and a paired state-query diagnostic. It never mutates or trains the model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from eval.eval_reentry_drift import prepare_recurrent_inputs, run_recurrent_block  # noqa: E402
from eval.eval_synthetic_depth_active_labels import (  # noqa: E402
    candidates_for_row,
    prompt_for_row,
    read_jsonl,
    score_candidates_all_loops,
)
from models.halting import masked_mean  # noqa: E402


def reverse_target_for_loop(row: dict[str, Any], loop: int) -> str | None:
    completions = list(row.get("loop_completions") or [])
    index = int(loop) - 1
    if index < 0 or index >= len(completions):
        return None
    return str(completions[index]).strip()


def _inverse_mapping(row: dict[str, Any]) -> dict[str, str]:
    mapping = {str(left): str(right) for left, right in row["mapping"].items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("reverse-chain continuation requires an injective mapping")
    return {right: left for left, right in mapping.items()}


def continued_reverse_target(row: dict[str, Any], loop: int) -> str | None:
    if int(loop) <= int(row["depth"]):
        return reverse_target_for_loop(row, loop)
    current = str(row["selected_start"])
    inverse = _inverse_mapping(row)
    for _ in range(int(loop) - int(row["depth"])):
        current = inverse[current]
    return current


def classify_abductive_prediction(row: dict[str, Any], prediction: str) -> str:
    prediction = str(prediction).strip()
    if prediction == str(row["selected_start"]):
        return "correct_start"
    if prediction == reverse_target_for_loop(row, 1):
        return "one_step_preimage"
    intermediates = {str(value) for value in row.get("selected_orbit", [])[1:-1]}
    if prediction in intermediates:
        return "other_orbit_intermediate"
    if prediction == str(row["observed_target"]):
        return "observed_target"
    if prediction in {str(value) for value in row.get("mapping", {})}:
        return "other_valid_name"
    return "junk"


def _rate(correct: int, total: int) -> dict[str, int | float]:
    return {"correct": int(correct), "total": int(total), "accuracy": correct / total if total else 0.0}


def summarize_prediction_rows(rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["split"]) == str(split)]
    active_counts: dict[tuple[int, int], list[int]] = {}
    loop1_counts: dict[int, list[int]] = {}
    diagonal_counts: dict[int, list[int]] = {}
    confusion: Counter[str] = Counter()
    above: Counter[str] = Counter()
    for row in selected:
        depth = int(row["depth"])
        loop = int(row["loop"])
        if bool(row["active"]):
            cell = active_counts.setdefault((depth, loop), [0, 0])
            cell[0] += int(bool(row["hit"]))
            cell[1] += 1
            if loop == 1:
                cell = loop1_counts.setdefault(depth, [0, 0])
                cell[0] += int(bool(row["hit"]))
                cell[1] += 1
            if loop == depth:
                cell = diagonal_counts.setdefault(depth, [0, 0])
                cell[0] += int(bool(row["hit"]))
                cell[1] += 1
                confusion[str(row["diagonal_confusion"])] += 1
        else:
            above[str(row["above_diagonal_behavior"])] += 1

    depths = sorted({depth for depth, _loop in active_counts})
    loops = sorted({loop for _depth, loop in active_counts})
    matrix = {
        str(depth): {
            str(loop): _rate(*active_counts.get((depth, loop), [0, 0]))
            for loop in loops
            if loop <= depth
        }
        for depth in depths
    }
    return {
        "split": split,
        "source_rows": len(
            {str(row.get("id", f"anonymous_row_{index}")) for index, row in enumerate(selected)}
        ),
        "active_matrix": matrix,
        "loop1_by_depth": {str(depth): _rate(*counts) for depth, counts in sorted(loop1_counts.items())},
        "diagonal_by_depth": {str(depth): _rate(*counts) for depth, counts in sorted(diagonal_counts.items())},
        "diagonal_confusion": dict(sorted(confusion.items())),
        "above_diagonal": dict(sorted(above.items())),
    }


def subset_by_depth(rows: list[dict[str, Any]], rows_per_depth: int) -> list[dict[str, Any]]:
    counts: Counter[int] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        depth = int(row["depth"])
        if counts[depth] >= int(rows_per_depth):
            continue
        selected.append(row)
        counts[depth] += 1
    return selected


def evaluate_rows(
    wrapper: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    split: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    loops = list(range(1, int(args.max_loops) + 1))
    for index, row in enumerate(rows, start=1):
        if index == 1 or index % int(args.progress_every) == 0 or index == len(rows):
            print(f"autopsy_progress split={split} row={index}/{len(rows)} depth={row['depth']}", flush=True)
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="name:")
        scores_by_loop = score_candidates_all_loops(wrapper, tokenizer, prompt, candidates, args, loop_counts=loops)
        depth = int(row["depth"])
        for loop in loops:
            prediction = max(scores_by_loop[loop].items(), key=lambda item: item[1])[0]
            target = reverse_target_for_loop(row, loop)
            active = target is not None
            behavior = None
            if not active:
                if prediction == continued_reverse_target(row, loop):
                    behavior = "iterate"
                elif prediction == str(row["selected_start"]):
                    behavior = "hold"
                else:
                    behavior = "other"
            output.append(
                {
                    "id": row["id"],
                    "split": split,
                    "depth": depth,
                    "loop": loop,
                    "prediction": prediction,
                    "target": target,
                    "active": active,
                    "hit": bool(active and prediction == target),
                    "above_diagonal_behavior": behavior,
                    "diagonal_confusion": (
                        classify_abductive_prediction(row, prediction) if loop == depth else None
                    ),
                }
            )
    return output


def _paired_depth1_question(row: dict[str, Any], observed_target: str) -> str:
    sentence = re.compile(r"After exactly \d+ handoffs, the key is with [^.]+\.")
    replacement = f"After exactly 1 handoffs, the key is with {observed_target}."
    question, substitutions = sentence.subn(replacement, str(row["question"]), count=1)
    if substitutions != 1:
        raise ValueError(f"Could not rewrite query sentence for row {row['id']}")
    return question


def _rank_and_margin(scores: dict[str, float], correct: str) -> dict[str, Any]:
    ordered = sorted(scores, key=scores.get, reverse=True)
    rank = ordered.index(correct) + 1
    best_decoy = max(value for name, value in scores.items() if name != correct)
    return {
        "correct": correct,
        "rank": rank,
        "reciprocal_rank": 1.0 / rank,
        "top1": rank == 1,
        "correct_cosine": scores[correct],
        "margin_over_best_decoy": scores[correct] - best_decoy,
    }


def state_query_probe(
    wrapper: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    examples: int,
    device: str,
) -> dict[str, Any]:
    selected = [row for row in rows if int(row["depth"]) == 2][: int(examples)]
    records: list[dict[str, Any]] = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for index, row in enumerate(selected, start=1):
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            entry, mask, causal, positions, cache, rotary = prepare_recurrent_inputs(
                wrapper, encoded["input_ids"], encoded["attention_mask"]
            )
            loop1_output = run_recurrent_block(wrapper, entry, causal, positions, cache, rotary)
            loop2_input = wrapper.bridge(loop1_output, prelude_hidden=entry)
            loop2_output = run_recurrent_block(wrapper, loop2_input, causal, positions, cache, rotary)

        names = [str(name) for name in row["mapping"]]
        paired_prompts = [f"{_paired_depth1_question(row, name)}\nAnswer:" for name in names]
        paired = tokenizer(paired_prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        with torch.no_grad():
            paired_entry, paired_mask, paired_causal, paired_positions, paired_cache, paired_rotary = (
                prepare_recurrent_inputs(wrapper, paired["input_ids"], paired["attention_mask"])
            )
            paired_output = run_recurrent_block(
                wrapper, paired_entry, paired_causal, paired_positions, paired_cache, paired_rotary
            )
        loop2_input_pool = masked_mean(loop2_input, mask).float()
        loop2_output_pool = masked_mean(loop2_output, mask).float()
        paired_entry_pool = masked_mean(paired_entry, paired_mask).float()
        paired_output_pool = masked_mean(paired_output, paired_mask).float()
        entry_cosines = F.cosine_similarity(loop2_input_pool, paired_entry_pool, dim=-1)
        output_cosines = F.cosine_similarity(loop2_output_pool, paired_output_pool, dim=-1)
        correct_query = str(row["selected_orbit"][-2])
        records.append(
            {
                "id": row["id"],
                "index": index,
                "correct_query": correct_query,
                "reentry_to_prompt_entry": _rank_and_margin(
                    dict(zip(names, (float(value) for value in entry_cosines.cpu().tolist()))),
                    correct_query,
                ),
                "loop2_output_to_paired_loop1_output": _rank_and_margin(
                    dict(zip(names, (float(value) for value in output_cosines.cpu().tolist()))),
                    correct_query,
                ),
            }
        )

    def aggregate(key: str) -> dict[str, float | int]:
        values = [row[key] for row in records]
        return {
            "n": len(values),
            "top1_rate": sum(int(value["top1"]) for value in values) / len(values) if values else 0.0,
            "mean_reciprocal_rank": (
                sum(float(value["reciprocal_rank"]) for value in values) / len(values) if values else 0.0
            ),
            "mean_margin_over_best_decoy": (
                sum(float(value["margin_over_best_decoy"]) for value in values) / len(values) if values else 0.0
            ),
        }

    return {
        "kind": "abductive_state_query_probe",
        "interpretation": "exploratory_similarity_ranking_not_a_pass_fail_gate",
        "reentry_to_prompt_entry": aggregate("reentry_to_prompt_entry"),
        "loop2_output_to_paired_loop1_output": aggregate("loop2_output_to_paired_loop1_output"),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_per_depth", type=int, default=16)
    parser.add_argument("--max_loops", type=int, default=8)
    parser.add_argument("--state_query_examples", type=int, default=8)
    parser.add_argument("--progress_every", type=int, default=16)
    parser.add_argument("--normalize_candidate_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_slow_candidate_score", action="store_true")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    args = parser.parse_args()

    train_rows = subset_by_depth(read_jsonl(args.train_jsonl), args.rows_per_depth)
    test_rows = subset_by_depth(read_jsonl(args.test_jsonl), args.rows_per_depth)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    predictions = evaluate_rows(wrapper, tokenizer, train_rows, split="train", args=args)
    predictions.extend(evaluate_rows(wrapper, tokenizer, test_rows, split="test", args=args))
    state_query = state_query_probe(
        wrapper,
        tokenizer,
        test_rows,
        examples=args.state_query_examples,
        device=args.device,
    )
    summary = {
        "kind": "abductive_curriculum_autopsy",
        "checkpoint": args.checkpoint,
        "rows_per_depth": args.rows_per_depth,
        "max_loops": args.max_loops,
        "train": summarize_prediction_rows(predictions, split="train"),
        "test": summarize_prediction_rows(predictions, split="test"),
        "state_query_probe": state_query,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
