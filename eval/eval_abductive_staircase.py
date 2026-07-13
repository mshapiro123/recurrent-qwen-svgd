"""Evaluate inverse-composition staircase checkpoints at active loop labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_abductive_curriculum_autopsy import (  # noqa: E402
    evaluate_rows,
    reverse_target_for_loop,
    subset_by_depth,
    summarize_prediction_rows,
)
from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from eval.eval_reentry_drift import prepare_recurrent_inputs, run_recurrent_block  # noqa: E402
from eval.eval_synthetic_depth_active_labels import prompt_for_row, read_jsonl, row_symbol_names  # noqa: E402
from eval.eval_synthetic_depth_probe import permutation_p95, ridge_multiclass_accuracy  # noqa: E402
from models.halting import masked_mean  # noqa: E402


def conditional_transition_success(rows: list[dict[str, Any]], *, loop: int) -> dict[str, int | float]:
    by_id_loop = {(str(row["id"]), int(row["loop"])): row for row in rows if bool(row.get("active"))}
    eligible = [
        row
        for (row_id, row_loop), row in by_id_loop.items()
        if row_loop == int(loop)
        and bool(by_id_loop.get((row_id, int(loop) - 1), {}).get("hit"))
    ]
    correct = sum(int(bool(row["hit"])) for row in eligible)
    return {
        "correct": correct,
        "total": len(eligible),
        "accuracy": correct / len(eligible) if eligible else 0.0,
    }


def _split_ids(ids: list[str], *, seed: int) -> tuple[list[int], list[int]]:
    ranked = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(f"{seed}|{ids[index]}".encode("utf-8")).digest(),
    )
    split = max(1, min(len(ranked) - 1, len(ranked) // 2))
    return ranked[:split], ranked[split:]


def target_decodability_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ids: list[str],
    n_classes: int,
    permutations: int,
    seed: int,
    l2: float = 1e-2,
) -> dict[str, Any]:
    if len(ids) != features.shape[0] or targets.shape[0] != features.shape[0]:
        raise ValueError("features, targets, and ids must have matching rows")
    if features.shape[0] < 4:
        return {"status": "insufficient_rows", "rows": int(features.shape[0])}
    train_indices, test_indices = _split_ids(ids, seed=seed)
    train_x = features[train_indices].float()
    test_x = features[test_indices].float()
    train_y = targets[train_indices].long()
    test_y = targets[test_indices].long()
    accuracy = ridge_multiclass_accuracy(
        train_x,
        train_y,
        test_x,
        test_y,
        n_classes=int(n_classes),
        l2=float(l2),
    )
    null = permutation_p95(
        train_x,
        train_y,
        test_x,
        test_y,
        n_classes=int(n_classes),
        l2=float(l2),
        permutations=int(permutations),
        seed=int(seed) + 97,
    )
    return {
        "status": "completed",
        "accuracy": float(accuracy),
        "permutation_p95": float(null),
        "lift_over_p95": float(accuracy - null),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
    }


def centered_linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[0] != right.shape[0] or left.shape[0] < 2:
        raise ValueError("CKA inputs must have the same sample count >= 2")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    cross = left.T @ right
    numerator = cross.square().sum()
    denominator = ((left.T @ left).square().sum() * (right.T @ right).square().sum()).sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _state_tensor(output: Any) -> torch.Tensor:
    states = output.loop_recurrent_states
    if states is None:
        raise RuntimeError("wrapper did not return loop recurrent states")
    if states.dim() == 5:
        return states[:, 0]
    if states.dim() == 4:
        return states
    raise RuntimeError(f"unexpected loop state shape: {tuple(states.shape)}")


def collect_state_records(
    wrapper: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_loops: int,
    device: str,
    progress_every: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        if row_index == 1 or row_index % int(progress_every) == 0 or row_index == len(rows):
            print(f"staircase_probe_progress row={row_index}/{len(rows)} depth={row['depth']}", flush=True)
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            output = wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                max_loops=int(max_loops),
                particle_update_mode="none",
                use_cache=False,
                return_dict=True,
                return_loop_recurrent_states=True,
            )
        states = _state_tensor(output)
        pooled = torch.stack(
            [masked_mean(states[:, loop_index], encoded["attention_mask"]) for loop_index in range(states.shape[1])],
            dim=1,
        )
        symbols = row_symbol_names(row)
        for loop in range(1, min(int(row["depth"]), int(max_loops)) + 1):
            target = reverse_target_for_loop(row, loop)
            if target is None:
                continue
            records.append(
                {
                    "id": str(row["id"]),
                    "depth": int(row["depth"]),
                    "loop": loop,
                    "target": symbols.index(str(target)),
                    "feature": pooled[0, loop - 1].detach().float().cpu(),
                }
            )
    return records


def decodability_by_loop(
    records: list[dict[str, Any]],
    *,
    max_loops: int,
    n_classes: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for loop in range(1, int(max_loops) + 1):
        selected = [record for record in records if int(record["loop"]) == loop]
        if not selected:
            continue
        result[str(loop)] = target_decodability_probe(
            torch.stack([record["feature"] for record in selected]),
            torch.tensor([record["target"] for record in selected]),
            ids=[str(record["id"]) for record in selected],
            n_classes=n_classes,
            permutations=permutations,
            seed=seed + loop,
        )
    return result


def _paired_depth1_question(row: dict[str, Any], target: str) -> str:
    question = str(row["question"])
    if row.get("table_direction") == "inverse_given":
        pattern = re.compile(r"Starting with [^,]+, after exactly \d+ reverse handoffs")
        question, count = pattern.subn(
            f"Starting with {target}, after exactly 1 reverse handoffs",
            question,
            count=1,
        )
    else:
        pattern = re.compile(r"After exactly \d+ handoffs, the key is with [^.]+\.")
        question, count = pattern.subn(
            f"After exactly 1 handoffs, the key is with {target}.",
            question,
            count=1,
        )
    if count != 1:
        raise ValueError(f"could not build paired depth-1 prompt for {row['id']}")
    return question


def stratified_loop2_cka(
    wrapper: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    device: str,
    minimum_per_stratum: int,
) -> dict[str, Any]:
    loop2_hits = {
        str(row["id"]): bool(row["hit"])
        for row in prediction_rows
        if int(row["loop"]) == 2 and bool(row["active"])
    }
    strata: dict[bool, list[tuple[torch.Tensor, torch.Tensor]]] = {True: [], False: []}
    for row in rows:
        row_id = str(row["id"])
        if row_id not in loop2_hits or int(row["depth"]) < 2:
            continue
        prompt = f"{str(row['question']).rstrip()}\nAnswer:"
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        paired_question = _paired_depth1_question(row, reverse_target_for_loop(row, 1) or "")
        paired = tokenizer(
            f"{paired_question.rstrip()}\nAnswer:",
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            entry, mask, causal, positions, cache, rotary = prepare_recurrent_inputs(
                wrapper, encoded["input_ids"], encoded["attention_mask"]
            )
            loop1 = run_recurrent_block(wrapper, entry, causal, positions, cache, rotary)
            reentry = wrapper.bridge(loop1, prelude_hidden=entry)
            paired_entry, paired_mask, *_ = prepare_recurrent_inputs(
                wrapper, paired["input_ids"], paired["attention_mask"]
            )
        strata[loop2_hits[row_id]].append(
            (
                masked_mean(reentry, mask)[0].detach().float().cpu(),
                masked_mean(paired_entry, paired_mask)[0].detach().float().cpu(),
            )
        )

    output: dict[str, Any] = {"minimum_per_stratum": int(minimum_per_stratum)}
    for hit, label in ((True, "loop2_correct"), (False, "loop2_incorrect")):
        pairs = strata[hit]
        if len(pairs) < int(minimum_per_stratum):
            output[label] = {"status": "insufficient_rows", "rows": len(pairs)}
            continue
        output[label] = {
            "status": "completed",
            "rows": len(pairs),
            "cka": centered_linear_cka(
                torch.stack([pair[0] for pair in pairs]),
                torch.stack([pair[1] for pair in pairs]),
            ),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_per_depth", type=int, default=64)
    parser.add_argument("--max_loops", type=int, required=True)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--run_probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_train_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=81001)
    parser.add_argument("--run_cka", action="store_true")
    parser.add_argument("--cka_minimum_per_stratum", type=int, default=32)
    parser.add_argument("--progress_every", type=int, default=32)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
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

    train_rows = [row for row in subset_by_depth(read_jsonl(args.train_jsonl), args.rows_per_depth) if int(row["depth"]) <= args.max_loops]
    test_rows = [row for row in subset_by_depth(read_jsonl(args.test_jsonl), args.rows_per_depth) if int(row["depth"]) <= args.max_loops]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    predictions = (
        evaluate_rows(wrapper, tokenizer, train_rows, split="train", args=args)
        if args.include_train_predictions
        else []
    )
    predictions.extend(evaluate_rows(wrapper, tokenizer, test_rows, split="test", args=args))
    test_predictions = [row for row in predictions if row["split"] == "test"]
    summary = {
        "kind": "abductive_staircase_diagnostic",
        "checkpoint": args.checkpoint,
        "max_loops": args.max_loops,
        "rows_per_depth": args.rows_per_depth,
        "train": (
            summarize_prediction_rows(predictions, split="train")
            if args.include_train_predictions
            else {"status": "skipped"}
        ),
        "test": summarize_prediction_rows(predictions, split="test"),
        "conditional_transition_success": {
            str(loop): conditional_transition_success(test_predictions, loop=loop)
            for loop in range(2, args.max_loops + 1)
        },
    }
    if args.run_probes:
        state_records = collect_state_records(
            wrapper,
            tokenizer,
            test_rows,
            max_loops=args.max_loops,
            device=args.device,
            progress_every=args.progress_every,
        )
        summary["target_decodability"] = decodability_by_loop(
            state_records,
            max_loops=args.max_loops,
            n_classes=20,
            permutations=args.permutations,
            seed=args.seed,
        )
    else:
        summary["target_decodability"] = {"status": "skipped_for_checkpoint_gate"}
    if args.run_probes and args.run_cka:
        summary["stratified_loop2_cka"] = stratified_loop2_cka(
            wrapper,
            tokenizer,
            test_rows,
            test_predictions,
            device=args.device,
            minimum_per_stratum=args.cka_minimum_per_stratum,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
