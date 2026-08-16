"""Run the authorized KP-1R cheap rung from the frozen T1 state cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_kp1_t1 import canonical_answer
from training.paper2_phase3_kp1r_t1_teacher import (
    KP1R_BOOTSTRAP_DRAWS,
    KP1R_BOOTSTRAP_SEED,
    KP1R_PERMUTATION_SEED,
    KP1R_PRIMARY_BATTERIES,
    answer_token_ids,
    battery_frequency_predictions,
    knowledge_margin_rows,
    probe_token_predictions,
    summarize_margin,
    target_entropy_audit,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")


def permutation_read(
    *,
    probe_predictions: Sequence[int],
    frequency_predictions: Sequence[int],
    targets: Sequence[int],
    batteries: Sequence[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    observed = sum(
        knowledge_margin_rows(probe_predictions, frequency_predictions, targets)
    ) / len(targets)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battery in enumerate(batteries):
        grouped[str(battery)].append(index)
    generator = random.Random(int(seed))
    null: list[float] = []
    for _ in range(int(draws)):
        permuted = list(int(value) for value in targets)
        for indexes in grouped.values():
            values = [permuted[index] for index in indexes]
            generator.shuffle(values)
            for index, value in zip(indexes, values):
                permuted[index] = value
        null.append(
            sum(
                knowledge_margin_rows(
                    probe_predictions, frequency_predictions, permuted
                )
            )
            / len(permuted)
        )
    p_value = (1 + sum(value >= observed for value in null)) / (len(null) + 1)
    return {
        "kind": "fixed-prediction_within-battery_eval-label_permutation",
        "draws": int(draws),
        "seed": int(seed),
        "observed_pooled_margin": float(observed),
        "null_mean_margin": float(sum(null) / len(null)),
        "one_sided_p_value": float(p_value),
    }


def surface_read(
    *,
    name: str,
    features: torch.Tensor,
    target_ids: Sequence[int],
    batteries: Sequence[str],
    splits: Sequence[str],
    embedding: torch.Tensor,
    ridge: float,
    permutation_draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = [index for index, value in enumerate(splits) if value == "probe_train"]
    evaluate = [index for index, value in enumerate(splits) if value == "probe_eval"]
    train_features = features[torch.tensor(train)]
    eval_features = features[torch.tensor(evaluate)]
    train_targets = [int(target_ids[index]) for index in train]
    eval_targets = [int(target_ids[index]) for index in evaluate]
    train_batteries = [str(batteries[index]) for index in train]
    eval_batteries = [str(batteries[index]) for index in evaluate]
    predictions, top10 = probe_token_predictions(
        train_features=train_features,
        train_target_ids=train_targets,
        eval_features=eval_features,
        output_embedding=embedding,
        ridge=float(ridge),
    )
    frequency = battery_frequency_predictions(
        train_targets, train_batteries, eval_batteries
    )
    margins = knowledge_margin_rows(predictions, frequency, eval_targets)
    margin_summary = summarize_margin(
        margins,
        eval_batteries,
        seed=KP1R_BOOTSTRAP_SEED,
        draws=KP1R_BOOTSTRAP_DRAWS,
    )
    rows = [
        {
            "surface": name,
            "source_index": int(source_index),
            "battery": battery,
            "target_id": int(target_id),
            "probe_prediction_id": int(prediction),
            "frequency_prediction_id": int(control),
            "probe_correct": bool(prediction == target_id),
            "frequency_correct": bool(control == target_id),
            "probe_top10_ids": [int(value) for value in local_top10],
        }
        for source_index, battery, target_id, prediction, control, local_top10 in zip(
            evaluate,
            eval_batteries,
            eval_targets,
            predictions,
            frequency,
            top10,
        )
    ]
    summary = {
        "surface": name,
        "train_rows": len(train),
        "eval_rows": len(evaluate),
        "probe_top1_accuracy": sum(row["probe_correct"] for row in rows) / len(rows),
        "frequency_top1_accuracy": sum(row["frequency_correct"] for row in rows) / len(rows),
        "knowledge_presence_margin": margin_summary,
        "label_permutation_control": permutation_read(
            probe_predictions=predictions,
            frequency_predictions=frequency,
            targets=eval_targets,
            batteries=eval_batteries,
            draws=int(permutation_draws),
            seed=KP1R_PERMUTATION_SEED,
        ),
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--state_cache", type=Path, required=True)
    parser.add_argument("--gap_rows", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutation_draws", type=int, default=10_000)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_score_only" or not lock.get("locked_before_scoring"):
        raise RuntimeError("KP-1R lock is not active")
    if lock["sealed_partitions"] != {
        "allowed": ["dev"],
        "confirm_scored": False,
        "eval_e_scored": False,
        "remain_sealed": True,
    }:
        raise RuntimeError("KP-1R sealed-partition contract changed")

    gap_rows = read_jsonl(args.gap_rows)
    panel = read_jsonl(args.panel)
    panel_by_id = {str(row["item_id"]): row for row in panel}
    if len(gap_rows) != 329 or set(panel_by_id) < {str(row["item_id"]) for row in gap_rows}:
        raise RuntimeError("KP-1R population differs from the banked 329-row gap")
    if any(str(panel_by_id[str(row["item_id"])]["partition"]) != "dev" for row in gap_rows):
        raise RuntimeError("KP-1R may read DEV rows only")

    model_spec = lock["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], cache_dir=args.model_cache
    )
    selected_rows = [panel_by_id[str(row["item_id"])] for row in gap_rows]
    target_sequences = [answer_token_ids(row, tokenizer) for row in selected_rows]
    target_ids = [values[0] for values in target_sequences]
    audit = target_entropy_audit(
        selected_rows, target_ids, enforce_batteries=KP1R_PRIMARY_BATTERIES
    )
    primary_indexes = [
        index
        for index, row in enumerate(gap_rows)
        if str(row["battery"]) in KP1R_PRIMARY_BATTERIES
    ]

    state_cache = torch.load(args.state_cache, map_location="cpu", weights_only=False)
    if state_cache.get("kind") != "paper2_phase3_t1_state_cache_v1":
        raise RuntimeError("KP-1R state cache kind changed")
    item_ids = [str(value) for value in state_cache["item_ids"]]
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    if len(item_index) != len(item_ids):
        raise RuntimeError("KP-1R state cache item IDs are not unique")
    gap_cache_indexes = torch.tensor([item_index[str(row["item_id"])] for row in gap_rows])
    core = state_cache["core_cells"]["p35_seed_0_ema_step_4400"][gap_cache_indexes].float()
    primary_tensor = torch.tensor(primary_indexes)
    features = {
        "cached_projected_substrate_layer_24_proxy": core[primary_tensor, -1, :],
        "p35_seed_0_loop_4_recurrent_cell_set": core[
            primary_tensor, 32:40, :
        ].reshape(len(primary_indexes), -1),
    }

    base = AutoModelForCausalLM.from_pretrained(
        model_spec["id"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    embedding = base.get_output_embeddings().weight.detach().cpu()
    del base

    primary_targets = [target_ids[index] for index in primary_indexes]
    primary_batteries = [str(gap_rows[index]["battery"]) for index in primary_indexes]
    primary_splits = [str(gap_rows[index]["probe_split"]) for index in primary_indexes]
    reads = {}
    row_receipts: list[dict[str, Any]] = []
    for name, values in features.items():
        read, rows = surface_read(
            name=name,
            features=values,
            target_ids=primary_targets,
            batteries=primary_batteries,
            splits=primary_splits,
            embedding=embedding,
            ridge=float(args.ridge),
            permutation_draws=int(args.permutation_draws),
        )
        reads[name] = read
        for row, source_index in zip(rows, [
            primary_indexes[index]
            for index, split in enumerate(primary_splits)
            if split == "probe_eval"
        ]):
            row["item_id"] = str(gap_rows[source_index]["item_id"])
            row["answer"] = canonical_answer(selected_rows[source_index])
        row_receipts.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.output_dir / "kp1r_cached_row_predictions.jsonl"
    write_jsonl(row_path, row_receipts)
    summary = {
        "kind": "paper2_phase3_kp1r_cached_summary_v1",
        "status": "complete_cpu_score_only",
        "authority": lock["authority"],
        "population_rows": len(gap_rows),
        "primary_rows_excluding_mbpp": len(primary_indexes),
        "target_entropy_audit": audit,
        "surfaces": reads,
        "row_predictions": {
            "path": str(row_path),
            "rows": len(row_receipts),
            "sha256": sha256_file(row_path),
        },
        "scope": {
            "cached_projected_layer24_is_proxy_not_raw_substrate": True,
            "strong_rung_still_required": True,
            "mbpp_primary": False,
        },
        "assertions": {
            "dev_only": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
