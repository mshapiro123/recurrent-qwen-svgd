"""Measure the locked pretraining forced-depth floor and select the D0 target branch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import (
    calibration_verdict,
    fit_depth_mapping,
    validate_cache_summary,
)
from training.speculative_depth_d0_spec import DRAFTER_CHECKPOINT_SHA256, validate_locked_d0


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def load_partition_cache(cache_summary: dict[str, Any], teacher: str, partition: str) -> dict[int, dict[str, Any]]:
    receipt = cache_summary["caches"][teacher][partition]
    rows: dict[int, dict[str, Any]] = {}
    for shard in receipt["shards"]:
        if sha256_file(shard["path"]) != shard["sha256"]:
            raise RuntimeError(f"D0 teacher-cache shard hash mismatch: {shard['path']}")
        payload = torch.load(shard["path"], map_location="cpu", weights_only=False)
        for row in payload["rows"]:
            rows[int(row["row_index"])] = row
    if len(rows) != int(receipt["rows"]):
        raise RuntimeError(f"D0 teacher cache row count mismatch: {teacher}/{partition}")
    return rows


def quartile_boundaries(values: torch.Tensor) -> list[float]:
    if values.numel() < 4:
        raise ValueError("D0 calibration needs at least four rejected positions")
    return [float(value) for value in torch.quantile(values.float(), torch.tensor([0.25, 0.5, 0.75]))]


def quartile(value: float, boundaries: list[float]) -> str:
    return f"q{1 + sum(float(value) > boundary for boundary in boundaries)}"


def batched(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.inference_mode()
def forced_predictions(
    wrapper: Any,
    rows: list[dict[str, Any]],
    *,
    original_vocab_size: int,
    device: str,
    batch_size: int,
    resume_dir: Path,
    checkpoint_sha256: str,
) -> list[list[list[int]]]:
    outputs: list[list[list[int]]] = []
    forced_depths = [1, 2, 3, 4, 5, 6]
    if batch_size != 1:
        raise ValueError("D0 six-loop full-vocabulary floor is locked to batch size 1")
    for group in batched(rows, batch_size):
        row_index = len(outputs)
        cache_path = resume_dir / f"row_{row_index:06d}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                cached.get("row_index") != row_index
                or cached.get("forced_depths") != forced_depths
                or cached.get("checkpoint_sha256") != checkpoint_sha256
            ):
                raise RuntimeError(f"D0 floor resume cache mismatch: {cache_path}")
            outputs.append(cached["predictions"])
            print(f"d0_floor_resume rows={len(outputs)}/{len(rows)}", flush=True)
            continue
        sequences = [list(row["input_ids"]) for row in group]
        maximum = max(len(values) for values in sequences)
        pad_id = 0
        input_ids = torch.full((len(group), maximum), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for index, values in enumerate(sequences):
            input_ids[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
            attention[index, : len(values)] = 1
        result = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            labels=None,
            max_loops=6,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
        )
        if result.loop_logits is None:
            raise RuntimeError("D0 floor requires per-loop logits")
        for index, values in enumerate(sequences):
            logits = result.loop_logits[index, 0, :, : len(values) - 1, :original_vocab_size]
            predictions = logits.argmax(dim=-1).transpose(0, 1).cpu().tolist()
            if any(len(row_predictions) != len(forced_depths) for row_predictions in predictions):
                raise RuntimeError("D0 floor did not execute exactly six loops")
            result = [[int(value) for value in row_predictions] for row_predictions in predictions]
            outputs.append(result)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "kind": "paper2_d0_floor_row_cache",
                    "row_index": row_index,
                    "forced_depths": forced_depths,
                    "checkpoint_sha256": checkpoint_sha256,
                    "predictions": result,
                },
                temporary,
            )
            os.replace(temporary, cache_path)
        print(f"d0_floor_progress rows={len(outputs)}/{len(rows)}", flush=True)
    return outputs


def aggregate(
    examples: list[dict[str, Any]], predictions: list[list[int]], boundaries: list[float]
) -> dict[str, Any]:
    counts: dict[tuple[str, str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    rows_out: list[dict[str, Any]] = []
    for example, predicted in zip(examples, predictions, strict=True):
        severity = quartile(float(example["kl"]), boundaries)
        matches_7b = [value == int(example["teacher_7b"]) for value in predicted]
        matches_14b = [value == int(example["teacher_14b"]) for value in predicted]
        for teacher, matches in (("teacher_7b", matches_7b), ("teacher_14b", matches_14b)):
            for depth, matched in enumerate(matches, start=1):
                for stratum in ("pooled", str(example["stratum"])):
                    for bin_name in ("all", severity):
                        cell = counts[(teacher, stratum, bin_name, depth)]
                        cell[1] += 1
                        cell[0] += int(matched)
        rows_out.append(
            {
                "row_index": int(example["row_index"]),
                "local_position": int(example["local_position"]),
                "stratum": str(example["stratum"]),
                "severity_bin": severity,
                "teacher_7b": int(example["teacher_7b"]),
                "teacher_14b": int(example["teacher_14b"]),
                "kl": float(example["kl"]),
                "rank": int(example["rank"]),
                "run_length": int(example["run_length"]),
                "teacher_entropy": float(example["teacher_entropy"]),
                "negative_drafter_logprob_under_teacher": float(
                    example["negative_drafter_logprob_under_teacher"]
                ),
                "required_depth": int(example["required_depth"]),
                "rejected_7b": bool(example.get("rejected_7b", True)),
                "rejected_14b": bool(example.get("rejected_14b", True)),
                "predictions": predicted,
                "matches_teacher_7b": matches_7b,
                "matches_teacher_14b": matches_14b,
            }
        )
    curves: dict[str, Any] = {}
    for teacher in ("teacher_7b", "teacher_14b"):
        curves[teacher] = {}
        for stratum in ("pooled", "general", "code"):
            curves[teacher][stratum] = {}
            for bin_name in ("all", "q1", "q2", "q3", "q4"):
                cells = [counts[(teacher, stratum, bin_name, depth)] for depth in range(1, 7)]
                curves[teacher][stratum][bin_name] = [
                    {"correct": correct, "total": total, "accuracy": correct / total if total else None}
                    for correct, total in cells
                ]
    return {"curves": curves, "rows": rows_out}


def split_examples_and_predictions(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Separate receipt metadata from predictions without mutating shared rows."""
    clean_examples: list[dict[str, Any]] = []
    predictions: list[list[int]] = []
    for index, source in enumerate(examples):
        if "predictions" not in source:
            raise RuntimeError(f"D0 floor example {index} is missing predictions")
        clean = dict(source)
        predicted = clean.pop("predictions")
        clean_examples.append(clean)
        predictions.append([int(value) for value in predicted])
    return clean_examples, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--private_rows_output", required=True)
    parser.add_argument("--resume_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--expected_checkpoint_sha256", default=DRAFTER_CHECKPOINT_SHA256)
    parser.add_argument("--measurement_label", default="pretraining_floor")
    args = parser.parse_args()

    prereg = read_json(args.preregistration)
    validate_locked_d0(prereg)
    cache_summary = read_json(args.teacher_cache_summary)
    validate_cache_summary(cache_summary)
    observed_checkpoint_sha256 = sha256_file(args.checkpoint)
    if observed_checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("D0 floor received a checkpoint with the wrong SHA-256")
    rows = read_jsonl(args.data_jsonl)
    cache_7b = load_partition_cache(cache_summary, "teacher_7b", "calibration")
    cache_14b = load_partition_cache(cache_summary, "teacher_14b", "calibration")
    rejected_kl = torch.cat(
        [row["teacher_to_plain_drafter_kl"][~row["accepted"]] for row in cache_7b.values()]
    )
    boundaries = quartile_boundaries(rejected_kl)
    tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    row_predictions = forced_predictions(
        wrapper,
        rows,
        original_vocab_size=resize.original_tokenizer_size,
        device=args.device,
        batch_size=args.batch_size,
        resume_dir=Path(args.resume_dir),
        checkpoint_sha256=observed_checkpoint_sha256,
    )
    examples: list[dict[str, Any]] = []
    dual_teacher_union: list[dict[str, Any]] = []
    for row_index, source in enumerate(rows):
        seven = cache_7b[row_index]
        fourteen = cache_14b[row_index]
        if seven["row_sha256"] != fourteen["row_sha256"]:
            raise RuntimeError("D0 dual-teacher calibration rows are not aligned")
        accepted_7b = seven["accepted"].tolist()
        accepted_14b = fourteen["accepted"].tolist()
        for local_position, accepted in enumerate(accepted_7b):
            if accepted and accepted_14b[local_position]:
                continue
            item = {
                "row_index": row_index,
                "local_position": local_position,
                "stratum": source["stratum"],
                "kl": float(seven["teacher_to_plain_drafter_kl"][local_position]),
                "teacher_7b": int(seven["teacher_greedy_token_id"][local_position]),
                "teacher_14b": int(fourteen["teacher_greedy_token_id"][local_position]),
                "rank": int(seven["drafter_token_rank_under_teacher"][local_position]),
                "run_length": int(seven["rejection_run_length"][local_position]),
                "teacher_entropy": float(seven["teacher_entropy"][local_position]),
                "negative_drafter_logprob_under_teacher": -float(
                    seven["drafter_token_logprob_under_teacher"][local_position]
                ),
                "rejected_7b": not bool(accepted),
                "rejected_14b": not bool(accepted_14b[local_position]),
                "predictions": row_predictions[row_index][local_position],
            }
            matches_7b = [
                token == item["teacher_7b"] for token in item["predictions"][:4]
            ]
            item["required_depth"] = next(
                (depth for depth, matched in enumerate(matches_7b, start=1) if matched),
                4,
            )
            dual_teacher_union.append(item)
            if not accepted:
                examples.append(item)
    examples, predictions = split_examples_and_predictions(examples)
    aggregated = aggregate(examples, predictions, boundaries)
    union_examples, union_predictions = split_examples_and_predictions(dual_teacher_union)
    union_aggregated = aggregate(union_examples, union_predictions, boundaries)
    bin_curves = {
        name: [
            float(cell["accuracy"])
            for cell in aggregated["curves"]["teacher_7b"]["pooled"][name]
        ]
        for name in ("q1", "q2", "q3", "q4")
    }
    verdict = calibration_verdict(bin_curves)
    mapping = fit_depth_mapping(examples)
    verdict["disagreement_to_depth_mapping"] = mapping
    private_path = Path(args.private_rows_output)
    write_json(
        private_path,
        {
            "boundaries": boundaries,
            "rows": aggregated.pop("rows"),
            "dual_teacher_union_rows": union_aggregated.pop("rows"),
        },
    )
    summary = {
        "kind": "paper2_d0_floor_calibration",
        "status": "complete",
        "measurement_label": args.measurement_label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": observed_checkpoint_sha256,
        "forced_depths": [1, 2, 3, 4, 5, 6],
        "rejected_positions": len(examples),
        "severity_quartile_boundaries": boundaries,
        "calibration": verdict,
        "curves": aggregated["curves"],
        "dual_teacher_union_positions": len(union_examples),
        "dual_teacher_union_curves": union_aggregated["curves"],
        "private_rows_sha256": sha256_file(private_path),
        "optimizer_steps": 0,
        "training_started": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
