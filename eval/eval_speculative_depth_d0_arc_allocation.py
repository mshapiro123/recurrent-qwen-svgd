"""Descriptive ARC-Easy/Challenge loop-allocation probe for Paper Two D0."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter
from eval.eval_mcq import MCQExample, format_prompt
from eval.eval_speculative_depth_d0 import first_stop, spearman
from eval.prepare_arc_mcq import row_to_mcq
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_spec import deterministic_argmax_fp32


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def prepared_examples(config: str, limit: int) -> list[MCQExample]:
    dataset = load_dataset("allenai/ai2_arc", config, split="validation")
    examples: list[MCQExample] = []
    for index in range(min(limit, len(dataset))):
        row = row_to_mcq(dict(dataset[index]), index=index, seed=0, shuffle_choices=False)
        examples.append(
            MCQExample(
                id=f"{config}:{row['id']}",
                question=str(row["question"]),
                choices=[(str(label), str(text)) for label, text in row["choices"].items()],
                answer=str(row["answer"]),
            )
        )
    return examples


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--limit_per_benchmark", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("D0 ARC allocation checkpoint SHA-256 mismatch")
    tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    control_ids = [int(value) for value in resize.control_token_ids[:2]]
    if control_ids != sorted(control_ids):
        raise AssertionError("D0 ARC control IDs must be in ascending token-id order")
    rows: list[dict[str, Any]] = []
    control_tie_cells = control_argmax_cells = 0
    for difficulty, config in enumerate(("ARC-Easy", "ARC-Challenge")):
        examples = prepared_examples(config, args.limit_per_benchmark)
        for index, example in enumerate(examples):
            prompt = format_prompt(example, "with_options")
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(args.device)
            output = wrapper(
                **encoded,
                labels=None,
                max_loops=4,
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
            if output.loop_logits is None:
                raise RuntimeError("D0 ARC allocation requires loop logits")
            controls, tied = deterministic_argmax_fp32(
                output.loop_logits[0, 0, :, :, control_ids], dim=-1
            )
            controls = controls.transpose(0, 1)
            control_tie_cells += int(tied.sum().item())
            control_argmax_cells += int(tied.numel())
            selected, exhausted = first_stop(controls, 4)
            values = selected.float().cpu().tolist()
            rows.append(
                {
                    "id": example.id,
                    "benchmark": config,
                    "difficulty_code": difficulty,
                    "answer_position_loops": values[-1],
                    "context_mean_loops": statistics.fmean(values[:-1]) if len(values) > 1 else values[-1],
                    "mean_loops": statistics.fmean(values),
                    "answer_minus_context_loops": values[-1] - statistics.fmean(values[:-1]) if len(values) > 1 else 0.0,
                    "answer_position_exhausted": bool(exhausted[-1].item()),
                }
            )
            if (index + 1) % 32 == 0:
                print(f"d0_arc_allocation {config} rows={index + 1}/{len(examples)}", flush=True)
    difficulty = [float(row["difficulty_code"]) for row in rows]
    answer_loops = [float(row["answer_position_loops"]) for row in rows]
    summary = {
        "kind": "paper2_d0_arc_allocation_probe",
        "status": "descriptive_complete",
        "checkpoint_sha256": args.checkpoint_sha256,
        "rows": len(rows),
        "by_benchmark": {
            config: {
                "rows": sum(row["benchmark"] == config for row in rows),
                "mean_answer_position_loops": statistics.fmean(
                    row["answer_position_loops"] for row in rows if row["benchmark"] == config
                ),
                "mean_context_loops": statistics.fmean(
                    row["context_mean_loops"] for row in rows if row["benchmark"] == config
                ),
                "mean_answer_minus_context_loops": statistics.fmean(
                    row["answer_minus_context_loops"] for row in rows if row["benchmark"] == config
                ),
            }
            for config in ("ARC-Easy", "ARC-Challenge")
        },
        "spearman_answer_depth_with_challenge_indicator": spearman(answer_loops, difficulty),
        "tie_diagnostics": {
            "policy": "fp32 logits; exact ties choose the lowest token id",
            "control_tie_cells": control_tie_cells,
            "control_argmax_cells": control_argmax_cells,
            "tie_rate": control_tie_cells / control_argmax_cells if control_argmax_cells else None,
        },
        "scope": "descriptive allocation only; no question-answering capability claim",
        "records": rows,
    }
    write_json(args.output_summary, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
