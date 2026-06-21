"""Prepare ARC multiple-choice rows as causal SFT prompt/completion JSONL.

This deliberately mirrors ``eval/eval_mcq.py`` prompt and label formatting so
short competence-recovery runs train the same option-scoring surface that the
MCQ benchmark evaluates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import MCQExample, format_completion, format_prompt  # noqa: E402
from eval.prepare_arc_mcq import row_to_mcq  # noqa: E402


def arc_row_to_sft(
    row: dict[str, Any],
    *,
    index: int,
    seed: int,
    shuffle_choices: bool,
    prompt_style: str,
    score_target: str,
) -> dict[str, Any]:
    mcq = row_to_mcq(row, index=index, seed=seed, shuffle_choices=shuffle_choices)
    choices = [(str(label), str(text)) for label, text in mcq["choices"].items()]
    answer = str(mcq["answer"])
    choice_text = dict(choices)[answer]
    example = MCQExample(
        id=str(mcq["id"]),
        question=str(mcq["question"]),
        choices=choices,
        answer=answer,
    )
    return {
        "prompt": format_prompt(example, prompt_style),
        "completion": format_completion(answer, choice_text, score_target),
        "cot_tokens": 1,
        "source_dataset": "ai2_arc",
        "category": str(row.get("config") or ""),
        "difficulty": None,
        "arc_id": str(mcq["id"]),
        "answer": answer,
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", default="allenai/ai2_arc")
    parser.add_argument("--config", default="ARC-Challenge")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="with_options")
    parser.add_argument("--score_target", choices=("label", "option_text", "label_and_text"), default="label")
    parser.add_argument("--no_shuffle_choices", action="store_true")
    parser.add_argument("--max_total_tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dataset = load_dataset(args.dataset_id, args.config, split=args.split)
    rows = list(dataset)
    random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    examples: list[dict[str, Any]] = []
    skipped = 0
    for idx, row in enumerate(rows):
        row_dict = dict(row)
        row_dict["config"] = args.config
        example = arc_row_to_sft(
            row_dict,
            index=idx,
            seed=args.seed,
            shuffle_choices=not args.no_shuffle_choices,
            prompt_style=args.prompt_style,
            score_target=args.score_target,
        )
        total_tokens = len(
            tokenizer(
                str(example["prompt"]) + str(example["completion"]),
                add_special_tokens=False,
            )["input_ids"]
        )
        if total_tokens > args.max_total_tokens:
            skipped += 1
            continue
        examples.append(example)

    write_jsonl(args.output_jsonl, examples)
    print(f"dataset_id={args.dataset_id}")
    print(f"config={args.config}")
    print(f"split={args.split}")
    print(f"rows={len(examples)}")
    print(f"skipped_rows={skipped}")
    print(f"output_jsonl={args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
