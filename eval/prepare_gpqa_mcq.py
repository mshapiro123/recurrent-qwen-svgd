"""Prepare a local GPQA-style MCQ JSONL for eval_mcq.py.

The generated JSONL contains question text and therefore should remain local.
Do not commit it. Benchmark result rows from eval_mcq.py intentionally omit
question and choice text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset


LABELS = ("A", "B", "C", "D", "E", "F")
QUESTION_KEYS = ("Question", "question", "prompt")
CORRECT_KEYS = ("Correct Answer", "Correct answer", "correct_answer", "answer")
ID_KEYS = ("Record ID", "record_id", "id", "ID")
INCORRECT_KEYS = (
    ("Incorrect Answer 1", "Incorrect answer 1", "incorrect_answer_1"),
    ("Incorrect Answer 2", "Incorrect answer 2", "incorrect_answer_2"),
    ("Incorrect Answer 3", "Incorrect answer 3", "incorrect_answer_3"),
)


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    raise KeyError(f"None of these keys were present with a value: {keys}")


def stable_rng(seed: int, row_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def row_to_mcq(row: dict[str, Any], index: int, seed: int, shuffle_choices: bool) -> dict[str, Any]:
    question = str(first_present(row, QUESTION_KEYS))
    correct = str(first_present(row, CORRECT_KEYS))
    row_id = str(next((row[key] for key in ID_KEYS if key in row and row[key] not in (None, "")), index))
    distractors = [str(first_present(row, keys)) for keys in INCORRECT_KEYS]
    options = [(True, correct)] + [(False, item) for item in distractors]
    if shuffle_choices:
        stable_rng(seed, row_id).shuffle(options)
    choices = {LABELS[i]: text for i, (_, text) in enumerate(options)}
    answer_index = next(i for i, (is_correct, _) in enumerate(options) if is_correct)
    return {
        "id": row_id,
        "question": question,
        "choices": choices,
        "answer": LABELS[answer_index],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", default="Idavidrein/gpqa")
    parser.add_argument("--config", default="gpqa_diamond")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_shuffle_choices", action="store_true")
    parser.add_argument("--print_configs", action="store_true")
    args = parser.parse_args()

    if args.print_configs:
        print("configs=", get_dataset_config_names(args.dataset_id))

    dataset = load_dataset(args.dataset_id, args.config, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            prepared = row_to_mcq(
                dict(row),
                index=idx,
                seed=args.seed,
                shuffle_choices=not args.no_shuffle_choices,
            )
            handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")

    print(f"dataset_id={args.dataset_id}")
    print(f"config={args.config}")
    print(f"split={args.split}")
    print(f"rows={len(dataset)}")
    print(f"output_jsonl={output_path}")
    print("raw_questions_are_local_do_not_commit=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
