"""Prepare ARC-Challenge/Easy MCQ JSONL for eval_mcq.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


LABELS = ("A", "B", "C", "D", "E", "F")


def stable_rng(seed: int, row_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def row_to_mcq(row: dict[str, Any], index: int, seed: int, shuffle_choices: bool) -> dict[str, Any]:
    row_id = str(row.get("id") or index)
    question = str(row["question"])
    raw_choices = row["choices"]
    texts = list(raw_choices["text"])
    source_labels = [str(label) for label in raw_choices["label"]]
    answer_key = str(row["answerKey"])

    if answer_key in source_labels:
        answer_index = source_labels.index(answer_key)
    else:
        folded = answer_key.casefold()
        answer_index = next(i for i, text in enumerate(texts) if str(text).strip().casefold() == folded)

    options = [(i == answer_index, str(text)) for i, text in enumerate(texts)]
    if len(options) > len(LABELS):
        raise ValueError(f"Too many choices for {row_id}: {len(options)}")
    if shuffle_choices:
        stable_rng(seed, row_id).shuffle(options)
    choices = {LABELS[i]: text for i, (_, text) in enumerate(options)}
    new_answer_index = next(i for i, (is_correct, _) in enumerate(options) if is_correct)
    return {
        "id": row_id,
        "question": question,
        "choices": choices,
        "answer": LABELS[new_answer_index],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", default="allenai/ai2_arc")
    parser.add_argument("--config", default="ARC-Challenge")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_shuffle_choices", action="store_true")
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
