"""Prepare ARC-AGI-format tasks as supervised JSONL for recurrent fine-tuning.

Rows are written in this project's causal JSONL format:

    {"prompt": chat_template(user_prompt), "completion": output_grid_json, ...}

For each task we can use:

- original test pairs with public outputs, when available;
- leave-one-out training pairs, where one train example becomes the query and
  the remaining train examples become demonstrations;
- safe color-permutation augmentation, applied consistently to every grid in
  a task instance.
- safe dihedral geometry augmentation, applied consistently to every grid in
  a task instance.

This script should be used on public ARC-AGI training tasks first. Do not train
on evaluation tasks you intend to report as held-out benchmark results.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc_agi_utils import (  # noqa: E402
    ArcAgiExample,
    ArcPair,
    Grid,
    format_grid_completion,
    load_arc_agi_examples,
    render_arc_prompt,
)


def apply_color_permutation(grid: Grid, permutation: list[int]) -> Grid:
    return [[permutation[cell] for cell in row] for row in grid]


GEOMETRY_TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)


def apply_geometry_transform(grid: Grid, transform: str) -> Grid:
    if transform == "identity":
        return [row[:] for row in grid]
    if transform == "rot90":
        return [[grid[row][col] for row in range(len(grid) - 1, -1, -1)] for col in range(len(grid[0]))]
    if transform == "rot180":
        return [list(reversed(row)) for row in reversed(grid)]
    if transform == "rot270":
        return [[grid[row][col] for row in range(len(grid))] for col in range(len(grid[0]) - 1, -1, -1)]
    if transform == "flip_h":
        return [list(reversed(row)) for row in grid]
    if transform == "flip_v":
        return [row[:] for row in reversed(grid)]
    if transform == "transpose":
        return [[grid[row][col] for row in range(len(grid))] for col in range(len(grid[0]))]
    if transform == "anti_transpose":
        return [
            [grid[row][col] for row in range(len(grid) - 1, -1, -1)]
            for col in range(len(grid[0]) - 1, -1, -1)
        ]
    raise ValueError(f"Unknown geometry transform: {transform}")


def permute_pair(pair: ArcPair, permutation: list[int]) -> ArcPair:
    return ArcPair(
        input=apply_color_permutation(pair.input, permutation),
        output=apply_color_permutation(pair.output, permutation) if pair.output is not None else None,
    )


def transform_pair(pair: ArcPair, transform: str) -> ArcPair:
    return ArcPair(
        input=apply_geometry_transform(pair.input, transform),
        output=apply_geometry_transform(pair.output, transform) if pair.output is not None else None,
    )


def permute_example(example: ArcAgiExample, permutation: list[int], suffix: str) -> ArcAgiExample:
    return ArcAgiExample(
        task_id=f"{example.task_id}:{suffix}",
        test_index=example.test_index,
        train=tuple(permute_pair(pair, permutation) for pair in example.train),
        test_input=apply_color_permutation(example.test_input, permutation),
        test_output=apply_color_permutation(example.test_output, permutation) if example.test_output is not None else None,
    )


def transform_example(example: ArcAgiExample, transform: str, suffix: str) -> ArcAgiExample:
    return ArcAgiExample(
        task_id=f"{example.task_id}:{suffix}",
        test_index=example.test_index,
        train=tuple(transform_pair(pair, transform) for pair in example.train),
        test_input=apply_geometry_transform(example.test_input, transform),
        test_output=apply_geometry_transform(example.test_output, transform) if example.test_output is not None else None,
    )


def random_color_permutation(rng: random.Random) -> list[int]:
    values = list(range(10))
    rng.shuffle(values)
    return values


def identity_permutation() -> list[int]:
    return list(range(10))


def parse_geometry_augmentations(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "0", "false"}:
        return ["identity"]
    if normalized == "all":
        return list(GEOMETRY_TRANSFORMS)
    transforms = [item.strip() for item in normalized.split(",") if item.strip()]
    unknown = set(transforms) - set(GEOMETRY_TRANSFORMS)
    if unknown:
        raise ValueError(f"Unknown geometry transforms: {sorted(unknown)}")
    return ["identity", *[item for item in transforms if item != "identity"]]


def leave_one_out_examples(task_examples: Iterable[ArcAgiExample]) -> list[ArcAgiExample]:
    """Create held-out train-pair examples for each task.

    ``load_arc_agi_examples`` returns one example per test pair. The training
    demonstrations are repeated across those examples, so this function
    de-duplicates by task id before making leave-one-out rows.
    """

    by_task: dict[str, ArcAgiExample] = {}
    for example in task_examples:
        by_task.setdefault(example.task_id, example)

    generated: list[ArcAgiExample] = []
    for task_id, example in sorted(by_task.items()):
        train_pairs = list(example.train)
        if len(train_pairs) < 2:
            continue
        for idx, heldout in enumerate(train_pairs):
            assert heldout.output is not None
            demonstrations = tuple(pair for j, pair in enumerate(train_pairs) if j != idx)
            generated.append(
                ArcAgiExample(
                    task_id=f"{task_id}:loo{idx}",
                    test_index=0,
                    train=demonstrations,
                    test_input=heldout.input,
                    test_output=heldout.output,
                )
            )
    return generated


def render_chat_prompt(tokenizer, example: ArcAgiExample, output_format: str) -> str:
    user_prompt = render_arc_prompt(example, output_format=output_format)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def example_to_jsonl_row(
    tokenizer,
    example: ArcAgiExample,
    *,
    append_eos: bool,
    source: str,
    output_format: str,
) -> dict[str, object] | None:
    if example.test_output is None:
        return None
    completion = format_grid_completion(example.test_output, output_format=output_format)
    if append_eos and tokenizer.eos_token:
        completion += tokenizer.eos_token
    prompt = render_chat_prompt(tokenizer, example, output_format)
    cot_tokens = max(1, len(tokenizer(completion, add_special_tokens=False)["input_ids"]))
    return {
        "prompt": prompt,
        "completion": completion,
        "cot_tokens": cot_tokens,
        "source_dataset": "arc-agi",
        "category": source,
        "difficulty": None,
        "task_id": example.task_id,
        "test_index": example.test_index,
    }


def write_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks_path", required=True)
    parser.add_argument("--solutions_path")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--val_jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--augment_color_permutations", type=int, default=0)
    parser.add_argument(
        "--augment_geometries",
        default="none",
        help="Geometry transforms to apply: none, all, or comma-separated values from "
        f"{','.join(GEOMETRY_TRANSFORMS)}.",
    )
    parser.add_argument("--include_original_test_pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_leave_one_out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_train_examples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid_format", default="json", choices=("json", "compact", "tagged"))
    parser.add_argument("--max_total_tokens", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    base_examples = load_arc_agi_examples(args.tasks_path, solutions_path=args.solutions_path, limit=args.limit)
    examples: list[tuple[ArcAgiExample, str]] = []
    if args.include_original_test_pairs:
        examples.extend((example, "arc_original_test_pair") for example in base_examples if example.test_output is not None)
    if args.include_leave_one_out:
        examples.extend((example, "arc_leave_one_out") for example in leave_one_out_examples(base_examples))

    rng = random.Random(args.seed)
    augmented: list[tuple[ArcAgiExample, str]] = []
    geometry_transforms = parse_geometry_augmentations(args.augment_geometries)
    for example, source in examples:
        for transform in geometry_transforms:
            transformed = example if transform == "identity" else transform_example(example, transform, transform)
            geometry_source = source if transform == "identity" else f"{source}:geom:{transform}"
            augmented.append((transformed, geometry_source))
            for aug_idx in range(args.augment_color_permutations):
                permutation = random_color_permutation(rng)
                if permutation == identity_permutation():
                    continue
                augmented.append(
                    (
                        permute_example(transformed, permutation, f"color{aug_idx}"),
                        f"{geometry_source}:color",
                    )
                )

    rows: list[dict[str, object]] = []
    skipped = 0
    for example, source in augmented:
        working = example
        if args.shuffle_train_examples and len(working.train) > 1:
            shuffled = list(working.train)
            rng.shuffle(shuffled)
            working = replace(working, train=tuple(shuffled))
        row = example_to_jsonl_row(
            tokenizer,
            working,
            append_eos=args.append_eos,
            source=source,
            output_format=args.grid_format,
        )
        if row is None:
            skipped += 1
            continue
        token_count = len(tokenizer(str(row["prompt"]) + str(row["completion"]), add_special_tokens=False)["input_ids"])
        if token_count > args.max_total_tokens:
            skipped += 1
            continue
        rows.append(row)

    rng.shuffle(rows)
    if args.val_jsonl:
        val_count = max(1, int(len(rows) * args.val_fraction)) if rows else 0
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]
    else:
        train_rows = rows
        val_rows = []

    write_jsonl(args.output_jsonl, train_rows)
    if args.val_jsonl:
        write_jsonl(args.val_jsonl, val_rows)

    print(f"tasks_path={args.tasks_path}")
    print(f"base_examples={len(base_examples)}")
    print(f"rendered_rows={len(rows)}")
    print(f"train_rows={len(train_rows)}")
    print(f"val_rows={len(val_rows)}")
    print(f"skipped_rows={skipped}")
    print(f"grid_format={args.grid_format}")
    print(f"geometry_transforms={','.join(geometry_transforms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
