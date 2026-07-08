"""Synthetic iterated-function task for recurrent-depth mechanism tests.

The task presents a shuffled table for a finite function ``f: S -> S`` and asks
for ``f^d(x)``.  Instances are generated with a distinct orbit prefix of length
``d + 1`` so the requested depth is real: the target cannot be reached earlier by
falling into a short cycle.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETTER_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NAME_SYMBOLS = (
    "Ben",
    "Sam",
    "Tom",
    "Max",
    "Ada",
    "Eve",
    "Kai",
    "Lee",
    "Ana",
    "Joe",
    "Amy",
    "Dan",
    "Ida",
    "Gus",
    "Mia",
    "Ray",
    "Sue",
    "Ted",
    "Una",
    "Val",
)


@dataclass(frozen=True)
class SyntheticDepthConfig:
    n_symbols: int = 16
    max_depth: int = 8
    rows_per_depth: int = 64
    seed: int = 0
    num_choices: int = 4
    max_target_loops: int = 8
    value_prefix: str = ""


@dataclass(frozen=True)
class SyntheticDepthInstance:
    instance_id: str
    split: str
    n_symbols: int
    depth: int
    start: int
    target: int
    mapping: dict[int, int]
    table_order: list[int]
    choices: list[int]
    answer_index: int
    orbit: list[int]


def symbol(value: int, *, prefix: str = "") -> str:
    if prefix == "letter:":
        idx = int(value)
        if idx < 0 or idx >= len(LETTER_SYMBOLS):
            raise ValueError(f"letter: value_prefix supports values 0-{len(LETTER_SYMBOLS) - 1}; got {value}")
        return LETTER_SYMBOLS[idx]
    if prefix == "name:":
        idx = int(value)
        if idx < 0 or idx >= len(NAME_SYMBOLS):
            raise ValueError(f"name: value_prefix supports values 0-{len(NAME_SYMBOLS) - 1}; got {value}")
        return NAME_SYMBOLS[idx]
    return f"{prefix}{value}" if prefix else str(value)


def apply_mapping(mapping: dict[int, int], start: int, depth: int) -> int:
    current = int(start)
    for _ in range(int(depth)):
        current = int(mapping[current])
    return current


def _seed_for(seed: int, split: str, depth: int, row_index: int) -> int:
    split_offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
    return int(seed) + split_offsets.get(split, 3_000_000) + int(depth) * 10_000 + int(row_index)


def build_instance(
    *,
    instance_id: str,
    n_symbols: int,
    depth: int,
    seed: int,
    split: str = "test",
    num_choices: int = 4,
) -> SyntheticDepthInstance:
    if n_symbols < 3:
        raise ValueError("n_symbols must be at least 3")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth >= n_symbols:
        raise ValueError("depth must be < n_symbols to guarantee a distinct d+1 orbit prefix")
    if not 2 <= num_choices <= min(len(LABELS), n_symbols):
        raise ValueError("num_choices must be between 2 and min(6, n_symbols)")

    rng = random.Random(seed)
    values = list(range(n_symbols))
    orbit = rng.sample(values, depth + 1)
    mapping: dict[int, int] = {}
    for left, right in zip(orbit, orbit[1:]):
        mapping[left] = right
    for value in values:
        if value not in mapping:
            mapping[value] = rng.choice(values)

    table_order = values[:]
    rng.shuffle(table_order)
    target = orbit[-1]
    distractors = [value for value in values if value != target]
    choices = rng.sample(distractors, num_choices - 1) + [target]
    rng.shuffle(choices)
    answer_index = choices.index(target)

    computed = apply_mapping(mapping, orbit[0], depth)
    if computed != target:
        raise AssertionError("Synthetic-depth construction failed to preserve target")

    return SyntheticDepthInstance(
        instance_id=instance_id,
        split=split,
        n_symbols=n_symbols,
        depth=depth,
        start=orbit[0],
        target=target,
        mapping=mapping,
        table_order=table_order,
        choices=choices,
        answer_index=answer_index,
        orbit=orbit,
    )


def build_permutation_instance(
    *,
    instance_id: str,
    n_symbols: int,
    depth: int,
    seed: int,
    split: str = "test",
    num_choices: int = 4,
) -> SyntheticDepthInstance:
    """Build an iterated-function row whose mapping is bijective.

    This is the permutation zero-shot control for the synthetic-depth line.  It
    keeps the same distinct orbit-prefix condition as ``build_instance`` while
    removing the many-to-one arbitrary-function statistics from the table.
    """

    if n_symbols < 3:
        raise ValueError("n_symbols must be at least 3")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth >= n_symbols:
        raise ValueError("depth must be < n_symbols to guarantee a distinct d+1 orbit prefix")
    if not 2 <= num_choices <= min(len(LABELS), n_symbols):
        raise ValueError("num_choices must be between 2 and min(6, n_symbols)")

    rng = random.Random(seed)
    values = list(range(n_symbols))
    orbit = rng.sample(values, depth + 1)
    mapping: dict[int, int] = {}
    for left, right in zip(orbit, orbit[1:]):
        mapping[left] = right

    used_images = set(orbit[1:])
    remaining_domains = [value for value in values if value not in mapping]
    remaining_images = [value for value in values if value not in used_images]
    rng.shuffle(remaining_images)
    for left, right in zip(remaining_domains, remaining_images):
        mapping[left] = right

    if sorted(mapping) != values or sorted(mapping.values()) != values:
        raise AssertionError("Permutation construction failed to produce a bijection")

    table_order = values[:]
    rng.shuffle(table_order)
    target = orbit[-1]
    distractors = [value for value in values if value != target]
    choices = rng.sample(distractors, num_choices - 1) + [target]
    rng.shuffle(choices)
    answer_index = choices.index(target)

    computed = apply_mapping(mapping, orbit[0], depth)
    if computed != target:
        raise AssertionError("Synthetic-depth permutation construction failed to preserve target")

    return SyntheticDepthInstance(
        instance_id=instance_id,
        split=split,
        n_symbols=n_symbols,
        depth=depth,
        start=orbit[0],
        target=target,
        mapping=mapping,
        table_order=table_order,
        choices=choices,
        answer_index=answer_index,
        orbit=orbit,
    )


def render_table(instance: SyntheticDepthInstance, *, value_prefix: str = "") -> str:
    return "\n".join(
        f"{symbol(left, prefix=value_prefix)} -> {symbol(instance.mapping[left], prefix=value_prefix)}"
        for left in instance.table_order
    )


def render_question(instance: SyntheticDepthInstance, *, value_prefix: str = "") -> str:
    return (
        "You are given a finite function f as a shuffled lookup table.\n"
        f"Function table:\n{render_table(instance, value_prefix=value_prefix)}\n\n"
        f"Start value: {symbol(instance.start, prefix=value_prefix)}\n"
        f"Apply f exactly {instance.depth} times.\n"
        "What is the final value?"
    )


def build_mcq_row(instance: SyntheticDepthInstance, *, value_prefix: str = "") -> dict[str, Any]:
    labels = LABELS[: len(instance.choices)]
    choices = {
        label: symbol(value, prefix=value_prefix)
        for label, value in zip(labels, instance.choices)
    }
    answer = labels[instance.answer_index]
    return {
        "id": instance.instance_id,
        "question": render_question(instance, value_prefix=value_prefix),
        "choices": choices,
        "answer": answer,
        "target": symbol(instance.target, prefix=value_prefix),
        "depth": instance.depth,
        "start": symbol(instance.start, prefix=value_prefix),
        "orbit": [symbol(value, prefix=value_prefix) for value in instance.orbit],
        "n_symbols": instance.n_symbols,
        "synthetic_task": "iterated_function",
    }


def render_mcq_prompt(instance: SyntheticDepthInstance, *, value_prefix: str = "") -> str:
    row = build_mcq_row(instance, value_prefix=value_prefix)
    rendered_choices = "\n".join(f"{label}. {text}" for label, text in row["choices"].items())
    return f"{row['question'].rstrip()}\n{rendered_choices}\nAnswer:"


def render_mcq_completion(
    instance: SyntheticDepthInstance,
    *,
    score_target: str,
    value_prefix: str = "",
) -> str:
    labels = LABELS[: len(instance.choices)]
    answer_label = labels[instance.answer_index]
    answer_text = symbol(instance.target, prefix=value_prefix)
    if score_target == "label":
        return f" {answer_label}"
    if score_target == "option_text":
        return f" {answer_text}"
    if score_target == "label_and_text":
        return f" {answer_label}. {answer_text}"
    raise ValueError(f"Unknown score_target={score_target!r}")


def serialized_mapping(instance: SyntheticDepthInstance, *, value_prefix: str = "") -> dict[str, str]:
    return {
        symbol(left, prefix=value_prefix): symbol(right, prefix=value_prefix)
        for left, right in instance.mapping.items()
    }


def build_sft_row(
    instance: SyntheticDepthInstance,
    *,
    max_target_loops: int,
    value_prefix: str = "",
) -> dict[str, Any]:
    prompt = (
        render_question(instance, value_prefix=value_prefix).rstrip()
        + "\nAnswer with only the final value.\nAnswer: "
    )
    target_loop_count = max(1, min(int(max_target_loops), int(instance.depth)))
    return {
        "prompt": prompt,
        "completion": f"{symbol(instance.target, prefix=value_prefix)}",
        "target_loop_count": target_loop_count,
        "synthetic_task": "iterated_function",
        "synthetic_depth": instance.depth,
        "depth": instance.depth,
        "n_symbols": instance.n_symbols,
        "start": symbol(instance.start, prefix=value_prefix),
        "target": symbol(instance.target, prefix=value_prefix),
        "mapping": serialized_mapping(instance, value_prefix=value_prefix),
        "orbit": [symbol(value, prefix=value_prefix) for value in instance.orbit],
        "instance_id": instance.instance_id,
    }


def build_mcq_sft_row(
    instance: SyntheticDepthInstance,
    *,
    max_target_loops: int,
    score_target: str = "option_text",
    value_prefix: str = "",
) -> dict[str, Any]:
    target_loop_count = max(1, min(int(max_target_loops), int(instance.depth)))
    mcq = build_mcq_row(instance, value_prefix=value_prefix)
    return {
        "prompt": render_mcq_prompt(instance, value_prefix=value_prefix),
        "completion": render_mcq_completion(
            instance,
            score_target=score_target,
            value_prefix=value_prefix,
        ),
        "target_loop_count": target_loop_count,
        "synthetic_task": "iterated_function",
        "synthetic_depth": instance.depth,
        "depth": instance.depth,
        "n_symbols": instance.n_symbols,
        "start": symbol(instance.start, prefix=value_prefix),
        "target": symbol(instance.target, prefix=value_prefix),
        "mapping": serialized_mapping(instance, value_prefix=value_prefix),
        "orbit": [symbol(value, prefix=value_prefix) for value in instance.orbit],
        "instance_id": instance.instance_id,
        "answer": mcq["answer"],
        "choices": mcq["choices"],
        "score_target": score_target,
        "prompt_style": "with_options",
    }


def _stable_shuffle_seed(text: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def build_chain_mcq_row(
    instance: SyntheticDepthInstance,
    *,
    max_target_loops: int,
    value_prefix: str = "",
) -> dict[str, Any]:
    max_active = min(int(max_target_loops), int(instance.depth))
    chain_targets = list(instance.orbit[1 : max_active + 1])
    choices = list(dict.fromkeys(chain_targets))
    for value in instance.table_order:
        if len(choices) >= len(instance.choices):
            break
        if value not in choices:
            choices.append(value)
    rng = random.Random(_stable_shuffle_seed(instance.instance_id + ":chain"))
    rng.shuffle(choices)
    labels = LABELS[: len(choices)]
    label_by_value = {value: label for label, value in zip(labels, choices)}
    choices_text = {
        label: symbol(value, prefix=value_prefix)
        for label, value in zip(labels, choices)
    }
    final_label = label_by_value[instance.orbit[max_active]]
    return {
        "id": instance.instance_id,
        "question": render_question(instance, value_prefix=value_prefix),
        "choices": choices_text,
        "answer": final_label,
        "target": symbol(instance.target, prefix=value_prefix),
        "depth": instance.depth,
        "start": symbol(instance.start, prefix=value_prefix),
        "mapping": serialized_mapping(instance, value_prefix=value_prefix),
        "orbit": [symbol(value, prefix=value_prefix) for value in instance.orbit],
        "n_symbols": instance.n_symbols,
        "score_target": "label",
        "prompt_style": "with_options",
        "intermediate_chain_supervision": True,
        "chain_answer_by_loop": {
            str(loop_idx): label_by_value[instance.orbit[loop_idx]]
            for loop_idx in range(1, max_active + 1)
        },
    }


def build_chain_label_sft_row(
    instance: SyntheticDepthInstance,
    *,
    max_target_loops: int,
    value_prefix: str = "",
) -> dict[str, Any]:
    mcq = build_chain_mcq_row(
        instance,
        max_target_loops=max_target_loops,
        value_prefix=value_prefix,
    )
    prompt = (
        mcq["question"].rstrip()
        + "\n"
        + "\n".join(f"{label}. {text}" for label, text in mcq["choices"].items())
        + "\nAnswer:"
    )
    loop_completions = [
        f" {mcq['chain_answer_by_loop'][str(loop_idx)]}"
        for loop_idx in range(1, min(int(max_target_loops), int(instance.depth)) + 1)
    ]
    return {
        "prompt": prompt,
        "completion": f" {mcq['answer']}",
        "loop_completions": loop_completions,
        "target_loop_count": len(loop_completions),
        "synthetic_task": "iterated_function",
        "synthetic_depth": instance.depth,
        "depth": instance.depth,
        "n_symbols": instance.n_symbols,
        "start": mcq["start"],
        "target": mcq["target"],
        "mapping": mcq["mapping"],
        "orbit": mcq["orbit"],
        "instance_id": instance.instance_id,
        "answer": mcq["answer"],
        "choices": mcq["choices"],
        "score_target": "label",
        "prompt_style": "with_options",
        "intermediate_chain_supervision": True,
        "chain_answer_by_loop": mcq["chain_answer_by_loop"],
    }


def build_chain_symbol_sft_row(
    instance: SyntheticDepthInstance,
    *,
    max_target_loops: int,
    value_prefix: str = "",
) -> dict[str, Any]:
    prompt = render_question(instance, value_prefix=value_prefix).rstrip() + "\nAnswer:"
    max_active = min(int(max_target_loops), int(instance.depth))
    loop_completions = [
        f" {symbol(instance.orbit[loop_idx], prefix=value_prefix)}"
        for loop_idx in range(1, max_active + 1)
    ]
    return {
        "prompt": prompt,
        "completion": loop_completions[-1],
        "loop_completions": loop_completions,
        "target_loop_count": len(loop_completions),
        "synthetic_task": "iterated_function",
        "synthetic_depth": instance.depth,
        "depth": instance.depth,
        "n_symbols": instance.n_symbols,
        "start": symbol(instance.start, prefix=value_prefix),
        "target": symbol(instance.target, prefix=value_prefix),
        "mapping": serialized_mapping(instance, value_prefix=value_prefix),
        "orbit": [symbol(value, prefix=value_prefix) for value in instance.orbit],
        "instance_id": instance.instance_id,
        "score_target": "full_symbols",
        "prompt_style": "question_only",
        "intermediate_chain_supervision": True,
        "chain_symbol_by_loop": {
            str(loop_idx): symbol(instance.orbit[loop_idx], prefix=value_prefix)
            for loop_idx in range(1, max_active + 1)
        },
    }


def build_dataset(
    config: SyntheticDepthConfig,
    *,
    split: str,
    permutation: bool = False,
) -> list[SyntheticDepthInstance]:
    rows: list[SyntheticDepthInstance] = []
    builder = build_permutation_instance if permutation else build_instance
    for depth in range(1, config.max_depth + 1):
        for idx in range(config.rows_per_depth):
            instance_id = f"{split}_d{depth:02d}_{idx:05d}"
            rows.append(
                builder(
                    instance_id=instance_id,
                    split=split,
                    n_symbols=config.n_symbols,
                    depth=depth,
                    seed=_seed_for(config.seed, split, depth, idx),
                    num_choices=config.num_choices,
                )
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_synthetic_depth_dataset(
    *,
    output_dir: str | Path,
    config: SyntheticDepthConfig,
    permutation: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    instances = {
        split: build_dataset(config, split=split, permutation=permutation)
        for split in ("train", "val", "test")
    }
    for split, split_instances in instances.items():
        _write_jsonl(
            out / f"{split}_sft.jsonl",
            [
                build_sft_row(
                    instance,
                    max_target_loops=config.max_target_loops,
                    value_prefix=config.value_prefix,
                )
                for instance in split_instances
            ],
        )
        _write_jsonl(
            out / f"{split}_mcq.jsonl",
            [build_mcq_row(instance, value_prefix=config.value_prefix) for instance in split_instances],
        )
        for score_target in ("option_text", "label", "label_and_text"):
            _write_jsonl(
                out / f"{split}_mcq_{score_target}_sft.jsonl",
                [
                    build_mcq_sft_row(
                        instance,
                        max_target_loops=config.max_target_loops,
                        score_target=score_target,
                        value_prefix=config.value_prefix,
                    )
                    for instance in split_instances
                ],
            )
        _write_jsonl(
            out / f"{split}_chain_label_sft.jsonl",
            [
                build_chain_label_sft_row(
                    instance,
                    max_target_loops=config.max_target_loops,
                    value_prefix=config.value_prefix,
                )
                for instance in split_instances
            ],
        )
        _write_jsonl(
            out / f"{split}_chain_symbol_sft.jsonl",
            [
                build_chain_symbol_sft_row(
                    instance,
                    max_target_loops=config.max_target_loops,
                    value_prefix=config.value_prefix,
                )
                for instance in split_instances
            ],
        )
        _write_jsonl(
            out / f"{split}_chain_mcq.jsonl",
            [
                build_chain_mcq_row(
                    instance,
                    max_target_loops=config.max_target_loops,
                    value_prefix=config.value_prefix,
                )
                for instance in split_instances
            ],
        )

    summary = {
        "kind": "synthetic_depth_dataset",
        "config": asdict(config),
        "mapping_family": "permutation" if permutation else "arbitrary_function",
        "rows": {split: len(split_instances) for split, split_instances in instances.items()},
        "depth_counts": {
            split: {
                str(depth): sum(1 for instance in split_instances if instance.depth == depth)
                for depth in range(1, config.max_depth + 1)
            }
            for split, split_instances in instances.items()
        },
        "orbit_guarantee": "distinct_prefix_length_depth_plus_one",
        "files": {
            split: {
                "sft": f"{split}_sft.jsonl",
                "mcq": f"{split}_mcq.jsonl",
                "mcq_option_text_sft": f"{split}_mcq_option_text_sft.jsonl",
                "mcq_label_sft": f"{split}_mcq_label_sft.jsonl",
                "mcq_label_and_text_sft": f"{split}_mcq_label_and_text_sft.jsonl",
                "chain_label_sft": f"{split}_chain_label_sft.jsonl",
                "chain_symbol_sft": f"{split}_chain_symbol_sft.jsonl",
                "chain_mcq": f"{split}_chain_mcq.jsonl",
            }
            for split in instances
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_symbols", type=int, default=16)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--rows_per_depth", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_choices", type=int, default=4)
    parser.add_argument("--max_target_loops", type=int, default=8)
    parser.add_argument("--value_prefix", default="")
    parser.add_argument("--permutation", action="store_true")
    args = parser.parse_args()

    summary = write_synthetic_depth_dataset(
        output_dir=args.output_dir,
        config=SyntheticDepthConfig(
            n_symbols=args.n_symbols,
            max_depth=args.max_depth,
            rows_per_depth=args.rows_per_depth,
            seed=args.seed,
            num_choices=args.num_choices,
            max_target_loops=args.max_target_loops,
            value_prefix=args.value_prefix,
        ),
        permutation=args.permutation,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
