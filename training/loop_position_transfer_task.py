"""Mixed-direction key-passing rows for the loop-position transfer micro-test."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable

from training.synthetic_depth_task import NAME_SYMBOLS


@dataclass(frozen=True)
class LoopPositionConfig:
    n_symbols: int = 20
    train_rows: int = 640
    rows_per_position: int = 128
    seed: int = 8_120_331
    forward_rehearsal_fraction: float = 0.30


def _seed(seed: int, split: str, role: str, position: int, index: int) -> int:
    material = f"{seed}|{split}|{role}|{position}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _permutation(n_symbols: int, rng: random.Random) -> dict[int, int]:
    images = list(range(n_symbols))
    rng.shuffle(images)
    return {index: images[index] for index in range(n_symbols)}


def _render_table(mapping: dict[int, int], order: list[int], names: tuple[str, ...]) -> str:
    return "\n".join(f"{names[left]} passes the key to {names[mapping[left]]}." for left in order)


def _build_row(
    config: LoopPositionConfig,
    *,
    split: str,
    role: str,
    prefix_length: int,
    index: int,
) -> dict[str, Any]:
    if config.n_symbols > len(NAME_SYMBOLS):
        raise ValueError("Loop-position name rendering supports at most 20 symbols")
    rng = random.Random(_seed(config.seed, split, role, prefix_length, index))
    names = NAME_SYMBOLS[: config.n_symbols]
    mapping = _permutation(config.n_symbols, rng)
    inverse = {right: left for left, right in mapping.items()}
    order = list(range(config.n_symbols))
    rng.shuffle(order)
    start = rng.randrange(config.n_symbols)
    chain = [start]
    if role == "mixed_forward_then_inverse":
        for _ in range(prefix_length):
            chain.append(mapping[chain[-1]])
        chain.append(inverse[chain[-1]])
        operation_sequence = ["forward"] * prefix_length + ["inverse"]
        instruction = (
            f"The key starts with {names[start]}. Pass the key forward for {prefix_length} "
            "days, then once backward. Who holds it?"
        )
    elif role == "pure_forward_rehearsal":
        forward_steps = max(1, prefix_length)
        for _ in range(forward_steps):
            chain.append(mapping[chain[-1]])
        operation_sequence = ["forward"] * forward_steps
        instruction = (
            f"The key starts with {names[start]}. Pass the key forward for {forward_steps} "
            "days. Who holds it?"
        )
    else:
        raise ValueError(f"Unknown curriculum role {role!r}")
    question = f"{_render_table(mapping, order, names)}\n\n{instruction}"
    chain_names = [names[value] for value in chain]
    row_id = f"loop_position_{split}_{role}_p{prefix_length}_{index:05d}"
    return {
        "id": row_id,
        "instance_id": row_id,
        "split": split,
        "question": question,
        "prompt": f"{question}\nAnswer:",
        "completion": f" {chain_names[-1]}",
        "loop_completions": [f" {value}" for value in chain_names[1:]],
        "target_loop_count": len(operation_sequence),
        "forward_loop_count": len(operation_sequence),
        "depth": len(operation_sequence),
        "n_symbols": config.n_symbols,
        "symbol_names": list(names),
        "start": names[start],
        "start_value": start,
        "target": names[chain[-1]],
        "orbit": chain_names,
        "chain_values": chain,
        "mapping": {names[left]: names[right] for left, right in mapping.items()},
        "mapping_values": {str(left): right for left, right in mapping.items()},
        "operation_sequence": operation_sequence,
        "curriculum_role": role,
        "forward_prefix_length": prefix_length,
        "inverse_loop_position": prefix_length + 1 if role == "mixed_forward_then_inverse" else None,
        "score_target": "full_symbols",
        "prompt_style": "question_only",
    }


def build_eval_rows(
    config: LoopPositionConfig,
    *,
    prefix_lengths: Iterable[int],
    split: str,
) -> list[dict[str, Any]]:
    return [
        _build_row(
            config,
            split=split,
            role="mixed_forward_then_inverse",
            prefix_length=int(prefix),
            index=index,
        )
        for prefix in prefix_lengths
        for index in range(config.rows_per_position)
    ]


def build_training_rows(config: LoopPositionConfig) -> list[dict[str, Any]]:
    rehearsal_rows = round(config.train_rows * config.forward_rehearsal_fraction)
    mixed_rows = config.train_rows - rehearsal_rows
    rows: list[dict[str, Any]] = []
    for index in range(mixed_rows):
        rows.append(
            _build_row(
                config,
                split="train",
                role="mixed_forward_then_inverse",
                prefix_length=index % 2,
                index=index,
            )
        )
    for index in range(rehearsal_rows):
        rows.append(
            _build_row(
                config,
                split="train",
                role="pure_forward_rehearsal",
                prefix_length=1 + index % 2,
                index=index,
            )
        )
    random.Random(config.seed).shuffle(rows)
    return rows


def row_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")
    ids = "\n".join(str(row["id"]) for row in rows).encode("utf-8")
    return {
        "rows": len(rows),
        "row_id_sha256": hashlib.sha256(ids).hexdigest(),
        "row_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_loop_position_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for row in rows:
        mapping = {int(left): int(right) for left, right in row["mapping_values"].items()}
        if sorted(mapping) != list(range(int(row["n_symbols"]))) or len(set(mapping.values())) != len(mapping):
            errors.append(f"{row['id']}: mapping is not bijective")
            continue
        chain = [int(value) for value in row["chain_values"]]
        operations = list(row["operation_sequence"])
        inverse = {right: left for left, right in mapping.items()}
        current = chain[0]
        computed = [current]
        for operation in operations:
            current = mapping[current] if operation == "forward" else inverse[current]
            computed.append(current)
        if computed != chain:
            errors.append(f"{row['id']}: stored chain does not match operations")
        if len(row["loop_completions"]) != len(operations):
            errors.append(f"{row['id']}: loop completion count mismatch")
        if int(row["target_loop_count"]) != len(operations):
            errors.append(f"{row['id']}: target loop count mismatch")
    return {"status": "passed" if not errors else "failed", "errors": errors, "manifest": row_manifest(rows)}
