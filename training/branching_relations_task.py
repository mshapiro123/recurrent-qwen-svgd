"""Exactly verifiable multi-valued forward-relation task for width screening."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Literal

from training.abductive_injective_task import PHASE_G_NAME_SYMBOLS


Rendering = Literal["verbal", "symbolic"]
STRATA = ("2", "3-4", "5-8", "9-16")


@dataclass(frozen=True)
class BranchingRelationsConfig:
    rows_per_depth: int = 128
    max_depth: int = 4
    seed: int = 9_401_773


def reachable_values(mapping: dict[int, tuple[int, int]], start: int, depth: int) -> list[int]:
    current = {int(start)}
    for _ in range(int(depth)):
        current = {successor for value in current for successor in mapping[int(value)]}
    return sorted(current)


def reachable_stratum(size: int) -> str:
    if size == 2:
        return "2"
    if 3 <= size <= 4:
        return "3-4"
    if 5 <= size <= 8:
        return "5-8"
    if 9 <= size <= 16:
        return "9-16"
    raise ValueError(f"Reachable set size {size} is outside the preregistered bins")


def _eligible_strata(depth: int) -> tuple[str, ...]:
    return STRATA[: int(depth)]


def _target_size(stratum: str, index: int, max_size: int) -> int:
    low, high = (2, 2) if stratum == "2" else tuple(int(value) for value in stratum.split("-"))
    high = min(high, max_size)
    return low + int(index) % (high - low + 1)


def _seed(seed: int, split: str, rendering: str, n_symbols: int, depth: int, index: int) -> int:
    material = f"{seed}|{split}|{rendering}|{n_symbols}|{depth}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _construct_mapping(
    *,
    n_symbols: int,
    depth: int,
    target_size: int,
    rng: random.Random,
) -> tuple[dict[int, tuple[int, int]], int, list[int]]:
    values = list(range(n_symbols))
    rng.shuffle(values)
    layer_sizes = [1] + [min(2**step, target_size) for step in range(1, depth + 1)]
    internal_total = sum(layer_sizes[:-1])
    if internal_total > n_symbols:
        raise ValueError("Not enough symbols for disjoint internal layers")
    internal_values = values[:internal_total]
    layers: list[list[int]] = []
    cursor = 0
    for size in layer_sizes[:-1]:
        layers.append(internal_values[cursor : cursor + size])
        cursor += size
    final_layer = values[:target_size]
    layers.append(final_layer)
    mapping: dict[int, tuple[int, int]] = {}
    for left_layer, right_layer in zip(layers, layers[1:]):
        for index, left in enumerate(left_layer):
            first = right_layer[(2 * index) % len(right_layer)]
            second = right_layer[(2 * index + 1) % len(right_layer)]
            if first == second:
                second = right_layer[(2 * index + 2) % len(right_layer)]
            mapping[left] = (first, second)
    for value in values:
        if value in mapping:
            continue
        pair = rng.sample(values, 2)
        mapping[value] = (pair[0], pair[1])
    start = layers[0][0]
    reachable = reachable_values(mapping, start, depth)
    if len(reachable) != target_size:
        raise AssertionError(f"Constructive branching row produced {len(reachable)} leaves, expected {target_size}")
    return mapping, start, reachable


def _symbols(n_symbols: int, rendering: Rendering) -> tuple[str, ...]:
    if rendering == "verbal":
        if n_symbols > len(PHASE_G_NAME_SYMBOLS):
            raise ValueError("Verbal rendering exceeds the available name set")
        return tuple(PHASE_G_NAME_SYMBOLS[:n_symbols])
    if n_symbols > 26:
        raise ValueError("Symbolic rendering supports at most 26 symbols")
    return tuple(chr(ord("A") + index) for index in range(n_symbols))


def _render_question(
    mapping: dict[int, tuple[int, int]],
    order: list[int],
    symbols: tuple[str, ...],
    start: int,
    depth: int,
    rendering: Rendering,
) -> str:
    if rendering == "verbal":
        table = "\n".join(
            f"{symbols[left]} passes the key to {symbols[mapping[left][0]]} or to {symbols[mapping[left][1]]}."
            for left in order
        )
        question = (
            f"The key starts with {symbols[start]}. Each day it is passed to one of the two listed people. "
            f"After exactly {depth} days, who could have it? Answer with one valid name."
        )
    else:
        table = "\n".join(
            f"{symbols[left]} -> {symbols[mapping[left][0]]} | {symbols[mapping[left][1]]}"
            for left in order
        )
        question = (
            f"Start at {symbols[start]}. Follow one listed branch per step for exactly {depth} steps. "
            "Which symbol could be reached? Answer with one valid symbol."
        )
    return f"{table}\n\n{question}"


def _build_row(
    config: BranchingRelationsConfig,
    *,
    split: str,
    rendering: Rendering,
    n_symbols: int,
    depth: int,
    index: int,
) -> dict[str, Any]:
    strata = _eligible_strata(depth)
    stratum = strata[index % len(strata)]
    within_stratum_index = index // len(strata)
    rng = random.Random(_seed(config.seed, split, rendering, n_symbols, depth, index))
    target_size = _target_size(stratum, within_stratum_index, min(2**depth, n_symbols))
    mapping, start, reachable = _construct_mapping(
        n_symbols=n_symbols,
        depth=depth,
        target_size=target_size,
        rng=rng,
    )
    symbols = _symbols(n_symbols, rendering)
    order = list(range(n_symbols))
    rng.shuffle(order)
    chain = [start]
    for _ in range(depth):
        chain.append(rng.choice(mapping[chain[-1]]))
    question = _render_question(mapping, order, symbols, start, depth, rendering)
    row_id = f"branching_{split}_{rendering}_n{n_symbols}_d{depth:02d}_{index:05d}"
    return {
        "id": row_id,
        "instance_id": row_id,
        "split": split,
        "question": question,
        "prompt": f"{question}\nAnswer:",
        "completion": f" {symbols[chain[-1]]}",
        "loop_completions": [f" {symbols[value]}" for value in chain[1:]],
        "target_loop_count": depth,
        "depth": depth,
        "n_symbols": n_symbols,
        "symbol_names": list(symbols),
        "rendering": rendering,
        "start": symbols[start],
        "start_value": start,
        "target": symbols[chain[-1]],
        "sampled_chain": [symbols[value] for value in chain],
        "sampled_chain_values": chain,
        "reachable_symbols": [symbols[value] for value in reachable],
        "reachable_values": reachable,
        "reachable_set_size": len(reachable),
        "reachable_set_stratum": reachable_stratum(len(reachable)),
        "coverage_denominator": len(reachable),
        "successors": {
            symbols[left]: [symbols[right] for right in mapping[left]] for left in range(n_symbols)
        },
        "successor_values": {str(left): list(mapping[left]) for left in range(n_symbols)},
        "score_target": "full_symbols",
        "prompt_style": "question_only",
        "posterior_chain_sampling": "uniform_local_branch_choice",
    }


def build_rows(
    config: BranchingRelationsConfig,
    *,
    split: str,
    rendering: Rendering,
    n_symbols: int,
) -> list[dict[str, Any]]:
    if config.rows_per_depth < 1:
        raise ValueError("rows_per_depth must be positive")
    return [
        _build_row(
            config,
            split=split,
            rendering=rendering,
            n_symbols=n_symbols,
            depth=depth,
            index=index,
        )
        for depth in range(1, config.max_depth + 1)
        for index in range(config.rows_per_depth)
    ]


def row_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")
    ids = "\n".join(str(row["id"]) for row in rows).encode("utf-8")
    return {
        "rows": len(rows),
        "row_id_sha256": hashlib.sha256(ids).hexdigest(),
        "row_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    depth_counts: dict[str, int] = {}
    strata: dict[str, dict[str, int]] = {}
    for row in rows:
        depth = int(row["depth"])
        depth_key = str(depth)
        depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
        mapping = {int(left): tuple(int(value) for value in rights) for left, rights in row["successor_values"].items()}
        if any(len(set(rights)) != 2 for rights in mapping.values()):
            errors.append(f"{row['id']}: every source must have two distinct successors")
        exact = reachable_values(mapping, int(row["start_value"]), depth)
        if exact != sorted(int(value) for value in row["reachable_values"]):
            errors.append(f"{row['id']}: stored reachable set is not exact")
        actual_stratum = reachable_stratum(len(exact))
        if actual_stratum != row["reachable_set_stratum"]:
            errors.append(f"{row['id']}: wrong reachable-set stratum")
        strata.setdefault(depth_key, {})[actual_stratum] = strata.setdefault(depth_key, {}).get(actual_stratum, 0) + 1
        chain = [int(value) for value in row["sampled_chain_values"]]
        if len(chain) != depth + 1 or any(chain[i + 1] not in mapping[chain[i]] for i in range(depth)):
            errors.append(f"{row['id']}: sampled chain is invalid")
        if chain[-1] not in exact:
            errors.append(f"{row['id']}: sampled answer is outside the reachable set")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "rows": len(rows),
        "depth_counts": depth_counts,
        "stratum_counts_by_depth": strata,
        "manifest": row_manifest(rows),
    }


def assess_validity_gate(
    scored_rows: list[dict[str, Any]],
    *,
    pooled_floor: float = 0.70,
    per_depth_floor: float = 0.55,
) -> dict[str, Any]:
    by_depth: dict[str, dict[str, Any]] = {}
    for depth in sorted({int(row["depth"]) for row in scored_rows}):
        selected = [row for row in scored_rows if int(row["depth"]) == depth]
        correct = sum(bool(row["valid"]) for row in selected)
        by_depth[str(depth)] = {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected) if selected else 0.0,
        }
    pooled_correct = sum(bool(row["valid"]) for row in scored_rows)
    pooled_accuracy = pooled_correct / len(scored_rows) if scored_rows else 0.0
    passed = (
        bool(scored_rows)
        and pooled_accuracy >= pooled_floor
        and all(row["accuracy"] >= per_depth_floor for row in by_depth.values())
    )
    return {
        "passed": passed,
        "pooled_correct": pooled_correct,
        "pooled_total": len(scored_rows),
        "pooled_accuracy": pooled_accuracy,
        "pooled_floor": pooled_floor,
        "per_depth_floor": per_depth_floor,
        "by_depth": by_depth,
    }
