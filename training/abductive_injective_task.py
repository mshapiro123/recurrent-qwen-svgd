"""Paired injective and multimodal abduction tasks for Phase G-alpha.

Each row gives a finite function, a recursion depth, and the observed endpoint.
The model must propose a start value whose depth-step orbit reaches that endpoint.
Injective controls have one valid start. Multimodal rows use a constructive
convergent fan with an exact set of two or more valid starts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from training.synthetic_depth_task import NAME_SYMBOLS, apply_mapping, build_permutation_instance


TaskMode = Literal["injective", "abductive"]


@dataclass(frozen=True)
class AbductiveInjectiveConfig:
    n_symbols: int = 20
    max_depth: int = 8
    rows_per_depth: int = 128
    seed: int = 1_104_729
    min_solutions: int = 2
    max_solutions: int = 4


@dataclass(frozen=True)
class AbductiveInstance:
    instance_id: str
    split: str
    mode: TaskMode
    n_symbols: int
    depth: int
    target: int
    mapping: dict[int, int]
    table_order: list[int]
    valid_starts: list[int]
    selected_start: int
    selected_orbit: list[int]


def orbit(mapping: dict[int, int], start: int, depth: int) -> list[int]:
    values = [int(start)]
    for _ in range(int(depth)):
        values.append(int(mapping[values[-1]]))
    return values


def exact_depth_preimages(mapping: dict[int, int], target: int, depth: int) -> list[int]:
    return sorted(
        int(start)
        for start in mapping
        if apply_mapping(mapping, int(start), int(depth)) == int(target)
    )


def _validate_common(n_symbols: int, depth: int) -> None:
    if n_symbols < 4:
        raise ValueError("n_symbols must be at least 4")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth >= n_symbols:
        raise ValueError("depth must be < n_symbols")
    if n_symbols > len(NAME_SYMBOLS):
        raise ValueError(f"n_symbols cannot exceed {len(NAME_SYMBOLS)} for name rendering")


def build_injective_abduction_instance(
    *,
    instance_id: str,
    n_symbols: int,
    depth: int,
    seed: int,
    split: str = "test",
) -> AbductiveInstance:
    _validate_common(n_symbols, depth)
    forward = build_permutation_instance(
        instance_id=instance_id,
        n_symbols=n_symbols,
        depth=depth,
        seed=seed,
        split=split,
    )
    valid_starts = exact_depth_preimages(forward.mapping, forward.target, depth)
    if valid_starts != [forward.start]:
        raise AssertionError("Permutation abduction control must have exactly one preimage")
    return AbductiveInstance(
        instance_id=instance_id,
        split=split,
        mode="injective",
        n_symbols=n_symbols,
        depth=depth,
        target=forward.target,
        mapping=forward.mapping,
        table_order=forward.table_order,
        valid_starts=valid_starts,
        selected_start=forward.start,
        selected_orbit=orbit(forward.mapping, forward.start, depth),
    )


def build_multimodal_abduction_instance(
    *,
    instance_id: str,
    n_symbols: int,
    depth: int,
    seed: int,
    solution_count: int,
    split: str = "test",
) -> AbductiveInstance:
    """Construct exactly ``solution_count`` depth-step preimages.

    Several starts merge immediately into a shared suffix. All unused values
    are self-loops, while the target exits to a safe sink, preventing accidental
    extra preimages at the requested depth.
    """

    _validate_common(n_symbols, depth)
    if solution_count < 2:
        raise ValueError("multimodal abduction requires at least two solutions")
    required = solution_count + depth + 1
    if required > n_symbols:
        raise ValueError(
            "n_symbols is too small for the requested exact fan: "
            f"need at least {required}, got {n_symbols}"
        )

    rng = random.Random(seed)
    values = list(range(n_symbols))
    rng.shuffle(values)
    target = values[0]
    shared_reverse = values[1:depth]
    starts = sorted(values[depth : depth + solution_count])
    sink = values[depth + solution_count]

    mapping = {value: value for value in range(n_symbols)}
    if depth == 1:
        for start in starts:
            mapping[start] = target
    else:
        first_shared = shared_reverse[-1]
        for start in starts:
            mapping[start] = first_shared
        for left, right in zip(reversed(shared_reverse), reversed(shared_reverse[:-1])):
            mapping[left] = right
        mapping[shared_reverse[0]] = target
    mapping[target] = sink
    mapping[sink] = sink

    valid_starts = exact_depth_preimages(mapping, target, depth)
    if valid_starts != starts:
        raise AssertionError(
            f"Constructed multimodal row has wrong preimages: expected={starts}, got={valid_starts}"
        )
    selected_start = rng.choice(valid_starts)
    selected_orbit = orbit(mapping, selected_start, depth)
    table_order = list(range(n_symbols))
    rng.shuffle(table_order)
    return AbductiveInstance(
        instance_id=instance_id,
        split=split,
        mode="abductive",
        n_symbols=n_symbols,
        depth=depth,
        target=target,
        mapping=mapping,
        table_order=table_order,
        valid_starts=valid_starts,
        selected_start=selected_start,
        selected_orbit=selected_orbit,
    )


def name(value: int) -> str:
    return NAME_SYMBOLS[int(value)]


def render_question(instance: AbductiveInstance) -> str:
    table = "\n".join(
        f"{name(left)} always passes the key to {name(instance.mapping[left])}."
        for left in instance.table_order
    )
    return (
        f"{table}\n\n"
        f"After exactly {instance.depth} handoffs, the key is with {name(instance.target)}.\n"
        "Who could have held the key before the first handoff? "
        "Answer with one valid name."
    )


def build_row(instance: AbductiveInstance) -> dict[str, Any]:
    reverse_chain = list(reversed(instance.selected_orbit[:-1]))
    valid_orbits = {
        name(start): [name(value) for value in orbit(instance.mapping, start, instance.depth)]
        for start in instance.valid_starts
    }
    question = render_question(instance)
    return {
        "id": instance.instance_id,
        "instance_id": instance.instance_id,
        "question": question,
        "prompt": f"{question}\nAnswer: ",
        "completion": name(instance.selected_start),
        "loop_completions": [name(value) for value in reverse_chain],
        "target_loop_count": instance.depth,
        "depth": instance.depth,
        "synthetic_depth": instance.depth,
        "synthetic_task": "iterated_function_abduction",
        "task_mode": instance.mode,
        "n_symbols": instance.n_symbols,
        "observed_target": name(instance.target),
        "selected_start": name(instance.selected_start),
        "selected_orbit": [name(value) for value in instance.selected_orbit],
        "valid_starts": [name(value) for value in instance.valid_starts],
        "valid_start_values": list(instance.valid_starts),
        "valid_orbits": valid_orbits,
        "coverage_denominator": len(instance.valid_starts),
        "mapping": {name(left): name(right) for left, right in instance.mapping.items()},
        "mapping_values": {str(left): int(right) for left, right in instance.mapping.items()},
        "score_target": "full_symbols",
        "prompt_style": "question_only",
        "intermediate_chain_supervision": True,
        "chain_direction": "reverse_preimage",
    }


def _seed_for(seed: int, split: str, mode: TaskMode, depth: int, row_index: int) -> int:
    split_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}.get(split, 3_000_000)
    mode_offset = 0 if mode == "injective" else 10_000_000
    return int(seed) + split_offset + mode_offset + int(depth) * 10_000 + int(row_index)


def build_rows(
    config: AbductiveInjectiveConfig,
    *,
    split: str,
    mode: TaskMode,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth in range(1, config.max_depth + 1):
        max_solutions = min(config.max_solutions, config.n_symbols - depth - 1)
        if mode == "abductive" and max_solutions < config.min_solutions:
            raise ValueError(f"depth {depth} leaves no room for the requested multimodal fan")
        for row_index in range(config.rows_per_depth):
            seed = _seed_for(config.seed, split, mode, depth, row_index)
            instance_id = f"{split}_{mode}_d{depth:02d}_{row_index:05d}"
            if mode == "injective":
                instance = build_injective_abduction_instance(
                    instance_id=instance_id,
                    n_symbols=config.n_symbols,
                    depth=depth,
                    seed=seed,
                    split=split,
                )
            else:
                solution_count = config.min_solutions + row_index % (max_solutions - config.min_solutions + 1)
                instance = build_multimodal_abduction_instance(
                    instance_id=instance_id,
                    n_symbols=config.n_symbols,
                    depth=depth,
                    seed=seed,
                    solution_count=solution_count,
                    split=split,
                )
            rows.append(build_row(instance))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def row_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = "\n".join(str(row["id"]) for row in rows).encode("utf-8")
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")
    depth_counts: dict[str, int] = {}
    solution_counts: dict[str, int] = {}
    for row in rows:
        depth = str(row["depth"])
        solutions = str(row["coverage_denominator"])
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        solution_counts[solutions] = solution_counts.get(solutions, 0) + 1
    return {
        "rows": len(rows),
        "depth_counts": depth_counts,
        "solution_counts": solution_counts,
        "row_id_sha256": hashlib.sha256(ids).hexdigest(),
        "row_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_rows(rows: list[dict[str, Any]], *, expected_mode: TaskMode) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen:
            errors.append(f"duplicate id: {row_id}")
        seen.add(row_id)
        mapping = {int(left): int(right) for left, right in row["mapping_values"].items()}
        target_name = str(row["observed_target"])
        try:
            target = NAME_SYMBOLS.index(target_name)
        except ValueError:
            errors.append(f"{row_id}: unknown target {target_name!r}")
            continue
        exact = [name(value) for value in exact_depth_preimages(mapping, target, int(row["depth"]))]
        if exact != list(row["valid_starts"]):
            errors.append(f"{row_id}: stored valid starts do not match exact preimages")
        if row.get("task_mode") != expected_mode:
            errors.append(f"{row_id}: expected mode {expected_mode}")
        if expected_mode == "injective" and len(exact) != 1:
            errors.append(f"{row_id}: injective control has {len(exact)} solutions")
        if expected_mode == "abductive" and len(exact) < 2:
            errors.append(f"{row_id}: abductive row has fewer than two solutions")
        if row.get("completion") not in exact:
            errors.append(f"{row_id}: completion is not a valid preimage")
        if int(row.get("coverage_denominator", -1)) != len(exact):
            errors.append(f"{row_id}: wrong coverage denominator")
        if len(row.get("loop_completions") or []) != int(row["depth"]):
            errors.append(f"{row_id}: loop completion count does not match depth")
    return {
        "status": "passed" if not errors else "failed",
        "mode": expected_mode,
        "rows": len(rows),
        "errors": errors,
        "manifest": row_manifest(rows),
    }

