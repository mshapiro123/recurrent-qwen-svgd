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
PreimageStratum = Literal["unique", "small", "large"]
PHASE_G_NAME_SYMBOLS = NAME_SYMBOLS + ("Eli", "Mia", "Leo", "Eva")


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
    preimage_stratum: PreimageStratum | None = None
    generator_kind: str = "constructive"
    generation_attempts: int = 1
    symbol_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PhaseGFrozenEvalConfig:
    """Locked G-alpha evaluation design for arbitrary N=24 functions."""

    n_symbols: int = 24
    depths: tuple[int, ...] = (1, 2, 3, 4)
    rows_per_stratum: int = 128
    seed: int = 7_194_203
    max_attempts_per_row: int = 100_000


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


def preimage_stratum(solution_count: int) -> PreimageStratum:
    count = int(solution_count)
    if count == 1:
        return "unique"
    if 2 <= count <= 4:
        return "small"
    if count >= 5:
        return "large"
    raise ValueError("A reachable target must have at least one exact preimage")


def _validate_common(n_symbols: int, depth: int) -> None:
    if n_symbols < 4:
        raise ValueError("n_symbols must be at least 4")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth >= n_symbols:
        raise ValueError("depth must be < n_symbols")
    if n_symbols > len(PHASE_G_NAME_SYMBOLS):
        raise ValueError(f"n_symbols cannot exceed {len(PHASE_G_NAME_SYMBOLS)} for name rendering")


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


def build_stratified_random_abduction_instance(
    *,
    instance_id: str,
    n_symbols: int,
    depth: int,
    seed: int,
    stratum: PreimageStratum,
    split: str = "test",
    max_attempts: int = 100_000,
) -> AbductiveInstance:
    """Sample an arbitrary non-bijective table conditioned on a preimage bin.

    A random start is used only to choose a reachable target. Once a mapping is
    accepted, the posterior-supervision chain is sampled uniformly from the
    independently enumerated exact preimage set.
    """

    _validate_common(n_symbols, depth)
    if stratum not in {"unique", "small", "large"}:
        raise ValueError(f"Unknown preimage stratum: {stratum!r}")
    rng = random.Random(int(seed))
    for attempt in range(1, int(max_attempts) + 1):
        mapping = {value: rng.randrange(n_symbols) for value in range(n_symbols)}
        if len(set(mapping.values())) == n_symbols:
            continue
        probe_start = rng.randrange(n_symbols)
        target = apply_mapping(mapping, probe_start, depth)
        valid_starts = exact_depth_preimages(mapping, target, depth)
        if preimage_stratum(len(valid_starts)) != stratum:
            continue
        selected_start = rng.choice(valid_starts)
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
            selected_orbit=orbit(mapping, selected_start, depth),
            preimage_stratum=stratum,
            generator_kind="arbitrary_random_mapping_conditioned_on_preimage_stratum",
            generation_attempts=attempt,
            symbol_names=PHASE_G_NAME_SYMBOLS[:n_symbols],
        )
    raise RuntimeError(
        f"Could not sample {stratum} row at depth {depth} after {max_attempts} attempts"
    )


def name(value: int, *, symbol_names: tuple[str, ...] | None = None) -> str:
    return (symbol_names or NAME_SYMBOLS)[int(value)]


def render_question(instance: AbductiveInstance) -> str:
    symbols = instance.symbol_names or NAME_SYMBOLS
    table = "\n".join(
        f"{name(left, symbol_names=symbols)} always passes the key to "
        f"{name(instance.mapping[left], symbol_names=symbols)}."
        for left in instance.table_order
    )
    return (
        f"{table}\n\n"
        f"After exactly {instance.depth} handoffs, the key is with "
        f"{name(instance.target, symbol_names=symbols)}.\n"
        "Who could have held the key before the first handoff? "
        "Answer with one valid name."
    )


def build_row(instance: AbductiveInstance) -> dict[str, Any]:
    symbols = instance.symbol_names or NAME_SYMBOLS
    reverse_chain = list(reversed(instance.selected_orbit[:-1]))
    valid_orbits = {
        name(start, symbol_names=symbols): [
            name(value, symbol_names=symbols)
            for value in orbit(instance.mapping, start, instance.depth)
        ]
        for start in instance.valid_starts
    }
    question = render_question(instance)
    selected_name = name(instance.selected_start, symbol_names=symbols)
    row = {
        "id": instance.instance_id,
        "instance_id": instance.instance_id,
        "question": question,
        "prompt": f"{question}\nAnswer:",
        "completion": f" {selected_name}",
        "loop_completions": [f" {name(value, symbol_names=symbols)}" for value in reverse_chain],
        "target_loop_count": instance.depth,
        "depth": instance.depth,
        "synthetic_depth": instance.depth,
        "synthetic_task": "iterated_function_abduction",
        "task_mode": instance.mode,
        "n_symbols": instance.n_symbols,
        "observed_target": name(instance.target, symbol_names=symbols),
        "selected_start": selected_name,
        "selected_orbit": [name(value, symbol_names=symbols) for value in instance.selected_orbit],
        "valid_starts": [name(value, symbol_names=symbols) for value in instance.valid_starts],
        "valid_start_values": list(instance.valid_starts),
        "valid_orbits": valid_orbits,
        "coverage_denominator": len(instance.valid_starts),
        "mapping": {
            name(left, symbol_names=symbols): name(right, symbol_names=symbols)
            for left, right in instance.mapping.items()
        },
        "mapping_values": {str(left): int(right) for left, right in instance.mapping.items()},
        "score_target": "full_symbols",
        "prompt_style": "question_only",
        "intermediate_chain_supervision": True,
        "chain_direction": "reverse_preimage",
    }
    if instance.preimage_stratum is not None:
        row.update(
            {
                "preimage_stratum": instance.preimage_stratum,
                "generator_kind": instance.generator_kind,
                "generation_attempts": instance.generation_attempts,
                "posterior_chain_sampling": "uniform_over_exact_valid_preimages",
                "symbol_names": list(symbols[: instance.n_symbols]),
            }
        )
    return row


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


def _phase_g_seed(seed: int, split: str, stratum: PreimageStratum, depth: int, row_index: int) -> int:
    material = f"{int(seed)}|{split}|{stratum}|{int(depth)}|{int(row_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def build_phase_g_frozen_rows(
    config: PhaseGFrozenEvalConfig,
    *,
    split: str,
) -> list[dict[str, Any]]:
    if config.n_symbols != 24:
        raise ValueError("The locked G-alpha frozen evaluation uses N=24")
    if not config.depths or any(depth < 1 or depth >= config.n_symbols for depth in config.depths):
        raise ValueError("Frozen evaluation depths must be non-empty and within the symbol range")
    rows: list[dict[str, Any]] = []
    for stratum in ("unique", "small", "large"):
        for row_index in range(config.rows_per_stratum):
            depth = int(config.depths[row_index % len(config.depths)])
            instance_id = f"phase_g_{split}_{stratum}_d{depth:02d}_{row_index:05d}"
            instance = build_stratified_random_abduction_instance(
                instance_id=instance_id,
                n_symbols=config.n_symbols,
                depth=depth,
                seed=_phase_g_seed(config.seed, split, stratum, depth, row_index),
                stratum=stratum,
                split=split,
                max_attempts=config.max_attempts_per_row,
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
    include_frozen_metadata = any("preimage_stratum" in row for row in rows)
    stratum_counts: dict[str, int] = {}
    generator_counts: dict[str, int] = {}
    for row in rows:
        depth = str(row["depth"])
        solutions = str(row["coverage_denominator"])
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        solution_counts[solutions] = solution_counts.get(solutions, 0) + 1
        if include_frozen_metadata:
            stratum = str(row["preimage_stratum"])
            stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
            generator = str(row["generator_kind"])
            generator_counts[generator] = generator_counts.get(generator, 0) + 1
    manifest = {
        "rows": len(rows),
        "depth_counts": depth_counts,
        "solution_counts": solution_counts,
        "row_id_sha256": hashlib.sha256(ids).hexdigest(),
        "row_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if include_frozen_metadata:
        manifest["stratum_counts"] = stratum_counts
        manifest["generator_counts"] = generator_counts
    return manifest


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
        symbols = tuple(str(item) for item in (row.get("symbol_names") or NAME_SYMBOLS))
        try:
            target = symbols.index(target_name)
        except ValueError:
            errors.append(f"{row_id}: unknown target {target_name!r}")
            continue
        exact = [
            name(value, symbol_names=symbols)
            for value in exact_depth_preimages(mapping, target, int(row["depth"]))
        ]
        if exact != list(row["valid_starts"]):
            errors.append(f"{row_id}: stored valid starts do not match exact preimages")
        if row.get("task_mode") != expected_mode:
            errors.append(f"{row_id}: expected mode {expected_mode}")
        if expected_mode == "injective" and len(exact) != 1:
            errors.append(f"{row_id}: injective control has {len(exact)} solutions")
        if expected_mode == "abductive" and len(exact) < 2:
            errors.append(f"{row_id}: abductive row has fewer than two solutions")
        if str(row.get("completion", "")).strip() not in exact:
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


def validate_phase_g_frozen_rows(
    rows: list[dict[str, Any]],
    *,
    rows_per_stratum: int,
) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    counts = {"unique": 0, "small": 0, "large": 0}
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen:
            errors.append(f"duplicate id: {row_id}")
        seen.add(row_id)
        if int(row.get("n_symbols", -1)) != 24:
            errors.append(f"{row_id}: expected N=24")
        mapping = {int(left): int(right) for left, right in row["mapping_values"].items()}
        if len(set(mapping.values())) == len(mapping):
            errors.append(f"{row_id}: mapping is bijective, expected arbitrary non-bijective table")
        symbols = tuple(str(item) for item in (row.get("symbol_names") or NAME_SYMBOLS))
        target = symbols.index(str(row["observed_target"]))
        exact_values = exact_depth_preimages(mapping, target, int(row["depth"]))
        exact_names = [name(value, symbol_names=symbols) for value in exact_values]
        if exact_names != list(row.get("valid_starts") or []):
            errors.append(f"{row_id}: stored preimages do not match forward-orbit enumeration")
        stratum = str(row.get("preimage_stratum"))
        actual_stratum = preimage_stratum(len(exact_values))
        if stratum != actual_stratum:
            errors.append(f"{row_id}: stratum {stratum!r} != {actual_stratum!r}")
        elif stratum in counts:
            counts[stratum] += 1
        if row.get("selected_start") not in exact_names:
            errors.append(f"{row_id}: sampled posterior chain is not valid")
        if row.get("posterior_chain_sampling") != "uniform_over_exact_valid_preimages":
            errors.append(f"{row_id}: posterior chain sampling contract missing")
        if int(row.get("coverage_denominator", -1)) != len(exact_values):
            errors.append(f"{row_id}: exact coverage denominator mismatch")
    expected_counts = {key: int(rows_per_stratum) for key in counts}
    if counts != expected_counts:
        errors.append(f"stratum counts {counts} != {expected_counts}")
    return {
        "status": "passed" if not errors else "failed",
        "rows": len(rows),
        "errors": errors,
        "stratum_counts": counts,
        "manifest": row_manifest(rows),
    }
