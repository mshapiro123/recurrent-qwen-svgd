"""Repeated-prompt multi-target rows for the Phase G guidance correction."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.branching_relations_task import (
    BranchingRelationsConfig,
    build_rows,
)


def _terminal_chains(row: dict[str, Any]) -> dict[str, list[str]]:
    """Return one deterministic exact-depth chain for every reachable terminal."""

    depth = int(row["depth"])
    successors = {
        str(source): tuple(str(target) for target in targets)
        for source, targets in row["successors"].items()
    }
    active = [(str(row["start"]), [str(row["start"])])]
    for _ in range(depth):
        active = [
            (target, chain + [target])
            for source, chain in active
            for target in successors[source]
        ]
    chains: dict[str, list[str]] = {}
    for target, chain in active:
        prior = chains.get(target)
        if prior is None or tuple(chain) < tuple(prior):
            chains[target] = chain
    reachable = {str(value) for value in row["reachable_symbols"]}
    if set(chains) != reachable:
        raise ValueError(
            "Exact-depth chain enumeration disagrees with stored reachable set: "
            f"chains={sorted(chains)}, reachable={sorted(reachable)}"
        )
    return chains


def _selected_targets(
    values: list[str],
    *,
    base_problem_id: str,
    targets_per_prompt: int | None,
) -> list[str]:
    if targets_per_prompt is None or targets_per_prompt >= len(values):
        return list(values)
    if targets_per_prompt < 1:
        raise ValueError("targets_per_prompt must be positive or None")
    ranked = sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{base_problem_id}|{value}".encode("utf-8")
        ).digest(),
    )
    selected = set(ranked[:targets_per_prompt])
    return [value for value in values if value in selected]


def expand_multitarget_row(
    row: dict[str, Any],
    *,
    targets_per_prompt: int | None = None,
) -> list[dict[str, Any]]:
    """Clone one prompt into rows that differ only in their valid target chain."""

    base_problem_id = str(row["id"])
    chains = _terminal_chains(row)
    symbol_order = [str(value) for value in row["symbol_names"]]
    targets = _selected_targets(
        [value for value in symbol_order if value in chains],
        base_problem_id=base_problem_id,
        targets_per_prompt=targets_per_prompt,
    )
    if len(targets) < 1:
        raise AssertionError("A multi-target row must retain at least one target")
    symbol_values = {symbol: index for index, symbol in enumerate(symbol_order)}
    variants: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        chain = chains[target]
        variant = copy.deepcopy(row)
        variant_id = f"{base_problem_id}__target_{target_index:02d}_{target}"
        variant.update(
            {
                "id": variant_id,
                "instance_id": variant_id,
                "base_problem_id": base_problem_id,
                "target_variant_index": target_index,
                "target_variant_count": len(targets),
                "target": target,
                "completion": f" {target}",
                "sampled_chain": chain,
                "sampled_chain_values": [symbol_values[value] for value in chain],
                "loop_completions": [f" {value}" for value in chain[1:]],
                "posterior_chain_sampling": "enumerated_distinct_terminal_target",
                "multitarget_target_selection": (
                    "all_reachable_targets"
                    if len(targets) == len(chains)
                    else f"deterministic_subset_{len(targets)}"
                ),
            }
        )
        variants.append(variant)
    return variants


def build_multitarget_rows(
    config: BranchingRelationsConfig,
    *,
    split: str,
    rendering: str,
    n_symbols: int,
    targets_per_prompt: int | None = None,
) -> list[dict[str, Any]]:
    """Build exact repeated-prompt rows with distinct valid terminal targets."""

    base_rows = build_rows(
        config,
        split=split,
        rendering=rendering,
        n_symbols=n_symbols,
    )
    return [
        variant
        for row in base_rows
        for variant in expand_multitarget_row(
            row,
            targets_per_prompt=targets_per_prompt,
        )
    ]


def validate_multitarget_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate prompt sharing, valid chains, and target support by base problem."""

    errors: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identifiers: set[str] = set()
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in identifiers:
            errors.append(f"duplicate or missing row id: {identifier!r}")
        identifiers.add(identifier)
        base_problem_id = str(row.get("base_problem_id", ""))
        if not base_problem_id:
            errors.append(f"{identifier}: missing base_problem_id")
            continue
        groups[base_problem_id].append(row)

    all_reachable_targets_covered = bool(groups)
    for base_problem_id, variants in groups.items():
        reference = variants[0]
        reference_payload = json.dumps(
            {
                "question": reference["question"],
                "prompt": reference["prompt"],
                "start": reference["start"],
                "depth": int(reference["depth"]),
                "successors": reference["successors"],
                "reachable_symbols": sorted(str(value) for value in reference["reachable_symbols"]),
            },
            sort_keys=True,
        )
        targets: list[str] = []
        for row in variants:
            payload = json.dumps(
                {
                    "question": row["question"],
                    "prompt": row["prompt"],
                    "start": row["start"],
                    "depth": int(row["depth"]),
                    "successors": row["successors"],
                    "reachable_symbols": sorted(str(value) for value in row["reachable_symbols"]),
                },
                sort_keys=True,
            )
            if payload != reference_payload:
                errors.append(f"{base_problem_id}: variants do not share an identical prompt problem")
            targets.append(str(row["target"]))
            depth = int(row["depth"])
            chain = [str(value) for value in row["sampled_chain"]]
            successors = {
                str(source): {str(target) for target in next_values}
                for source, next_values in row["successors"].items()
            }
            if (
                len(chain) != depth + 1
                or not chain
                or chain[0] != str(row["start"])
                or any(chain[index + 1] not in successors[chain[index]] for index in range(depth))
                or chain[-1] != str(row["target"])
            ):
                errors.append(f"{row['id']}: invalid stored target chain")
            expected_loop_completions = [f" {value}" for value in chain[1:]]
            if row["loop_completions"] != expected_loop_completions:
                errors.append(f"{row['id']}: loop completions do not match stored chain")
            if int(row.get("target_variant_count", -1)) != len(variants):
                errors.append(f"{row['id']}: target_variant_count disagrees with group size")
        reachable = {str(value) for value in reference["reachable_symbols"]}
        if not set(targets).issubset(reachable):
            errors.append(f"{base_problem_id}: target outside reachable set")
        if len(set(targets)) != len(targets):
            errors.append(f"{base_problem_id}: duplicate targets within prompt group")
        if set(targets) != reachable:
            all_reachable_targets_covered = False

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "rows": len(rows),
        "base_problem_groups": len(groups),
        "groups_with_multiple_targets": sum(len(values) > 1 for values in groups.values()),
        "all_reachable_targets_covered": all_reachable_targets_covered,
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
