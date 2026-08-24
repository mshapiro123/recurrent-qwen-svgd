"""Descriptive paired diagnostics for the registered Bicameral W1 cells."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from training.paper2_bicameral_w1 import bootstrap_mean_ci


FAMILIES = ("l0a", "l0b", "l0c", "l0d", "l0g")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    if x.numel() < 2 or float(x.std(unbiased=False)) == 0.0 or float(y.std(unbiased=False)) == 0.0:
        return None
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _paired_rows(
    own: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    comparator_by_id = {str(row["item_id"]): row for row in comparator}
    if len(comparator_by_id) != len(comparator):
        raise RuntimeError("W1 comparator contains duplicate item IDs")
    pairs = []
    for row in own:
        item_id = str(row["item_id"])
        other = comparator_by_id.get(item_id)
        if other is None:
            raise RuntimeError(f"W1 comparator is missing {item_id}")
        if (
            str(other["battery"]) != str(row["battery"])
            or int(other["seed"]) != int(row["seed"])
            or float(other["baseline_margin"]) != float(row["baseline_margin"])
            or str(other["evaluator"]) != str(row["evaluator"])
            or str(other["schedule"]) != str(row["schedule"])
        ):
            raise RuntimeError(f"W1 paired provenance changed for {item_id}")
        pairs.append((row, other))
    if len(pairs) != len(comparator):
        raise RuntimeError("W1 paired population changed")
    return pairs


def paired_summary(
    own: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pairs = _paired_rows(own, comparator)
    own_delta = [float(row["margin_delta"]) for row, _other in pairs]
    comparator_delta = [float(other["margin_delta"]) for _row, other in pairs]
    advantage = [left - right for left, right in zip(own_delta, comparator_delta)]
    by_battery: dict[str, list[float]] = defaultdict(list)
    for (row, _other), value in zip(pairs, advantage):
        by_battery[str(row["battery"])].append(value)
    return {
        "rows": len(pairs),
        "own_mean": _mean(own_delta),
        "comparator_mean": _mean(comparator_delta),
        "own_positive_fraction": sum(value > 0.0 for value in own_delta) / len(own_delta),
        "comparator_positive_fraction": sum(value > 0.0 for value in comparator_delta)
        / len(comparator_delta),
        "own_comparator_row_correlation": _correlation(own_delta, comparator_delta),
        "paired_advantage": bootstrap_mean_ci(advantage, seed=bootstrap_seed),
        "paired_advantage_positive_fraction": sum(value > 0.0 for value in advantage)
        / len(advantage),
        "by_battery": {
            battery: {
                **bootstrap_mean_ci(values, seed=bootstrap_seed + index + 1),
                "positive_fraction": sum(value > 0.0 for value in values) / len(values),
            }
            for index, (battery, values) in enumerate(sorted(by_battery.items()))
        },
    }


def baseline_margin_quartiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["baseline_margin"]), str(row["item_id"])))
    result = []
    for index in range(4):
        start = index * len(ordered) // 4
        stop = (index + 1) * len(ordered) // 4
        selected = ordered[start:stop]
        margins = [float(row["baseline_margin"]) for row in selected]
        deltas = [float(row["margin_delta"]) for row in selected]
        result.append(
            {
                "quartile": index + 1,
                "rows": len(selected),
                "baseline_margin_min": min(margins),
                "baseline_margin_max": max(margins),
                "mean_delta": _mean(deltas),
                "positive_delta_fraction": sum(value > 0.0 for value in deltas) / len(deltas),
            }
        )
    return result


def analyze_seed(private_dir: Path, seed: int) -> dict[str, Any]:
    cells = {
        arm: read_jsonl(private_dir / f"seed_{seed}_full_{arm}.jsonl")
        for arm in (
            "l0a",
            "l0b",
            "l0c",
            "l0d",
            "l0g",
            "l4",
            "l5_a",
            "l5_b",
            "l5_c",
            "l5_d",
            "l5_g",
        )
    }
    if any(len(rows) != 2048 for rows in cells.values()):
        raise RuntimeError(f"W1 seed {seed} descriptive population changed")
    families = {}
    for index, family in enumerate(FAMILIES):
        letter = family[-1]
        families[family] = {
            "versus_shuffle": paired_summary(
                cells[family], cells[f"l5_{letter}"], bootstrap_seed=20260840 + 10 * seed + index
            ),
            "versus_random": paired_summary(
                cells[family], cells["l4"], bootstrap_seed=20260860 + 10 * seed + index
            ),
            "baseline_margin_quartiles": baseline_margin_quartiles(cells[family]),
        }
    return {
        "seed": seed,
        "rows": 2048,
        "families": families,
        "role": "descriptive_only_does_not_change_registered_winner_rule",
    }


def run(seed_directories: Mapping[int, Path]) -> dict[str, Any]:
    seeds = {str(seed): analyze_seed(path, seed) for seed, path in sorted(seed_directories.items())}
    return {
        "kind": "paper2_bicameral_w1_paired_diagnostics_v1",
        "status": "complete",
        "seeds": seeds,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_0_private", type=Path)
    parser.add_argument("--seed_1_private", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        seed: path
        for seed, path in ((0, args.seed_0_private), (1, args.seed_1_private))
        if path is not None
    }
    if not inputs:
        raise RuntimeError("at least one W1 seed directory is required")
    result = run(inputs)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
