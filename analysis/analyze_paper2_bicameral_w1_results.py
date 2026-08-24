"""Aggregate Bicameral W1 seed receipts under the registered winner rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from training.paper2_bicameral_w1 import resolve_phase_a


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_seed(path: Path, expected_seed: int) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("kind") != "paper2_bicameral_w1_seed_v1":
        raise RuntimeError(f"wrong W1 seed receipt kind: {path}")
    if receipt.get("status") != "full_complete_score_only":
        raise RuntimeError(f"W1 seed receipt is incomplete: {path}")
    if int(receipt.get("seed", -1)) != expected_seed or int(receipt.get("rows", -1)) != 2048:
        raise RuntimeError(f"W1 seed identity changed: {path}")
    if receipt.get("optimizer_constructed") or int(receipt.get("optimizer_steps", -1)) != 0:
        raise RuntimeError(f"W1 score-only contract violated: {path}")
    if receipt.get("confirm_scored") or receipt.get("eval_e_scored"):
        raise RuntimeError(f"W1 sealed partition contract violated: {path}")
    return receipt


def aggregate(seed_receipts: list[Mapping[str, Any]], w3: Mapping[str, Any]) -> dict[str, Any]:
    cells = [dict(cell) for receipt in seed_receipts for cell in receipt["cells"]]
    decision = resolve_phase_a(cells)
    by_family: dict[str, Any] = {}
    for family in ("l0a", "l0b", "l0c", "l0d", "l0g"):
        family_cells = [cell for cell in cells if cell["arm"] == family]
        shuffle_cells = [cell for cell in cells if cell["arm"] == f"l5_{family[-1]}"]
        if len(family_cells) != 2 or len(shuffle_cells) != 2:
            raise RuntimeError(f"W1 family {family} lacks both seed/control cells")
        by_family[family] = {
            "seed_means": [float(cell["mean"]) for cell in sorted(family_cells, key=lambda x: x["seed"])],
            "seed_ci_low": [float(cell["ci_low"]) for cell in sorted(family_cells, key=lambda x: x["seed"])],
            "shuffle_seed_means": [
                float(cell["mean"]) for cell in sorted(shuffle_cells, key=lambda x: x["seed"])
            ],
            "pooled_seed_mean": sum(float(cell["mean"]) for cell in family_cells) / 2.0,
        }

    if w3.get("kind") != "paper2_bicameral_w3_desk_wave_v1":
        raise RuntimeError("wrong W3 receipt kind")
    dm5 = {
        seed: max(payload["dm5"]["cells"], key=lambda cell: cell["mean_cosine"])
        for seed, payload in w3["seeds"].items()
    }
    x6 = {
        seed: {
            "common_mode_energy_fraction": payload["x6"]["common_mode_energy_fraction"],
            "residual_direction_1_energy_fraction": payload["x6"]["directions"][0][
                "residual_energy_fraction"
            ],
            "residual_direction_1_battery_eta_squared": payload["x6"]["directions"][0][
                "battery_association"
            ]["eta_squared"],
            "residual_direction_1_cluster_eta_squared": payload["x6"]["directions"][0][
                "cluster_association"
            ]["eta_squared"],
        }
        for seed, payload in w3["seeds"].items()
    }
    return {
        "kind": "paper2_bicameral_w1_aggregate_v1",
        "status": "phase_a_complete",
        "phase_a": {"decision": decision, "families": by_family},
        "w3_desk": {"dm5_best_rank_cells": dm5, "x6": x6},
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_0", type=Path, required=True)
    parser.add_argument("--seed_1", type=Path, required=True)
    parser.add_argument("--w3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [load_seed(args.seed_0, 0), load_seed(args.seed_1, 1)]
    w3 = json.loads(args.w3.read_text(encoding="utf-8"))
    result = aggregate(seeds, w3)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
