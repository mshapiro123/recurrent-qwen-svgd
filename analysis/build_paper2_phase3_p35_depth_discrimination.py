"""Build the no-training D1 plus K+ depth-discrimination receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_K = (1, 2, 3, 4)
EXPLORATORY_K = (5, 6)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell(input_dir: Path, seed: int, k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stem = f"seed_{seed}_k_{k}"
    rows_path = input_dir / f"{stem}.jsonl"
    summary_path = input_dir / f"{stem}.json"
    rows = read_jsonl(rows_path)
    summary = read_json(summary_path)
    if len(rows) != 1024 or len({str(row["item_id"]) for row in rows}) != 1024:
        raise RuntimeError(f"{stem} coverage changed")
    if int(summary["flow_loops"]) != k:
        raise RuntimeError(f"{stem} K identity changed")
    expected_clamp = k in EXPLORATORY_K
    if bool(summary["clamped_extension"]) != expected_clamp:
        raise RuntimeError(f"{stem} clamp classification changed")
    if float(summary["evaluation_gate_ceiling"]) != 0.02:
        raise RuntimeError(f"{stem} ceiling changed")
    if any(bool(summary[key]) for key in ("confirm_scored", "eval_e_scored", "optimizer_constructed")):
        raise RuntimeError(f"{stem} violated score-only scope")
    if int(summary["optimizer_steps"]) != 0:
        raise RuntimeError(f"{stem} stepped an optimizer")
    return rows, summary


def counts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "correct": sum(bool(row["augmented_correct"]) for row in rows),
        "accuracy": sum(bool(row["augmented_correct"]) for row in rows) / len(rows),
    }


def build(input_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    panel_shas = set()
    rows_by_seed_k: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for seed in (0, 1):
        for k in (*PRIMARY_K, *EXPLORATORY_K):
            rows, summary = cell(input_dir, seed, k)
            rows_by_seed_k[(seed, k)] = rows
            panel_shas.add(str(summary["panel_sha256"]))
            by_battery = {
                battery: counts([row for row in rows if str(row["battery"]) == battery])
                for battery in sorted({str(row["battery"]) for row in rows})
            }
            cells[f"seed_{seed}_k_{k}"] = {
                "seed": seed,
                "k": k,
                "scope": "registered" if k in PRIMARY_K else "exploratory_clamped",
                "pooled": counts(rows),
                "by_battery": by_battery,
                "rows_sha256": sha256_file(input_dir / f"seed_{seed}_k_{k}.jsonl"),
                "summary_sha256": sha256_file(input_dir / f"seed_{seed}_k_{k}.json"),
            }
    if len(panel_shas) != 1:
        raise RuntimeError("K+ cells did not share one DEV panel")

    marginal: dict[str, Any] = {}
    for seed in (0, 1):
        previous: dict[str, dict[str, Any]] | None = None
        for k in (*PRIMARY_K, *EXPLORATORY_K):
            rows = rows_by_seed_k[(seed, k)]
            current = {str(row["item_id"]): row for row in rows}
            if previous is not None:
                if set(previous) != set(current):
                    raise RuntimeError("K+ paired row identity changed")
                batteries = sorted({str(row["battery"]) for row in rows})
                per_battery = {}
                for battery in batteries:
                    ids = [item_id for item_id, row in current.items() if str(row["battery"]) == battery]
                    fixes = sum(
                        not bool(previous[item_id]["augmented_correct"])
                        and bool(current[item_id]["augmented_correct"])
                        for item_id in ids
                    )
                    regressions = sum(
                        bool(previous[item_id]["augmented_correct"])
                        and not bool(current[item_id]["augmented_correct"])
                        for item_id in ids
                    )
                    per_battery[battery] = {
                        "rows": len(ids), "fixes": fixes, "regressions": regressions,
                        "net_rows": fixes - regressions,
                        "accuracy_delta": (fixes - regressions) / len(ids),
                    }
                all_ids = sorted(current)
                fixes = sum(
                    not bool(previous[item_id]["augmented_correct"])
                    and bool(current[item_id]["augmented_correct"])
                    for item_id in all_ids
                )
                regressions = sum(
                    bool(previous[item_id]["augmented_correct"])
                    and not bool(current[item_id]["augmented_correct"])
                    for item_id in all_ids
                )
                marginal[f"seed_{seed}_k_{k}_minus_k_{k-1}"] = {
                    "seed": seed, "from_k": k - 1, "to_k": k,
                    "scope": "registered" if k <= 4 else "exploratory_clamped",
                    "pooled": {
                        "rows": len(all_ids), "fixes": fixes, "regressions": regressions,
                        "net_rows": fixes - regressions,
                        "accuracy_delta": (fixes - regressions) / len(all_ids),
                    },
                    "by_battery": per_battery,
                }
            previous = current

    k4_deltas = [marginal[f"seed_{seed}_k_4_minus_k_3"]["pooled"]["net_rows"] for seed in (0, 1)]
    return {
        "kind": "paper2_phase3_p35_depth_discrimination_v1",
        "status": "complete_dev_score_only",
        "authority": lock["authority"],
        "panel_sha256": next(iter(panel_shas)),
        "cells": cells,
        "marginal_improvement": marginal,
        "registered_k4_marginal_positive_both_seeds": all(value > 0 for value in k4_deltas),
        "d1_archive_read": {
            "status": "folded_into_k_plus",
            "reason": "P3.4/P3.5 archive contains no landed per-K coda rows",
            "future_prediction_kl": "unavailable; registered task graph keeps draft head inactive",
        },
        "interpretation_rule": {
            "positive_at_k4": "depth remains underexploited and deeper or persistent computation re-enters the lever queue",
            "saturated_by_k2_or_k3": "depth is not the binding constraint and E3 weakens",
            "k5_k6": "exploratory trajectory only; cannot determine the registered conclusion",
        },
        "scope": {
            "dev_only": True,
            "checkpoint_selection_barred": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input_dir, read_json(args.lock))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
