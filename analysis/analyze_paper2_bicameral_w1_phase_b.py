"""Analyze W1 Phase B with paired row receipts and registered branch language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from training.paper2_bicameral_w1 import bootstrap_mean_ci


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def paired(root: Path, seed: int, first: str, second: str) -> dict[str, Any]:
    left = read_jsonl(root / "private" / f"seed_{seed}_phase_b_{first}.jsonl")
    right = read_jsonl(root / "private" / f"seed_{seed}_phase_b_{second}.jsonl")
    right_by_id = {str(row["item_id"]): row for row in right}
    if len(left) != 2048 or len(right_by_id) != 2048:
        raise RuntimeError("W1 Phase B paired population changed")
    deltas = [
        float(row["margin_delta"]) - float(right_by_id[str(row["item_id"])]["margin_delta"])
        for row in left
    ]
    result = bootstrap_mean_ci(deltas, seed=20260824 + seed)
    result.update({"first": first, "second": second})
    return result


def analyze(root: Path) -> dict[str, Any]:
    summaries = {seed: read_json(root / "receipts" / f"seed_{seed}_phase_b_summary.json") for seed in (0, 1)}
    cells = {
        seed: {str(cell["arm"]): cell for cell in summary["cells"]}
        for seed, summary in summaries.items()
    }
    for seed, summary in summaries.items():
        if summary.get("status") != "complete_score_only" or summary.get("optimizer_constructed"):
            raise RuntimeError(f"W1 Phase B seed {seed} is not a complete score-only receipt")
        if summary.get("confirm_scored") or summary.get("eval_e_scored"):
            raise RuntimeError("W1 sealed partition contract changed")

    comparisons = {
        str(seed): {
            "l1_minus_l2": paired(root, seed, "l1", "l2"),
            "l1_minus_l3": paired(root, seed, "l1", "l3"),
            "l2_minus_l3": paired(root, seed, "l2", "l3"),
        }
        for seed in (0, 1)
    }
    l1_positive = all(float(cells[seed]["l1"]["ci_low"]) > 0 for seed in (0, 1))
    l2_positive = all(float(cells[seed]["l2"]["ci_low"]) > 0 for seed in (0, 1))
    l1_l2 = [comparisons[str(seed)]["l1_minus_l2"] for seed in (0, 1)]
    cluster_better = all(float(value["ci_low"]) > 0 for value in l1_l2)
    global_better = all(float(value["ci_high"]) < 0 for value in l1_l2)
    unresolved = all(float(value["ci_low"]) <= 0 <= float(value["ci_high"]) for value in l1_l2)
    if l1_positive and l2_positive and cluster_better:
        branch = "CLUSTER-CAUSAL"
        adjudication = False
    elif l1_positive and l2_positive and unresolved:
        branch = "GLOBAL-STEER"
        adjudication = False
    elif l1_positive and l2_positive and global_better:
        branch = "GLOBAL-STEER-WITH-GLOBAL-ADVANTAGE"
        adjudication = True
    elif not l1_positive and not l2_positive:
        branch = "ROW-LEVEL-ONLY"
        adjudication = False
    else:
        branch = "SEED-SPLIT-OR-UNMAPPED"
        adjudication = True

    l6_arms = sorted(arm for arm in cells[0] if arm.startswith("l6_"))
    l6_positive = [
        arm for arm in l6_arms if all(float(cells[seed][arm]["ci_low"]) > 0 for seed in (0, 1))
    ]
    assignments = {
        seed: torch.load(root / "private" / f"seed_{seed}_phase_b_assignments.pt", map_location="cpu", weights_only=False)
        for seed in (0, 1)
    }
    labels0 = torch.as_tensor(assignments[0]["assignments"], dtype=torch.long)
    labels1 = torch.as_tensor(assignments[1]["assignments"], dtype=torch.long)
    assignment_agreement = float((labels0 == labels1).float().mean())
    result = {
        "kind": "paper2_bicameral_w1_phase_b_analysis_v1",
        "status": "complete",
        "registered_winner": "l0c",
        "branch": branch,
        "strategy_adjudication_required": adjudication,
        "branch_reason": {
            "l1_positive_both_seeds": l1_positive,
            "l2_positive_both_seeds": l2_positive,
            "l1_minus_l2_cluster_better_both": cluster_better,
            "l1_minus_l2_global_better_both": global_better,
            "l1_minus_l2_unresolved_both": unresolved,
        },
        "comparisons": comparisons,
        "l6_key": "H-signal" if l6_positive else "H-noise",
        "l6_positive_arms_both_seeds": l6_positive,
        "cells": {
            str(seed): {
                arm: {
                    "mean": float(cell["mean"]),
                    "ci_low": float(cell["ci_low"]),
                    "ci_high": float(cell["ci_high"]),
                }
                for arm, cell in sorted(seed_cells.items())
            }
            for seed, seed_cells in cells.items()
        },
        "cluster_extension": {
            str(seed): summaries[seed]["extension"] for seed in (0, 1)
        },
        "cross_seed_assignment_agreement": assignment_agreement,
        "interpretive_boundary": {
            "assignment_feature": "oracle-derived unit negative-CE correction gradient",
            "l1_l3_oracle_routed": True,
            "l2_deployable_fixed_target": True,
            "cluster_semantics": "battery-family association measured; abstract-mode claim prohibited",
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
