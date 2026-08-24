"""Analyze paired W1 generative cells without treating seeds as row replicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PAIRS = {
    "l0a_vs_shuffle": ("l0a", "l5_a"),
    "l0c_vs_shuffle": ("l0c", "l5_c"),
    "l1_vs_shuffle": ("l1", "l5_l1"),
    "l2_vs_shuffle": ("l2", "l5_l2"),
    "l3_vs_shuffle": ("l3", "l5_l3"),
    "l0a_vs_random": ("l0a", "l4"),
    "l0c_vs_random": ("l0c", "l4"),
    "l1_vs_random": ("l1", "l4"),
    "l2_vs_random": ("l2", "l4"),
    "l3_vs_random": ("l3", "l4"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def exact_mcnemar_p(fixes: int, regressions: int) -> float:
    discordant = fixes + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(fixes, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_summary(
    parent: list[dict[str, Any]],
    control: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    draws: int = 20_000,
) -> dict[str, Any]:
    parent_by_id = {str(row["item_id"]): row for row in parent}
    control_by_id = {str(row["item_id"]): row for row in control}
    if set(parent_by_id) != set(control_by_id):
        raise RuntimeError("Paired generation cells do not contain identical item ids")
    ids = sorted(parent_by_id)
    parent_correct = np.asarray(
        [bool(parent_by_id[item_id]["augmented_correct"]) for item_id in ids], dtype=np.int8
    )
    control_correct = np.asarray(
        [bool(control_by_id[item_id]["augmented_correct"]) for item_id in ids], dtype=np.int8
    )
    difference = parent_correct - control_correct
    fixes = int(np.sum(difference == 1))
    regressions = int(np.sum(difference == -1))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = difference[rng.integers(0, len(ids), size=(draws, len(ids)))].mean(axis=1)
    summary: dict[str, Any] = {
        "rows": len(ids),
        "parent_correct": int(parent_correct.sum()),
        "control_correct": int(control_correct.sum()),
        "fixes": fixes,
        "regressions": regressions,
        "unchanged": int(np.sum(difference == 0)),
        "net_rows": fixes - regressions,
        "paired_difference": float(difference.mean()),
        "paired_difference_ci_low": float(np.quantile(bootstrap, 0.025)),
        "paired_difference_ci_high": float(np.quantile(bootstrap, 0.975)),
        "mcnemar_exact_p": exact_mcnemar_p(fixes, regressions),
        "prediction_identity_rows": sum(
            str(parent_by_id[item_id].get("prediction"))
            == str(control_by_id[item_id].get("prediction"))
            for item_id in ids
        ),
    }
    by_battery: dict[str, Any] = {}
    batteries = sorted({str(parent_by_id[item_id]["battery"]) for item_id in ids})
    for battery in batteries:
        mask = np.asarray(
            [str(parent_by_id[item_id]["battery"]) == battery for item_id in ids], dtype=bool
        )
        local = difference[mask]
        local_fixes = int(np.sum(local == 1))
        local_regressions = int(np.sum(local == -1))
        by_battery[battery] = {
            "rows": int(mask.sum()),
            "parent_correct": int(parent_correct[mask].sum()),
            "control_correct": int(control_correct[mask].sum()),
            "fixes": local_fixes,
            "regressions": local_regressions,
            "net_rows": local_fixes - local_regressions,
            "paired_difference": float(local.mean()),
            "mcnemar_exact_p": exact_mcnemar_p(local_fixes, local_regressions),
        }
    summary["by_battery"] = by_battery
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    private = args.input_dir / "private"
    receipts = args.input_dir / "receipts"
    output: dict[str, Any] = {
        "kind": "paper2_bicameral_w1_generation_analysis_v1",
        "source_bundle_manifest_sha256": sha256_file(
            args.input_dir / "generation_bundle_manifest.json"
        ),
        "pairs": {},
        "cross_seed": {},
        "interpretive_scope": {
            "l0a_l0c": "oracle-target-assisted; causal same-row contrasts only",
            "l1_l3": "population target values with oracle-derived row routing; not deployable-grade",
            "l2": "fixed global population target; deployable-grade",
            "l4": "random-direction control",
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }

    loaded: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for seed in (0, 1):
        summary = json.loads(
            (receipts / f"seed_{seed}_generation_summary.json").read_text(encoding="utf-8")
        )
        if summary["status"] != "complete_score_only":
            raise RuntimeError(f"Seed {seed} generation summary is incomplete")
        for arm in {arm for pair in PAIRS.values() for arm in pair}:
            loaded[(seed, arm)] = read_jsonl(private / f"seed_{seed}_generation_{arm}.jsonl")
        output["pairs"][str(seed)] = {}
        for index, (name, (parent_arm, control_arm)) in enumerate(PAIRS.items()):
            output["pairs"][str(seed)][name] = {
                "parent_arm": parent_arm,
                "control_arm": control_arm,
                **paired_summary(
                    loaded[(seed, parent_arm)],
                    loaded[(seed, control_arm)],
                    bootstrap_seed=20260824 + 100 * seed + index,
                ),
            }

    for arm in sorted({arm for pair in PAIRS.values() for arm in pair}):
        seed0 = {str(row["item_id"]): row for row in loaded[(0, arm)]}
        seed1 = {str(row["item_id"]): row for row in loaded[(1, arm)]}
        ids = sorted(seed0)
        if ids != sorted(seed1):
            raise RuntimeError(f"Cross-seed item ids differ for arm {arm}")
        output["cross_seed"][arm] = {
            "rows": len(ids),
            "correctness_identity_rows": sum(
                bool(seed0[item_id]["augmented_correct"])
                == bool(seed1[item_id]["augmented_correct"])
                for item_id in ids
            ),
            "prediction_identity_rows": sum(
                str(seed0[item_id].get("prediction")) == str(seed1[item_id].get("prediction"))
                for item_id in ids
            ),
        }

    output["registered_read"] = {
        "margin_branch": "GLOBAL-STEER-WITH-GLOBAL-ADVANTAGE_REQUIRES_STRATEGY_ADJUDICATION",
        "l6_read": "H-noise",
        "generation_read": "TARGETS-NOT-ANSWER-GRADE",
        "basis": (
            "L0a proves a same-row oracle target can causally alter answers, while the only "
            "deployable fixed population target (L2) is prediction-identical to its control "
            "in both seeds. Oracle-routed L1 harms and oracle-routed L3 provides only a small "
            "shuffle contrast while remaining below the random-direction control."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
