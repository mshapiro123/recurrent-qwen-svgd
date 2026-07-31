"""Compute the locked DC1 Stage A verdict from one immutable row cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, int | float]:
    scored = sum(int(row["scored_positions"]) for row in rows)
    helps = sum(int(row["arms"][arm]["helps"]) for row in rows)
    hurts = sum(int(row["arms"][arm]["hurts"]) for row in rows)
    if scored <= 0:
        raise ValueError("immutable cache has no scored positions")
    return {
        "scored_positions": scored,
        "helps": helps,
        "hurts": hurts,
        "net": helps - hurts,
        "u": (helps - hurts) / scored,
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]], *, arm: str, replicates: int, seed: int, ci: float
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("row-cluster bootstrap requires at least one row")
    rng = random.Random(seed)
    values = np.empty(replicates, dtype=np.float64)
    row_count = len(rows)
    for replicate in range(replicates):
        sampled = [rows[rng.randrange(row_count)] for _ in range(row_count)]
        values[replicate] = float(aggregate(sampled, arm)["u"])
    alpha = (1.0 - ci) / 2.0
    return {
        "cluster": "eval_c_source_row",
        "rows": row_count,
        "replicates": replicates,
        "seed": seed,
        "ci": ci,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def compute_verdict(
    rows: list[dict[str, Any]], prereg: dict[str, Any]
) -> dict[str, Any]:
    trained = aggregate(rows, "trained_append_k1")
    untrained = aggregate(rows, "untrained_append_k1")
    bootstrap_policy = prereg["evaluation"]["bootstrap"]
    bootstrap = cluster_bootstrap(
        rows,
        arm="trained_append_k1",
        replicates=int(bootstrap_policy["replicates"]),
        seed=int(bootstrap_policy["seed"]),
        ci=float(bootstrap_policy["ci"]),
    )
    hurts_ratio = (
        float(trained["hurts"]) / float(untrained["hurts"])
        if int(untrained["hurts"]) > 0
        else (0.0 if int(trained["hurts"]) == 0 else float("inf"))
    )
    helps_ratio = (
        float(trained["helps"]) / float(untrained["helps"])
        if int(untrained["helps"]) > 0
        else (float("inf") if int(trained["helps"]) > 0 else 1.0)
    )
    qualifies_policy = prereg["bands"]["qualifies"]
    partial_policy = prereg["bands"]["partial_domestication"]
    qualifies = (
        float(trained["u"]) >= float(qualifies_policy["u_point_min"])
        and float(bootstrap["lower"])
        >= float(qualifies_policy["u_ci_lower_min"])
    )
    partial = (
        hurts_ratio <= float(partial_policy["hurts_vs_untrained_max_ratio"])
        and helps_ratio >= float(partial_policy["helps_vs_untrained_min_ratio"])
        and float(trained["u"])
        < float(partial_policy["u_point_max_exclusive"])
    )
    hurts_reduction = 1.0 - hurts_ratio
    no_material_threshold = prereg["bands"]["no_material_improvement"]
    no_material = hurts_reduction < float(
        no_material_threshold["hurts_reduction_below"]
    )
    if qualifies:
        verdict = "qualifies"
        consequence = prereg["consequences"]["qualifies"]
    elif partial:
        verdict = "partial_domestication"
        consequence = prereg["consequences"]["partial"]
    else:
        verdict = "none"
        consequence = prereg["consequences"]["none"]
    return {
        "kind": "paper2_dc1_stage_a_registered_verdict",
        "verdict": verdict,
        "consequence": consequence,
        "trained_append": trained,
        "untrained_append": untrained,
        "trained_vs_untrained": {
            "hurts_ratio": hurts_ratio,
            "hurts_reduction": hurts_reduction,
            "helps_ratio": helps_ratio,
        },
        "bootstrap": bootstrap,
        "criteria": {
            "qualifies": qualifies,
            "partial_domestication": partial,
            "no_material_improvement_threshold_met": no_material,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--immutable_cache", required=True)
    parser.add_argument("--expected_cache_sha256", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--output_summary", required=True)
    args = parser.parse_args()

    actual_hash = sha256_file(args.immutable_cache)
    if actual_hash != args.expected_cache_sha256:
        raise RuntimeError("immutable EVAL-C scoring cache SHA-256 mismatch")
    prereg = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    if prereg.get("locked_before_training") is not True:
        raise RuntimeError("Stage A preregistration is not locked")
    rows = read_jsonl(args.immutable_cache)
    result = compute_verdict(rows, prereg)
    result["immutable_cache_sha256"] = actual_hash
    result["prereg_sha256"] = sha256_file(args.prereg)
    output = Path(args.output_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
