"""Apply the locked cross-arm decision rule to matched-alpha DEV pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from training.paper2_phase2_matched_alpha import (
    PROTOCOL_LOCK_COMMIT,
    paired_bootstrap_interval,
    practical_equivalence,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _alpha_key(value: float) -> str:
    return f"{float(value):g}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decide(summary: dict[str, Any], *, bootstrap_draws: int = 10_000) -> dict[str, Any]:
    if summary.get("protocol_lock_commit") != PROTOCOL_LOCK_COMMIT:
        raise RuntimeError("pilot summary does not carry the locked protocol commit")
    by_alpha: dict[str, list[dict[str, Any]]] = {}
    for arm in summary["arms"]:
        by_alpha.setdefault(_alpha_key(float(arm["alpha"])), []).append(arm)
    if not {"0", "0.5", "1"}.issubset(by_alpha):
        raise RuntimeError("decision requires the three locked alpha arms")

    pooled: dict[str, dict[str, Any]] = {}
    pooled_rows: dict[str, torch.Tensor] = {}
    for alpha, arms in sorted(by_alpha.items(), key=lambda item: float(item[0])):
        arms = sorted(arms, key=lambda row: int(row["seed"]))
        if len(arms) != 2 or [int(row["seed"]) for row in arms] != [0, 1]:
            raise RuntimeError(f"alpha {alpha} does not have the locked seed pair")
        row_values = []
        for arm in arms:
            receipt = arm["final_rows"]
            path = Path(receipt["path"])
            if _sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"final-row receipt hash mismatch: {path}")
            rows = torch.load(path, map_location="cpu", weights_only=False)
            row_values.append(rows["accepted_length"].float())
        if row_values[0].shape != row_values[1].shape:
            raise RuntimeError(f"seed row alignment failed for alpha {alpha}")
        pooled_rows[alpha] = torch.stack(row_values).mean(0)
        seed_means = [float(value.mean()) for value in row_values]
        valid = all(
            arm["status"] == "complete" and bool(arm["final"]["quality_noninferior"])
            for arm in arms
        )
        pooled[alpha] = {
            "valid": valid,
            "seed_means": seed_means,
            "seed_spread": abs(seed_means[0] - seed_means[1]),
            "mean_accepted_length": float(pooled_rows[alpha].mean()),
            "mean_flow_validation_loss": sum(
                float(arm["final"]["flow_validation_loss"]) for arm in arms
            ) / 2.0,
            "mean_gradient_cv": sum(
                sum(float(value) for value in arm["gradient_atlases"][-1]["module_gradient_norm_cv"].values()) / 3.0
                for arm in arms
            ) / 2.0,
            "mean_clip_fraction": sum(
                sum(float(value) for value in arm["clip_events"].values()) / 3.0
                for arm in arms
            ) / 2.0,
        }

    comparisons: dict[str, dict[str, Any]] = {}
    reference = pooled_rows["0.5"]
    reference_mean = float(reference.mean())
    for alpha in sorted((value for value in pooled if value != "0.5"), key=float):
        difference = pooled_rows[alpha] - reference
        mean, low, high = paired_bootstrap_interval(
            difference, seed=20260804 + int(float(alpha) * 100), draws=bootstrap_draws
        )
        band_equivalent = practical_equivalence(
            difference_ci=(low, high), reference_mean=reference_mean, relative_band=0.02
        )
        noise_equivalent = max(pooled[alpha]["seed_spread"], pooled["0.5"]["seed_spread"]) >= abs(mean)
        comparisons[f"{alpha}_vs_0.5"] = {
            "mean_difference": mean,
            "paired_bootstrap_95_interval": [low, high],
            "relative_equivalence_band": [-0.02 * abs(reference_mean), 0.02 * abs(reference_mean)],
            "interval_inside_band": band_equivalent,
            "between_seed_spread_exceeds_arm_difference": noise_equivalent,
            "equivalent": band_equivalent or noise_equivalent,
        }

    valid = [alpha for alpha, row in pooled.items() if row["valid"]]
    result: dict[str, Any] = {
        "kind": "paper2_phase2_matched_alpha_decision",
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "adequacy_precondition_met": bool(summary["adequacy_precondition_met"]),
        "pooled": pooled,
        "comparisons": comparisons,
        "selected_alpha": None,
        "refinement_required": None,
        "status": "no_selection",
    }
    if not summary["adequacy_precondition_met"]:
        if any(arm["status"] != "complete" for arm in summary["arms"]):
            result["reason"] = "one_or_more_arms_aborted_before_adequacy"
        elif summary.get("extended_once"):
            result["reason"] = "adequacy_failed_after_registered_extension"
        else:
            result["reason"] = "adequacy_failed_before_registered_extension"
        return result
    if "0.5" not in valid:
        result["reason"] = "registered_default_failed_quality_or_assertions"
        return result
    best = max(valid, key=lambda alpha: (
        pooled[alpha]["mean_accepted_length"],
        -pooled[alpha]["mean_flow_validation_loss"],
        -pooled[alpha]["mean_gradient_cv"],
        -pooled[alpha]["mean_clip_fraction"],
    ))
    refinement_present = "0.25" in pooled or "0.75" in pooled
    if (
        not refinement_present
        and best in {"0", "1"}
        and not comparisons[f"{best}_vs_0.5"]["equivalent"]
    ):
        result["status"] = "refinement_required"
        result["refinement_required"] = 0.25 if best == "0" else 0.75
        result["reason"] = f"alpha_{best}_beats_0.5_outside_equivalence"
        return result
    result["status"] = "selected_dev_configuration"
    result["selected_alpha"] = (
        0.5
        if comparisons.get(f"{best}_vs_0.5", {}).get("equivalent", best == "0.5")
        else float(best)
    )
    result["reason"] = "alpha_0.5_wins_registered_equivalence" if result["selected_alpha"] == 0.5 else "ranked_verified_acceptance"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    result = decide(payload, bootstrap_draws=args.bootstrap_draws)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
