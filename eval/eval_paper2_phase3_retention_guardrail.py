"""Calibrate P3.3 token-retention rules on the exact e2 estimator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t


REQUESTED_DROPS = (0.005, 0.010, 0.020)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _upper_95(successes: int, campaigns: int) -> float:
    if successes == campaigns:
        return 1.0
    return float(beta_distribution.ppf(0.95, successes + 1, campaigns - successes))


def sequential_pair_statistic(
    *, campaigns: int, looks: int, correlation: float, seed: int
) -> np.ndarray:
    """Return min_t max(z_t,z_t+1) for correlated standard-normal looks."""

    if campaigns <= 0 or looks < 2:
        raise ValueError("retention calibration requires campaigns > 0 and looks >= 2")
    if not -1.0 < correlation < 1.0:
        raise ValueError("adjacent-look correlation must be inside (-1, 1)")
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(campaigns)
    previous = latent.copy()
    pair_min = np.full(campaigns, np.inf)
    innovation_scale = math.sqrt(1.0 - correlation**2)
    for _ in range(1, looks):
        latent = correlation * latent + innovation_scale * rng.standard_normal(campaigns)
        pair_min = np.minimum(pair_min, np.maximum(previous, latent))
        previous = latent
    return pair_min


def evaluate_rule(
    pair_statistic: np.ndarray,
    *,
    rows: int,
    discordance: float,
    alpha: float,
    decision_margin: float,
    true_drop: float,
) -> dict[str, Any]:
    if rows <= 1 or not 0.0 < discordance <= 1.0:
        raise ValueError("retention calibration needs rows > 1 and positive discordance")
    standard_error = math.sqrt(discordance / rows)
    critical = float(student_t.ppf(1.0 - alpha, df=rows - 1))
    threshold = (decision_margin + true_drop) / standard_error - critical
    actions = int(np.count_nonzero(pair_statistic < threshold))
    campaigns = int(pair_statistic.shape[0])
    return {
        "rows": rows,
        "one_sided_alpha": alpha,
        "decision_margin_relative_to_init": decision_margin,
        "sustained_drop": -true_drop,
        "campaigns": campaigns,
        "campaigns_with_action": actions,
        "estimated_action_probability": actions / campaigns,
        "conservative_upper_95_probability": _upper_95(actions, campaigns),
        "consecutive_looks": 2,
        "paired_null_discordance": discordance,
        "standard_error": standard_error,
        "critical_value": critical,
        "simulation_method": "correlated_gaussian_paired_mean_exact_consecutive_pair_statistic",
    }


def _step0_noise(
    rows: Sequence[Mapping[str, Any]], prior: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    if set(by_seed) != {0, 1}:
        raise ValueError("retention step-zero receipt must contain seeds 0 and 1")
    counts = {seed: len(values) for seed, values in by_seed.items()}
    if set(counts.values()) != {1024}:
        raise ValueError(f"retention step-zero rows changed: {counts}")
    by_seed_discordance = {
        str(seed): float(np.mean([not bool(row["retained"]) for row in values]))
        for seed, values in by_seed.items()
    }
    prior_candidate = prior["binding_noise_model_candidate"]
    prior_discordance = float(prior_candidate["paired_discordant_probability"])
    # The exact panel supplies estimator-specific step-zero discordance. The banked
    # token-level trajectory remains the conservative floor and supplies look-to-look
    # dependence, which cannot be estimated before training creates multiple looks.
    discordance = max(*by_seed_discordance.values(), prior_discordance, 1.0 / 1024.0)
    return discordance, {
        "exact_panel_step0_nonretention_by_seed": by_seed_discordance,
        "prior_token_trajectory_discordance": prior_discordance,
        "selected_discordance": discordance,
        "selection_rule": "maximum exact-panel seed rate, prior token-trajectory rate, and 1/n",
        "hybrid_model_disclosed": True,
    }


def calibrate(
    *,
    step0_rows: Sequence[Mapping[str, Any]],
    prior_empirical_summary: Mapping[str, Any],
    panel_sha256: str,
    campaigns: int = 100_000,
    looks: int = 20,
    seed: int = 20260811,
    tier_s_fwer: float = 1e-4,
    tier_w_null_ceiling: float = 0.005,
) -> dict[str, Any]:
    discordance, noise = _step0_noise(step0_rows, prior_empirical_summary)
    prior_candidate = prior_empirical_summary["binding_noise_model_candidate"]
    correlation = float(prior_candidate["adjacent_checkpoint_autocorrelation"])
    pair_statistic = sequential_pair_statistic(
        campaigns=campaigns, looks=looks, correlation=correlation, seed=seed
    )
    alphas = (0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001)
    margins = tuple(-value for value in np.arange(0.0, 0.0105, 0.0005))
    effect_grid = tuple(float(value) for value in np.arange(0.0005, 0.0505, 0.0005))

    candidates = []
    for alpha in alphas:
        for margin in margins:
            null = evaluate_rule(
                pair_statistic,
                rows=1024,
                discordance=discordance,
                alpha=alpha,
                decision_margin=margin,
                true_drop=0.0,
            )
            requested = {
                f"drop_{drop:.3f}": evaluate_rule(
                    pair_statistic,
                    rows=1024,
                    discordance=discordance,
                    alpha=alpha,
                    decision_margin=margin,
                    true_drop=drop,
                )
                for drop in REQUESTED_DROPS
            }
            delta_cat = None
            delta_cat_power = None
            for drop in effect_grid:
                result = evaluate_rule(
                    pair_statistic,
                    rows=1024,
                    discordance=discordance,
                    alpha=alpha,
                    decision_margin=margin,
                    true_drop=drop,
                )
                if result["estimated_action_probability"] >= 0.99:
                    delta_cat = drop
                    delta_cat_power = result
                    break
            candidates.append(
                {
                    "one_sided_alpha": alpha,
                    "decision_margin_relative_to_init": margin,
                    "null": null,
                    "requested_power": requested,
                    "delta_cat": delta_cat,
                    "delta_cat_power": delta_cat_power,
                }
            )

    tier_s_eligible = [
        row
        for row in candidates
        if row["delta_cat"] is not None
        and row["null"]["conservative_upper_95_probability"] <= tier_s_fwer
    ]
    if not tier_s_eligible:
        raise RuntimeError("no Tier-S retention rule meets the registered false-stop target")
    tier_s = min(
        tier_s_eligible,
        key=lambda row: (
            float(row["delta_cat"]),
            -row["requested_power"]["drop_0.005"]["estimated_action_probability"],
            -float(row["one_sided_alpha"]),
            -float(row["decision_margin_relative_to_init"]),
        ),
    )
    tier_w_eligible = [
        row
        for row in candidates
        if row["null"]["conservative_upper_95_probability"] <= tier_w_null_ceiling
    ]
    if not tier_w_eligible:
        raise RuntimeError("no Tier-W retention rule stays in the prior null-warning class")
    tier_w = max(
        tier_w_eligible,
        key=lambda row: (
            row["requested_power"]["drop_0.005"]["estimated_action_probability"],
            row["requested_power"]["drop_0.010"]["estimated_action_probability"],
            float(row["one_sided_alpha"]),
            float(row["decision_margin_relative_to_init"]),
        ),
    )
    receipt = {
        "kind": "paper2_phase3_p33_token_retention_guardrail_recalibration_v1",
        "status": "complete_before_optimizer_construction",
        "panel_rows": 1024,
        "panel_sha256": panel_sha256,
        "looks": looks,
        "campaigns": campaigns,
        "seed": seed,
        "estimand": "fraction of positions where augmented top1 matches frozen base top1",
        "threshold_reference": "step0_init_relative",
        "noise_model": {
            **noise,
            "adjacent_checkpoint_autocorrelation": correlation,
            "autocorrelation_source": "banked Option B token-level DEV checkpoint trajectory",
        },
        "tier_s": {
            **tier_s,
            "familywise_false_stop_target": tier_s_fwer,
            "power_target": 0.99,
            "consecutive_looks": 2,
        },
        "tier_w": {
            **tier_w,
            "null_warning_ceiling": tier_w_null_ceiling,
            "consecutive_looks": 2,
        },
        "superseded_task_scale_thresholds": {
            "delta_cat_minus_8p5_points": "void_for_p33",
            "tier_w_0p441_percent": "void_for_p33_except_as_null_warning_class",
        },
        "task_level_capability_scoring": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    receipt["assertions"] = {
        "exact_panel_1024_both_seeds": len(step0_rows) == 2048,
        "twenty_looks": looks == 20,
        "tier_s_fwer_met": tier_s["null"]["conservative_upper_95_probability"] <= tier_s_fwer,
        "tier_s_power_met_at_delta_cat": (
            tier_s["delta_cat"] is not None
            and tier_s["delta_cat_power"] is not None
            and tier_s["delta_cat_power"]["estimated_action_probability"] >= 0.99
        ),
        "tier_w_null_class_met": tier_w["null"]["conservative_upper_95_probability"] <= tier_w_null_ceiling,
        "requested_powers_reported_both_tiers": all(
            set(receipt[tier]["requested_power"])
            == {"drop_0.005", "drop_0.010", "drop_0.020"}
            for tier in ("tier_s", "tier_w")
        ),
        "task_scoring_absent": True,
        "optimizer_absent": True,
        "training_steps_zero": True,
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0_rows", type=Path, required=True)
    parser.add_argument("--prior_empirical_summary", type=Path, required=True)
    parser.add_argument("--panel_sha256", required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--campaigns", type=int, default=100_000)
    parser.add_argument("--looks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.step0_rows.read_text(encoding="utf-8").splitlines()
        if line
    ]
    prior = json.loads(args.prior_empirical_summary.read_text(encoding="utf-8"))
    receipt = calibrate(
        step0_rows=rows,
        prior_empirical_summary=prior,
        panel_sha256=args.panel_sha256,
        campaigns=args.campaigns,
        looks=args.looks,
        seed=args.seed,
    )
    failed = [name for name, passed in receipt["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"retention guardrail calibration assertions failed: {failed}")
    write_json(args.output_summary, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
