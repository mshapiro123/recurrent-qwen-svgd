"""Calibrate Phase 3 Tier-S and Tier-W rules on empirical DEV noise."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def paired_discordance(reference_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    floor = [
        row
        for row in reference_rows
        if row["partition"] == "dev" and row["battery_role"] == "floor_retention_only"
    ]
    if not floor:
        raise ValueError("guardrail calibration requires floor/retention DEV rows")
    by_family = {}
    for battery in sorted({str(row["battery"]) for row in floor}):
        selected = [row for row in floor if row["battery"] == battery]
        by_family[battery] = {
            "rows": len(selected),
            "base_teacher_discordance": float(
                np.mean(
                    [
                        bool(row["base_correct"]) != bool(row["teacher_14b_correct"])
                        for row in selected
                    ]
                )
            ),
        }
    return {
        "rows": len(floor),
        "pooled_base_teacher_discordance": float(
            np.mean(
                [
                    bool(row["base_correct"]) != bool(row["teacher_14b_correct"])
                    for row in floor
                ]
            )
        ),
        "by_family": by_family,
    }


def simulate_rule(
    *,
    rows: int,
    looks: int,
    discordance: float,
    correlation: float,
    alpha: float,
    decision_margin: float,
    true_difference: float,
    campaigns: int,
    seed: int,
    batch_campaigns: int = 1_024,
    consecutive_looks: int = 2,
) -> dict[str, Any]:
    """Simulate consecutive paired-UCB events from sufficient statistics.

    The empirical inputs are paired discordance and adjacent-look correlation.
    At the registered panel sizes, the paired mean is simulated with its
    correlated Gaussian limit instead of materializing ``campaigns * rows``
    ternary observations at every look.  This preserves the paired-mean
    variance and sequential dependence while making the 100k-campaign panel
    sensitivity practical on a CPU runtime.
    """

    if consecutive_looks < 2 or consecutive_looks > looks:
        raise ValueError("consecutive_looks must be between two and the campaign look count")
    if abs(true_difference) > discordance:
        raise ValueError("true difference cannot exceed paired discordance")
    paired_variance = discordance - true_difference**2
    if paired_variance <= 0.0:
        raise ValueError("paired variance must be positive")
    standard_error = math.sqrt(paired_variance / rows)
    critical = float(student_t.ppf(1.0 - alpha, df=rows - 1))
    rng = np.random.default_rng(seed)
    campaigns_with_action = 0
    action_count = 0
    completed = 0
    scale = math.sqrt(1.0 - correlation**2)
    while completed < campaigns:
        batch = min(batch_campaigns, campaigns - completed)
        latent = rng.standard_normal(batch)
        streak = np.zeros(batch, dtype=np.int16)
        actions = np.zeros(batch, dtype=np.int32)
        for look in range(looks):
            if look:
                latent = correlation * latent + scale * rng.standard_normal(batch)
            mean = true_difference + standard_error * latent
            below = mean + critical * standard_error < decision_margin
            streak = np.where(below, streak + 1, 0)
            action = streak >= consecutive_looks
            actions += action
            # An action consumes its run so one long breach cannot double count it.
            streak = np.where(action, 0, streak)
        campaigns_with_action += int((actions > 0).sum())
        action_count += int(actions.sum())
        completed += batch
    probability = campaigns_with_action / campaigns
    upper_95 = (
        1.0
        if campaigns_with_action == campaigns
        else float(
            beta_distribution.ppf(
                0.95,
                campaigns_with_action + 1,
                campaigns - campaigns_with_action,
            )
        )
    )
    return {
        "rows": rows,
        "looks": looks,
        "discordance": discordance,
        "adjacent_checkpoint_autocorrelation": correlation,
        "one_sided_alpha": alpha,
        "decision_margin": decision_margin,
        "true_mean_difference": true_difference,
        "campaigns": campaigns,
        "campaigns_with_action": campaigns_with_action,
        "estimated_action_probability": probability,
        "conservative_upper_95_probability": upper_95,
        "actions_total": action_count,
        "expected_actions_per_campaign": action_count / campaigns,
        "consecutive_looks": consecutive_looks,
        "simulation_method": "correlated_gaussian_paired_mean_sufficient_statistic",
        "paired_variance": paired_variance,
        "standard_error": standard_error,
    }


def calibrate_panel(
    *,
    rows: int,
    looks: int,
    discordance: float,
    correlation: float,
    campaigns: int,
    seed: int,
    alphas: list[float],
    effect_grid: list[float],
) -> dict[str, Any]:
    nulls = []
    for index, alpha in enumerate(sorted(set(alphas), reverse=True)):
        result = simulate_rule(
            rows=rows,
            looks=looks,
            discordance=discordance,
            correlation=correlation,
            alpha=alpha,
            decision_margin=-0.03,
            true_difference=0.0,
            campaigns=campaigns,
            seed=seed + rows * 100_000 + index,
        )
        result["familywise_false_stop_target_met"] = (
            result["conservative_upper_95_probability"] <= 1e-4
        )
        nulls.append(result)
    passing = [row for row in nulls if row["familywise_false_stop_target_met"]]
    if not passing:
        raise RuntimeError(f"no Tier-S alpha meets false-stop target at n={rows}")
    selected_null = max(passing, key=lambda row: row["one_sided_alpha"])
    alpha = float(selected_null["one_sided_alpha"])

    powers = {}
    delta_cat = None
    for index, drop in enumerate(sorted(set(effect_grid))):
        if drop > discordance + 1e-12:
            continue
        result = simulate_rule(
            rows=rows,
            looks=looks,
            discordance=discordance,
            correlation=correlation,
            alpha=alpha,
            decision_margin=-0.03,
            true_difference=-drop,
            campaigns=campaigns,
            seed=seed + rows * 1_000_000 + index + 100,
        )
        powers[f"drop_{drop:.3f}"] = result
        if delta_cat is None and result["estimated_action_probability"] >= 0.99:
            delta_cat = drop

    tier_w = {
        "null": simulate_rule(
            rows=rows,
            looks=looks,
            discordance=discordance,
            correlation=correlation,
            alpha=0.10,
            decision_margin=-0.03,
            true_difference=0.0,
            campaigns=campaigns,
            seed=seed + rows * 2_000_000,
        )
    }
    for offset, drop in enumerate((0.03, 0.05), start=1):
        if drop <= discordance + 1e-12:
            tier_w[f"power_drop_{drop:.2f}"] = simulate_rule(
                rows=rows,
                looks=looks,
                discordance=discordance,
                correlation=correlation,
                alpha=0.10,
                decision_margin=-0.03,
                true_difference=-drop,
                campaigns=campaigns,
                seed=seed + rows * 2_000_000 + offset,
            )
    requested = {}
    for drop in (0.03, 0.05):
        key = f"drop_{drop:.3f}"
        requested[key] = powers.get(key)
    if delta_cat is not None:
        requested["delta_cat"] = powers[f"drop_{delta_cat:.3f}"]
    return {
        "rows": rows,
        "tier_s": {
            "decision_margin": -0.03,
            "delta_cat_definition": (
                "smallest sustained true drop detected with >=0.99 probability; "
                "the paired-UCB decision margin remains -0.03"
            ),
            "delta_cat": delta_cat,
            "selected_null": selected_null,
            "requested_power": requested,
            "alpha_candidates": nulls,
            "all_power_candidates": powers,
        },
        "tier_w": {
            "rule": "one-sided 90% paired upper bound below -0.03 twice consecutively",
            **tier_w,
        },
    }


def build_receipt(
    *,
    reference_rows: list[Mapping[str, Any]],
    prior_empirical_summary: Mapping[str, Any],
    panel_sizes: list[int],
    campaigns: int,
    seed: int,
    looks: int,
) -> dict[str, Any]:
    discordance_receipt = paired_discordance(reference_rows)
    prior = prior_empirical_summary["binding_noise_model_candidate"]
    # Base/teacher discordance is a lower-cost proxy for the new battery; the
    # larger prior paired discordance is retained conservatively.
    discordance = max(
        float(discordance_receipt["pooled_base_teacher_discordance"]),
        float(prior["paired_discordant_probability"]),
        0.05,
    )
    correlation = float(prior["adjacent_checkpoint_autocorrelation"])
    max_effect = min(discordance, 0.30)
    effect_grid = [round(value, 3) for value in np.arange(0.03, max_effect + 0.0001, 0.005)]
    alphas = [0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    panels = [
        calibrate_panel(
            rows=rows,
            looks=looks,
            discordance=discordance,
            correlation=correlation,
            campaigns=campaigns,
            seed=seed,
            alphas=alphas,
            effect_grid=effect_grid,
        )
        for rows in sorted(set(panel_sizes))
    ]
    return {
        "kind": "paper2_phase3_three_tier_guardrail_recalibration_v1",
        "status": "complete_empirical_dev_hybrid_noise_model",
        "seed": seed,
        "campaigns": campaigns,
        "looks": looks,
        "noise_model": {
            "paired_discordance_source": "P3.1 floor-retention DEV base-versus-14B rows",
            "checkpoint_autocorrelation_source": "banked Option B DEV checkpoint trajectory",
            "hybrid_model_disclosed": True,
            "discordance": discordance,
            "adjacent_checkpoint_autocorrelation": correlation,
            "p31_reference_discordance": discordance_receipt,
            "prior_empirical_noise_model": prior,
        },
        "panel_sensitivity": panels,
        "tier_e": {
            "rule": "macro and micro pooled CONFIRM deltas must both be non-negative",
            "evaluated_now": False,
            "confirm_scoring_spent": False,
        },
        "assertions": {
            "panel_sizes_256_512_1024": [row["rows"] for row in panels] == [256, 512, 1024],
            "tier_s_false_stop_target_met_all_panels": all(
                row["tier_s"]["selected_null"]["familywise_false_stop_target_met"]
                for row in panels
            ),
            "tier_w_reports_expected_false_demotions": all(
                "expected_actions_per_campaign" in row["tier_w"]["null"] for row in panels
            ),
            "confirm_unscored": True,
            "optimizer_absent": True,
            "training_steps_zero": True,
        },
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_rows", type=Path, required=True)
    parser.add_argument("--prior_empirical_summary", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--panel_sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--campaigns", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--looks", type=int, default=20)
    args = parser.parse_args()
    reference = [
        json.loads(line)
        for line in args.reference_rows.read_text(encoding="utf-8").splitlines()
        if line
    ]
    prior = json.loads(args.prior_empirical_summary.read_text(encoding="utf-8"))
    result = build_receipt(
        reference_rows=reference,
        prior_empirical_summary=prior,
        panel_sizes=args.panel_sizes,
        campaigns=args.campaigns,
        seed=args.seed,
        looks=args.looks,
    )
    failed = [name for name, passed in result["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"guardrail recalibration assertions failed: {failed}")
    write_json(args.output_summary, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
