"""P3.4 task-estimator guardrail and oracle-collateral calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import beta as beta_distribution

from eval.eval_paper2_phase3_guardrail_recalibration import calibrate_panel
from training.paper2_phase3_p34 import P34_GATE_CEILINGS


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def task_noise_model(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_rows: int = 1_024,
    bootstrap_draws: int = 2_000,
    seed: int = 20260812,
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("P3.4 task calibration requires scored DEV rows")
    if any(str(row.get("partition", "")).casefold() != "dev" for row in records):
        raise RuntimeError("P3.4 task calibration received a non-DEV row")
    if any(str(row.get("battery_role", "")).casefold() in {"confirm", "eval_e"} for row in records):
        raise RuntimeError("P3.4 task calibration touched a sealed role")
    conditions = sorted(
        {(int(row["seed"]), int(row["look"])) for row in records}
    )
    vectors: dict[tuple[int, int], np.ndarray] = {}
    item_order: list[str] | None = None
    for condition in conditions:
        selected = sorted(
            [
                row
                for row in records
                if (int(row["seed"]), int(row["look"])) == condition
            ],
            key=lambda row: str(row["item_id"]),
        )
        ids = [str(row["item_id"]) for row in selected]
        if len(ids) != expected_rows or len(ids) != len(set(ids)):
            raise RuntimeError(
                f"P3.4 task panel changed at condition {condition}: {len(ids)}"
            )
        if item_order is None:
            item_order = ids
        elif ids != item_order:
            raise RuntimeError("P3.4 task panel item order differs between conditions")
        vectors[condition] = np.asarray(
                [int(bool(row["augmented_correct"])) - int(bool(row["base_correct"])) for row in selected],
                dtype=np.float64,
            )
    matrix = np.stack([vectors[condition] for condition in conditions])
    discordance = float(np.mean(np.abs(matrix)))
    adjacent_pairs = []
    for source_seed in sorted({condition[0] for condition in conditions}):
        seed_conditions = [condition for condition in conditions if condition[0] == source_seed]
        adjacent_pairs.extend(zip(seed_conditions[:-1], seed_conditions[1:]))
    correlations: list[float] = []
    for left_condition, right_condition in adjacent_pairs:
        left = vectors[left_condition]
        right = vectors[right_condition]
        if float(left.std()) > 0.0 and float(right.std()) > 0.0:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    if adjacent_pairs and not correlations:
        raise RuntimeError("P3.4 task trajectory cannot identify adjacent-look autocorrelation")
    correlation = float(np.median(correlations)) if correlations else 0.0
    correlation = min(0.999, max(-0.999, correlation))
    if correlation < 0.0:
        raise RuntimeError("negative P3.4 task autocorrelation requires strategy review")

    if bootstrap_draws <= 0:
        raise ValueError("P3.4 task correlation bootstrap requires positive draws")
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _draw in range(bootstrap_draws):
        indexes = rng.integers(0, expected_rows, size=expected_rows)
        draw_correlations = []
        for left_condition, right_condition in adjacent_pairs:
            left = vectors[left_condition][indexes]
            right = vectors[right_condition][indexes]
            if float(left.std()) > 0.0 and float(right.std()) > 0.0:
                draw_correlations.append(float(np.corrcoef(left, right)[0, 1]))
        if draw_correlations:
            bootstrap.append(float(np.median(draw_correlations)))
    if not bootstrap:
        raise RuntimeError("P3.4 task correlation bootstrap produced no estimates")
    upper = min(0.999, max(correlation, float(np.quantile(bootstrap, 0.95))))
    return {
        "rows_per_look": expected_rows,
        "conditions": [
            {"seed": source_seed, "checkpoint_index": look}
            for source_seed, look in conditions
        ],
        "source_condition_count": len(conditions),
        "source_seeds": sorted({condition[0] for condition in conditions}),
        "paired_discordance": discordance,
        "adjacent_checkpoint_autocorrelation": correlation,
        "adjacent_correlations": correlations,
        "autocorrelation_bootstrap_draws": bootstrap_draws,
        "autocorrelation_bootstrap_upper_95": upper,
        "autocorrelation_sensitivity_band": [correlation, upper],
        "mean_augmented_minus_base": [
            float(vectors[condition].mean()) for condition in conditions
        ],
        "estimator": "paired augmented-minus-base task correctness on fixed DEV items",
    }


def task_guardrail_receipt(
    *,
    rows: Iterable[Mapping[str, Any]],
    campaigns: int,
    seed: int,
    campaign_looks: int,
    bootstrap_draws: int = 2_000,
) -> dict[str, Any]:
    noise = task_noise_model(rows, bootstrap_draws=bootstrap_draws, seed=seed)
    if noise["source_condition_count"] != 6 or noise["source_seeds"] != [0, 1]:
        raise RuntimeError("P3.4 guardrail calibration requires three checkpoints for each seed")
    if campaign_looks != 20:
        raise RuntimeError("P3.4 wave-1 calibration certificate requires exactly 20 looks")
    discordance = max(float(noise["paired_discordance"]), 0.005)
    max_effect = min(discordance, 0.30)
    effect_grid = [round(value, 3) for value in np.arange(0.005, max_effect + 0.0001, 0.005)]
    if not effect_grid:
        effect_grid = [0.005]
    sensitivity = []
    for index, correlation in enumerate(noise["autocorrelation_sensitivity_band"]):
        sensitivity.append(
            {
                "correlation_role": "empirical" if index == 0 else "bootstrap_upper_95",
                "correlation": float(correlation),
                "result": calibrate_panel(
                    rows=1_024,
                    looks=campaign_looks,
                    discordance=discordance,
                    correlation=float(correlation),
                    campaigns=campaigns,
                    seed=seed + index * 10_000_000,
                    alphas=[0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
                    effect_grid=effect_grid,
                ),
            }
        )
    conservative = sensitivity[-1]
    return {
        "kind": "paper2_phase3_p34_task_guardrail_calibration_v1",
        "status": "complete_dev_task_estimator",
        "noise_model": noise,
        "campaign_looks": campaign_looks,
        "sensitivity_band": sensitivity,
        "conservative_edge": conservative,
        "tier_s_contract": {
            "familywise_false_stop_ceiling": 1e-4,
            "power_floor": 0.99,
            "two_consecutive_looks": True,
        },
        "tier_w_contract": {
            "drop_class": -0.03,
            "two_consecutive_looks": True,
            "consequence": "demote one controller rung and flag strategy review",
        },
        "tier_e_evaluated": False,
        "sealed_partitions_scored": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def clopper_pearson_upper(*, flips: int, rows: int, confidence: float = 0.95) -> float:
    if rows <= 0 or flips < 0 or flips > rows:
        raise ValueError("invalid collateral binomial counts")
    if flips == rows:
        return 1.0
    return float(beta_distribution.ppf(confidence, flips + 1, rows - flips))


def chi_receipt(
    summaries: Iterable[Mapping[str, Any]], *, margin: float
) -> dict[str, Any]:
    if not 0.0 <= float(margin) <= 1.0:
        raise ValueError("P3.4 chi margin must be explicitly bound as a probability")
    records = sorted([dict(row) for row in summaries], key=lambda row: float(row["gate_ceiling"]))
    ceilings = tuple(float(row["gate_ceiling"]) for row in records)
    if ceilings != P34_GATE_CEILINGS:
        raise RuntimeError(f"P3.4 chi calibration requires all registered ceilings: {ceilings}")
    rows = []
    for record in records:
        count = int(record["rows"])
        flips = int(record["collateral_flips"])
        upper = clopper_pearson_upper(flips=flips, rows=count)
        rows.append(
            {
                "gate_ceiling": float(record["gate_ceiling"]),
                "rows": count,
                "collateral_flips": flips,
                "collateral_rate": flips / count,
                "clopper_pearson_upper_95": upper,
                "upper_plus_strategy_margin": min(1.0, upper + float(margin)),
                "estimator": str(record["estimator"]),
            }
        )
    chi_max = max(row["upper_plus_strategy_margin"] for row in rows)
    return {
        "kind": "paper2_phase3_p34_chi_max_calibration_v1",
        "status": "complete_bound_margin",
        "rows": rows,
        "strategy_margin": float(margin),
        "chi_max": chi_max,
        "controller_estimator": "oracle-write collateral on the audit population at matched gate ceiling",
        "estimator_same_clause": len({row["estimator"] for row in rows}) == 1,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--rows", type=Path, required=True)
    task.add_argument("--output", type=Path, required=True)
    task.add_argument("--campaigns", type=int, default=100_000)
    task.add_argument("--seed", type=int, default=20260812)
    task.add_argument("--campaign_looks", type=int, required=True)
    task.add_argument("--bootstrap_draws", type=int, default=2_000)
    chi = subparsers.add_parser("chi")
    chi.add_argument("--summaries", type=Path, required=True)
    chi.add_argument("--output", type=Path, required=True)
    chi.add_argument("--margin", type=float, required=True)
    args = parser.parse_args()
    if args.command == "task":
        result = task_guardrail_receipt(
            rows=read_jsonl(args.rows),
            campaigns=args.campaigns,
            seed=args.seed,
            campaign_looks=args.campaign_looks,
            bootstrap_draws=args.bootstrap_draws,
        )
    else:
        result = chi_receipt(json.loads(args.summaries.read_text(encoding="utf-8")), margin=args.margin)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
