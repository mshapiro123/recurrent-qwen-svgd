"""Build the score-blind P3.1 split ledger and false-stop calibration receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.paper2_phase3_p31 import (
    PairedNullDesign,
    build_split_ledger,
    estimate_empirical_paired_design,
    simulate_false_stop_probability,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def calibrate_false_stop(
    *,
    rows: list[int],
    alphas: list[float],
    looks: int,
    campaigns: int,
    seed: int,
    discordant_probability: float,
    adjacent_correlation: float,
    power_drops: list[float] | None = None,
) -> dict[str, Any]:
    if power_drops is None:
        power_drops = [0.03, 0.05]
    receipts = []
    for row_count in sorted(set(rows)):
        design = PairedNullDesign(
            rows=row_count,
            looks=looks,
            discordant_probability=discordant_probability,
            adjacent_correlation=adjacent_correlation,
        )
        for alpha in sorted(set(alphas), reverse=True):
            candidate_seed = seed + row_count * 10_000 + round(alpha * 1e9)
            false_stop = simulate_false_stop_probability(
                design,
                alpha=alpha,
                campaigns=campaigns,
                seed=candidate_seed,
                true_mean_difference=0.0,
            )
            power = {
                f"drop_{round(drop * 100):d}_points": simulate_false_stop_probability(
                    PairedNullDesign(
                        rows=row_count,
                        looks=looks,
                        discordant_probability=max(discordant_probability, float(drop)),
                        adjacent_correlation=adjacent_correlation,
                    ),
                    alpha=alpha,
                    campaigns=campaigns,
                    seed=candidate_seed + index + 1,
                    true_mean_difference=-float(drop),
                )
                for index, drop in enumerate(power_drops)
            }
            receipts.append(
                {
                    "rows": row_count,
                    "one_sided_alpha": alpha,
                    "false_stop": false_stop,
                    "power": power,
                }
            )
    passing = [
        receipt
        for receipt in receipts
        if receipt["false_stop"]["target_met_by_conservative_upper"]
    ]
    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda receipt: (receipt["rows"], -receipt["one_sided_alpha"]),
        )[0]
    return {
        "kind": "paper2_phase3_p31_false_stop_and_power_grid_v2",
        "status": "candidate_selected" if selected else "no_candidate_meets_target",
        "selection_rule": "minimum rows, then largest one-sided alpha meeting conservative upper target",
        "selection_uses_power": False,
        "power_effects": [f"-{drop:.3f}" for drop in power_drops],
        "power_gate": None,
        "selected": selected,
        "candidates": receipts,
        "scores_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_jsonl", type=Path, required=True)
    parser.add_argument("--dataset_revisions_json", type=Path, required=True)
    parser.add_argument("--reader_versions_json", type=Path, required=True)
    parser.add_argument("--output_ledger", type=Path, required=True)
    parser.add_argument("--output_simulation", type=Path, required=True)
    parser.add_argument("--candidate_rows", type=int, nargs="+", default=[512])
    parser.add_argument(
        "--candidate_alphas",
        type=float,
        nargs="+",
        default=[0.00005],
    )
    parser.add_argument("--looks", type=int, default=20)
    parser.add_argument("--campaigns", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--discordant_probability", type=float, default=0.20)
    parser.add_argument("--adjacent_correlation", type=float, default=0.80)
    parser.add_argument("--power_drops", type=float, nargs="+", default=[0.03, 0.05])
    parser.add_argument("--empirical_differences_json", type=Path)
    args = parser.parse_args()

    ledger = build_split_ledger(
        read_jsonl(args.rows_jsonl),
        dataset_revisions=json.loads(args.dataset_revisions_json.read_text(encoding="utf-8")),
        reader_versions=json.loads(args.reader_versions_json.read_text(encoding="utf-8")),
    )
    empirical = None
    discordance = args.discordant_probability
    correlation = args.adjacent_correlation
    if args.empirical_differences_json is not None:
        empirical = estimate_empirical_paired_design(
            json.loads(args.empirical_differences_json.read_text(encoding="utf-8"))
        )
        discordance = empirical["paired_discordant_probability"]
        correlation = empirical["adjacent_checkpoint_autocorrelation"]
    simulation = calibrate_false_stop(
        rows=args.candidate_rows,
        alphas=args.candidate_alphas,
        looks=args.looks,
        campaigns=args.campaigns,
        seed=args.seed,
        discordant_probability=discordance,
        adjacent_correlation=correlation,
        power_drops=args.power_drops,
    )
    simulation["calibration_status"] = (
        "binding_empirical_dev_noise_model" if empirical is not None else "planning_forecast_only"
    )
    simulation["empirical_noise_estimate"] = empirical
    write_json(args.output_ledger, ledger)
    write_json(args.output_simulation, simulation)
    print(json.dumps({"ledger": ledger, "simulation": simulation}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
