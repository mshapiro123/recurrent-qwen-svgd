"""Build the score-blind P3.1 split ledger and false-stop calibration receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.paper2_phase3_p31 import (
    PairedNullDesign,
    build_split_ledger,
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
) -> dict[str, Any]:
    receipts = []
    for row_count in sorted(set(rows)):
        design = PairedNullDesign(
            rows=row_count,
            looks=looks,
            discordant_probability=discordant_probability,
            adjacent_correlation=adjacent_correlation,
        )
        for alpha in sorted(set(alphas), reverse=True):
            receipts.append(
                simulate_false_stop_probability(
                    design,
                    alpha=alpha,
                    campaigns=campaigns,
                    seed=seed + row_count * 10_000 + round(alpha * 1e9),
                )
            )
    passing = [receipt for receipt in receipts if receipt["target_met_by_conservative_upper"]]
    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda receipt: (receipt["rows"], -receipt["one_sided_alpha"]),
        )[0]
    return {
        "kind": "paper2_phase3_p31_false_stop_grid_v1",
        "status": "candidate_selected" if selected else "no_candidate_meets_target",
        "selection_rule": "minimum rows, then largest one-sided alpha meeting conservative upper target",
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
    parser.add_argument("--candidate_rows", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--candidate_alphas", type=float, nargs="+", default=[0.001, 0.0005, 0.0001]
    )
    parser.add_argument("--looks", type=int, default=20)
    parser.add_argument("--campaigns", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--discordant_probability", type=float, default=0.20)
    parser.add_argument("--adjacent_correlation", type=float, default=0.80)
    args = parser.parse_args()

    ledger = build_split_ledger(
        read_jsonl(args.rows_jsonl),
        dataset_revisions=json.loads(args.dataset_revisions_json.read_text(encoding="utf-8")),
        reader_versions=json.loads(args.reader_versions_json.read_text(encoding="utf-8")),
    )
    simulation = calibrate_false_stop(
        rows=args.candidate_rows,
        alphas=args.candidate_alphas,
        looks=args.looks,
        campaigns=args.campaigns,
        seed=args.seed,
        discordant_probability=args.discordant_probability,
        adjacent_correlation=args.adjacent_correlation,
    )
    write_json(args.output_ledger, ledger)
    write_json(args.output_simulation, simulation)
    print(json.dumps({"ledger": ledger, "simulation": simulation}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
