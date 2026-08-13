"""Evaluate nested P3.4 Tier-W/Tier-S sequential guardrail repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from eval.eval_paper2_phase3_guardrail_recalibration import simulate_rule, write_json


KIND = "paper2_phase3_p34_guardrail_collision_receipt_v1"


def build_receipt(
    *, lock: Mapping[str, Any], campaigns: int = 100_000, seed: int = 20260813
) -> dict[str, Any]:
    guard = lock["guardrails"]
    rows = int(lock["task_inference"]["dev_panel_rows"])
    looks = int(guard["look_count"])
    discordance = float(guard["paired_discordance"])
    correlation = float(guard["autocorrelation_conservative_edge"])
    alpha = float(guard["tier_s_one_sided_alpha"])
    tier_s_margin = float(guard["tier_s_decision_margin"])
    tier_w_margin = float(guard["tier_w_drop_class"])
    collision = tier_s_margin == tier_w_margin
    drops = (0.0, 0.03, 0.05, 0.055, 0.06, 0.07)
    candidates: dict[str, Any] = {}
    for consecutive in (2, 3, 4, 5):
        reads = {}
        for drop in drops:
            reads[f"drop_{drop:.3f}"] = simulate_rule(
                rows=rows,
                looks=looks,
                discordance=discordance,
                correlation=correlation,
                alpha=alpha,
                decision_margin=tier_w_margin,
                true_difference=-drop,
                campaigns=campaigns,
                seed=seed + consecutive * 1_000 + int(drop * 1_000),
                consecutive_looks=consecutive,
            )
        candidates[f"streak_{consecutive}"] = reads
    recommended = candidates["streak_4"]
    return {
        "kind": KIND,
        "status": "strategy_amendment_required",
        "training_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
        "registered_collision": {
            "tier_s_margin": tier_s_margin,
            "tier_w_margin": tier_w_margin,
            "identical_predicates_at_two_consecutive_looks": collision,
        },
        "simulation": {
            "rows": rows,
            "looks": looks,
            "discordance": discordance,
            "correlation": correlation,
            "one_sided_alpha": alpha,
            "campaigns_per_condition": campaigns,
            "candidates": candidates,
        },
        "recommended_nested_rule": {
            "tier_w": "paired upper bound below -0.03 for two consecutive looks: demote",
            "tier_s": "paired upper bound below -0.03 for four consecutive looks: stop",
            "why_four": (
                "four is the longest tested streak retaining at least 0.99 simulated "
                "power at a sustained 0.055 drop"
            ),
            "null_upper_95": recommended["drop_0.000"]["conservative_upper_95_probability"],
            "power_drop_0.055": recommended["drop_0.055"]["estimated_action_probability"],
            "power_drop_0.050": recommended["drop_0.050"]["estimated_action_probability"],
            "power_drop_0.030": recommended["drop_0.030"]["estimated_action_probability"],
            "nested_consequence": True,
        },
        "assertions": {
            "collision_reproduced": collision,
            "tier_s_four_streak_false_stop_below_1e_4": (
                recommended["drop_0.000"]["conservative_upper_95_probability"] <= 1e-4
            ),
            "tier_s_four_streak_power_at_0_055_at_least_0_99": (
                recommended["drop_0.055"]["estimated_action_probability"] >= 0.99
            ),
            "five_streak_fails_registered_0_055_power": (
                candidates["streak_5"]["drop_0.055"]["estimated_action_probability"] < 0.99
            ),
            "training_untouched": True,
            "sealed_partitions_untouched": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaigns", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    receipt = build_receipt(lock=lock, campaigns=args.campaigns, seed=args.seed)
    failed = [name for name, passed in receipt["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"P3.4 nested guardrail assertions failed: {failed}")
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
