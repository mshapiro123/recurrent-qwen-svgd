"""Extract the registered GSM8K-only W2-prime FS-1/D1 rider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.paper2_tm0 import atomic_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_d_summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.phase_d_summary.read_text(encoding="utf-8"))
    seeds = {}
    for seed in ("0", "1"):
        fit = source["seeds"][seed]["targets"]["l0a"]["feature_sets"]["fs1_md"]["fit"]
        conditional = fit["conditional_cosine"]
        seeds[seed] = {
            "pooled": conditional["pooled"],
            "gsm8k": conditional["per_battery"]["gsm8k"],
            "gsm8k_minus_pooled": (
                conditional["per_battery"]["gsm8k"]["mean"]
                - conditional["pooled"]["mean"]
            ),
        }
    receipt = {
        "kind": "paper2_tm0_w2p_r1_gsm8k_granularity_v1",
        "source_sha256": sha256_file(args.phase_d_summary),
        "estimator": "existing_FS1_D1_fit_read_no_refit",
        "seeds": seeds,
        "interpretation": (
            "WEAK_DEPLOYABLE_SIGNAL_SURVIVES_WITHIN_GSM8K"
            if all(row["gsm8k"]["ci_low"] > 0.0 for row in seeds.values())
            else "BATTERY_IDENTITY_CONFOUNDED"
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
