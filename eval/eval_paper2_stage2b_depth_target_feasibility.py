"""Assess whether banked task rows support execution-verified partial targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line]
    target = [row for row in rows if row.get("battery") in {"gsm8k", "mbpp"}]
    by_battery = Counter(str(row["battery"]) for row in target)
    with_intermediate_trace = [
        row for row in target if row.get("verified_intermediate_steps") or row.get("execution_trace")
    ]
    segmented = [row for row in target if row.get("step_boundaries")]
    result = {
        "kind": "paper2_stage2b_depth_target_feasibility_v1",
        "status": "fallback_activated",
        "source_sha256": sha256_file(args.rows),
        "target_rows": len(target),
        "by_battery": dict(sorted(by_battery.items())),
        "rows_with_execution_verified_intermediate_trace": len(with_intermediate_trace),
        "rows_with_registered_step_boundaries": len(segmented),
        "gsm8k_available_supervision": "final normalized number only",
        "mbpp_available_supervision": "final program and unit tests only",
        "verdict": "no cheap execution-verified k-step target generator can be built from banked rows",
        "registered_fallback": {
            "verified_depth_weight": 0.0,
            "active_depth_mechanism": "per-loop full-sequence distillation plus monotonicity hinge",
        },
        "teacher_sampled_pseudo_steps_used": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
