"""Bind Phase G posterior-control thresholds to an exact frozen row manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.phase_g_multitarget_spec import build_posterior_control_gate_lock


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_multi_target_groups", type=int, required=True)
    parser.add_argument("--min_teacher_target_rate", type=float, required=True)
    parser.add_argument("--min_teacher_prior_target_lift", type=float, required=True)
    parser.add_argument("--min_teacher_switching_groups", type=int, required=True)
    parser.add_argument("--max_teacher_prior_target_lift_p_value", type=float, required=True)
    args = parser.parse_args()
    lock = build_posterior_control_gate_lock(
        read_jsonl(args.control_jsonl),
        {
            "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS": args.min_multi_target_groups,
            "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE": args.min_teacher_target_rate,
            "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT": args.min_teacher_prior_target_lift,
            "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS": args.min_teacher_switching_groups,
            "STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE": args.max_teacher_prior_target_lift_p_value,
        },
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
