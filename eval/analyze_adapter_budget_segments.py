"""Post-hoc depth-segment localization for the adapter-budget Arm E result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.adapter_budget_arm import paired_binary_test


DEFAULT_SEGMENTS = {
    "trained_support_d1_8": tuple(range(1, 9)),
    "near_extrapolation_d9_11": tuple(range(9, 12)),
    "far_extrapolation_d12_14": tuple(range(12, 15)),
    "reported_tail_d11_14": tuple(range(11, 15)),
    "all_d1_14": tuple(range(1, 15)),
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def hit(row: dict[str, Any]) -> bool:
    for key in ("same_reader_final_hit", "hit", "correct"):
        if key in row:
            return bool(row[key])
    raise ValueError(f"Row has no recognized correctness field: {row.get('id')}")


def analyze_segments(
    arm_a_rows: list[dict[str, Any]],
    arm_e_rows: list[dict[str, Any]],
    *,
    segments: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    left = {str(row["id"]): row for row in arm_a_rows}
    right = {str(row["id"]): row for row in arm_e_rows}
    if len(left) != len(arm_a_rows) or len(right) != len(arm_e_rows):
        raise ValueError("Arm A and Arm E rows must each have unique IDs")
    if set(left) != set(right):
        raise ValueError("Arm A and Arm E must contain identical row IDs")

    output: dict[str, Any] = {}
    for name, depths in (segments or DEFAULT_SEGMENTS).items():
        depth_set = set(depths)
        row_ids = sorted(
            row_id for row_id, row in left.items() if int(row["depth"]) in depth_set
        )
        if not row_ids:
            raise ValueError(f"Segment {name} selected no rows")
        for row_id in row_ids:
            if int(left[row_id]["depth"]) != int(right[row_id]["depth"]):
                raise ValueError(f"Depth mismatch for {row_id}")

        arm_a_hits = [hit(left[row_id]) for row_id in row_ids]
        arm_e_hits = [hit(right[row_id]) for row_id in row_ids]
        paired = paired_binary_test(arm_e_hits, arm_a_hits)
        total = len(row_ids)
        arm_a_correct = sum(arm_a_hits)
        arm_e_correct = sum(arm_e_hits)
        output[name] = {
            "depths": list(depths),
            "rows": total,
            "arm_a": {
                "correct": arm_a_correct,
                "total": total,
                "accuracy": arm_a_correct / total,
            },
            "arm_e": {
                "correct": arm_e_correct,
                "total": total,
                "accuracy": arm_e_correct / total,
            },
            "accuracy_delta": (arm_e_correct - arm_a_correct) / total,
            "paired": paired,
        }

    return {
        "kind": "adapter_budget_posthoc_depth_segment_localization",
        "analysis_status": "post_hoc_localization_not_preregistered_gate",
        "segments": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm_a_rows", required=True)
    parser.add_argument("--arm_e_rows", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    payload = analyze_segments(read_jsonl(args.arm_a_rows), read_jsonl(args.arm_e_rows))
    payload["arm_a_rows"] = args.arm_a_rows
    payload["arm_e_rows"] = args.arm_e_rows
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
