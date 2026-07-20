"""Score the parameter-matched layerwise oracle-control localization probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.oracle_intrablock_control_spec import (  # noqa: E402
    score_oracle_intrablock_control,
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm_summary", required=True)
    parser.add_argument("--single_entry_control_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    arm = read_json(args.arm_summary)
    control = read_json(args.single_entry_control_summary)
    if arm.get("status") != "finished":
        raise AssertionError("Layerwise oracle arm must finish before scoring")
    if control.get("route") != "film" or control.get("status") != "finished":
        raise AssertionError("Historical control must be the finished single-entry FiLM arm")
    result = score_oracle_intrablock_control(arm)
    result["arm_summary"] = args.arm_summary
    result["single_entry_control_summary"] = args.single_entry_control_summary
    result["single_entry_control"] = {
        "passed": bool(control["passed"]),
        "checks": control["checks"],
    }
    Path(args.output_json).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checks = arm["checks"]
    lines = [
        "# Phase G Distributed Oracle-Control Localization",
        "",
        f"- Measured reading: `{result['measured_reading']}`",
        f"- Interpretation: `{result['interpretation']}`",
        "- Automatic successor authorized: `False`",
        "",
        "| Route | Non-default control | Overall control | Legality | Terminal validity | Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, payload in (
        ("single-entry FiLM", control),
        ("shared layerwise FiLM", arm),
    ):
        arm_checks = payload["checks"]
        lines.append(
            f"| {label} | "
            f"{arm_checks['nondefault_branch_control']['observed']:.4f} | "
            f"{arm_checks['overall_transition_control']['observed']:.4f} | "
            f"{arm_checks['transition_legality']['observed']:.4f} | "
            f"{arm_checks['terminal_validity']['observed']:.4f} | "
            f"{payload['passed']} |"
        )
    lines.extend(
        [
            "",
            "The conditioner parameterization, rows, optimizer, dose, and gates "
            "match the historical FiLM control. The intervention changes only "
            "command access from once before the recurrent block to once before "
            "each recurrent layer.",
            "",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
