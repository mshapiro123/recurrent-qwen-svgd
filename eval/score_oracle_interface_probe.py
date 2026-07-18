"""Combine the additive and FiLM oracle-interface arms into a terminal verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.oracle_interface_probe_spec import score_oracle_interface_probe  # noqa: E402


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm_summaries", nargs=2, required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    arms = [read_json(path) for path in args.arm_summaries]
    if any(arm.get("status") != "finished" for arm in arms):
        raise AssertionError("Both oracle interface arms must finish before scoring")
    result = score_oracle_interface_probe(arms)
    result["arm_summaries"] = list(args.arm_summaries)

    Path(args.output_json).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase G Oracle Re-entry Interface Probe",
        "",
        f"- Measured reading: `{result['measured_reading']}`",
        f"- Interpretation: `{result['interpretation']}`",
        "- Automatic successor authorized: `False`",
        "",
        "| Route | Non-default control | Overall control | Legality | Terminal validity | Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in ("additive", "film"):
        arm = result["arms"][route]
        checks = arm["checks"]
        lines.append(
            f"| {route} | "
            f"{checks['nondefault_branch_control']['observed']:.4f} | "
            f"{checks['overall_transition_control']['observed']:.4f} | "
            f"{checks['transition_legality']['observed']:.4f} | "
            f"{checks['terminal_validity']['observed']:.4f} | "
            f"{arm['passed']} |"
        )
    lines.extend(
        [
            "",
            "This is a terminal oracle-capacity probe. It contains no variational "
            "training and cannot directly authorize coverage, selection, halting, "
            "particle, or SVGD experiments.",
            "",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
