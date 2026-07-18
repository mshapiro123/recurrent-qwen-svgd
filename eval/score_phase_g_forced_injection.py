"""Score both Phase G forced-injection arms against the locked causal gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.phase_g_forced_injection_spec import score_forced_injection_probe


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm_summaries", nargs=2, required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--run_summary")
    parser.add_argument("--runtime_status_json")
    args = parser.parse_args()

    arms = [read_json(path) for path in args.arm_summaries]
    if any(arm.get("status") != "finished" for arm in arms):
        raise AssertionError("Both forced-injection arm summaries must be finished")
    if any(not arm.get("factor_1_exact_equivalence") for arm in arms):
        raise AssertionError("Both arms must pass factor-1 equivalence before scoring")
    if any(not arm.get("frozen_lineage_unchanged") for arm in arms):
        raise AssertionError("Both arms must preserve the frozen keeper lineage")
    result = score_forced_injection_probe(arms)
    result["arm_summaries"] = list(args.arm_summaries)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase G Forced-Injection Causal Probe",
        "",
        f"- Measured verdict: `{result['measured_verdict']}`",
        f"- Authorization: `{result['authorization']}`",
        f"- Next step: `{result['next_step']}`",
        "",
        "| Arm | Factor | Switching groups | Target fidelity | K=1 validity |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in result["points"]:
        lines.append(
            f"| {point['arm']} | {point['factor']:g} | "
            f"{point['switching_groups']}/32 | "
            f"{point['selected_target_fidelity']:.4f} | "
            f"{point['K1_validity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This is an evaluation-only magnitude intervention on the preserved A0 "
            "posterior checkpoints. It does not authorize coverage, selection, halting, "
            "or particle experiments unless the locked CHANNEL-EXISTS gate passes.",
            "",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    if args.run_summary:
        run_summary = read_json(args.run_summary)
        run_summary["gate"] = result
        run_summary["status"] = result["status"]
        run_summary["training_performed"] = False
        run_summary["coverage_performed"] = False
        Path(args.run_summary).write_text(
            json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.runtime_status_json:
        runtime_status = {
            "kind": "stage5_phase_g_forced_injection_runtime_status",
            "stage": "forced_injection_gate",
            "status": result["status"],
            "measured_verdict": result["measured_verdict"],
            "authorization": result["authorization"],
        }
        runtime_path = Path(args.runtime_status_json)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            json.dumps(runtime_status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["measured_verdict"] == "CHANNEL-EXISTS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
