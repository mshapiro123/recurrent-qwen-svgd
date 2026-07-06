"""Publish the Phase-A surpass comparison preregistration.

This is intentionally CPU/lightweight.  It does not train the dense control
arms; it freezes the comparison definition so those arms can be run without
moving the goalposts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, write_json
from colab.stage5_phase_a_surpass import phase_a_preregistration


def write_markdown(run_dir: Path, payload: dict) -> None:
    prereg = payload["preregistration"]
    lines = [
        f"# Phase-A Surpass Preregistration - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Frozen eval set: `{prereg['train_distribution']['eval_frozen_set']}`",
        f"- Primary gate: {prereg['surpass_gate']['primary']}",
        "",
        "## Arms",
    ]
    for key, value in prereg["arms"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Compute Ledger",
            f"- Looped arm: {prereg['compute_ledger']['looped_arm_context_growth']}",
            f"- Scratchpad arm: {prereg['compute_ledger']['scratchpad_arm_context_growth']}",
            f"- Policy: {prereg['compute_ledger']['flop_claim_policy']}",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_PHASE_A_PLAN_RUN_ID") or time.strftime(
        "stage5_phase_a_surpass_prereg_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "stage5_phase_a_surpass_preregistration",
        "run_id": run_id,
        "status": "preregistered",
        "preregistration": phase_a_preregistration(),
        "implementation_status": {
            "A_looped": "available from support-8 dose arm",
            "B_dense_direct": "pending dense synthetic direct SFT runner",
            "C_dense_scratchpad": "pending dense serialized-orbit scratchpad SFT runner",
            "D_dense_1p5b_direct_optional": "pending optional larger dense exchange-rate arm",
        },
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Preregister Stage 5 Phase-A surpass comparison {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
