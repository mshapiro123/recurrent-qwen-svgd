"""Run the preregistered seed-26 plateau dose.

Seed 26 failed to clear the support-6 canonical frontier after the first
2,000-step rescue dose.  This script applies exactly one additional fixed dose
from that checkpoint and classifies the result as UNIFIED, PLATEAU, or
AMBIGUOUS under the locked countdown-plan rule.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.run_stage5_support6_replication_receipts import score_payload  # noqa: E402
from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json  # noqa: E402


DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/stage5_support6_dosed_seed_resolution_20260708_004504_seed_20260726_dose2000/summary.json"
)
UNIFIED_FRONTIER = 8.0
PLATEAU_MIN_GAIN = 0.3


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def plateau_outcome(*, pre_frontier: float, post_frontier: float) -> dict[str, Any]:
    gain = float(post_frontier) - float(pre_frontier)
    if float(post_frontier) >= UNIFIED_FRONTIER:
        status = "seed26_unified"
        decision = "UNIFIED: seed 26 cleared the canonical frontier after the final fixed dose."
    elif gain < PLATEAU_MIN_GAIN:
        status = "seed26_plateau"
        decision = "PLATEAU: seed 26 stayed below the frontier and gained less than the locked minimum."
    else:
        status = "seed26_ambiguous"
        decision = "AMBIGUOUS: seed 26 stayed below the frontier but still gained materially."
    return {
        "unified_frontier": UNIFIED_FRONTIER,
        "plateau_min_gain": PLATEAU_MIN_GAIN,
        "pre_frontier": float(pre_frontier),
        "post_frontier": float(post_frontier),
        "gain": gain,
        "status": status,
        "decision": decision,
    }


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    outcome = payload.get("plateau_outcome", {})
    lines = [
        f"# Support-6 Seed-26 Plateau Test - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Extra steps: `{payload['extra_steps']}`",
        f"- Pre frontier: `{outcome.get('pre_frontier')}`",
        f"- Post frontier: `{outcome.get('post_frontier')}`",
        f"- Gain: `{outcome.get('gain')}`",
        f"- Decision: {outcome.get('decision')}",
        f"- Child summary: `{payload.get('child_summary')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_SEED26_PLATEAU_RUN_ID") or time.strftime(
        "stage5_support6_seed26_plateau_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_summary = os.environ.get("STAGE5_SEED26_PLATEAU_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
    extra_steps = int(os.environ.get("STAGE5_SEED26_PLATEAU_STEPS", "2000"))
    seed = os.environ.get("STAGE5_SEED26_PLATEAU_SEED", "20260726")
    source_payload = read_json(source_summary)
    pre_score = score_payload(source_payload)
    pre_frontier = float(pre_score["canonical_frontier"])
    child_id = f"{run_id}_dose{extra_steps}"

    payload: dict[str, Any] = {
        "kind": "stage5_support6_seed26_plateau_test",
        "run_id": run_id,
        "status": "started",
        "source_summary": source_summary,
        "source_score": pre_score,
        "extra_steps": extra_steps,
        "seed": seed,
        "child_run_id": child_id,
    }
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)

    run(
        [sys.executable, "colab/run_stage5_depth_support_route_comparison.py"],
        env={
            "STAGE5_ROUTE_RUN_ID": child_id,
            "STAGE5_ROUTE_INIT_CHECKPOINT": source_summary,
            "STAGE5_ROUTE_TRAIN_SEED": seed,
            "STAGE5_ROUTE_TOTAL_STEPS": str(extra_steps),
            "STAGE5_ROUTE_TRAIN_MAX_DEPTH": "6",
            "STAGE5_ROUTE_EVAL_MAX_DEPTH": "10",
            "STAGE5_ROUTE_ROWS_PER_DEPTH": os.environ.get("STAGE5_SEED26_PLATEAU_ROWS_PER_DEPTH", "256"),
            "STAGE5_ROUTE_FROZEN_ROWS_PER_DEPTH": "128",
            "STAGE5_ROUTE_FROZEN_EVAL_ID": "stage5_synthetic_depth_frozen_eval_v1",
            "STAGE5_ROUTE_BACKUP_CHECKPOINTS_TO_DRIVE": os.environ.get(
                "STAGE5_SEED26_PLATEAU_BACKUP_CHECKPOINTS_TO_DRIVE", "1"
            ),
            "STAGE5_ROUTE_DTYPE": os.environ.get("STAGE5_SEED26_PLATEAU_DTYPE", "bfloat16"),
        },
    )

    child_summary = ROOT / "outputs" / "stage5" / child_id / "summary.json"
    child_payload = read_json(child_summary)
    post_score = score_payload(child_payload)
    outcome = plateau_outcome(pre_frontier=pre_frontier, post_frontier=float(post_score["canonical_frontier"]))
    payload.update(
        {
            "status": outcome["status"],
            "child_summary": path_for_cli(child_summary),
            "post_score": post_score,
            "plateau_outcome": outcome,
            "decision": outcome["decision"],
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 support-6 seed26 plateau {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
