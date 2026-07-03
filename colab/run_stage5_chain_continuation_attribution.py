"""Run the chain-only continuation attribution ablation.

This is the control for the post-anneal extrapolation jump: start from the same
pre-anneal corrected-chain checkpoint, spend the same continuation budget, keep
per-loop chain labels active, then evaluate the same held-out 5-8 protocol.
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

from colab.stage5_chain_consolidation_utils import ROOT, path_for_cli, publish_run, read_json, write_json
from colab.run_stage5_post_anneal_readouts import compact_extrapolation


def run(cmd: list[str | os.PathLike[str]], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    extrap = payload.get("attribution_extrapolation") or {}
    lines = [
        f"# Chain-Continuation Attribution - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Continuation loss mode: `{payload.get('loop_loss_mode')}`",
        f"- Source checkpoint: `{payload.get('init_checkpoint')}`",
        f"- Final active diagonal through train horizon: `{payload.get('final_eval', {}).get('active', {}).get('active_diagonal')}`",
        f"- Extrapolation active diagonal: `{extrap.get('active_diagonal')}`",
        f"- Extrapolation read: `{extrap.get('extrapolation_read')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_CHAIN_CONTINUATION_RUN_ID") or time.strftime(
        "stage5_chain_continuation_attribution_%Y%m%d_%H%M%S"
    )
    parent_summary = ROOT / "outputs" / "stage5" / run_id / "summary.json"
    run(
        [sys.executable, "colab/run_stage5_chain_anneal_to_outcome.py"],
        env={
            "STAGE5_ANNEAL_RUN_ID": run_id,
            "STAGE5_ANNEAL_LOOP_LOSS_MODE": "per_loop_labels",
            "STAGE5_ANNEAL_INIT_CHECKPOINT": os.environ.get(
                "STAGE5_CHAIN_CONTINUATION_INIT_CHECKPOINT",
                "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            ),
            "STAGE5_ANNEAL_N_SYMBOLS": os.environ.get("STAGE5_CHAIN_CONTINUATION_N_SYMBOLS", "16"),
            "STAGE5_ANNEAL_MAX_DEPTH": os.environ.get("STAGE5_CHAIN_CONTINUATION_MAX_DEPTH", "4"),
            "STAGE5_ANNEAL_ROWS_PER_DEPTH": os.environ.get("STAGE5_CHAIN_CONTINUATION_ROWS_PER_DEPTH", "256"),
            "STAGE5_ANNEAL_HELDOUT_ROWS_PER_DEPTH": os.environ.get(
                "STAGE5_CHAIN_CONTINUATION_HELDOUT_ROWS_PER_DEPTH", "64"
            ),
            "STAGE5_ANNEAL_TOTAL_STEPS": os.environ.get("STAGE5_CHAIN_CONTINUATION_TOTAL_STEPS", "2000"),
            "STAGE5_ANNEAL_HOLD_FRAC": os.environ.get("STAGE5_CHAIN_CONTINUATION_SAVE_MID_FRAC", "0.5"),
            "STAGE5_ANNEAL_PRELUDE_LR_MULT": os.environ.get("STAGE5_CHAIN_CONTINUATION_PRELUDE_LR_MULT", "10.0"),
            "STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE": os.environ.get(
                "STAGE5_CHAIN_CONTINUATION_BACKUP_CHECKPOINTS_TO_DRIVE", "1"
            ),
            "STAGE5_ANNEAL_DTYPE": os.environ.get("STAGE5_CHAIN_CONTINUATION_DTYPE", "bfloat16"),
        },
    )
    if not parent_summary.exists():
        raise FileNotFoundError(parent_summary)

    extrap_run_id = os.environ.get("STAGE5_CHAIN_CONTINUATION_EXTRAP_RUN_ID", f"{run_id}_depth_extrapolation")
    run(
        [sys.executable, "colab/run_stage5_depth_extrapolation_eval.py"],
        env={
            "STAGE5_EXTRAP_RUN_ID": extrap_run_id,
            "STAGE5_EXTRAP_CHECKPOINT": path_for_cli(parent_summary),
            "STAGE5_EXTRAP_N_SYMBOLS": os.environ.get("STAGE5_CHAIN_CONTINUATION_N_SYMBOLS", "16"),
            "STAGE5_EXTRAP_DEPTHS": os.environ.get("STAGE5_CHAIN_CONTINUATION_EXTRAP_DEPTHS", "1,2,3,4,5,6,7,8"),
            "STAGE5_EXTRAP_ROWS_PER_DEPTH": os.environ.get("STAGE5_CHAIN_CONTINUATION_EXTRAP_ROWS_PER_DEPTH", "128"),
            "STAGE5_EXTRAP_MAX_LOOPS": os.environ.get("STAGE5_CHAIN_CONTINUATION_EXTRAP_MAX_LOOPS", "8"),
            "STAGE5_EXTRAP_DTYPE": os.environ.get("STAGE5_CHAIN_CONTINUATION_DTYPE", "bfloat16"),
        },
    )
    extrap_summary = ROOT / "outputs" / "stage5" / extrap_run_id / "summary.json"
    payload = read_json(parent_summary)
    payload.update(
        {
            "kind": "stage5_chain_continuation_attribution",
            "status": "finished_with_extrapolation",
            "attribution_extrapolation": compact_extrapolation(read_json(extrap_summary)),
            "decision_read": {
                **dict(payload.get("decision_read") or {}),
                "attribution_question": "chain_only_continuation_vs_outcome_anneal",
                "compare_against": "outputs/stage5/stage5_post_anneal_readouts_20260703_191158/summary.json",
            },
        }
    )
    write_json(parent_summary, payload)
    write_markdown(parent_summary.parent, payload)
    publish_run(
        parent_summary.parent,
        message=f"Record Stage 5 chain-continuation attribution {run_id} [skip ci]",
        update_pointer=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
