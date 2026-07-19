"""E3a: zero-shot natural-surface transfer from the frozen Arm E checkpoint."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.adapter_parity_common import (
    ARM_E_FINAL_SHA256,
    ROOT,
    lora_eval_args,
    path_for_cli,
    read_json,
    restore_arm_e_checkpoint,
    run,
    write_json,
)
from colab.stage5_chain_consolidation_utils import publish_run
from training.adapter_parity_battery import score_e3a_transfer


DATA_ROOT = ROOT / "outputs/stage5/stage5_natural_surface_transfer_20260708_230229/data"
FULL_BLOCK = ROOT / "outputs/stage5/stage5_natural_surface_receipts_20260709_210151/summary.json"


def _eval(run_dir: Path, *, family: str, checkpoint: Path) -> dict[str, Any]:
    data = DATA_ROOT / f"{family}_test_chain_mcq.jsonl"
    out = run_dir / "eval" / family
    rows = out / "rows.jsonl"
    summary = out / "summary.json"
    if not summary.exists():
        run(
            [
                sys.executable,
                "eval/eval_synthetic_depth_final_symbol.py",
                "--data_jsonl",
                path_for_cli(data),
                "--checkpoint",
                path_for_cli(checkpoint),
                "--output_jsonl",
                path_for_cli(rows),
                "--output_summary",
                path_for_cli(summary),
                "--max_loops",
                "12",
                "--threshold",
                "0.71",
                "--prompt_style",
                "question_only",
                "--value_prefix",
                "name:",
                *lora_eval_args(),
            ]
        )
    payload = read_json(summary)
    total = payload["same_reader_total"]
    return {
        "data_jsonl": path_for_cli(data),
        "summary": path_for_cli(summary),
        "same_reader_total": total,
        "by_depth": payload["by_depth"],
        "reporting": score_e3a_transfer(
            correct=int(total["correct"]),
            total=int(total["total"]),
        ),
    }


def main() -> int:
    run_id = os.environ.get("STAGE5_ADAPTER_E3A_RUN_ID", "stage5_adapter_parity_e3a_20260719")
    run_dir = ROOT / "outputs/stage5" / run_id
    checkpoint, restore = restore_arm_e_checkpoint(run_dir / "restored" / "arm_e_final.pt")
    full_block = read_json(FULL_BLOCK)["same_reader"]["step_6000"]
    results = {family: _eval(run_dir, family=family, checkpoint=checkpoint) for family in ("relay", "pointer")}
    payload = {
        "kind": "stage5_adapter_parity_e3a",
        "run_id": run_id,
        "status": "finished",
        "source_checkpoint_sha256": ARM_E_FINAL_SHA256,
        "restore_receipt": restore,
        "protocol": {
            "training": "none_eval_only",
            "depths": list(range(1, 13)),
            "forced_loops": "row_depth",
            "reader": "same_reader_full_symbol",
            "bands": {"strong_floor": 0.70, "partial_floor": 0.40},
        },
        "results": results,
        "full_block_reference": {
            family: {
                "same_reader_total": full_block[f"{family}_original"]["same_reader_total"],
                "by_depth": full_block[f"{family}_original"]["by_depth"],
            }
            for family in ("relay", "pointer")
        },
    }
    write_json(run_dir / "summary.json", payload)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Arm E E3a - {run_id}",
                "",
                f"- Relay: `{results['relay']['same_reader_total']}` (`{results['relay']['reporting']['band']}`)",
                f"- Pointer: `{results['pointer']['same_reader_total']}` (`{results['pointer']['reporting']['band']}`)",
                "- Comparison is descriptive against the full-block keeper, not a parity test.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    publish_run(run_dir, message=f"Record Arm E E3a verbal transfer {run_id} [skip ci]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
