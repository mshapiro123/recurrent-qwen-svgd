"""Run the locked CPU-only KP-1R cached-state rung."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from colab.run_stage5_paper2_phase3_p34_a2 import DRIVE_STAGE5, rsync, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_kp1r_cached_20260816"
SOURCE_RUN_ID = "stage5_paper2_phase3_kp1_t1_20260816"
LOCK_PATH = ROOT / "training/paper2_phase3_kp1r_t1_teacher_lock.json"
AUTHORITY_PATH = ROOT / "docs/STRATEGY_KP1_T1_RESPONSE_20260816.md"
PANEL_PATH = (
    ROOT
    / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 5 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("KP-1R cached rung requires at least 5 GiB local scratch")


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    authority = lock["authority"]
    if AUTHORITY_PATH.stat().st_size != int(authority["bytes"]):
        raise RuntimeError("KP-1R authority byte count changed")
    if sha256_file(AUTHORITY_PATH) != authority["sha256"]:
        raise RuntimeError("KP-1R authority SHA changed")
    if sha256_file(PANEL_PATH) != lock["source_files"]["panel_sha256"]:
        raise RuntimeError("KP-1R DEV panel SHA changed")

    drive_run = DRIVE_STAGE5 / RUN_ID
    receipts = drive_run / "receipts"
    private = drive_run / "private"
    status_path = receipts / "status.json"
    local = ROOT / "outputs/stage5" / RUN_ID
    receipts.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)

    def status(value: str, **details: object) -> None:
        write_json(
            status_path,
            {
                "kind": "paper2_phase3_kp1r_cached_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                **details,
            },
        )
        print(f"kp1r_cached_status={value} details={details}", flush=True)

    try:
        scratch = scratch_root()
        state_cache = scratch / "t1_state_cache.pt"
        gap_rows = scratch / "kp1_gap_rows.jsonl"
        status("staging_hash_locked_cached_inputs", scratch=str(scratch))
        rsync(DRIVE_STAGE5 / SOURCE_RUN_ID / "private/t1_state_cache.pt", state_cache)
        rsync(DRIVE_STAGE5 / SOURCE_RUN_ID / "private/kp1_gap_rows.jsonl", gap_rows)
        for key, path in (
            ("t1_state_cache_sha256", state_cache),
            ("kp1_gap_rows_sha256", gap_rows),
        ):
            observed = sha256_file(path)
            expected = lock["source_files"][key]
            if observed != expected:
                raise RuntimeError(
                    f"KP-1R cached source SHA mismatch: {key} expected={expected} observed={observed}"
                )
        status("scoring_cpu_cached_primary_surfaces", optimizer_steps=0)
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_phase3_kp1r_cached",
                "--lock",
                str(LOCK_PATH),
                "--state_cache",
                str(state_cache),
                "--gap_rows",
                str(gap_rows),
                "--panel",
                str(PANEL_PATH),
                "--model_cache",
                str(scratch / "hf_model_cache"),
                "--output_dir",
                str(local),
                "--permutation_draws",
                str(lock["kp1r"]["permutation_draws"]),
            ]
        )
        summary = local / "summary.json"
        rows = local / "kp1r_cached_row_predictions.jsonl"
        if not summary.is_file() or not rows.is_file():
            raise RuntimeError("KP-1R cached evaluator did not produce both required receipts")
        shutil.copy2(summary, receipts / "summary.json")
        shutil.copy2(rows, private / rows.name)
        status(
            "complete",
            summary_sha256=sha256_file(summary),
            row_predictions_sha256=sha256_file(rows),
            confirm_scored=False,
            eval_e_scored=False,
            optimizer_steps=0,
        )
        print(summary.read_text(encoding="utf-8"), flush=True)
        return 0
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
