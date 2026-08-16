"""Run the bounded score-only KP-1R strong rung and teacher fingerprints."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from colab.run_stage5_paper2_phase3_kp1_t1 import stage_chain_with_verified_p34
from colab.run_stage5_paper2_phase3_p34_a2 import (
    DRIVE_STAGE5,
    MIGRATED_SHA,
    P33_SHA,
    rsync,
    sha256_file,
    write_json,
)
from colab.run_stage5_paper2_phase3_p35 import I1_SHA
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA, P35_SHA


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_kp1r_t1_teacher_20260816"
SOURCE_RUN_ID = "stage5_paper2_phase3_kp1_t1_20260816"
P35_ID = "stage5_paper2_phase3_p35_20260815"
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
        if root.exists() and shutil.disk_usage(root).free >= 45 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("KP-1R/T1 teacher requires at least 45 GiB local scratch")


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    authority = lock["authority"]
    if AUTHORITY_PATH.stat().st_size != int(authority["bytes"]):
        raise RuntimeError("KP-1R/T1 teacher authority byte count changed")
    if sha256_file(AUTHORITY_PATH) != authority["sha256"]:
        raise RuntimeError("KP-1R/T1 teacher authority SHA changed")
    if sha256_file(PANEL_PATH) != lock["source_files"]["panel_sha256"]:
        raise RuntimeError("KP-1R/T1 teacher DEV panel SHA changed")

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
                "kind": "paper2_phase3_kp1r_t1_teacher_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                **details,
            },
        )
        print(f"kp1r_t1_teacher_status={value} details={details}", flush=True)

    try:
        scratch = scratch_root()
        state_cache = scratch / "t1_state_cache.pt"
        gap_rows = scratch / "kp1_gap_rows.jsonl"
        status("staging_hash_locked_inputs", scratch=str(scratch))
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
                    f"KP-1R/T1 teacher source SHA mismatch: {key} expected={expected} observed={observed}"
                )

        chain = stage_chain_with_verified_p34(
            scratch / "chain_seed_0", seed=0, expected_p34=P34_SHA[0]
        )
        p35 = scratch / "seed_0_p35_ema_step_4400.pt"
        rsync(DRIVE_STAGE5 / P35_ID / "private/arm_s_seed_0/ema_step_4400.pt", p35)
        if sha256_file(p35) != P35_SHA[0]:
            raise RuntimeError("KP-1R/T1 teacher P3.5 seed-0 endpoint SHA mismatch")
        chain_manifest = {
            "kind": "paper2_phase3_kp1r_t1_teacher_chain_manifest_v1",
            "checkpoint": {
                "paths": {name: str(path) for name, path in chain.items()} | {"p35": str(p35)},
                "sha256": {
                    "migrated": MIGRATED_SHA[0],
                    "p33": P33_SHA[0],
                    "i1": I1_SHA[0],
                    "p34": P34_SHA[0],
                    "p35": P35_SHA[0],
                },
            },
            "all_checkpoint_sha256_verified": True,
            "optimizer_constructed": False,
        }
        chain_path = private / "chain_manifest.json"
        write_json(chain_path, chain_manifest)
        status(
            "scoring_student_then_teacher_sequentially",
            chain_manifest_sha256=sha256_file(chain_path),
            optimizer_steps=0,
        )
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_phase3_kp1r_t1_teacher",
                "--lock",
                str(LOCK_PATH),
                "--panel",
                str(PANEL_PATH),
                "--gap_rows",
                str(gap_rows),
                "--state_cache",
                str(state_cache),
                "--chain_manifest",
                str(chain_path),
                "--student_cache",
                str(scratch / "student_hf_cache"),
                "--teacher_cache",
                str(scratch / "teacher_hf_cache"),
                "--output_dir",
                str(local),
                "--private_dir",
                str(private),
                "--permutation_draws",
                str(lock["kp1r"]["permutation_draws"]),
            ]
        )
        summary = local / "summary.json"
        pre_model = local / "pre_model_target_audit.json"
        if not summary.is_file() or not pre_model.is_file():
            raise RuntimeError("KP-1R/T1 teacher evaluator omitted required public receipts")
        shutil.copy2(summary, receipts / "summary.json")
        shutil.copy2(pre_model, receipts / "pre_model_target_audit.json")
        status(
            "complete",
            summary_sha256=sha256_file(summary),
            pre_model_target_audit_sha256=sha256_file(pre_model),
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
