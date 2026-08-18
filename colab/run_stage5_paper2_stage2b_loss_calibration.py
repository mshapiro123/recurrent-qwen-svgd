"""Run the no-training Stage 2B full-sequence loss calibration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from colab.run_stage5_paper2_phase3_p34_a2 import DRIVE_STAGE5, MIGRATED_SHA, P33_SHA, rsync
from colab.run_stage5_paper2_phase3_p35 import I1_SHA, stage_chain
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA, P35_ID, P35_SHA
from training.paper2_stage2b_data import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_stage2b_loss_calibration_20260818"
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
OLD_DATA = (
    DRIVE_STAGE5
    / "stage5_paper2_dc1_preflight_20260729/private/dev_c/dev_c.jsonl"
)
NEW_DATA = (
    DRIVE_STAGE5
    / "stage5_paper2_phase2_option_b_teacher_cache_20260806/private/new_documents_target.jsonl"
)
OLD_SHA = "05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d"
NEW_SHA = "bd0c84984f1dd47d1ee5dc06172afb7ea9728443d7712ca82ffa836d54edff9b"
CALIBRATION_MANIFEST_SHA = "5d9e6784f47ddd9a18f9d966e5e6d3392d60159a9872bafba0d02ef386a34b21"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 50 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2B loss calibration requires 50 GiB local scratch")


def main() -> int:
    scratch = scratch_root()
    receipts = DRIVE_RUN / "receipts"
    private = DRIVE_RUN / "private"
    status_path = receipts / "status.json"

    def status(value: str, **details: Any) -> None:
        payload = {
            "kind": "paper2_stage2b_loss_calibration_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
            **details,
        }
        write_json(status_path, payload)
        print(f"stage2b_loss_calibration_status={value} details={details}", flush=True)

    try:
        status("staging_sources")
        old = scratch / "dev_c.jsonl"
        new = scratch / "new_documents_target.jsonl"
        rsync(OLD_DATA, old)
        rsync(NEW_DATA, new)
        if sha256_file(old) != OLD_SHA or sha256_file(new) != NEW_SHA:
            raise RuntimeError("Stage 2B source corpus SHA mismatch")
        data_dir = scratch / "data_prelock"
        run([
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_stage2b_data",
            "--old-data",
            str(old),
            "--new-data",
            str(new),
            "--output-dir",
            str(data_dir),
        ])
        data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
        if data_summary["calibration"]["manifest_sha256"] != CALIBRATION_MANIFEST_SHA:
            raise RuntimeError("Stage 2B calibration selection changed")
        receipts.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_dir / "summary.json", receipts / "data_prelock_summary.json")
        shutil.copy2(data_dir / "loss_calibration_rows.jsonl", private / "loss_calibration_rows.jsonl")

        teacher_cache = private / "calibration_teacher_top128.pt"
        if not teacher_cache.is_file():
            status("caching_pinned_14b_teacher")
            run([
                sys.executable,
                "-u",
                "-m",
                "eval.cache_paper2_stage2b_calibration_teacher",
                "--rows",
                str(data_dir / "loss_calibration_rows.jsonl"),
                "--expected-manifest-sha256",
                CALIBRATION_MANIFEST_SHA,
                "--model-cache",
                str(scratch / "hf_teacher_cache"),
                "--output",
                str(teacher_cache),
            ])
        shutil.copy2(
            teacher_cache.with_suffix(".receipt.json"),
            receipts / "calibration_teacher_top128_receipt.json",
        )

        for seed in (0, 1):
            output = receipts / f"loss_calibration_seed_{seed}.json"
            if output.is_file():
                continue
            status("calibrating_seed", seed=seed)
            chain = stage_chain(scratch / f"chain_seed_{seed}", seed=seed, expected_p34=P34_SHA[seed])
            p35 = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
            rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", p35)
            if sha256_file(p35) != P35_SHA[seed]:
                raise RuntimeError(f"Stage 2B seed-{seed} P3.5 endpoint SHA mismatch")
            run([
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_stage2b_loss_calibration",
                "--seed",
                str(seed),
                "--teacher-cache",
                str(teacher_cache),
                "--migrated",
                str(chain["migrated"]),
                "--migrated-sha256",
                MIGRATED_SHA[seed],
                "--p33",
                str(chain["p33"]),
                "--p33-sha256",
                P33_SHA[seed],
                "--i1",
                str(chain["i1"]),
                "--i1-sha256",
                I1_SHA[seed],
                "--p34",
                str(chain["p34"]),
                "--p34-sha256",
                P34_SHA[seed],
                "--p35",
                str(p35),
                "--p35-sha256",
                P35_SHA[seed],
                "--model-cache",
                str(scratch / "hf_student_cache"),
                "--output",
                str(output),
            ])
        status(
            "complete_no_optimizer",
            data_summary_sha256=sha256_file(receipts / "data_prelock_summary.json"),
            teacher_receipt_sha256=sha256_file(receipts / "calibration_teacher_top128_receipt.json"),
            seed_0_receipt_sha256=sha256_file(receipts / "loss_calibration_seed_0.json"),
            seed_1_receipt_sha256=sha256_file(receipts / "loss_calibration_seed_1.json"),
        )
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
