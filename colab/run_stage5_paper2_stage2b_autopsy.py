"""Prelock inventory or execute the signed Stage 2B-A score-only autopsy."""

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
from training.paper2_stage2b_autopsy import sha256_file, validate_autopsy_lock


ROOT = Path(__file__).resolve().parents[1]
MODE = os.environ.get("STAGE2B_AUTOPSY_MODE", "prelock").strip().lower()
SOURCE_RUN_ID = "stage5_paper2_stage2b_depth_20260819"
RUN_ID = "stage5_paper2_stage2b_autopsy_20260820"
LOSS_RUN_ID = "stage5_paper2_stage2b_loss_calibration_20260818"
AMPLITUDE_RUN_ID = "stage5_paper2_phase3_p35_amplitude_t1_20260816"
DRIVE_SOURCE = DRIVE_STAGE5 / SOURCE_RUN_ID
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
LOCK = ROOT / "training/paper2_stage2b_autopsy_lock.json"
PANEL = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
BASE_SCORES = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
REFERENCE_ROWS = DRIVE_STAGE5 / "stage5_paper2_phase3_p31_completion_20260810/private/p31_partitioned_rows.jsonl"
DEV2_MANIFEST = DRIVE_SOURCE / "private/dev2/dev2_manifest.jsonl"
CALIBRATION_TEACHER = DRIVE_STAGE5 / LOSS_RUN_ID / "private/calibration_teacher_top128.pt"
STOP_SHA = {
    0: "50cbf437adda668812dbe53a015792d3dc8ebc02cb785fba594c512b64bf2f58",
    1: "830bbfa11dca4d3b9ed56db96a7c40c887f56fb4a5227555edc1bd447b6662bc",
}
INITIALIZATION_SCORE_SHA = {
    0: "13732e986949aa2bcec5b4060947a262b6c3a980305659cf7ca604d61df08815",
    1: "f3495dd32904bcef4388a02272d8a67fb01eb9fa54d82ebb4eeb341a2667dff1",
}
INITIALIZATION_SCORE_0P02_SHA = {
    0: "baf14141ff75e8c6a280a68f91b6b20256b19bdc2fef615717193feaccf71a02",
    1: "ad965c1b462d953d9c503ab12e193b7bdc3cd422cd74cf6b2525ffb06c444442",
}
ONSET_STEPS = (20, 60, 100, 200, 300, 500, 700)
TRAINING_SUMMARY_SHA = {
    0: "90b6e4c9fea538b7876349550e8caa02e5094c2f02d4535c8b7ecff4397669b0",
    1: "faafb98887555a0fa7fe876ffc33f35b0c61f9fa35b9e574ff667f1835c1fb23",
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, log_path: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8", newline="") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def archive_incomplete_status(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "complete_score_only":
        return None
    digest = sha256_file(path)
    destination = DRIVE_RUN / "receipts/superseded" / f"{label}__{digest[:16]}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
    return {"source": str(path), "path": str(destination), "sha256": digest}


def scratch_root() -> Path:
    for candidate in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if candidate.exists() and shutil.disk_usage(candidate).free >= 80 * 1024**3:
            target = candidate / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2B-A requires at least 80 GiB local scratch")


def _trajectory_candidates(seed: int, step: int) -> tuple[Path, ...]:
    private = DRIVE_SOURCE / f"private/seed_{seed}"
    return (
        private / f"ema_step_{step:05d}.pt",
        private / f"resume_step_{step:05d}.pt",
        private / "checkpoints" / f"ema_step_{step:05d}.pt",
        private / "checkpoints" / f"resume_step_{step:05d}.pt",
    )


def prelock(scratch: Path) -> dict[str, Any]:
    local_manifest = scratch / "dev2_manifest.jsonl"
    rsync(DEV2_MANIFEST, local_manifest)
    frozen = DRIVE_RUN / "private/prelock/dev2_autopsy_subsample.jsonl"
    staged = scratch / "dev2_autopsy_subsample.jsonl"
    run([
        sys.executable,
        "-u",
        "-m",
        "eval.eval_paper2_stage2b_autopsy",
        "--freeze-dev2-subsample",
        "--dev2_manifest",
        str(local_manifest),
        "--dev2_subsample_manifest",
        str(staged),
    ])
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staged, frozen)
    shutil.copy2(staged.with_suffix(".receipt.json"), frozen.with_suffix(".receipt.json"))

    inventory: dict[str, Any] = {}
    missing = []
    for seed in (0, 1):
        entries = {
            "0": {"construction": "deterministic signed initialization", "available": True},
            "1000": {
                "path": str(DRIVE_SOURCE / f"private/seed_{seed}/ema_step_01000.pt"),
                "sha256": STOP_SHA[seed],
                "available": (DRIVE_SOURCE / f"private/seed_{seed}/ema_step_01000.pt").is_file(),
            },
        }
        for step in ONSET_STEPS:
            found = next((path for path in _trajectory_candidates(seed, step) if path.is_file()), None)
            if found is None:
                entries[str(step)] = {
                    "available": False,
                    "candidates": [str(path) for path in _trajectory_candidates(seed, step)],
                }
                missing.append({"seed": seed, "step": step})
            else:
                entries[str(step)] = {
                    "available": True,
                    "path": str(found),
                    "sha256": sha256_file(found),
                }
        inventory[str(seed)] = entries
    receipt = {
        "kind": "paper2_stage2b_autopsy_prelock_v1",
        "status": "complete_prelock" if not missing else "blocked_on_historical_checkpoint_inventory",
        "dev2_subsample": {
            "path": str(frozen),
            "sha256": sha256_file(frozen),
            "receipt_sha256": sha256_file(frozen.with_suffix(".receipt.json")),
        },
        "trajectory_checkpoint_inventory": inventory,
        "missing_trajectory_cells": missing,
        "source_runner_checkpoint_behavior": "resume.pt was atomically overwritten every 20 steps; named snapshots were emitted only at registered looks",
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "model_loaded": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(DRIVE_RUN / "receipts/prelock.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def execute(scratch: Path) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_autopsy_lock(lock, require_signature=True)
    dev2_subsample = DRIVE_RUN / "private/prelock/dev2_autopsy_subsample.jsonl"
    if sha256_file(dev2_subsample) != lock["dev2_subsample"]["manifest_sha256"]:
        raise RuntimeError("Stage 2B-A frozen DEV-2 subsample changed")
    reference = scratch / "p31_partitioned_rows.jsonl"
    rsync(REFERENCE_ROWS, reference)
    teacher = scratch / "calibration_teacher_top128.pt"
    rsync(CALIBRATION_TEACHER, teacher)
    if sha256_file(teacher) != lock["heldout_training_slice"]["teacher_cache_sha256"]:
        raise RuntimeError("Stage 2B-A heldout teacher cache changed")
    archived_attempts = []
    archived = archive_incomplete_status(
        DRIVE_RUN / "receipts/status.json", label="run_status_before_resume"
    )
    if archived:
        archived_attempts.append(archived)
    summaries = []
    for seed in (0, 1):
        output = DRIVE_RUN / f"receipts/seed_{seed}"
        private = DRIVE_RUN / f"private/seed_{seed}"
        summary = output / "summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if (
                payload.get("status") != "complete_score_only"
                or int(payload.get("seed", -1)) != seed
                or payload.get("lock_sha256") != sha256_file(LOCK)
                or payload.get("optimizer_steps") != 0
                or payload.get("confirm_scored") is not False
                or payload.get("eval_e_scored") is not False
            ):
                raise RuntimeError(f"Stage 2B-A completed seed receipt failed resume validation: {seed}")
            summaries.append({"seed": seed, "path": str(summary), "sha256": sha256_file(summary)})
            print(f"stage2b_seed_resume seed={seed} status=complete_score_only", flush=True)
            continue
        archived = archive_incomplete_status(
            output / "status.json", label=f"seed_{seed}_status_before_resume"
        )
        if archived:
            archived_attempts.append(archived)
        chain = stage_chain(scratch / f"chain_seed_{seed}", seed=seed, expected_p34=P34_SHA[seed])
        p35 = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
        rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", p35)
        if sha256_file(p35) != P35_SHA[seed]:
            raise RuntimeError("Stage 2B-A P3.5 endpoint changed")
        stop = scratch / f"seed_{seed}_ema_step_01000.pt"
        rsync(DRIVE_SOURCE / f"private/seed_{seed}/ema_step_01000.pt", stop)
        if sha256_file(stop) != STOP_SHA[seed]:
            raise RuntimeError("Stage 2B-A stop checkpoint changed")
        initialization = scratch / f"seed_{seed}_initialization_dev1.jsonl"
        rsync(
            DRIVE_STAGE5 / AMPLITUDE_RUN_ID / f"private/amplitude_surface/seed_{seed}_ceiling_0p05.jsonl",
            initialization,
        )
        if sha256_file(initialization) != INITIALIZATION_SCORE_SHA[seed]:
            raise RuntimeError("Stage 2B-A initialization score receipt changed")
        initialization_0p02 = scratch / f"seed_{seed}_initialization_dev1_0p02.jsonl"
        rsync(
            DRIVE_STAGE5 / AMPLITUDE_RUN_ID / f"private/amplitude_surface/seed_{seed}_ceiling_0p02.jsonl",
            initialization_0p02,
        )
        if sha256_file(initialization_0p02) != INITIALIZATION_SCORE_0P02_SHA[seed]:
            raise RuntimeError("Stage 2B-A initialization 0.02 score receipt changed")
        training_summary = scratch / f"seed_{seed}_training_summary.json"
        rsync(DRIVE_SOURCE / f"receipts/seed_{seed}/summary.json", training_summary)
        if sha256_file(training_summary) != TRAINING_SUMMARY_SHA[seed]:
            raise RuntimeError("Stage 2B-A contemporaneous training summary changed")
        command = [
            sys.executable, "-u", "-m", "eval.eval_paper2_stage2b_autopsy",
            "--seed", str(seed), "--lock", str(LOCK),
            "--dev1_panel", str(PANEL), "--dev2_manifest", str(DEV2_MANIFEST),
            "--dev2_subsample_manifest", str(dev2_subsample), "--reference_rows", str(reference),
            "--base_scores", str(BASE_SCORES), "--initialization_scores", str(initialization),
            "--initialization_scores_0p02", str(initialization_0p02),
            "--heldout_teacher_cache", str(teacher), "--stop_checkpoint", str(stop),
            "--migrated", str(chain["migrated"]), "--migrated_sha256", MIGRATED_SHA[seed],
            "--p33", str(chain["p33"]), "--p33_sha256", P33_SHA[seed],
            "--i1", str(chain["i1"]), "--i1_sha256", I1_SHA[seed],
            "--p34", str(chain["p34"]), "--p34_sha256", P34_SHA[seed],
            "--p35", str(p35), "--p35_sha256", P35_SHA[seed],
            "--model_cache", str(scratch / "hf_student_cache"),
            "--output_dir", str(output), "--private_dir", str(private),
            "--training_summary", str(training_summary),
        ]
        run(
            command,
            log_path=DRIVE_RUN / "receipts/logs" / f"seed_{seed}_latest.log",
        )
        summaries.append({"seed": seed, "path": str(summary), "sha256": sha256_file(summary)})
    receipt = {
        "kind": "paper2_stage2b_autopsy_execution_v1",
        "status": "complete_score_only",
        "seed_summaries": summaries,
        "superseded_incomplete_attempts": archived_attempts,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(DRIVE_RUN / "receipts/status.json", receipt)
    return receipt


def main() -> int:
    if MODE not in {"prelock", "run"}:
        raise RuntimeError("STAGE2B_AUTOPSY_MODE must be prelock or run")
    scratch = scratch_root()
    status_path = DRIVE_RUN / "receipts/status.json"
    try:
        result = prelock(scratch) if MODE == "prelock" else execute(scratch)
        atomic_json(status_path, result)
        return 0 if result.get("status") != "blocked_on_historical_checkpoint_inventory" else 2
    except Exception as error:
        atomic_json(
            status_path,
            {
                "kind": "paper2_stage2b_autopsy_status_v1",
                "status": "failed",
                "mode": MODE,
                "updated_at_unix": time.time(),
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "confirm_scored": False,
                "eval_e_scored": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
