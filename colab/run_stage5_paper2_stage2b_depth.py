"""Stage, cache, preflight, and run the signed Stage 2B-D campaign."""

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
from training.paper2_stage2b_runtime import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_stage2b_depth_20260819"
AMPLITUDE_RUN_ID = "stage5_paper2_phase3_p35_amplitude_t1_20260816"
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
LOCK = ROOT / "training/paper2_stage2b_depth_executed_lock.json"
OLD_DATA = DRIVE_STAGE5 / "stage5_paper2_dc1_preflight_20260729/private/dev_c/dev_c.jsonl"
NEW_DATA = DRIVE_STAGE5 / "stage5_paper2_phase2_option_b_teacher_cache_20260806/private/new_documents_target.jsonl"
REFERENCE_ROWS = DRIVE_STAGE5 / "stage5_paper2_phase3_p31_completion_20260810/private/p31_partitioned_rows.jsonl"
REFERENCE_SCORES = DRIVE_STAGE5 / "stage5_paper2_phase3_p31_completion_20260810/private/p31_merged_dev_verified_scores.jsonl"
PANEL = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
BASE_SCORES = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
CORPUS_SHA = "2e3e4f8cc98f997854381a98819539f835b68830c75a75f7d0f24a9b91c4e135"
DEV2_SHA = "6b9ebf40128ed21b0351710e9f828bcacb096512704f02f34274a3b8adcc0adb"
DEV2_RECEIPT_SHA = "d9f1aee6c9f951376c1fa946deb5933481e1b9d180d22a259e3ba6e68751c3b7"
INITIALIZATION_SCORE_SHA = {
    0: "13732e986949aa2bcec5b4060947a262b6c3a980305659cf7ca604d61df08815",
    1: "f3495dd32904bcef4388a02272d8a67fb01eb9fa54d82ebb4eeb341a2667dff1",
}
MODE = os.environ.get("STAGE2B_MODE", "cache").strip().lower()
SEED = int(os.environ.get("STAGE2B_SEED", "0"))
TARGET_STEP = int(os.environ.get("STAGE2B_TARGET_STEP", "5000"))


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def scratch_root() -> Path:
    for candidate in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if candidate.exists() and shutil.disk_usage(candidate).free >= 80 * 1024**3:
            target = candidate / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2B requires at least 80 GiB local scratch")


def archive_previous_failure(status_path: Path) -> Path | None:
    if not status_path.is_file():
        return None
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if payload.get("status") != "failed":
        return None
    timestamp = int(float(payload.get("updated_at_unix", 0)))
    archive = status_path.parent / "archaeology" / (
        f"{status_path.stem}_failed_{timestamp}.json"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copyfile(status_path, archive)
    return archive


def prepare_common(scratch: Path) -> dict[str, Path]:
    old = scratch / "dev_c.jsonl"
    new = scratch / "new_documents_target.jsonl"
    rsync(OLD_DATA, old)
    rsync(NEW_DATA, new)
    data = scratch / "data"
    run([
        sys.executable, "-u", "-m", "eval.prepare_paper2_stage2b_data",
        "--old-data", str(old), "--new-data", str(new), "--output-dir", str(data),
    ])
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if summary["corpus"]["corpus_sha256"] != CORPUS_SHA:
        raise RuntimeError("Stage 2B rebuilt corpus changed")
    reference = scratch / "p31_partitioned_rows.jsonl"
    rsync(REFERENCE_ROWS, reference)
    reference_scores = scratch / "p31_merged_dev_verified_scores.jsonl"
    rsync(REFERENCE_SCORES, reference_scores)
    dev2_dir = DRIVE_RUN / "private/dev2"
    dev2 = dev2_dir / "dev2_manifest.jsonl"
    dev2_receipt = dev2_dir / "dev2_manifest_receipt.json"
    if (
        not dev2.is_file()
        or not dev2_receipt.is_file()
        or sha256_file(dev2) != DEV2_SHA
        or sha256_file(dev2_receipt) != DEV2_RECEIPT_SHA
    ):
        staged_dev2 = scratch / "dev2_staged"
        shutil.rmtree(staged_dev2, ignore_errors=True)
        run([
            sys.executable, "-u", "-m", "eval.prepare_paper2_stage2b_dev2",
            "--reference-rows", str(reference),
            "--reference-scores", str(reference_scores),
            "--dev1-rows", str(PANEL),
            "--output-dir", str(staged_dev2),
        ])
        staged_manifest = staged_dev2 / "dev2_manifest.jsonl"
        staged_receipt = staged_dev2 / "dev2_manifest_receipt.json"
        if sha256_file(staged_manifest) != DEV2_SHA:
            raise RuntimeError("Stage 2B staged DEV-2 manifest changed")
        if sha256_file(staged_receipt) != DEV2_RECEIPT_SHA:
            raise RuntimeError("Stage 2B staged DEV-2 receipt changed")
        dev2_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged_manifest, dev2)
        shutil.copyfile(staged_receipt, dev2_receipt)
    if sha256_file(dev2) != DEV2_SHA:
        raise RuntimeError("Stage 2B DEV-2 manifest changed")
    if sha256_file(dev2_receipt) != DEV2_RECEIPT_SHA:
        raise RuntimeError("Stage 2B DEV-2 receipt changed")
    return {"data": data, "reference": reference, "dev2": dev2}


def main() -> int:
    if MODE not in {"cache", "preflight", "train"} or SEED not in {0, 1}:
        raise RuntimeError("STAGE2B_MODE/SEED is invalid")
    scratch = scratch_root()
    receipts = DRIVE_RUN / "receipts"
    status_path = receipts / ("cache_status.json" if MODE == "cache" else f"seed_{SEED}_status.json")
    archived_failure = archive_previous_failure(status_path)
    if archived_failure is not None:
        print(f"stage2b_archived_previous_failure={archived_failure}", flush=True)

    def status(value: str, **details: Any) -> None:
        atomic_json(
            status_path,
            {
                "kind": "paper2_stage2b_colab_status_v1",
                "status": value,
                "mode": MODE,
                "seed": SEED,
                "target_step": TARGET_STEP,
                "updated_at_unix": time.time(),
                "confirm_scored": False,
                "eval_e_scored": False,
                **details,
            },
        )
        print(f"stage2b_status={value} details={details}", flush=True)

    try:
        status("preparing_registered_inputs")
        common = prepare_common(scratch)
        teacher_drive = DRIVE_RUN / "private/teacher_cache"
        if MODE == "cache":
            status("building_resumable_14b_teacher_cache")
            run([
                sys.executable, "-u", "-m", "eval.cache_paper2_stage2b_training_teacher",
                "--rows", str(common["data"] / "training_corpus.jsonl"),
                "--expected-corpus-sha256", CORPUS_SHA,
                "--model-cache", str(scratch / "hf_teacher_cache"),
                "--output-dir", str(teacher_drive),
            ])
            status(
                "complete_cache_only_release_gpu",
                teacher_index_sha256=sha256_file(teacher_drive / "index.json"),
                optimizer_constructed=False,
                optimizer_steps=0,
            )
            return 0

        if not (teacher_drive / "index.json").is_file():
            raise RuntimeError("Stage 2B teacher cache has not landed")
        teacher_local = scratch / "teacher_cache"
        rsync(teacher_drive, teacher_local)
        chain = stage_chain(scratch / f"chain_seed_{SEED}", seed=SEED, expected_p34=P34_SHA[SEED])
        p35 = scratch / f"seed_{SEED}_p35_ema_step_4400.pt"
        rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{SEED}/ema_step_4400.pt", p35)
        if sha256_file(p35) != P35_SHA[SEED]:
            raise RuntimeError("Stage 2B P3.5 source endpoint changed")
        initialization = scratch / f"seed_{SEED}_initialization_dev1.jsonl"
        rsync(
            DRIVE_STAGE5 / AMPLITUDE_RUN_ID
            / f"private/amplitude_surface/seed_{SEED}_ceiling_0p05.jsonl",
            initialization,
        )
        if sha256_file(initialization) != INITIALIZATION_SCORE_SHA[SEED]:
            raise RuntimeError("Stage 2B initialization score receipt changed")
        output = receipts / f"seed_{SEED}"
        private = DRIVE_RUN / f"private/seed_{SEED}"
        command = [
            sys.executable, "-u", "-m", "training.run_paper2_stage2b_depth",
            "--seed", str(SEED), "--lock", str(LOCK),
            "--teacher_cache_index", str(teacher_local / "index.json"),
            "--dev1_panel", str(PANEL), "--dev2_manifest", str(common["dev2"]),
            "--reference_rows", str(common["reference"]), "--base_scores", str(BASE_SCORES),
            "--initialization_scores", str(initialization),
            "--migrated", str(chain["migrated"]), "--migrated_sha256", MIGRATED_SHA[SEED],
            "--p33", str(chain["p33"]), "--p33_sha256", P33_SHA[SEED],
            "--i1", str(chain["i1"]), "--i1_sha256", I1_SHA[SEED],
            "--p34", str(chain["p34"]), "--p34_sha256", P34_SHA[SEED],
            "--p35", str(p35), "--p35_sha256", P35_SHA[SEED],
            "--model_cache", str(scratch / "hf_student_cache"),
            "--output_dir", str(output), "--private_dir", str(private),
            "--target_step", str(TARGET_STEP), "--resume_interval", "100",
        ]
        if MODE == "preflight":
            command.append("--preflight_only")
        status("running_preflight" if MODE == "preflight" else "training_registered_seed")
        run(command)
        summary = output / ("preoptimizer_receipt.json" if MODE == "preflight" else "summary.json")
        payload = json.loads(summary.read_text(encoding="utf-8"))
        status(
            "complete_preflight_no_optimizer" if MODE == "preflight" else payload["status"],
            summary_sha256=sha256_file(summary),
            step=payload.get("step", 0),
            optimizer_constructed=(False if MODE == "preflight" else True),
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
