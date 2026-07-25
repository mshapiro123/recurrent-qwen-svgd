"""Run the locked seed-1 T1-lite-R replication with raw-primary scoring."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_paper2_t1_lite import (
    CANARY,
    FROZEN_EVAL,
    REFERENCE_RECEIPT,
    T0_RECEIPT,
    path_for_cli,
    prepare_registered_data,
    read_json,
    run,
    sha256_file,
    write_json,
)
from training.internal_think_token_t1_r_spec import (
    ORIGINAL_T1_LOCK,
    ORIGINAL_T1_LOCK_SHA256,
    phase_t1_lite_r_locked,
    validate_phase_t1_lite_r_locked,
)


RUN_ID = os.environ.get("STAGE5_PAPER2_T1_LITE_R_RUN_ID", "stage5_paper2_t1_lite_r_20260725")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_T1_LITE_R_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/{RUN_ID}",
    )
)
SEED0_SUMMARY = ROOT / "outputs/stage5/stage5_paper2_t1_lite_20260724/summary.json"
SEED0_RAW_EVAL = ROOT / "outputs/stage5/stage5_paper2_t1_lite_20260724/eval/raw_secondary/summary.json"


def publish(message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in sorted(RUN_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md", ".log"}:
            continue
        if path.name.endswith("control.jsonl") or path.name == "causal_override_progress.jsonl":
            continue
        subprocess.run(["git", "add", "-f", path.relative_to(ROOT).as_posix()], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


def copy_eval_from_drive(label: str) -> Path:
    source = DRIVE_ROOT / "eval" / label
    destination = RUN_DIR / "eval" / label
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.jsonl"))
    return destination


def assert_replication_basis(prereg: dict[str, Any]) -> dict[str, Any]:
    original_lock = ROOT / ORIGINAL_T1_LOCK
    if sha256_file(original_lock) != ORIGINAL_T1_LOCK_SHA256:
        raise RuntimeError("T1-lite-R original preregistration hash mismatch")
    if not SEED0_SUMMARY.exists() or not SEED0_RAW_EVAL.exists():
        raise FileNotFoundError("T1-lite-R seed-0 basis receipts are missing")
    seed0 = read_json(SEED0_SUMMARY)
    raw = read_json(SEED0_RAW_EVAL)
    gated = raw.get("gated", {})
    control = gated.get("control", {})
    if int(gated.get("forced_correct", -1)) != 967:
        raise RuntimeError("T1-lite-R seed-0 raw preservation basis drifted")
    if int(control.get("exact_selected_depth_correct", -1)) != 1024:
        raise RuntimeError("T1-lite-R seed-0 raw selection basis drifted")
    if seed0.get("verdict") != "registered_negative":
        raise RuntimeError("T1-lite-R seed-0 registered verdict drifted")
    return {
        "seed0_registered_verdict": seed0["verdict"],
        "seed0_raw_forced_correct": gated["forced_correct"],
        "seed0_raw_exact_selection_correct": control["exact_selected_depth_correct"],
        "strategy_authorization": prereg["governing_document"],
    }


def main() -> int:
    prereg = phase_t1_lite_r_locked()
    validate_phase_t1_lite_r_locked(prereg)
    if subprocess.run(["git", "merge-base", "--is-ancestor", "ae2793ac", "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("T1-lite-R checkout predates its preregistration lock commit")
    t0 = read_json(T0_RECEIPT)
    if t0.get("status") != "passed_all_five_contracts":
        raise RuntimeError(f"T1-lite-R Phase T0 receipt is not green: {t0.get('status')}")
    reference = read_json(REFERENCE_RECEIPT)
    expected_reference_sha = prereg["fresh_base_lineages"]["full_block"]["nonhalting_reference"]["checkpoint_sha256"]
    arm_a = reference.get("checkpoint_receipts", {}).get("A", {})
    if arm_a.get("status") != "verified" or arm_a.get("sha256") != expected_reference_sha:
        raise RuntimeError(f"T1-lite-R full-block reference receipt mismatch: {arm_a}")
    basis = assert_replication_basis(prereg)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RUN_DIR / "preregistration.json", prereg)
    write_json(RUN_DIR / "replication_basis.json", basis)
    data = prepare_registered_data(RUN_DIR, seed=1)
    write_json(RUN_DIR / "data_manifest.json", data)
    training_dir = RUN_DIR / "train"
    result = run(
        [
            sys.executable,
            "training/run_internal_think_token_t1_lite.py",
            "--train_jsonl",
            data["train"],
            "--pilot_jsonl",
            data["pilot"],
            "--canary_jsonl",
            data["canary"],
            "--output_dir",
            path_for_cli(training_dir),
            "--backup_dir",
            str(DRIVE_ROOT / "checkpoints"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_T1_LITE_R_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("ATTN_IMPLEMENTATION", "default"),
            "--seed",
            "1",
            "--registered_contract",
            "t1_lite_r",
        ],
        allowed=(0, 2),
    )
    publish(f"Record T1-lite-R training and stage manifests {RUN_ID} [skip ci]")
    if result.returncode == 2:
        return 2

    training = read_json(training_dir / "training_summary.json")
    manifest = training.get("stage_checkpoint_manifest") or {}
    if manifest.get("complete") is not True:
        raise RuntimeError("T1-lite-R stage checkpoint manifest is not complete")
    eval_specs = (
        ("raw_primary", training["raw_checkpoint"], True),
        ("continuous_ema_shadow", training["continuous_ema_checkpoint"], False),
        ("stage_reset_ema_shadow", training["stage_reset_ema_checkpoint"], False),
    )
    evaluations: dict[str, Any] = {}
    for label, checkpoint, causal in eval_specs:
        drive_eval = DRIVE_ROOT / "eval" / label
        command = [
            sys.executable,
            "eval/eval_internal_think_token_t1_lite.py",
            "--checkpoint",
            checkpoint,
            "--gated_jsonl",
            data["frozen_eval"],
            "--extrapolation_jsonl",
            data["frozen_eval"],
            "--calibration_jsonl",
            data["calibration"],
            "--output_dir",
            str(drive_eval),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_T1_LITE_R_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("ATTN_IMPLEMENTATION", "default"),
            "--batch_size",
            os.environ.get("STAGE5_PAPER2_T1_LITE_R_EVAL_BATCH_SIZE", "8"),
        ]
        if causal:
            command.extend(
                [
                    "--run_causal_sweep",
                    "--causal_progress_path",
                    str(DRIVE_ROOT / "causal_override_progress.jsonl"),
                ]
            )
        run(command)
        local_eval = copy_eval_from_drive(label)
        evaluations[label] = read_json(local_eval / "summary.json")
        publish(f"Record T1-lite-R {label} evaluation {RUN_ID} [skip ci]")

    verdict = evaluations["raw_primary"]["registered_gates"]
    if verdict is None:
        raise RuntimeError("T1-lite-R raw primary did not produce registered gates")
    summary = {
        "kind": "stage5_paper2_t1_lite_r",
        "run_id": RUN_ID,
        "status": "finished",
        "registered_attempt": 2,
        "seed": 1,
        "primary_weights": "raw_final_step",
        "verdict": verdict["verdict"],
        "all_four_passed": verdict["all_four_passed"],
        "preregistration": path_for_cli(RUN_DIR / "preregistration.json"),
        "replication_basis": basis,
        "data": data,
        "training_summary": path_for_cli(training_dir / "training_summary.json"),
        "stage_checkpoint_manifest": path_for_cli(training_dir / "stage_checkpoint_manifest.json"),
        "evaluations": {
            label: path_for_cli(RUN_DIR / "eval" / label / "summary.json")
            for label in evaluations
        },
        "passive_shadows_are_not_registered_endpoints": True,
        "d0_training_authorized": False,
        "c_track_training_authorized": False,
    }
    write_json(RUN_DIR / "summary.json", summary)
    write_json(DRIVE_ROOT / "summary.json", summary)
    publish(f"Finish registered Paper Two T1-lite-R {RUN_ID} [skip ci]")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

