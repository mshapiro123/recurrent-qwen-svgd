"""Run the bounded repeated-prompt posterior-control gate for corrected Phase G.

This runner deliberately stops before any coverage comparison. Its only job is
to show that the posterior uses the selected valid chain on a held-out prompt
set where the same problem appears with multiple valid targets.
"""

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

from colab import run_stage5_phase_g_alpha as alpha  # noqa: E402
from colab.run_stage5_phase_g_multitarget_prepare import prepare_data  # noqa: E402
from training.branching_relations_task import (  # noqa: E402
    BranchingRelationsConfig,
    build_rows,
    row_manifest,
    validate_rows,
)
from training.phase_g_multitarget_spec import (  # noqa: E402
    assert_posterior_control_gate_lock,
    resolve_posterior_control_gate_lock_path,
)


A1_TARGET_ENTROPY = 0.1432
EXPECTED_A1_CALIBRATION_MANIFEST = {
    "rows": 512,
    "row_id_sha256": "755291d493cac76515228c16e73c1478038228e678b499c2bed2fa28be1871a8",
    "row_sha256": "71b89e00316683102fbb56a2f244121b1f8fe378a80c269059a3bb5076e9e64e",
}
LOCKED_A0_THRESHOLDS: dict[str, float | int] = {
    "STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS": 32,
    "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE": 0.60,
    "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT": 0.15,
    "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS": 24,
    "STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE": 0.05,
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(command: list[str], *, allow_blocked: bool = False) -> int:
    printable = "$ " + " ".join(map(str, command))
    print(printable, flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        alpha.append_runtime_transcript(line)
    return_code = process.wait()
    if return_code and not (allow_blocked and return_code == 2):
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def publish_receipts(run_dir: Path, message: str) -> None:
    """Publish all textual receipts while keeping checkpoints on Drive only."""

    subprocess.run(
        ["git", "pull", "--rebase", "--autostash", "origin", "main"],
        cwd=ROOT,
        check=False,
    )
    receipt_suffixes = {".json", ".jsonl", ".md", ".log"}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix in receipt_suffixes:
            subprocess.run(
                ["git", "add", "-f", path.relative_to(ROOT).as_posix()],
                cwd=ROOT,
                check=True,
            )
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", "origin", "main"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def deterministic_control_screen(*, run_dir: Path, keeper: Path) -> Path:
    output_dir = run_dir / "deterministic" / "posterior_control"
    rows_path = output_dir / "rows.jsonl"
    if rows_path.exists() and (output_dir / "summary.json").exists():
        print(f"resume_deterministic_control_screen={output_dir}", flush=True)
        return rows_path
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "eval/eval_branching_relations.py",
            "--data_jsonl",
            str((run_dir / "data" / "posterior_control.jsonl").relative_to(ROOT)),
            "--checkpoint",
            str(keeper),
            "--output_jsonl",
            str(rows_path.relative_to(ROOT)),
            "--output_summary",
            str((output_dir / "summary.json").relative_to(ROOT)),
            "--bridge_projection_mode",
            "split",
            "--dtype",
            "bfloat16",
            "--adapter_dtype",
            "float32",
            "--device",
            "cuda",
        ]
    )
    return rows_path


def verify_a1_continuity_receipts() -> dict[str, Any]:
    """Verify original coverage constants and manifests before A0 consumes GPU."""

    test_rows = build_rows(
        BranchingRelationsConfig(rows_per_depth=128, max_depth=4),
        split="test",
        rendering="verbal",
        n_symbols=20,
    )
    calibration_rows = build_rows(
        BranchingRelationsConfig(rows_per_depth=128, max_depth=4),
        split="calibration",
        rendering="verbal",
        n_symbols=20,
    )
    for name, rows in (("test", test_rows), ("calibration", calibration_rows)):
        validation = validate_rows(rows)
        if validation["status"] != "passed":
            raise AssertionError(f"Invalid frozen A1 {name} rows: {validation['errors'][:5]}")
    if row_manifest(test_rows) != alpha.EXPECTED_TEST_MANIFEST:
        raise AssertionError("A1 frozen coverage test manifest differs from the original receipt")
    if row_manifest(calibration_rows) != EXPECTED_A1_CALIBRATION_MANIFEST:
        raise AssertionError("A1 frozen coverage calibration manifest differs from the original receipt")
    if not alpha.DETERMINISTIC_TEST_ROWS.exists():
        raise FileNotFoundError(
            "Missing frozen deterministic test receipt required for the A1 coverage comparison: "
            f"{alpha.DETERMINISTIC_TEST_ROWS}"
        )
    deterministic_ids = [
        json.loads(line)["id"]
        for line in alpha.DETERMINISTIC_TEST_ROWS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if deterministic_ids != [row["id"] for row in test_rows]:
        raise AssertionError("A1 deterministic receipt IDs disagree with the frozen coverage rows")
    if A1_TARGET_ENTROPY != 0.1432:
        raise AssertionError("A1 entropy-match constant changed")
    return {
        "target_entropy": A1_TARGET_ENTROPY,
        "entropy_tolerance": 0.1 * A1_TARGET_ENTROPY,
        "test_manifest": alpha.EXPECTED_TEST_MANIFEST,
        "calibration_manifest": EXPECTED_A1_CALIBRATION_MANIFEST,
        "deterministic_test_receipt": alpha.repo_relative_text(
            alpha.DETERMINISTIC_TEST_ROWS
        ),
    }


def run_guidance_arm(
    *,
    run_dir: Path,
    drive_artifacts: Path,
    drive_checkpoints: Path,
    keeper: Path,
    deterministic_rows: Path,
    thresholds: dict[str, float | int],
    coefficient: float,
    steps: int,
    label: str,
) -> tuple[dict[str, Any], int]:
    """Train and score one pre-authorized Phase G posterior-control arm."""

    train_dir = run_dir / "train" / label
    raw_path = train_dir / f"phase_g_raw_step_{steps}.pt"
    ema_path = train_dir / f"phase_g_ema_step_{steps}.pt"
    drive_raw = drive_checkpoints / f"{label}_raw.pt"
    drive_ema = drive_checkpoints / f"{label}_ema.pt"
    progress_path = train_dir / "training_progress.pt"
    drive_progress = drive_checkpoints / f"{label}_progress.pt"
    resume_state = alpha.phase_g_arm_resume_state(
        summary_path=train_dir / "summary.json",
        raw_path=raw_path,
        ema_path=ema_path,
        drive_raw_path=drive_raw,
        drive_ema_path=drive_ema,
        progress_path=progress_path,
        drive_progress_path=drive_progress,
    )
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="training",
        status="resume_preflight",
        arm=label,
        kl_coefficient=coefficient,
        resume_state=resume_state,
    )
    if resume_state == "drive_complete":
        train_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_raw, raw_path)
        shutil.copy2(drive_ema, ema_path)
    elif resume_state != "local_complete":
        alpha.write_runtime_status(
            run_dir,
            drive_artifacts,
            stage="training",
            status="started",
            arm=label,
            kl_coefficient=coefficient,
        )
        run(
            [
                sys.executable,
                "training/train_phase_g_alpha.py",
                "--train_jsonl",
                str((run_dir / "data" / "train.jsonl").relative_to(ROOT)),
                "--keeper",
                str(keeper),
                "--expected_keeper_sha256",
                alpha.KEEPER_SHA256,
                "--output_dir",
                str(train_dir.relative_to(ROOT)),
                "--steps",
                str(steps),
                "--kl_coefficient",
                str(coefficient),
                "--sampling_policy",
                "base_problem_uniform",
                "--injection_scale_init",
                "0.001",
                "--checkpoint_every",
                os.environ.get("STAGE5_PHASE_G_MULTITARGET_CHECKPOINT_EVERY", "100"),
                "--progress_checkpoint",
                str(progress_path),
                "--progress_backup_path",
                str(drive_progress),
                "--progress_backup_dir",
                str(drive_artifacts / "train" / label),
                "--seed",
                os.environ.get("STAGE5_PHASE_G_MULTITARGET_SEED", "20260718"),
                "--device",
                "cuda",
                "--dtype",
                "bfloat16",
            ]
        )
    train_summary = read_json(train_dir / "summary.json")
    if train_summary.get("config", {}).get("sampling_policy") != "base_problem_uniform":
        raise AssertionError("Phase G A0 training did not use base-problem-uniform sampling")
    if int(train_summary.get("frozen_gradient_assertions", 0)) != steps:
        raise AssertionError("Phase G A0 did not assert frozen gradients after every step")
    selected_checkpoint = alpha.resolve_repo_path(train_summary["ema_checkpoint"])
    if not selected_checkpoint.exists():
        raise FileNotFoundError(f"Missing completed EMA checkpoint: {selected_checkpoint}")
    drive_checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(alpha.resolve_repo_path(train_summary["raw_checkpoint"]), drive_raw)
    shutil.copy2(selected_checkpoint, drive_ema)
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(run_dir, f"Train Phase G multi-target {label} {run_dir.name} [skip ci]")

    control_eval = run_dir / "posterior_control_eval" / label
    cache_path = drive_artifacts / "posterior_control_eval" / label / "row_cache.jsonl"
    run(
        [
            sys.executable,
            "eval/eval_phase_g_alpha.py",
            "--data_jsonl",
            str((run_dir / "data" / "posterior_control.jsonl").relative_to(ROOT)),
            "--deterministic_rows_jsonl",
            str(deterministic_rows.relative_to(ROOT)),
            "--keeper",
            str(keeper),
            "--expected_keeper_sha256",
            alpha.KEEPER_SHA256,
            "--guidance_checkpoint",
            str(selected_checkpoint),
            "--output_dir",
            str(control_eval.relative_to(ROOT)),
            "--resume_cache_path",
            str(cache_path),
            "--sample_counts",
            "1",
            "--no-include_temperature",
            "--no-include_iso_compute",
            "--include_posterior_teacher",
            "--skip_k1_parity",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
        ]
    )
    audit_path = run_dir / "posterior_control" / label / "audit.json"
    run(
        [
            sys.executable,
            "eval/analyze_phase_g_multimodal_supervision.py",
            "--train_jsonl",
            str((run_dir / "data" / "train.jsonl").relative_to(ROOT)),
            "--test_jsonl",
            str((run_dir / "data" / "posterior_control.jsonl").relative_to(ROOT)),
            "--row_cache_jsonl",
            str(cache_path),
            "--output_json",
            str(audit_path.relative_to(ROOT)),
            "--output_md",
            str((run_dir / "posterior_control" / label / "audit.md").relative_to(ROOT)),
        ]
    )
    gate_path = run_dir / "posterior_control" / label / "gate.json"
    gate_result = run(
        [
            sys.executable,
            "eval/score_phase_g_posterior_control.py",
            "--audit_json",
            str(audit_path.relative_to(ROOT)),
            "--output_json",
            str(gate_path.relative_to(ROOT)),
            "--min_multi_target_groups",
            str(thresholds["STAGE5_PHASE_G_MULTITARGET_MIN_GROUPS"]),
            "--min_teacher_target_rate",
            str(thresholds["STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE"]),
            "--min_teacher_prior_target_lift",
            str(thresholds["STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT"]),
            "--max_teacher_prior_target_lift_p_value",
            str(thresholds["STAGE5_PHASE_G_MULTITARGET_MAX_TEACHER_PRIOR_TARGET_LIFT_PVALUE"]),
            "--min_teacher_switching_groups",
            str(thresholds["STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS"]),
        ],
        allow_blocked=True,
    )
    gate = read_json(gate_path)
    arm = {
        "label": label,
        "kl_coefficient": coefficient,
        "ema_checkpoint": alpha.repo_relative_text(selected_checkpoint),
        "training_summary": str((train_dir / "summary.json").relative_to(ROOT)),
        "gate": gate,
    }
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="posterior_control",
        status="passed" if gate_result == 0 else "blocked",
        arm=label,
        kl_coefficient=coefficient,
        gate=gate,
    )
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(run_dir, f"Score Phase G multi-target {label} {run_dir.name} [skip ci]")
    return arm, gate_result


def main(run_id: str | None = None) -> int:
    gate_lock_path = resolve_posterior_control_gate_lock_path(
        ROOT,
        os.environ.get("STAGE5_PHASE_G_MULTITARGET_GATE_LOCK"),
    )
    run_id = run_id or os.environ.get(
        "STAGE5_PHASE_G_MULTITARGET_RUN_ID",
        "stage5_phase_g_multitarget_control_20260718",
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    drive_checkpoints = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / run_id
    )
    if drive_artifacts.exists():
        shutil.copytree(drive_artifacts, run_dir, dirs_exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    alpha.configure_runtime_transcript(run_dir / "runtime.log")
    steps = int(os.environ.get("STAGE5_PHASE_G_MULTITARGET_STEPS", "1000"))
    requested_coefficient = os.environ.get("STAGE5_PHASE_G_MULTITARGET_KL", "0.001")
    if steps < 1:
        raise ValueError("Phase G multi-target steps must be positive")
    if float(requested_coefficient) != 0.001:
        raise ValueError(
            "A0 is locked to the primary KL coefficient 0.001. The only allowed "
            "confirmation coefficient is the internally contingent 0.0001 arm."
        )
    summary: dict[str, Any] = {
        "kind": "stage5_phase_g_multitarget_posterior_control",
        "status": "started",
        "run_id": run_id,
        "keeper_sha256": alpha.KEEPER_SHA256,
        "steps": steps,
        "primary_kl_coefficient": 0.001,
        "contingent_confirmation_kl_coefficient": 0.0001,
        "sampling_policy": "base_problem_uniform",
        "gate_lock": alpha.repo_relative_text(gate_lock_path),
    }
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="startup",
        status="started",
        run_id=run_id,
        gate_lock=alpha.repo_relative_text(gate_lock_path),
    )

    summary["a1_continuity_receipts"] = verify_a1_continuity_receipts()
    keeper = alpha.restore_keeper(run_dir)
    prepared = prepare_data(
        run_dir / "data",
        train_rows_per_depth=128,
        control_rows_per_depth=8,
    )
    if prepared["train"]["validation"]["groups_with_multiple_targets"] != prepared["train"]["validation"]["base_problem_groups"]:
        raise AssertionError("Multi-target train rows do not expose multiple targets for every prompt")
    train_rows = [
        json.loads(line)
        for line in (run_dir / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    control_rows = [
        json.loads(line)
        for line in (run_dir / "data" / "posterior_control.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_problem_ids = {str(row["base_problem_id"]) for row in train_rows}
    control_problem_ids = {str(row["base_problem_id"]) for row in control_rows}
    if train_problem_ids & control_problem_ids:
        raise AssertionError("Train and posterior-control base_problem_id sets overlap")
    if len(control_problem_ids) != 32 or len(control_rows) != 106:
        raise AssertionError(
            "A0 control surface must be the locked 32-group/106-variant held-out set"
        )
    thresholds = assert_posterior_control_gate_lock(
        read_json(gate_lock_path),
        control_rows,
    )
    if thresholds != LOCKED_A0_THRESHOLDS:
        raise AssertionError(
            "Phase G A0 gate lock does not match the strategy-locked margin table"
        )
    summary["data"] = prepared
    summary["locked_gate_thresholds"] = thresholds
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(run_dir, f"Prepare Phase G multi-target control {run_id} [skip ci]")

    deterministic_rows = deterministic_control_screen(run_dir=run_dir, keeper=keeper)
    primary_arm, gate_result = run_guidance_arm(
        run_dir=run_dir,
        drive_artifacts=drive_artifacts,
        drive_checkpoints=drive_checkpoints,
        keeper=keeper,
        deterministic_rows=deterministic_rows,
        thresholds=thresholds,
        coefficient=0.001,
        steps=steps,
        label="kl_0p001",
    )
    summary["arms"] = [primary_arm]
    if gate_result == 2:
        confirmation_arm, gate_result = run_guidance_arm(
            run_dir=run_dir,
            drive_artifacts=drive_artifacts,
            drive_checkpoints=drive_checkpoints,
            keeper=keeper,
            deterministic_rows=deterministic_rows,
            thresholds=thresholds,
            coefficient=0.0001,
            steps=steps,
            label="kl_0p0001_confirmation",
        )
        summary["arms"].append(confirmation_arm)
    selected_arm = next(
        (arm for arm in summary["arms"] if arm["gate"]["status"] == "passed"),
        None,
    )
    summary["selected_arm"] = selected_arm["label"] if selected_arm else None
    summary["status"] = (
        "posterior_control_passed"
        if selected_arm
        else "blocked_posterior_control_after_confirmation"
    )
    summary["posterior_control_gate"] = summary["arms"][-1]["gate"]
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="posterior_control",
        status=summary["status"],
        gate=summary["posterior_control_gate"],
    )
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(run_dir, f"Record Phase G multi-target control {run_id} [skip ci]")
    return gate_result


def guarded_main() -> int:
    run_id = os.environ.get(
        "STAGE5_PHASE_G_MULTITARGET_RUN_ID",
        "stage5_phase_g_multitarget_control_20260718",
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    try:
        return main(run_id)
    except BaseException as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        alpha.record_runtime_failure(run_dir, drive_artifacts, exc)
        raise
    finally:
        alpha.configure_runtime_transcript(None)


if __name__ == "__main__":
    raise SystemExit(guarded_main())
