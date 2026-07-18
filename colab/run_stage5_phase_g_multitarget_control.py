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
from training.phase_g_multitarget_spec import (  # noqa: E402
    assert_posterior_control_gate_lock,
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def required_gate_lock_path(environment: dict[str, str]) -> Path:
    raw = environment.get("STAGE5_PHASE_G_MULTITARGET_GATE_LOCK", "").strip()
    if not raw:
        raise RuntimeError(
            "STAGE5_PHASE_G_MULTITARGET_GATE_LOCK must identify a committed "
            "pre-training posterior-control gate-lock JSON"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase G posterior-control gate lock: {path}")
    return path


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


def main(run_id: str | None = None) -> int:
    gate_lock_path = required_gate_lock_path(os.environ)
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
    coefficient = float(os.environ.get("STAGE5_PHASE_G_MULTITARGET_KL", "0.001"))
    if steps < 1 or coefficient < 0.0:
        raise ValueError("Phase G multi-target steps and KL must be nonnegative")
    summary: dict[str, Any] = {
        "kind": "stage5_phase_g_multitarget_posterior_control",
        "status": "started",
        "run_id": run_id,
        "keeper_sha256": alpha.KEEPER_SHA256,
        "steps": steps,
        "kl_coefficient": coefficient,
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

    keeper = alpha.restore_keeper(run_dir)
    prepared = prepare_data(run_dir / "data")
    if prepared["train"]["validation"]["groups_with_multiple_targets"] != prepared["train"]["validation"]["base_problem_groups"]:
        raise AssertionError("Multi-target train rows do not expose multiple targets for every prompt")
    if prepared["train"]["base_question_sha256"] == prepared["control"]["base_question_sha256"]:
        raise AssertionError("Train and posterior-control prompt manifests overlap")
    thresholds = assert_posterior_control_gate_lock(
        read_json(gate_lock_path),
        [
            json.loads(line)
            for line in (run_dir / "data" / "posterior_control.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ],
    )
    summary["data"] = prepared
    summary["locked_gate_thresholds"] = thresholds
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(run_dir, f"Prepare Phase G multi-target control {run_id} [skip ci]")

    deterministic_rows = deterministic_control_screen(run_dir=run_dir, keeper=keeper)
    train_dir = run_dir / "train" / "guided"
    raw_path = train_dir / f"phase_g_raw_step_{steps}.pt"
    ema_path = train_dir / f"phase_g_ema_step_{steps}.pt"
    drive_raw = drive_checkpoints / "guided_raw.pt"
    drive_ema = drive_checkpoints / "guided_ema.pt"
    progress_path = train_dir / "training_progress.pt"
    drive_progress = drive_checkpoints / "guided_progress.pt"
    resume_state = alpha.phase_g_arm_resume_state(
        summary_path=train_dir / "summary.json",
        raw_path=raw_path,
        ema_path=ema_path,
        drive_raw_path=drive_raw,
        drive_ema_path=drive_ema,
        progress_path=progress_path,
        drive_progress_path=drive_progress,
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
            resume_state=resume_state,
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
                "--checkpoint_every",
                os.environ.get("STAGE5_PHASE_G_MULTITARGET_CHECKPOINT_EVERY", "100"),
                "--progress_checkpoint",
                str(progress_path),
                "--progress_backup_path",
                str(drive_progress),
                "--progress_backup_dir",
                str(drive_artifacts / "train" / "guided"),
                "--seed",
                os.environ.get("STAGE5_PHASE_G_MULTITARGET_SEED", "20260718"),
                "--device",
                "cuda",
                "--dtype",
                "bfloat16",
            ]
        )
    train_summary = read_json(train_dir / "summary.json")
    selected_checkpoint = alpha.resolve_repo_path(train_summary["ema_checkpoint"])
    if not selected_checkpoint.exists():
        raise FileNotFoundError(f"Missing completed EMA checkpoint: {selected_checkpoint}")
    drive_checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(alpha.resolve_repo_path(train_summary["raw_checkpoint"]), drive_raw)
    shutil.copy2(selected_checkpoint, drive_ema)
    summary["ema_checkpoint"] = alpha.repo_relative_text(selected_checkpoint)
    summary["training"] = {
        "summary": str((train_dir / "summary.json").relative_to(ROOT)),
        "frozen_gradient_assertions": train_summary.get("frozen_gradient_assertions"),
        "sampling_policy": train_summary.get("config", {}).get("sampling_policy"),
    }
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(run_dir, f"Train Phase G multi-target control {run_id} [skip ci]")

    control_eval = run_dir / "posterior_control_eval"
    cache_path = drive_artifacts / "posterior_control_eval" / "row_cache.jsonl"
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
    audit_path = run_dir / "posterior_control_audit.json"
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
            str((run_dir / "posterior_control_audit.md").relative_to(ROOT)),
        ]
    )
    gate_path = run_dir / "posterior_control_gate.json"
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
            "--min_teacher_prior_distinct_prediction_lift",
            str(thresholds["STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_DISTINCT_LIFT"]),
        ],
        allow_blocked=True,
    )
    gate = read_json(gate_path)
    summary["posterior_control_gate"] = gate
    summary["status"] = (
        "posterior_control_passed" if gate_result == 0 else "blocked_posterior_control"
    )
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="posterior_control",
        status=summary["status"],
        gate=gate,
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
