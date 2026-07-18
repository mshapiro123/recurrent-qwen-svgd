from __future__ import annotations

import json
import sys
from pathlib import Path

from colab import run_stage5_phase_g_alpha as runner


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_runtime_status_is_mirrored_to_drive(tmp_path: Path) -> None:
    run_dir = tmp_path / "local"
    drive_dir = tmp_path / "drive"

    payload = runner.write_runtime_status(
        run_dir,
        drive_dir,
        stage="training",
        status="started",
        arm="kl_0p0001",
    )

    local = read_json(run_dir / "runtime_status.json")
    mirrored = read_json(drive_dir / "runtime_status.json")
    assert local == mirrored == payload
    assert payload["kind"] == "stage5_phase_g_alpha_runtime_status"
    assert payload["stage"] == "training"
    assert payload["arm"] == "kl_0p0001"


def test_runtime_failure_preserves_stage_and_traceback(tmp_path: Path) -> None:
    run_dir = tmp_path / "local"
    drive_dir = tmp_path / "drive"
    transcript = run_dir / "runtime.log"
    runner.configure_runtime_transcript(transcript)
    runner.append_runtime_transcript("important child failure detail\n")
    runner.write_runtime_status(
        run_dir,
        drive_dir,
        stage="training",
        status="started",
        arm="kl_0p001",
    )

    try:
        raise RuntimeError("deliberate phase-g failure")
    except RuntimeError as exc:
        payload = runner.record_runtime_failure(run_dir, drive_dir, exc)

    assert payload["status"] == "error"
    assert payload["stage"] == "training"
    assert payload["arm"] == "kl_0p001"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["exception"] == "deliberate phase-g failure"
    assert any("deliberate phase-g failure" in line for line in payload["traceback_tail"])
    assert payload["child_log_tail"] == ["important child failure detail"]
    assert read_json(drive_dir / "runtime_error.json") == payload
    runner.configure_runtime_transcript(None)


def test_child_output_is_written_to_durable_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "runtime.log"
    runner.configure_runtime_transcript(transcript)
    try:
        return_code = runner.run(
            [
                sys.executable,
                "-c",
                "print('phase-g-child-output'); raise SystemExit(2)",
            ],
            allow_blocked=True,
        )
    finally:
        runner.configure_runtime_transcript(None)

    assert return_code == 2
    text = transcript.read_text(encoding="utf-8")
    assert "phase-g-child-output" in text
    assert "return_code=2" in text


def test_arm_resume_state_distinguishes_partial_and_drive_complete(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    drive_dir = tmp_path / "drive"
    train_dir.mkdir()
    drive_dir.mkdir()
    summary = train_dir / "summary.json"
    raw = train_dir / "raw.pt"
    ema = train_dir / "ema.pt"
    drive_raw = drive_dir / "raw.pt"
    drive_ema = drive_dir / "ema.pt"

    summary.write_text("{}\n", encoding="utf-8")
    assert runner.phase_g_arm_resume_state(
        summary_path=summary,
        raw_path=raw,
        ema_path=ema,
        drive_raw_path=drive_raw,
        drive_ema_path=drive_ema,
    ) == "partial_requires_restart"

    progress = drive_dir / "progress.pt"
    progress.write_bytes(b"progress")
    assert runner.phase_g_arm_resume_state(
        summary_path=summary,
        raw_path=raw,
        ema_path=ema,
        drive_raw_path=drive_raw,
        drive_ema_path=drive_ema,
        drive_progress_path=progress,
    ) == "in_progress_resumable"

    drive_raw.write_bytes(b"raw")
    drive_ema.write_bytes(b"ema")
    assert runner.phase_g_arm_resume_state(
        summary_path=summary,
        raw_path=raw,
        ema_path=ema,
        drive_raw_path=drive_raw,
        drive_ema_path=drive_ema,
    ) == "drive_complete"
