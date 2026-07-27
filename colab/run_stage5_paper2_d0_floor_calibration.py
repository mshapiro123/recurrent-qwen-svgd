"""Run and publish the registered D0 forced-depth floor calibration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import D0_RUN_ID, validate_cache_summary
from training.speculative_depth_d0_spec import DRAFTER_CHECKPOINT_SHA256
from colab.run_stage5_paper2_d0_teacher_cache import (
    CHECKPOINT_ALIAS,
    CHECKPOINT_STAGE_STATE,
    resolve_checkpoint_source,
)


# Safety marker: floor calibration only no optimizer no training
LOCK_RUN = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
RUN_DIR = ROOT / "outputs" / "stage5" / D0_RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_RUN_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{D0_RUN_ID}",
    )
)
def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
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
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_event(status: str, **details: Any) -> None:
    payload = {"kind": "paper2_d0_floor_status", "status": status, **details}
    print("d0_floor_status:", json.dumps(payload, sort_keys=True), flush=True)
    try:
        write_json(DRIVE_ROOT / "receipts" / "floor_status.json", payload)
    except Exception as error:
        print(f"d0_floor_status_write_failed={error!r}", flush=True)


def restore_calibration() -> Path:
    manifest = read_json(LOCK_RUN / "data_manifest.json")
    receipt = manifest["artifacts"]["calibration"]
    source = Path(receipt["drive_path"])
    destination = RUN_DIR / "private_inputs" / "calibration.jsonl"
    print(f"d0_floor_calibration_preflight path={source} exists={source.exists()}", flush=True)
    if not source.exists():
        raise FileNotFoundError(f"D0 locked calibration partition is missing on Drive: {source}")
    observed = sha256_file(source)
    print(
        f"d0_floor_calibration_sha observed={observed} expected={receipt['sha256']}",
        flush=True,
    )
    if observed != receipt["sha256"]:
        raise RuntimeError(
            f"D0 locked calibration partition hash mismatch: observed={observed} "
            f"expected={receipt['sha256']}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != receipt["sha256"]:
        shutil.copy2(source, destination)
    return destination


def restore_checkpoint() -> tuple[Path, list[dict[str, Any]]]:
    explicit = os.environ.get("STAGE5_PAPER2_D0_DRAFTER_CHECKPOINT", "").strip()
    candidates = ([Path(explicit)] if explicit else []) + [CHECKPOINT_ALIAS, CHECKPOINT_STAGE_STATE]
    source, diagnostics = resolve_checkpoint_source(
        candidates, expected_sha256=DRAFTER_CHECKPOINT_SHA256
    )
    print("d0_floor_checkpoint_resolution:", json.dumps(diagnostics, sort_keys=True), flush=True)
    destination = RUN_DIR / "restored" / "t1_lite_r_raw_step_10500.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != DRAFTER_CHECKPOINT_SHA256:
        shutil.copy2(source, destination)
    return destination, diagnostics


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 floor produced no aggregate receipt changes")
    run(["git", "commit", "-m", "Record Paper Two D0 floor calibration [skip ci]"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return commit


def main() -> int:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    status_event("preflight_started", run_id=D0_RUN_ID, drive_root=str(DRIVE_ROOT))
    cache_summary_path = RUN_DIR / "labeling" / "summary.json"
    print(f"d0_floor_cache_summary path={cache_summary_path} exists={cache_summary_path.exists()}", flush=True)
    cache_summary = read_json(cache_summary_path)
    validate_cache_summary(cache_summary)
    status_event(
        "teacher_cache_validated",
        cache_summary=str(cache_summary_path),
        lock_commit=cache_summary.get("lock_commit"),
    )
    calibration = restore_calibration()
    status_event("calibration_restored", calibration=str(calibration))
    checkpoint, checkpoint_resolution = restore_checkpoint()
    status_event(
        "checkpoint_restored",
        checkpoint=str(checkpoint),
        checkpoint_sha256=DRAFTER_CHECKPOINT_SHA256,
        resolution=checkpoint_resolution,
    )
    floor_summary = RUN_DIR / "floor" / "summary.json"
    private_rows = DRIVE_ROOT / "private" / "floor" / "floor_rows.json"
    status_event("floor_evaluation_started", output_summary=str(floor_summary))
    run(
        [
            sys.executable,
            "eval/eval_speculative_depth_d0_floor.py",
            "--preregistration",
            str(LOCK_RUN / "preregistration.json"),
            "--data_jsonl",
            str(calibration),
            "--teacher_cache_summary",
            str(cache_summary_path),
            "--checkpoint",
            str(checkpoint),
            "--output_summary",
            str(floor_summary),
            "--private_rows_output",
            str(private_rows),
            "--resume_dir",
            str(DRIVE_ROOT / "private" / "floor" / "row_cache_pretraining"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_D0_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
            "--batch_size",
            os.environ.get("STAGE5_PAPER2_D0_FLOOR_BATCH_SIZE", "1"),
        ]
    )
    summary = read_json(floor_summary)
    drive_receipt = DRIVE_ROOT / "receipts" / "floor_summary.json"
    drive_receipt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(floor_summary, drive_receipt)
    summary_md = RUN_DIR / "floor" / "summary.md"
    summary_md.write_text(
        "# Paper Two D0 Floor Calibration\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Rejected calibration positions: `{summary['rejected_positions']}`\n"
        f"- Selected branch: `{summary['calibration']['branch']}`\n"
        "- Forced depths: 1 through 6\n"
        "- Optimizer steps: 0\n",
        encoding="utf-8",
    )
    commit = publish([floor_summary, summary_md])
    status_event("complete", publish_commit=commit, summary=str(floor_summary))
    print(json.dumps({**summary, "publish_commit": commit}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        status_event(
            "errored",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
