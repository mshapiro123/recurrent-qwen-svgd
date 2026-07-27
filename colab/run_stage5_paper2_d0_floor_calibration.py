"""Run and publish the registered D0 forced-depth floor calibration."""

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

from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import D0_RUN_ID, validate_cache_summary
from training.speculative_depth_d0_spec import DRAFTER_CHECKPOINT_SHA256


# Safety marker: floor calibration only no optimizer no training
LOCK_RUN = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
RUN_DIR = ROOT / "outputs" / "stage5" / D0_RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_RUN_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{D0_RUN_ID}",
    )
)
CHECKPOINT_SOURCE = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_DRAFTER_CHECKPOINT",
        "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
        "stage5_paper2_t1_lite_r_20260725/checkpoints/t1_lite_r_raw_step_10500.pt",
    )
)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    process = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def restore_calibration() -> Path:
    manifest = read_json(LOCK_RUN / "data_manifest.json")
    receipt = manifest["artifacts"]["calibration"]
    source = Path(receipt["drive_path"])
    destination = RUN_DIR / "private_inputs" / "calibration.jsonl"
    if not source.exists() or sha256_file(source) != receipt["sha256"]:
        raise RuntimeError("D0 locked calibration partition is missing or corrupt on Drive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != receipt["sha256"]:
        shutil.copy2(source, destination)
    return destination


def restore_checkpoint() -> Path:
    if not CHECKPOINT_SOURCE.exists() or sha256_file(CHECKPOINT_SOURCE) != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError("D0 floor cannot restore the locked drafter checkpoint")
    destination = RUN_DIR / "restored" / CHECKPOINT_SOURCE.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != DRAFTER_CHECKPOINT_SHA256:
        shutil.copy2(CHECKPOINT_SOURCE, destination)
    return destination


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
    cache_summary_path = RUN_DIR / "labeling" / "summary.json"
    cache_summary = read_json(cache_summary_path)
    validate_cache_summary(cache_summary)
    calibration = restore_calibration()
    checkpoint = restore_checkpoint()
    floor_summary = RUN_DIR / "floor" / "summary.json"
    private_rows = DRIVE_ROOT / "private" / "floor" / "floor_rows.json"
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
    print(json.dumps({**summary, "publish_commit": commit}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
