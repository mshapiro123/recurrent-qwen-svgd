"""Freeze EVAL-B, run its one allowed 7B cache pass, and publish aggregates."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from training.speculative_depth_d0_corpus import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_dc0_20260728"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
LOCK = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
D0_DRIVE = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d0_20260726")
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")
CHECKPOINT_SHA = "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def restore_checkpoint() -> Path:
    candidates = [
        D0_DRIVE / "private/training/d0_ema_step_4000.pt",
        D0_DRIVE / "private/train/d0_ema_step_4000.pt",
        D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
    ]
    for source in candidates:
        if source.exists() and sha256_file(source) == CHECKPOINT_SHA:
            destination = RUN_DIR / "runtime/d0_ema_step_4000.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_file(destination) != CHECKPOINT_SHA:
                shutil.copy2(source, destination)
            return destination
    raise FileNotFoundError(f"post-D0 EMA checkpoint not found: {candidates}")


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record fresh DC0 EVAL-B cache receipt [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    checkpoint = restore_checkpoint()
    data = DRIVE_RUN / "private/eval_b/eval_b.jsonl"
    cache_root = DRIVE_RUN / "private/eval_b/teacher_cache"
    private_summary = DRIVE_RUN / "private/eval_b/teacher_cache_summary.json"
    public_summary = RUN_DIR / "eval_b/summary.json"
    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-u",
            "eval/prepare_paper2_dc0_eval_b.py",
            "--data_manifest",
            str(LOCK / "data_manifest.json"),
            "--checkpoint",
            str(checkpoint),
            "--expected_checkpoint_sha256",
            CHECKPOINT_SHA,
            "--output_data",
            str(data),
            "--private_cache_root",
            str(cache_root),
            "--private_cache_summary",
            str(private_summary),
            "--output_summary",
            str(public_summary),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_DC0_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("STAGE5_DC0_ATTN", "sdpa"),
        ]
    )
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(public_summary, receipt_dir / "eval_b_summary.json")
    commit = publish(public_summary)
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
