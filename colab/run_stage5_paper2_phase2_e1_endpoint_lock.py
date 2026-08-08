"""Compute and publish final Option-B endpoint identities for the E1 lock."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_e1_eval_d_20260808"
OPTION_B_ID = "stage5_paper2_phase2_option_b_20260807"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
REGISTRATION = ROOT / "training/paper2_phase2_e1_confirmation_preregistration.draft.json"
OUTPUT = RUN_DIR / "receipts/e1_endpoint_lock_preparation.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_OPTION_B = DRIVE_ROOT / OPTION_B_ID / "private/option_b"
DRIVE_RECEIPTS = DRIVE_ROOT / RUN_ID / "receipts"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "eval.prepare_paper2_phase2_e1_endpoint_lock",
        "--registration",
        str(REGISTRATION),
        "--output",
        str(OUTPUT),
    ]
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            path = DRIVE_OPTION_B / f"seed_{seed}_{arm}/resume.pt"
            if not path.is_file():
                raise FileNotFoundError(f"missing final Option B endpoint: {path}")
            command.extend([f"--seed_{seed}_{arm}", str(path)])
    run(command)
    receipt = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if receipt.get("ready_for_lock_transcription") is not True:
        raise RuntimeError("E1 endpoint integrity receipt is not ready for lock")
    if receipt.get("read_once_scoring_spent") is not False:
        raise RuntimeError("E1 endpoint integrity pass spent read-once scoring")
    DRIVE_RECEIPTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, DRIVE_RECEIPTS / OUTPUT.name)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", OUTPUT.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record E1 endpoint lock identities [skip ci]"])
        run(["git", "push", "origin", "main"])
    print("E1 endpoint lock identities landed; EVAL-D remains unscored.", flush=True)


if __name__ == "__main__":
    main()
