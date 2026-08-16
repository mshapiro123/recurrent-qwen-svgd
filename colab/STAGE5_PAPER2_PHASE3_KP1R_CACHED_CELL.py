"""Colab launcher for the locked CPU-only KP-1R cached-state rung."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_PHASE3_KP1R_CACHED_VERSION = "paper2_phase3_kp1r_cached_v1"
# Bootstrap marker: colab/run_stage5_paper2_phase3_kp1r_cached.py
# Safety marker: 329-row repaired answer-bearing target entropy audit before scoring
# Safety marker: CPU cached states only no optimizer no training
# Safety marker: CONFIRM and EVAL-E remain sealed
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = os.environ.get("GH_TOKEN", "").strip()


def run(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(command).replace(GH, "****") if GH else " ".join(command)
    print("$", printable, flush=True)
    subprocess.run(command, cwd=cwd or ROOT, check=True, env=os.environ.copy())


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git" if GH else f"https://github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "fetch", "origin", f"refs/heads/{SOURCE_BRANCH}:refs/remotes/origin/{SOURCE_BRANCH}"])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase3_kp1r_t1_teacher.py",
        "tests/test_stage5_notebooks.py::test_kp1r_cached_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "-m", "colab.run_stage5_paper2_phase3_kp1r_cached"])
