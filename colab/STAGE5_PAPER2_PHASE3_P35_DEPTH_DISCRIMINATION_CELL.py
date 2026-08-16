"""Colab launcher for the authorized no-training D1 plus K+ probe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_PHASE3_P35_DEPTH_DISCRIMINATION_VERSION = "paper2_phase3_p35_depth_discrimination_v1"
# Safety marker: Arm-S EMA K1-K4 registered and K5-K6 disclosed last-step clamp
# Safety marker: DEV score-only no optimizer no training CONFIRM and EVAL-E sealed
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
run(["nvidia-smi"])
url = f"https://x-access-token:{GH}@github.com/{REPO}.git" if GH else f"https://github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "fetch", "origin", f"refs/heads/{SOURCE_BRANCH}:refs/remotes/origin/{SOURCE_BRANCH}"])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_paper2_phase3_p35_depth_discrimination.py",
    "tests/test_paper2_phase3_p34_task_inference.py",
    "tests/test_stage5_notebooks.py::test_p35_depth_discrimination_target_is_wired_and_guarded",
])
run([sys.executable, "-u", "-m", "colab.run_stage5_paper2_phase3_p35_depth_discrimination"])
