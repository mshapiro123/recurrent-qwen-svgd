"""A100 launcher for the no-training Stage 2B loss calibration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime


STAGE5_PAPER2_STAGE2B_LOSS_CALIBRATION_VERSION = "paper2_stage2b_loss_calibration_v1"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ["STAGE5_BOOTSTRAP_REF"]


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


drive.mount("/content/drive", force_remount=False)
if ROOT.exists():
    run(["git", "fetch", "origin", "main"])
    run(["git", "reset", "--hard", REF])
else:
    token = os.environ.get("GH_TOKEN")
    clone = "https://github.com/mshapiro123/recurrent-qwen-svgd.git"
    if token:
        clone = f"https://x-access-token:{token}@github.com/mshapiro123/recurrent-qwen-svgd.git"
    run(["git", "clone", clone, str(ROOT)], cwd=Path("/content"))
    run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_stage2b_depth.py"])
run([sys.executable, "-u", "-m", "colab.run_stage5_paper2_stage2b_loss_calibration"])
print("Stage 2B loss calibration complete; releasing runtime.", flush=True)
runtime.unassign()
