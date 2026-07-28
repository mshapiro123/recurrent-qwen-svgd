"""CPU Colab launcher for D0 expert-choice Rung 0."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata

STAGE5_PAPER2_D0_EXPERT_CHOICE_RUNG0_VERSION = "paper2_d0_expert_choice_rung0_v1"
# Safety marker: CPU-only frozen OOF score reconstruction no model no optimizer no training
# Safety marker: budgets 0.5 1 2 5 10 20 27 percent global and causal windows 256 1024
# Safety marker: pre-D0 floor harm asymmetry archaeology included
# Safety marker: Pharma Initiatives Drive authorization
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main")
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "reset", "--hard", REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_d0_expert_choice_rung0.py"])
run([sys.executable, "colab/run_stage5_paper2_d0_expert_choice_rung0.py"])
