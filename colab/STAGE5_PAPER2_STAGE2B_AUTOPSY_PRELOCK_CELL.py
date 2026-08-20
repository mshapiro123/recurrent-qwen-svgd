"""Colab CPU launcher for the Stage 2B-A prelock inventory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_STAGE2B_AUTOPSY_PRELOCK_VERSION = "paper2_stage2b_autopsy_prelock_v1"
# Safety marker: no model load no optimizer no training CONFIRM and EVAL-E sealed
# Contract marker: freeze 256-row DEV-2 subsample before autopsy model contact
# Contract marker: inventory historical checkpoints without substitution
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = os.environ.get("GH_TOKEN", "").strip()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or (ROOT if ROOT.exists() else Path("/content")), check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git" if GH else f"https://github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "fetch", "origin", REF])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_stage2b_autopsy.py"])
os.environ["STAGE2B_AUTOPSY_MODE"] = "prelock"
result = subprocess.run(
    [sys.executable, "-u", "-m", "colab.run_stage5_paper2_stage2b_autopsy"],
    cwd=ROOT,
)
if result.returncode not in {0, 2}:
    raise subprocess.CalledProcessError(result.returncode, result.args)
print(f"Stage 2B-A prelock returned code {result.returncode}; code 2 is a receipted checkpoint-inventory block.", flush=True)
