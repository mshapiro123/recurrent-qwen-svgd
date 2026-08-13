"""Colab launcher for one approved P3.4 campaign arm."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_PHASE3_P34_VERSION = "paper2_phase3_p34_campaign_v1"
# Safety marker: approved three-session P3.4 campaign main seeds zero one slot seed zero
# Safety marker: exactly twenty DEV looks no CONFIRM no EVAL-E contact
# Safety marker: deterministic resumable windows and hash-bound executed lock
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = os.environ.get("GH_TOKEN", "").strip()
SEED = int(os.environ.get("P34_SEED", "0"))
ARM = os.environ.get("P34_ARM", "main").strip()


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
    "tests/test_paper2_phase3_p34.py",
    "tests/test_paper2_phase3_p34_task_inference.py",
    "tests/test_paper2_phase3_p34_runner.py",
])
run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_p34.py", "--seed", str(SEED), "--arm", ARM])
print(f"P3.4 campaign landed arm={ARM} seed={SEED}; release this GPU session.", flush=True)
