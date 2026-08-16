"""Colab launcher for the ratified P3.5 preflight or one registered arm."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_PHASE3_P35_VERSION = "paper2_phase3_p35_campaign_v1"
# Bootstrap marker: colab/run_stage5_paper2_phase3_p35.py
# Safety marker: all three exact preflights pass before any P3.5 optimizer exists
# Safety marker: EMA primary pinned 0.02 ceiling repaired v2 causal cache only
# Safety marker: no persistence no CONFIRM no EVAL-E contact
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = os.environ.get("GH_TOKEN", "").strip()
MODE = os.environ.get("P35_MODE", "preflight").strip()
ARM = os.environ.get("P35_ARM", "stabilized").strip()
SEED = int(os.environ.get("P35_SEED", "0"))


def run(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(command).replace(GH, "****") if GH else " ".join(command)
    print("$", printable, flush=True)
    subprocess.run(command, cwd=cwd or ROOT, check=True, env=os.environ.copy())


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"])
url = (
    f"https://x-access-token:{GH}@github.com/{REPO}.git"
    if GH
    else f"https://github.com/{REPO}.git"
)
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(
    [
        "git",
        "fetch",
        "origin",
        f"refs/heads/{SOURCE_BRANCH}:refs/remotes/origin/{SOURCE_BRANCH}",
    ]
)
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase3_p35.py",
        "tests/test_stage5_notebooks.py::test_phase3_p35_target_is_wired_and_ratified",
    ]
)
command = [
    sys.executable,
    "-u",
    "-m",
    "colab.run_stage5_paper2_phase3_p35",
    "--mode",
    MODE,
]
if MODE == "train":
    command.extend(["--arm", ARM, "--seed", str(SEED)])
run(command)
print(f"P3.5 landed mode={MODE} arm={ARM} seed={SEED}; release this GPU session.", flush=True)
