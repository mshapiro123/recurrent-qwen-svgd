"""Colab CPU launcher for final E1 endpoint lock identities."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_E1_ENDPOINT_LOCK_VERSION = "paper2_phase2_e1_endpoint_lock_v1"
# Safety marker: CPU-only checkpoint-container integrity no model instantiation
# Safety marker: final Option B file hashes semantic trainable-state digests step 20000
# Safety marker: no EVAL-D contact no outcome score no optimizer no training
# Safety marker: read-once E1 scoring remains unspent before lock
# Safety marker: colab/run_stage5_paper2_phase2_e1_endpoint_lock.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "reset", "--hard", REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_phase2_e1_endpoint_lock.py"])
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_e1_endpoint_lock.py"])
