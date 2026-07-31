"""CPU launcher for Phase-2 Stage A cache post-processing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_ORACLE_OVERLAP_VERSION = "paper2_phase2_oracle_overlap_v1"
# Safety marker: cache-only Stage A oracle and hurt overlap no model no scoring no training
# Safety marker: EVAL-C read-once scoring remains spent and is never reexecuted
# Safety marker: position tensors remain private aggregate receipt only
# Safety marker: colab/run_stage5_paper2_phase2_oracle_overlap.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy(), check=True)


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
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_phase2_oracle_overlap.py"])
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_oracle_overlap.py"])
print("Phase-2 cache-only oracle and hurt-overlap receipt landed.", flush=True)
