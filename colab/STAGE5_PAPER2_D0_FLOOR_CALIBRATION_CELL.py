"""Colab launcher for the registered D0 forced-depth floor calibration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_D0_FLOOR_CALIBRATION_VERSION = "paper2_d0_floor_calibration_v1"
# Safety marker: floor calibration only no optimizer no training
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN and HF_TOKEN, "D0 floor requires GH_TOKEN and HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH_TOKEN, "****"), flush=True)
    process = subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy())
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


drive.mount("/content/drive", force_remount=False)
clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", clone_url])
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
    run(["git", "reset", "--hard", SYNC_REF])
else:
    run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "reset", "--hard", SYNC_REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_speculative_depth_d0_postlock.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_d0_floor_calibration_target",
    ]
)
run([sys.executable, "colab/run_stage5_paper2_d0_floor_calibration.py"])
print("D0 floor calibration completed and aggregate receipts published.", flush=True)
