"""Colab launcher for the zero-GPU D0 prelaunch receipts."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_D0_PRELAUNCH_VERSION = "paper2_d0_prelaunch_v1"
# Safety marker: read-only prelaunch post-processing no model no optimizer no evaluation partition
# Safety marker: authenticated figure-review addendum
# Safety marker: binned target policy receipt before training
# Safety marker: teacher demand uses each teachers own rejection population
# Safety marker: Drive mount retry with explicit Pharma Initiatives account authorization
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
assert GH_TOKEN, "D0 prelaunch requires GH_TOKEN in Colab secrets."


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH_TOKEN, "****"), flush=True)
    process = subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy())
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def ensure_drive() -> None:
    my_drive = Path("/content/drive/MyDrive")
    if my_drive.is_dir():
        print("Drive already mounted at /content/drive", flush=True)
        return
    failures: list[str] = []
    for attempt, force_remount in enumerate((False, True), start=1):
        try:
            print(
                f"Drive mount attempt {attempt}/2 force_remount={int(force_remount)}. "
                "Authorize the Pharma Initiatives Google account when prompted.",
                flush=True,
            )
            drive.mount(
                "/content/drive",
                force_remount=force_remount,
                timeout_ms=240_000,
            )
        except (ValueError, TimeoutError) as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        if my_drive.is_dir():
            return
        time.sleep(3)
    raise RuntimeError(
        "Google Drive could not be mounted after two attempts. In a separate Colab cell, run "
        "`from google.colab import drive; drive.mount('/content/drive', force_remount=True)` "
        "and authorize the Pharma Initiatives account, then rerun this launch cell. "
        f"Failures: {failures}"
    )


ensure_drive()
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
        "tests/test_speculative_depth_d0_launch_amendment.py",
        "tests/test_speculative_depth_d0_postlock.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_d0_prelaunch_target",
    ]
)
run([sys.executable, "colab/run_stage5_paper2_d0_prelaunch.py"])
print("D0 prelaunch receipts completed and landed.", flush=True)
