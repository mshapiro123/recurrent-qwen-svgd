"""Colab launcher for the Phase 3 empirical DEV calibration receipt."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE3_EMPIRICAL_CALIBRATION_VERSION = (
    "paper2_phase3_empirical_calibration_v1"
)
# Safety marker: empirical DEV paired differences from twenty positive-step Option B looks
# Safety marker: seed-wise conservative envelope no cross-seed pseudo-replication
# Safety marker: familywise false-stop target 0.0001 plus three-point and five-point power
# Safety marker: CPU-only saved-row post-processing no model no optimizer no training no CONFIRM scoring
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=400)
    process = subprocess.Popen(
        command,
        cwd=cwd or ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode:
        print("phase3_empirical_calibration_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("phase3_empirical_calibration_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
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
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase3_empirical_calibration.py",
        "tests/test_paper2_phase3_p31.py",
        "tests/test_stage5_notebooks.py::test_phase3_empirical_calibration_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_empirical_calibration.py"])
print("Phase 3 empirical DEV calibration landed; P3.3 remains unauthorized.", flush=True)
