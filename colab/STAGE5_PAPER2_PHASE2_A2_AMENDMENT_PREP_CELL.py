"""Colab CPU launcher for A2 calibration reconciliation and amendment prep."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_A2_AMENDMENT_PREP_VERSION = "paper2_phase2_a2_amendment_prep_v1"
# Safety marker: CPU-only public and private calibration receipt reconciliation
# Safety marker: no model load no optimizer updates no A2 training
# Safety marker: both 51-batch private receipts verified against public hashes
# Safety marker: directional contract draft remains unlocked pending strategy review
# Safety marker: legacy point shares are step-zero initialization only
# Safety marker: V1d receipt rides with the amendment handoff
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=500)
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
        print("a2_amendment_prep_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a2_amendment_prep_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


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
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase2_a2_amendment_prep.py",
        "tests/test_stage5_notebooks.py::test_phase2_a2_amendment_prep_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_a2_amendment_prep.py"])
print("A2 amendment draft published; A2 training remains closed pending strategy lock.", flush=True)
