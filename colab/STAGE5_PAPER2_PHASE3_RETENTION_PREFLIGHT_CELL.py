"""Colab launcher for the no-update P3.3 e2 retention preflight."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE3_RETENTION_PREFLIGHT_VERSION = "paper2_phase3_retention_preflight_v1"
# Safety marker: exact 1024-position token-retention estimator balanced across four horizons
# Safety marker: task-scale thresholds void and task capability scoring absent in P3.3
# Safety marker: step-zero init-relative calibration before optimizer construction
# Safety marker: Tier-S familywise false stop 1e-4 and 99 percent delta-cat power
# Safety marker: L4 read-only model pass then CPU simulation no optimizer no training
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = os.environ.get("GH_TOKEN") or userdata.get("GH_TOKEN")
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
        print("phase3_retention_preflight_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("phase3_retention_preflight_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"])
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "fetch", "origin", f"refs/heads/{SOURCE_BRANCH}:refs/remotes/origin/{SOURCE_BRANCH}"])
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
        "tests/test_paper2_phase3_p33_prep.py",
        "tests/test_paper2_phase3_retention_step0.py",
        "tests/test_paper2_phase3_retention_guardrail.py",
        "tests/test_stage5_notebooks.py::test_phase3_retention_preflight_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_retention_preflight.py"])
print("P3.3 e2 retention panel and calibration landed; no optimizer was constructed.", flush=True)
