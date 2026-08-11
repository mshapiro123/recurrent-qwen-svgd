"""Colab launcher for one registered P3.3 aimed-writeback seed."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE3_P33_VERSION = "paper2_phase3_p33_v1"
# Safety marker: e2 token-retention estimator only and no task capability scoring
# Safety marker: exact 1000 updates and 20 looks one every 50 updates
# Safety marker: A1 A2 A3 A4 A5 verified before optimizer construction
# Safety marker: pi_dir pi_dep gate recall precision chi and Tier-1 observatory
# Safety marker: resumable one A100 session per seed and end-of-run A_state battery
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
SEED = int(os.environ.get("PAPER2_P33_SEED", "0"))
GH = os.environ.get("GH_TOKEN") or userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."
assert SEED in (0, 1), "PAPER2_P33_SEED must be 0 or 1."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=500)
    process = subprocess.Popen(
        command, cwd=cwd or ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    if process.wait():
        print("p33_failure_tail_begin\n" + "\n".join(tail) + "\np33_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(process.returncode, command)


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
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_paper2_phase3_p33.py",
    "tests/test_paper2_phase3_p33_prep.py",
    "tests/test_paper2_phase3_p33_runner.py",
    "tests/test_stage5_notebooks.py::test_phase3_p33_target_is_wired_and_guarded",
])
run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_p33.py", "--seed", str(SEED)])
print(f"P3.3 seed {SEED} landed; release this A100 session after artifact verification.", flush=True)
