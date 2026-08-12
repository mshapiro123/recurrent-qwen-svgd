"""Colab launcher for the read-only P3.3 verification and BF16 re-score."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE3_P33_VERIFICATION_VERSION = "paper2_phase3_p33_verification_v1"
# Safety marker: read-only final P3.3 checkpoints and zero optimizer steps
# Safety marker: BF16 serving reader pinned end to end
# Safety marker: V1 nonzero margin deltas V2 shared deployed path V3 forced-open radius 0.15
# Safety marker: P3.4 remains unauthorized and i1 does not launch from this target
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = os.environ.get("GH_TOKEN") or userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None, allowed: tuple[int, ...] = (0,)) -> None:
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
    code = process.wait()
    if code not in allowed:
        print("verification_failure_tail_begin\n" + "\n".join(tail) + "\nverification_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(code, command)


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
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase3_p33_verification.py",
        "tests/test_paper2_phase3_p33.py",
        "tests/test_stage5_notebooks.py::test_phase3_p33_verification_target_is_wired_and_guarded",
    ]
)
run(
    [sys.executable, "-u", "colab/run_stage5_paper2_phase3_p33_verification.py"],
    allowed=(0, 2),
)
print("P3.3 verification landed. Inspect the positive-control verdict before i1.", flush=True)
