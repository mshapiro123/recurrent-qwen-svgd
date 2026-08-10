"""Colab launcher for authorized Phase 3.1/3.2 no-training receipts."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE3_P31_P32_RECEIPTS_VERSION = "paper2_phase3_p31_p32_receipts_v1"
# Safety marker: exact pinned datasets score-blind source assembly and CONFIRM seal only
# Safety marker: 512-row alpha-0.00005 planning forecast plus three-point and five-point power
# Safety marker: both E1 full-system seed checkpoints hash-pinned and migrated bit-exact
# Safety marker: P3.2 strict writes permissive distillation coverage receipt before cache finality
# Safety marker: no P3.3 optimizer no P3.3 training no CONFIRM scoring
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=600)
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
        print("phase3_receipts_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("phase3_receipts_launcher_failure_tail_end", flush=True)
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
        "tests/test_paper2_phase3_p31.py",
        "tests/test_paper2_phase3_p31_sources.py",
        "tests/test_paper2_phase3_p32.py",
        "tests/test_paper2_phase3_gate_migration.py",
        "tests/test_stage5_notebooks.py::test_phase3_p31_p32_receipts_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_p31_p32_receipts.py"])
print("Phase 3.1/3.2 receipts landed in Drive; P3.3 remains unauthorized.", flush=True)
