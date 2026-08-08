"""Colab launcher for the locked read-once Phase-2 E1 confirmation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_E1_CONFIRMATION_VERSION = (
    "paper2_phase2_e1_confirmation_v1"
)
# Safety marker: lock commit ebe4ea4b before scorer construction
# Safety marker: read-once EVAL-D scoring four final Option B endpoints
# Safety marker: paired document bootstrap 10000 replicates seed 20260808
# Safety marker: no optimizer no parameter gradients no training EVAL-E untouched
# Safety marker: A100 80GB required by locked conservative resource note
# Safety marker: failed score-bearing pass cannot rerun without strategy review
# Safety marker: colab/run_stage5_paper2_phase2_e1_confirmation.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd or ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code:
        print("\nE1 confirmation launcher tail:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert shutil.which("nvidia-smi"), "Attach an A100-SXM4-80GB runtime and rerun."
run(["nvidia-smi"], Path("/content"))
memory = max(
    int(value.strip())
    for value in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if value.strip()
)
assert memory >= 70000, (
    "The locked E1 resource note conservatively requires A100 80GB; "
    f"observed {memory} MiB."
)

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
        "tests/test_paper2_phase2_e1_confirmation.py",
        "tests/test_paper2_phase2_e1_runner.py",
    ]
)
run(
    [
        sys.executable,
        "-u",
        "-m",
        "colab.run_stage5_paper2_phase2_e1_confirmation",
    ]
)
print("Read-once E1 confirmation completed and published.", flush=True)
