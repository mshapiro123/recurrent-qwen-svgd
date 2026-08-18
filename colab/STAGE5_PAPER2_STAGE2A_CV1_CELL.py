"""Colab launcher for the score-only Stage 2A CV-1 and D5 diagnostics."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_STAGE2A_CV1_VERSION = "paper2_stage2a_cv1_d5_v1"
# Contract marker: score-only crossed values and gate dose no optimizer no training
# Contract marker: both comparators per battery dose zero bit exact initialization
# Contract marker: D5 retrieval relevance descriptive no gate
# Contract marker: DEV reuse only CONFIRM and EVAL-E remain sealed
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."
assert HF, "Missing HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF


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
        print("stage2a_cv1_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("stage2a_cv1_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
memory = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    text=True,
)
memory_mib = int(memory.strip().splitlines()[0])
print(f"stage2a_cv1_gpu_memory_mib={memory_mib}", flush=True)
assert memory_mib >= 22_000, "Stage 2A CV-1 requires a 24GB-class GPU or larger."
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists() and (ROOT / ".git").is_dir():
    run(["git", "remote", "set-url", "origin", url])
else:
    if ROOT.exists():
        shutil.rmtree(ROOT)
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
        "tests/test_paper2_stage2a_cv1.py",
        "tests/test_paper2_phase3_p34_task_inference.py",
        "tests/test_stage5_notebooks.py::test_stage2a_cv1_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_stage2a_cv1.py"])
print("Stage 2A CV-1 and D5 landed on DEV; CONFIRM and EVAL-E remain sealed.", flush=True)
