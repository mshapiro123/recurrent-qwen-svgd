"""Colab launcher for the score-blind Stage 2A content and geometry pass."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_STAGE2A_CONTENT_GEOMETRY_VERSION = "paper2_stage2a_content_geometry_v1"
# Safety marker: 32B concurrence plus frozen prompt states no optimizer no training
# Safety marker: dynamic power-of-two slots validation excluded before concurrence
# Safety marker: verified-train only CONFIRM and EVAL-E remain sealed
# Safety marker: resumable verifier and state caches publish exact hashes
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
        print("stage2a_content_geometry_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("stage2a_content_geometry_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
memory = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
)
memory_mib = int(memory.strip().splitlines()[0])
print(f"stage2a_content_geometry_gpu_memory_mib={memory_mib}", flush=True)
assert memory_mib >= 79_000, "The pinned 32B bf16 concurrence pass requires an A100 80GB."
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
        "tests/test_sidecar_v2_data_spine.py",
        "tests/test_paper2_stage2a_content_geometry.py",
        "tests/test_paper2_stage2a_lock.py",
        "tests/test_stage5_notebooks.py::test_stage2a_content_geometry_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_stage2a_content_geometry.py"])
print("Stage 2A score-blind content and geometry pass landed; training remains disabled.", flush=True)
