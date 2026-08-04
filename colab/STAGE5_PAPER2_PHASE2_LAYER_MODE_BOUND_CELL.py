"""Colab high-RAM CPU launcher for the corrected r2 layer-mode bound."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import psutil
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_LAYER_MODE_BOUND_VERSION = "paper2_phase2_layer_mode_bound_v1"
# Safety marker: CPU high RAM cached concat RRR three common randomized SVD seeds no training
# Safety marker: r2 supersedes dense LAPACK and forbids dense covariance materialization
# Safety marker: 0.25 point within-arm spread gate precedes 0.5 point paired CI swap rule
# Safety marker: Drive per-seed resume cache and failure status preserve long CPU work
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."
assert HF, "Missing HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=300)
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
        print("launcher_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("launcher_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
ram_gib = psutil.virtual_memory().total / 2**30
available_gib = psutil.virtual_memory().available / 2**30
print(f"cpu_memory_preflight total_gib={ram_gib:.1f} available_gib={available_gib:.1f}", flush=True)
assert ram_gib >= 45 and available_gib >= 38, (
    "Layer-mode concat fit requires Colab's 51-GiB high-RAM CPU tier with at least "
    f"45 GiB total and 38 GiB available; observed total={ram_gib:.1f}, "
    f"available={available_gib:.1f}."
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
        "tests/test_paper2_phase2_layer_mode_bound.py",
        "tests/test_paper2_phase2_arbitration.py",
        "tests/test_paper2_dc2_student.py",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_layer_mode_bound.py"])
print("Phase-2 r2 layer-mode bound landed; no model training ran.", flush=True)
