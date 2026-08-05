"""Colab launcher for amended Phase-2 staged A1 continuation."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_STAGED_A1_RESUME_VERSION = "paper2_phase2_staged_a1_resume_v1"
# Safety marker: amendment lock committed before resumed optimizer steps
# Safety marker: exact audited step-200 source checkpoints copied and preserved
# Safety marker: matched 51-by-128 training estimator owns hard share verdicts
# Safety marker: flow at least 0.50 and probe at most 0.25 inequality contract
# Safety marker: preserve share descriptive and preserve loss alarm log only
# Safety marker: fixed weights no periodic or automatic recalibration
# Safety marker: stop at step 1000 for strategy review no automatic extension
# Safety marker: A2 remains closed and cannot launch from this target
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."
assert HF, "Missing HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None, *, allowed: tuple[int, ...] = (0,)) -> None:
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
    if returncode not in allowed:
        print("staged_a1_resume_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("staged_a1_resume_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
assert torch.cuda.is_available(), "Staged A1 continuation requires CUDA."
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / 2**30
print(f"gpu_preflight name={properties.name} vram_gib={vram_gib:.1f}", flush=True)
assert properties.major >= 8 and vram_gib >= 35, (
    "Staged A1 continuation requires an A100 40GB or larger runtime; "
    f"observed {properties.name} with {vram_gib:.1f} GiB."
)
os.environ["STAGE5_STAGED_A1_RESUME_GPU_NAME"] = properties.name
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
        "tests/test_paper2_phase2_staged_a1_resume.py",
        "tests/test_paper2_phase2_staged_repilot.py",
        "tests/test_paper2_phase2_staged_repilot_preregistration.py",
        "tests/test_stage5_notebooks.py::test_phase2_staged_a1_resume_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_staged_a1_resume.py"], allowed=(0, 2))
print("Amended A1 continuation reached a terminal receipt; A2 was not launched.", flush=True)
