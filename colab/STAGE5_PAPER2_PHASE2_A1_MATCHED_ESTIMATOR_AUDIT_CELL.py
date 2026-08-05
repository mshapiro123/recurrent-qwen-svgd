"""Colab launcher for the read-only A1 matched-estimator audit."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_A1_MATCHED_ESTIMATOR_AUDIT_VERSION = (
    "paper2_phase2_a1_matched_estimator_audit_v1"
)
# Safety marker: read-only exact 51-batch calibration estimator reconstruction
# Safety marker: initialization and saved step-200 checkpoints for both seeds
# Safety marker: training population owns hard flow and probe share inequalities
# Safety marker: fixed 51-batch DEV population-shift check is descriptive only
# Safety marker: bootstrap intervals and per-batch contract frequencies
# Safety marker: zero optimizer updates and model hashes before and after every measurement
# Safety marker: emits resume_saved_step_200 or fresh_rerun_required without launching training
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
        print("a1_matched_audit_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a1_matched_audit_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
assert torch.cuda.is_available(), "The matched-estimator audit requires CUDA."
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / 2**30
print(f"gpu_preflight name={properties.name} vram_gib={vram_gib:.1f}", flush=True)
assert properties.major >= 8 and vram_gib >= 20, (
    "The audit requires an Ampere-class L4 24GB or larger runtime; "
    f"observed {properties.name} with {vram_gib:.1f} GiB."
)
os.environ["STAGE5_A1_AUDIT_GPU_NAME"] = properties.name
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
        "tests/test_paper2_phase2_a1_matched_estimator_audit.py",
        "tests/test_paper2_phase2_staged_repilot.py",
        "tests/test_stage5_notebooks.py::test_phase2_a1_matched_estimator_audit_is_wired",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_a1_matched_estimator_audit.py"])
print("A1 matched-estimator audit completed; no training or A2 launch occurred.", flush=True)
