"""Colab launcher for the locked Phase-2 staged A1 experiment."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_STAGED_A1_VERSION = "paper2_phase2_staged_a1_v1"
# Safety marker: locked gradient-only 100-batch calibration with zero optimizer updates
# Safety marker: A1 trains flow only and freezes initializer bridge control draft
# Safety marker: execution gates closed with counterfactual preservation isolated from execution
# Safety marker: static 60 20 20 gradient shares audited at step 200
# Safety marker: trust observe-only endpoint-ratio catastrophe tripwire and generous clip ceiling
# Safety marker: A1 ends for strategy review and cannot launch A2
# Safety marker: two seeds alpha 0.5 one registered extension maximum 2000 steps
# Safety marker: A100 40GB cached-state path minimum 35 GiB visible VRAM
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
        print("staged_a1_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("staged_a1_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
assert torch.cuda.is_available(), "Staged A1 requires CUDA."
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / 2**30
print(f"gpu_preflight name={properties.name} vram_gib={vram_gib:.1f}", flush=True)
assert properties.major >= 8, (
    f"Staged A1 requires Ampere-class CUDA or newer; observed {properties.name}."
)
assert vram_gib >= 35, (
    "Staged A1 requires an A100 40GB or larger runtime; "
    f"observed {properties.name} with {vram_gib:.1f} GiB."
)
os.environ["STAGE5_STAGED_A1_GPU_NAME"] = properties.name
os.environ["STAGE5_STAGED_A1_GPU_VRAM_GIB"] = f"{vram_gib:.3f}"
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
        "tests/test_paper2_phase2_staged_repilot.py",
        "tests/test_paper2_phase2_staged_repilot_preregistration.py",
        "tests/test_paper2_dc2_student.py",
        "tests/test_stage5_notebooks.py::test_phase2_staged_a1_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_staged_a1.py"], allowed=(0, 2))
print("Phase-2 staged A1 reached a terminal receipt; A2 was not launched.", flush=True)
