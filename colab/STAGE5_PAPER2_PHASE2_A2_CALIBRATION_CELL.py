"""Colab launcher for the authorized zero-update Phase-2 A2 calibration."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_A2_CALIBRATION_VERSION = "paper2_phase2_a2_calibration_v1"
# Safety marker: banked A1 pass checkpoints for both seeds asserted by SHA256
# Safety marker: zero optimizer updates and no A2 training from this target
# Safety marker: actual A2 graph bridge control and draft only
# Safety marker: raw and weighted per-loss gradient distributions
# Safety marker: pairwise conflict cosines including cumulative KL versus local CE
# Safety marker: legacy 35 35 10 20 shares are initialization targets only
# Safety marker: candidate clip tripwire is observed p99 times 10 not an active shaper
# Safety marker: full module and frozen flow hashes unchanged before and after
# Safety marker: strategy amendment required before A2 training
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
        print("a2_calibration_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a2_calibration_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
assert torch.cuda.is_available(), "A2 calibration requires CUDA."
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / 2**30
print(f"gpu_preflight name={properties.name} vram_gib={vram_gib:.1f}", flush=True)
assert properties.major >= 8 and vram_gib >= 35, (
    "A2 calibration requires an A100 40GB or larger runtime; "
    f"observed {properties.name} with {vram_gib:.1f} GiB."
)
os.environ["STAGE5_PHASE2_A2_CALIBRATION_GPU_NAME"] = properties.name
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
        "tests/test_paper2_phase2_a2_calibration.py",
        "tests/test_paper2_phase2_staged_repilot.py",
        "tests/test_paper2_phase2_staged_repilot_preregistration.py",
        "tests/test_stage5_notebooks.py::test_phase2_a2_calibration_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_a2_calibration.py"])
print("Zero-update A2 calibration completed; A2 training remains closed.", flush=True)
