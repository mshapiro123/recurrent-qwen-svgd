"""Colab launcher for the locked Phase-2 A2 step-200 continuation."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_A2_RESUME_VERSION = "paper2_phase2_a2_resume_v2"
# Safety marker: stopping-semantics-only resume amendment locked before updates
# Safety marker: all four exact step-two-hundred checkpoints asserted by SHA256
# Safety marker: historical full-arm quality abort is the only clearable abort
# Safety marker: endpoint point and Wilson quality gates remain unchanged
# Safety marker: in-flight point tripwire is step-zero retention minus 0.003
# Safety marker: in-flight Wilson floor below 0.990 stops immediately
# Safety marker: negative three-evaluation retention slope warns without stopping
# Safety marker: optimizer losses rows directional contract extension and verdict unchanged
# Safety marker: V1d receipt attached to A2 completion handoff
# Safety marker: executed schedule counts optimizer updates only
# Safety marker: rejected pre-update batch is preserved as tripwire evidence
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."
assert HF, "Missing HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF
os.environ["STAGE5_PHASE2_A2_RESUME_MODE"] = "1"


def run(command: list[str], cwd: Path | None = None, allowed: tuple[int, ...] = (0,)) -> int:
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
    if returncode not in allowed:
        print("a2_resume_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("a2_resume_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)
    return returncode


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
assert torch.cuda.is_available(), "A2 continuation requires CUDA."
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / 2**30
print(f"gpu_preflight name={properties.name} vram_gib={vram_gib:.1f}", flush=True)
assert properties.major >= 8 and vram_gib >= 35, (
    "A2 continuation requires an A100 40GB or larger runtime; "
    f"observed {properties.name} with {vram_gib:.1f} GiB."
)
os.environ["STAGE5_PHASE2_A2_GPU_NAME"] = properties.name
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
        "tests/test_paper2_phase2_a2.py",
        "tests/test_paper2_phase2_staged_repilot_preregistration.py",
        "tests/test_stage5_notebooks.py::test_phase2_a2_resume_is_wired_and_guarded",
    ]
)
returncode = run(
    [sys.executable, "-u", "colab/run_stage5_paper2_phase2_a2.py"],
    allowed=(0, 2),
)
print(
    "Phase-2 A2 continuation completed with a positive verdict."
    if returncode == 0
    else "Phase-2 A2 continuation completed with a registered non-positive or blocked verdict.",
    flush=True,
)
