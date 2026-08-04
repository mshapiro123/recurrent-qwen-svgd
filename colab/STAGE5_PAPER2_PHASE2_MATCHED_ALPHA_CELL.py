"""Colab A100-80GB launcher for the locked Phase-2 matched-alpha pilots."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_MATCHED_ALPHA_VERSION = "paper2_phase2_matched_alpha_v1"
# Safety marker: locked six-arm DEV-only alpha 0 0.5 1 seeds 0 1 matched pilots
# Safety marker: zero-loop bit identity frozen LM heads K at most four and document isolation asserted
# Safety marker: 14B functional probe uses the hashed 14B LM head not the student tied embedding
# Safety marker: one registered extension and conditional alpha 0.25 or 0.75 refinement only
# Safety marker: per-arm Drive resume plus local-scratch immutable Stage0A staging
# Safety marker: paired bootstrap quality gate gradient atlas and scripted decision
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
    tail: deque[str] = deque(maxlen=400)
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
        print("launcher_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("launcher_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
assert torch.cuda.is_available(), "Matched-alpha pilots require CUDA."
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / 2**30
print(f"gpu_preflight name={properties.name} vram_gib={vram_gib:.1f}", flush=True)
assert vram_gib >= 70, (
    "Matched-alpha pilots require an A100 80GB class runtime; "
    f"observed {properties.name} with {vram_gib:.1f} GiB."
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
        "tests/test_paper2_phase2_matched_alpha.py",
        "tests/test_paper2_dc2_student.py",
        "tests/test_stage5_notebooks.py::test_phase2_matched_alpha_target_is_wired_and_guarded",
    ]
)
run(
    [sys.executable, "-u", "colab/run_stage5_paper2_phase2_matched_alpha.py"],
    allowed=(0, 2),
)
print("Phase-2 matched-alpha pilot target reached a registered terminal receipt.", flush=True)
