```python
"""Pinned A100-40GB launcher for the locked recirculation Phase-A sweep."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from google.colab import runtime, userdata


STAGE5_PAPER2_RECIRCULATION_PHASE_A_VERSION = "paper2_recirculation_phase_a_v1_cli_transport"
MODE = "locked score-only Phase-A heatmap refinement and battery no optimizer"
REPO = "mshapiro123/recurrent-qwen-svgd"
REF = os.environ["STAGE5_BOOTSTRAP_REF"]
ROOT = Path("/content/recurrent-qwen-svgd")
TESTS = [
    "tests/test_paper2_recirculation.py",
    "tests/test_stage5_notebooks.py::test_paper2_recirculation_phase_a_target_is_wired_and_guarded",
]


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command)
    token = os.environ.get("GH_TOKEN", "")
    if token:
        printable = printable.replace(token, "****")
    print("$", printable, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


print("recirculation_phase_a_transport=cli_archives_no_drive_mount", flush=True)
run(["nvidia-smi"])
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
memory_mib = (
    torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    if torch.cuda.is_available()
    else 0
)
print(f"recirculation_phase_a_gpu_preflight name={name} memory_mib={memory_mib}", flush=True)
assert name == "NVIDIA A100-SXM4-40GB" and 40000 <= memory_mib <= 42000, (
    "Recirculation Phase A is pinned to NVIDIA A100-SXM4-40GB; "
    f"observed {name} with {memory_mib} MiB."
)

phase0_archive = Path(os.environ["RECIRCULATION_PHASE0_ARCHIVE"])
assert phase0_archive.is_file(), f"Banked Phase-0 archive is missing: {phase0_archive}"
gh = os.environ.get("GH_TOKEN", "") or userdata.get("GH_TOKEN") or ""
hf = os.environ.get("HF_TOKEN", "") or userdata.get("HF_TOKEN") or ""
if hf:
    os.environ["HF_TOKEN"] = hf
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf
clone_url = (
    f"https://x-access-token:{gh}@github.com/{REPO}.git"
    if gh
    else f"https://github.com/{REPO}.git"
)
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
    run(["git", "fetch", "origin", REF], cwd=ROOT)
else:
    run(["git", "clone", clone_url, str(ROOT)])
run(["git", "checkout", "-B", "codex/bicameral-stage0", REF], cwd=ROOT)
run(["git", "reset", "--hard", REF], cwd=ROOT)
run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==5.14.1",
        "datasets==5.0.0",
        "matplotlib==3.10.0",
    ],
    cwd=ROOT,
)
run([sys.executable, "-m", "pytest", "-q", *TESTS], cwd=ROOT)
run(
    [sys.executable, "-u", "-m", "colab.run_stage5_paper2_recirculation_phase_a"],
    cwd=ROOT,
)

print("paper2_recirculation_phase_a_complete=true", flush=True)
if os.environ.get("RECIRCULATION_KEEP_RUNTIME", "0").strip().lower() not in {
    "1",
    "true",
    "yes",
}:
    runtime.unassign()
```
