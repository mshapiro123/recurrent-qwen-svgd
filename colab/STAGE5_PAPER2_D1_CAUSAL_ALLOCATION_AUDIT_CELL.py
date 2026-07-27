"""Colab launcher for the read-only D0 causal allocation audit."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_D1_CAUSAL_ALLOCATION_AUDIT_VERSION = "paper2_d1_causal_allocation_audit_v1"
# Safety marker: read-only A100 post-D0 audit no optimizer no backward no checkpoint writes
# Safety marker: exact replay equivalence to banked D0 A100 anchors
# Safety marker: source-row grouped five-fold cross-fit seed 20260727
# Safety marker: D1 continue only when next loop helps stop on hurts or neutral
# Safety marker: teacher top1 top2 margin unavailable no teacher reload
# Safety marker: 100000-position label-train forced-depth-4 dry run
# Safety marker: evaluation partition post-hoc D0 verdict unchanged
# Safety marker: Pharma Initiatives Drive authorization
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN and HF_TOKEN, "D1 audit requires GH_TOKEN and HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
os.environ["STAGE5_PAPER2_D1_ALLOW_TRAINING"] = "0"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH_TOKEN, "****"), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd or ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


if not Path("/content/drive/MyDrive").is_dir():
    print("Authorize the Pharma Initiatives Google account for Drive.", flush=True)
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
gpu_memory = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
)
available_mib = max(int(value.strip()) for value in gpu_memory.splitlines() if value.strip())
assert available_mib >= 35000, f"D1 audit requires an A100-class GPU; observed {available_mib} MiB."
clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", clone_url])
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
    run(["git", "reset", "--hard", SYNC_REF])
else:
    run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "reset", "--hard", SYNC_REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_speculative_depth_d1_causal_allocation.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_d1_causal_allocation_audit_target",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_d1_causal_allocation_audit.py"])
print("D0 causal allocation audit and D1 label dry run completed and published.", flush=True)
