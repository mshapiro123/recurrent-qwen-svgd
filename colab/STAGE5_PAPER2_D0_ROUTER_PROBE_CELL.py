"""Colab launcher for the read-only D0 deployable-router probe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_D0_ROUTER_PROBE_VERSION = "paper2_d0_router_probe_v2_floor_equivalent_capture"
# Safety marker: read-only L4 feature extraction no model optimizer no model training
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
assert GH_TOKEN and HF_TOKEN, "D0 router probe requires GH_TOKEN and HF_TOKEN."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


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


drive.mount("/content/drive", force_remount=False)
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
run(["nvidia-smi"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_speculative_depth_router_probe.py",
        "tests/test_speculative_depth_router_feasibility.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_d0_router_probe_target",
    ]
)
run([sys.executable, "colab/run_stage5_paper2_d0_router_probe.py"])
print("D0 deployable-router probe completed and published.", flush=True)
