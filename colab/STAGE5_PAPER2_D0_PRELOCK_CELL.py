"""Colab launcher for the authorized D0 density probe and lock job only."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_D0_PRELOCK_CELL_VERSION = "paper2_d0_prelock_density_and_hash_v2_stack_smol"
# Safety marker: Draft 7 authenticated governing hash
# Safety marker: CC-MAIN-2025-26 pinned FineWeb-Edu dump
# Safety marker: the-stack-smol direct Hugging Face content Stack v1 lineage
# Safety marker: no AWS dependency no Software Heritage raw API
# Safety marker: seed-1 raw checkpoint SHA required
# Safety marker: density probe only before lock
# Safety marker: no labeling proper no 14B forward no optimizer no training
# Safety marker: post-lock launcher must be created after lock commit
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN in Colab secrets."
assert HF_TOKEN, "Missing HF_TOKEN in Colab secrets; bigcode/the-stack-smol is gated."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
print("HF token loaded", flush=True)
print("Code corpus access: direct Hugging Face text; AWS is not used", flush=True)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH_TOKEN, "****"), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd or (ROOT if ROOT.exists() else None),
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


def sync_repo() -> None:
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
    run(["git", "log", "--oneline", "-5"])


try:
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_speculative_depth_d0_spec.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_d0_prelock_target",
        ]
    )
    run([sys.executable, "colab/run_stage5_paper2_d0_prelock.py"])
    print("D0 preregistration lock landed. Stop here for lock review; labeling proper has not run.", flush=True)
except Exception:
    print("D0 pre-lock job errored; leaving runtime connected. No substitution is authorized.", flush=True)
    raise
