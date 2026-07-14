"""Colab cell for the idle-lane, eval-only multi-channel bridge precursor."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL_VERSION = "multichannel_bridge_precursor_v3_seeded_replication"
# Safety markers: eval/eval_multichannel_bridge_precursor.py
# Safety markers: tests/test_multichannel_bridge_precursor.py
# Safety markers: prelude_ablation_basis random_orthogonal_partitions STAGE5_MULTICHANNEL_MODE
# Safety marker: STAGE5_MULTICHANNEL_SEED_SUMMARY

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get("STAGE5_MULTICHANNEL_DISCONNECT", "0").strip().lower() in {"1", "true", "yes", "y"}


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
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
    else:
        run(["git", "clone", clone_url, str(ROOT)])
    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)


try:
    run(["nvidia-smi"])
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_bridge.py",
            "tests/test_multichannel_bridge_precursor.py",
            "tests/test_recurrent_wrapper_tiny.py",
        ],
        cwd=ROOT,
    )
    run([sys.executable, "colab/run_stage5_multichannel_bridge_precursor.py"], cwd=ROOT, env=os.environ.copy())
    print("Multi-channel bridge precursor finished.", flush=True)
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Multi-channel bridge precursor errored; leaving runtime connected for diagnosis.", flush=True)
    raise
