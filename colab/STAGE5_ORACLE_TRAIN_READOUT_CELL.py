"""Colab launcher for the Paper Two WP1 read-only oracle train-row readout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_ORACLE_TRAIN_READOUT_CELL_VERSION = "paper2_wp1_oracle_train_readout_v1"
# Safety markers: posthoc diagnostic only no training no parameter mutation
# Safety markers: registered BOTH_FAIL verdict immutable
# Safety markers: seed 20260722 matched 106 rows 32 groups 305 transitions
# Safety markers: full 1899 rows 512 groups 5617 transitions
# Safety markers: fit >=0.85 no-fit <=0.25 partial between
# Safety markers: tests/test_oracle_train_readout_spec.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get(
    "STAGE5_ORACLE_TRAIN_READOUT_DISCONNECT",
    "0",
).lower() in {"1", "true", "yes"}


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


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
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
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
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
            "tests/test_oracle_interface_probe.py",
            "tests/test_oracle_train_readout_spec.py",
            "tests/test_stage5_notebooks.py::test_paper2_wp1_oracle_train_readout_target_is_read_only",
        ]
    )
    os.environ.update(
        {
            "STAGE5_ORACLE_TRAIN_READOUT_DTYPE": "bfloat16",
            "STAGE5_ORACLE_TRAIN_READOUT_DISCONNECT": "0",
        }
    )
    run([sys.executable, "colab/run_stage5_phase_g_oracle_train_readout.py"])
    print(
        "Paper Two WP1 finished. BOTH_FAIL remains the registered held-out verdict.",
        flush=True,
    )
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Paper Two WP1 errored; leaving runtime connected.", flush=True)
    raise
