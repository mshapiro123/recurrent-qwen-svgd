"""Colab launcher for the terminal Phase G oracle re-entry interface probe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_ORACLE_INTERFACE_PROBE_CELL_VERSION = "oracle_interface_probe_v1"
# Safety markers: training/train_oracle_interface_probe.py
# Safety markers: eval/eval_oracle_interface_probe.py
# Safety markers: additive film parameter-matched
# Safety markers: nondefault_branch_control >=0.85
# Safety markers: overall_transition_control >=0.90
# Safety markers: transition_legality >=0.95 terminal_validity >=0.71
# Safety markers: zeroed_conditioning_identity frozen_keeper_lineage
# Safety markers: 106 rows 32 groups 305 transitions
# Safety markers: no KL no coverage no selector no particles no SVGD
# Safety markers: tests/test_oracle_reentry_conditioner.py
# Safety markers: tests/test_oracle_interface_probe.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get(
    "STAGE5_ORACLE_INTERFACE_DISCONNECT",
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


def run(command: list[str]) -> None:
    printable = " ".join(command).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT if ROOT.exists() else None,
        env=os.environ.copy(),
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
        raise subprocess.CalledProcessError(return_code, command)


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        subprocess.run(["git", "clone", clone_url, str(ROOT)], check=True)
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


try:
    subprocess.run(["nvidia-smi"], check=True)
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_oracle_reentry_conditioner.py",
            "tests/test_oracle_interface_probe.py",
            "tests/test_recurrent_wrapper_tiny.py",
            "tests/test_stage5_notebooks.py::test_oracle_interface_probe_target_is_terminal_and_locked",
        ]
    )
    os.environ.update(
        {
            "STAGE5_ORACLE_INTERFACE_STEPS": "1500",
            "STAGE5_ORACLE_INTERFACE_SEED": "20260718",
            "STAGE5_ORACLE_INTERFACE_BOTTLENECK_DIM": "256",
            "STAGE5_ORACLE_INTERFACE_DTYPE": "bfloat16",
        }
    )
    run([sys.executable, "colab/run_stage5_oracle_interface_probe.py"])
    print(
        "Phase G oracle interface probe finished. No successor is automatic; "
        "review gate.md with strategy.",
        flush=True,
    )
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Oracle interface probe errored; leaving runtime connected.", flush=True)
    raise
