"""Colab launcher for the executable Paper Two Phase T0 preflight."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_PAPER2_PHASE_T0_CELL_VERSION = "paper2_internal_token_t0_v1"
# Safety markers: tokenizer collision exactly three rows tie policy
# Safety markers: visible generation masks all three control logits
# Safety markers: one loop identity max abs diff below 1e-3
# Safety markers: requested executed selected loop counts agree under forcing
# Safety markers: no training no checkpoint written
# Safety markers: tests/test_internal_think_token_runtime.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get("STAGE5_PAPER2_T0_DISCONNECT", "0").lower() in {
    "1",
    "true",
    "yes",
}


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
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_internal_think_token_spec.py",
            "tests/test_internal_think_token_runtime.py",
            "tests/test_stage5_notebooks.py::test_paper2_t0_target_executes_five_contracts_without_training",
        ]
    )
    os.environ.update(
        {
            "STAGE5_PAPER2_T0_DTYPE": "bfloat16",
            "STAGE5_PAPER2_T0_DISCONNECT": "0",
        }
    )
    run([sys.executable, "colab/run_stage5_paper2_phase_t0_preflight.py"])
    print("Paper Two Phase T0 passed. This cell does not authorize T1 training.", flush=True)
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Paper Two Phase T0 errored; leaving runtime connected.", flush=True)
    raise
