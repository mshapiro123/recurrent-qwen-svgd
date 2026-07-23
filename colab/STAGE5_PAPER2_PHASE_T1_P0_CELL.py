"""Colab launcher for the authorized, uncitable Paper Two T1 P0 pilot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PAPER2_PHASE_T1_P0_CELL_VERSION = "paper2_internal_token_t1_p0_v1"
# Safety marker: P0 pilot only registered T1 remains locked
# Safety marker: ten cells lambda 0.5 1 2 ratio 1 3.5 7 plus lambda zero reference
# Safety marker: seed 9999 1500 steps checkpoints 500 1000 1500
# Safety marker: dedicated 256 row pilot slice never enters a registered set
# Safety marker: exact 70 percent control 30 percent mechanism rehearsal
# Safety marker: no silent sweep extension when both recalls miss 0.60
# Safety marker: exact normalized trie multi-token candidate scoring
# Safety marker: complete ten-cell calibration grid before coefficient lock
# Safety marker: tests/test_internal_think_token_t1.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
DISCONNECT = os.environ.get("STAGE5_PAPER2_T1_P0_DISCONNECT", "0").lower() in {
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


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> int:
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
    if code not in allowed_returncodes:
        raise subprocess.CalledProcessError(code, command)
    if code == 2:
        print("P0 reached a preregistered blocked or partial exit (code 2).", flush=True)
    return code


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
            "tests/test_internal_think_token_t1.py",
            "tests/test_internal_think_token_t1_spec.py",
            "tests/test_internal_think_token_runtime.py",
            "tests/test_stage5_notebooks.py::test_paper2_t1_p0_target_is_pilot_only_and_resumable",
        ]
    )
    os.environ.setdefault("STAGE5_PAPER2_T1_P0_DTYPE", "bfloat16")
    os.environ.setdefault("STAGE5_PAPER2_T1_P0_EVAL_BATCH_SIZE", "4")
    run(
        [sys.executable, "colab/run_stage5_paper2_phase_t1_p0.py"],
        allowed_returncodes=(0, 2),
    )
    print("P0 pilot finished or reached its preregistered no-selection exit.", flush=True)
    print("Registered T1-lite remains locked until Draft 3 and preregistration.json are committed.", flush=True)
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Paper Two T1 P0 errored; leaving runtime connected.", flush=True)
    raise
