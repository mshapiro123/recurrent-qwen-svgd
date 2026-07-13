"""Colab cell: run selected full-model AdamW Phase A dense controls."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PHASE_A_DENSE_FULL_CELL_VERSION = "phase_a_dense_full_v1"
# Safety marker: phase_a_dense_full
# Safety marker: adamw_full_fp32_state
# Safety marker: STAGE5_PHASE_A_DENSE_ARMS
# Safety marker: tests/test_stage5_phase_a_dense_full.py
# Safety marker: tests/test_train_dense_full.py
# Safety marker: eval/eval_synthetic_depth_dense.py

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = os.environ.get(name) or userdata.get(name)
        except Exception:
            value = os.environ.get(name)
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


def redact(text: str) -> str:
    safe = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            safe = safe.replace(token, "****")
    return safe


def run(cmd: list[str], *, cwd: Path = ROOT, env=None) -> None:
    print("$", redact(" ".join(cmd)), flush=True)
    process = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        tail = (tail + [safe])[-240:]
    returncode = process.wait()
    if returncode:
        print("FAILED_COMMAND_TAIL_START\n" + "".join(tail) + "FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)


def main() -> None:
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
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
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_phase_a_dense_full.py",
            "tests/test_train_dense_full.py",
            "tests/test_stage5_phase_a_surpass.py",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_PHASE_A_DENSE_ARMS", "B,C")
    default_suffix = "".join(arm.lower() for arm in env["STAGE5_PHASE_A_DENSE_ARMS"].split(","))
    env.setdefault("STAGE5_PHASE_A_DENSE_RUN_ID", f"stage5_phase_a_dense_full_{default_suffix}_20260713")
    env.setdefault("STAGE5_PHASE_A_DENSE_LR", "2e-6")
    env.setdefault("STAGE5_PHASE_A_DENSE_SEED", "931337")
    env.setdefault("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_phase_a_dense_full.py"], env=env)
    if os.environ.get("STAGE5_PHASE_A_DENSE_DISCONNECT", "0").lower() in {"1", "true", "yes", "y"}:
        runtime.unassign()
    else:
        print("Leaving runtime connected for Phase A dense review.", flush=True)


try:
    main()
except Exception:
    print("Phase A dense-full run errored; leaving runtime connected.", flush=True)
    raise
