"""Colab cell: run the deterministic Phase G Experiment 1 gates."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PHASE_G_EXPERIMENT1_CELL_VERSION = "phase_g_experiment1_v1"
# Safety marker: phase_g_experiment1
# Safety marker: colab/run_stage5_phase_g_experiment1.py
# Safety marker: eval/eval_abductive_coverage.py
# Safety marker: eval/eval_synthetic_diagonal_guardrail.py
# Safety marker: STAGE5_PHASE_G_EXP1_MAX_STEPS
# Safety marker: deterministic controls; latent, learned halting, LPRM, and SVGD disabled

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


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


def redact(text: str) -> str:
    safe = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            safe = safe.replace(token, "****")
    return safe


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        tail.append(safe)
        tail = tail[-200:]
    returncode = process.wait()
    if returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("".join(tail), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


def main() -> None:
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
            "tests/test_abductive_injective_task.py",
            "tests/test_eval_abductive_coverage.py",
            "tests/test_eval_synthetic_diagonal_guardrail.py",
            "tests/test_stage5_phase_g_gate_prepare.py",
            "tests/test_stage5_phase_g_experiment1.py",
            "tests/test_train_unfrozen_recurrent.py",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_PHASE_G_EXP1_RUN_ID", "stage5_phase_g_experiment1_20260712")
    env.setdefault("STAGE5_PHASE_G_EXP1_MAX_STEPS", "1000")
    env.setdefault("STAGE5_PHASE_G_EXP1_DATA_SEED", "1104729")
    env.setdefault("STAGE5_PHASE_G_EXP1_DTYPE", "bfloat16")
    env.setdefault("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_phase_g_experiment1.py"], env=env)
    if os.environ.get("STAGE5_PHASE_G_EXP1_DISCONNECT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        print("Disconnecting Colab runtime after Phase G Experiment 1.", flush=True)
        runtime.unassign()
    else:
        print("Leaving Colab runtime connected after Phase G Experiment 1.", flush=True)


try:
    main()
except Exception:
    print("Phase G Experiment 1 errored; leaving runtime connected.", flush=True)
    raise
