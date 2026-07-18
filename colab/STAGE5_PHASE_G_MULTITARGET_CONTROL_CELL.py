"""Colab launcher for the locked Phase G A0 posterior-control gate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PHASE_G_MULTITARGET_CONTROL_CELL_VERSION = "phase_g_a0_margin_lock_v1"
# Safety markers: docs/STAGE5_PHASE_G_A0_MARGIN_LOCK_20260718.json
# Safety markers: base_problem_uniform kl_0p001 kl_0p0001_confirmation
# Safety markers: STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE=0.60
# Safety markers: STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT=0.15
# Safety markers: STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS=24
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get("STAGE5_PHASE_G_MULTITARGET_DISCONNECT", "0").lower() in {
    "1", "true", "yes",
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


def run(command: list[str], *, allow_blocked: bool = False) -> int:
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
    if return_code and not (allow_blocked and return_code == 2):
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


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
            "tests/test_phase_g_guidance.py",
            "tests/test_phase_g_training.py",
            "tests/test_phase_g_branching.py",
            "tests/test_phase_g_alpha_spec.py",
            "tests/test_phase_g_multitarget_task.py",
            "tests/test_phase_g_sampling.py",
            "tests/test_phase_g_multitarget_spec.py",
            "tests/test_stage5_phase_g_multitarget_control.py",
            "tests/test_score_phase_g_posterior_control.py",
            "tests/test_analyze_phase_g_multimodal_supervision.py",
            "tests/test_recurrent_wrapper_tiny.py",
        ]
    )
    os.environ.update(
        {
            "STAGE5_PHASE_G_MULTITARGET_GATE_LOCK": "docs/STAGE5_PHASE_G_A0_MARGIN_LOCK_20260718.json",
            "STAGE5_PHASE_G_MULTITARGET_STEPS": "1000",
            "STAGE5_PHASE_G_MULTITARGET_KL": "0.001",
            "STAGE5_PHASE_G_MULTITARGET_SEED": "20260718",
            "STAGE5_PHASE_G_MULTITARGET_CHECKPOINT_EVERY": "100",
        }
    )
    result = run(
        [sys.executable, "colab/run_stage5_phase_g_multitarget_control.py"],
        allow_blocked=True,
    )
    print(
        "Phase G A0 posterior-control gate passed."
        if result == 0
        else "Phase G A0 reached its preregistered blocked exit after the one permitted confirmation.",
        flush=True,
    )
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Phase G A0 posterior-control run errored; leaving runtime connected.", flush=True)
    raise
