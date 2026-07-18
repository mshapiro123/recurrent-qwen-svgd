"""Colab launcher for the frozen-substrate Phase G-alpha program."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PHASE_G_ALPHA_CELL_VERSION = "phase_g_alpha_v1"
# Safety markers: training/train_phase_g_alpha.py eval/eval_phase_g_alpha.py
# Safety markers: phase_g_prior_head phase_g_posterior_head phase_g_injection_scale
# Safety markers: STAGE5_PHASE_G_ALPHA_KL_SWEEP blocked exit
# Safety marker: STAGE5_PHASE_G_ALPHA_TRAJECTORY_MICROBATCH_SIZE
# Safety marker: STAGE5_PHASE_G_ALPHA_CHECKPOINT_EVERY progress_backup_path
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get("STAGE5_PHASE_G_ALPHA_DISCONNECT", "0").lower() in {
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
    receipt_dir = Path(
        "/content/drive/MyDrive/recurrent-qwen-svgd-manuscript/"
        "stage5_closure_receipts_20260717"
    )
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for relative in (
        "docs/STAGE5_PEFT_SELECTOR_CLOSURE_HANDOFF_20260717.md",
        "docs/STAGE5_STRATEGY_CLOSURE_ADDENDA_10_11_20260717.md",
        "docs/part1_claim_evidence_ledger.json",
        "docs/PHASE_G_ALPHA_GUIDED_STOCHASTIC_TRANSITION_SPEC.md",
    ):
        shutil.copy2(ROOT / relative, receipt_dir / Path(relative).name)
    print(f"Closure receipts copied to {receipt_dir}", flush=True)
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
            "tests/test_recurrent_wrapper_tiny.py",
        ]
    )
    return_code = run(
        [sys.executable, "colab/run_stage5_phase_g_alpha.py"],
        allow_blocked=True,
    )
    print(
        "Phase G-alpha finished."
        if return_code == 0
        else "Phase G-alpha reached a preregistered blocked exit.",
        flush=True,
    )
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Phase G-alpha errored; leaving runtime connected for diagnosis.", flush=True)
    raise
