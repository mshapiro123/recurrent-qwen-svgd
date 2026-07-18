"""Colab launcher for the one authorized Phase G forced-injection probe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PHASE_G_FORCED_INJECTION_PROBE_CELL_VERSION = "forced_injection_causal_probe_v1"
# Safety markers: eval/eval_phase_g_forced_injection.py
# Safety markers: STAGE5_PHASE_G_FORCED_INJECTION_FACTORS=1,3,10,30,100
# Safety markers: CHANNEL-EXISTS switching >=16/32 K1 validity >0.50
# Safety markers: NO-CHANNEL switching <8/32
# Safety markers: factor_1_exact_equivalence frozen_lineage_unchanged
# Safety markers: tests/test_phase_g_forced_injection.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get(
    "STAGE5_PHASE_G_FORCED_INJECTION_DISCONNECT",
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
            "tests/test_phase_g_forced_injection.py",
            "tests/test_recurrent_wrapper_tiny.py",
            "tests/test_stage5_notebooks.py::test_phase_g_forced_injection_target_is_eval_only_and_locked",
        ]
    )
    os.environ.update(
        {
            "STAGE5_PHASE_G_FORCED_INJECTION_FACTORS": "1,3,10,30,100",
            "STAGE5_PHASE_G_FORCED_INJECTION_DTYPE": "bfloat16",
        }
    )
    result = run(
        [sys.executable, "colab/run_stage5_phase_g_forced_injection_probe.py"],
        allow_blocked=True,
    )
    print(
        "Phase G forced-injection probe found a magnitude-responsive channel."
        if result == 0
        else "Phase G forced-injection probe did not authorize a successor.",
        flush=True,
    )
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Phase G forced-injection probe errored; leaving runtime connected.", flush=True)
    raise
