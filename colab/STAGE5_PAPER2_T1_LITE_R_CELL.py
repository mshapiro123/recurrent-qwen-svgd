"""Colab launcher for the locked T1-lite-R seed-1 replication."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PAPER2_T1_LITE_R_CELL_VERSION = "paper2_t1_lite_r_locked_v1"
# Safety marker: locked before launcher commit ae2793ac
# Safety marker: seed 1 raw final-step primary
# Safety marker: continuous EMA and stage-reset EMA passive shadows
# Safety marker: atomic hashed stage states 500 2500 6500 8500 10500
# Safety marker: gate4 exact 4608 forced stop plus 1024 forced continue
# Safety marker: D0 build-only no labeling GPU no training
# Safety marker: C track design-stage RG-12 unauthorized
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
DISCONNECT = os.environ.get("STAGE5_PAPER2_T1_LITE_R_DISCONNECT", "0").lower() in {"1", "true", "yes"}


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


def run(command: list[str], *, cwd: Path | None = None, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command).replace(GH_TOKEN, "****"), flush=True)
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
    if code not in allowed:
        raise subprocess.CalledProcessError(code, command)
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
            "tests/test_internal_think_token_t1_r_spec.py",
            "tests/test_internal_think_token_t1_lite.py",
            "tests/test_internal_think_token_runtime.py",
            "tests/test_recurrent_wrapper_tiny.py",
        ]
    )
    os.environ.setdefault("STAGE5_PAPER2_T1_LITE_R_DTYPE", "bfloat16")
    os.environ.setdefault("STAGE5_PAPER2_T1_LITE_R_EVAL_BATCH_SIZE", "8")
    code = run([sys.executable, "colab/run_stage5_paper2_t1_lite_r.py"], allowed=(0, 2))
    print(f"T1-lite-R completed with registered exit code {code}.", flush=True)
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Paper Two T1-lite-R errored; leaving runtime connected for diagnosis.", flush=True)
    raise
