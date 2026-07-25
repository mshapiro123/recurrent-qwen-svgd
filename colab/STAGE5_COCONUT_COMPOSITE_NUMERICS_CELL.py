"""Colab launcher for the read-only COCONUT numerical follow-up."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_COCONUT_COMPOSITE_NUMERICS_CELL_VERSION = "coconut_composite_numerics_v1"
# Safety marker: recompute only sliced cache retired
# Safety marker: fixed-weight fixed-direction epsilon stability sweep
# Safety marker: original 10 percent derivative criterion unchanged
# Safety marker: fp32 full bf16 and fp32-master bf16-autocast fixed prompts
# Safety marker: per-example gradient cosine threshold 0.99 unchanged
# Safety marker: no training no checkpoint RG-12 unauthorized
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
DISCONNECT = os.environ.get("STAGE5_COCONUT_NUMERICS_DISCONNECT", "0").lower() in {"1", "true", "yes"}


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


def run(command: list[str], *, cwd: Path | None = None, accepted: tuple[int, ...] = (0,)) -> int:
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
    code = int(process.wait())
    if code not in accepted:
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
            "tests/test_coconut_composite.py",
            "tests/test_coconut_composite_numerics.py",
            "tests/test_recurrent_wrapper_tiny.py",
        ]
    )
    code = run([sys.executable, "colab/run_stage5_coconut_composite_numerics.py"], accepted=(0, 2))
    if code == 2:
        print("COCONUT numerical follow-up needs review; RG-12 remains unauthorized.", flush=True)
    elif DISCONNECT:
        runtime.unassign()
except Exception:
    print("COCONUT numerical follow-up errored; leaving runtime connected.", flush=True)
    raise
