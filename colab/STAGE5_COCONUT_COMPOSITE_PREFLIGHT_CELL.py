"""Colab launcher for the no-training COCONUT composite integrity battery."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_COCONUT_COMPOSITE_PREFLIGHT_CELL_VERSION = "coconut_composite_rg1_rg11_v1"
# Safety markers: no training RG-12 remains unrun
# Safety markers: H times L feedback and H plus one times L total applications
# Safety markers: full recompute reference sliced cache L1 only
# Safety markers: finite difference cache checkpointing bfloat16 equivalence
# Safety markers: frozen adapter backbone gradient transparent
# Safety markers: tests/test_coconut_composite.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get("STAGE5_COCONUT_PREFLIGHT_DISCONNECT", "0").lower() in {
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


def run(command: list[str], *, cwd: Path | None = None, accepted: tuple[int, ...] = (0,)) -> int:
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
            "tests/test_coconut_composite.py",
            "tests/test_recurrent_wrapper_tiny.py",
        ]
    )
    code = run(
        [sys.executable, "colab/run_stage5_coconut_composite_preflight.py"],
        accepted=(0, 2),
    )
    if code == 2:
        print("Composite preflight landed a red contract; leaving runtime connected.", flush=True)
    elif DISCONNECT:
        runtime.unassign()
except Exception:
    print("Composite preflight errored; leaving runtime connected.", flush=True)
    raise
