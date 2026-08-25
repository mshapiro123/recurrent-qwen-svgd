"""Pinned launcher for the W2-prime prompt-only D4 cache."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


STAGE5_PAPER2_BICAMERAL_W2P_D4_VERSION = "paper2_bicameral_w2p_d4_v1"
MODE = "forward-only prompt cache no optimizer no teacher no sealed evaluation"
CACHE_MODULE = "eval.cache_paper2_bicameral_w2p_d4"
REPO = "mshapiro123/recurrent-qwen-svgd"
REF = os.environ["STAGE5_BOOTSTRAP_REF"]
ROOT = Path("/content/recurrent-qwen-svgd")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command)
    token = os.environ.get("GH_TOKEN", "")
    if token:
        printable = printable.replace(token, "****")
    print("$", printable, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


token = os.environ.get("GH_TOKEN", "")
clone_url = f"https://x-access-token:{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
run(["nvidia-smi"])
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
    run(["git", "fetch", "origin", REF], cwd=ROOT)
else:
    run(["git", "clone", clone_url, str(ROOT)])
run(["git", "reset", "--hard", REF], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_bicameral_w2p.py",
        "tests/test_bicameral.py",
    ],
    cwd=ROOT,
)
run([sys.executable, "-u", "colab/run_stage5_paper2_bicameral_w2p_d4.py"], cwd=ROOT)
