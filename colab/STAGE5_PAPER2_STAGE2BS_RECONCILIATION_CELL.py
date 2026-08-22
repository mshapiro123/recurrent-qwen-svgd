"""Colab launcher for the authorized Stage 2B-S serving-graph reconciliation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_STAGE2BS_RECONCILIATION_VERSION = "paper2_stage2bs_reconciliation_v1"
# Contract marker: immutable initialization paired tensor trace before any optimizer
# Contract marker: P3.5 one-shot graph versus Stage 2B recurrent serving graph
# Contract marker: DEV-1 only CONFIRM and EVAL-E remain sealed
# Contract marker: no optimizer no training all retained artifacts SHA-256 banked
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = os.environ.get("GH_TOKEN", "").strip()


def run(command: list[str], cwd: Path | None = None, *, capture: bool = False) -> str:
    rendered = " ".join(command).replace(GH, "****") if GH else " ".join(command)
    print("$", rendered, flush=True)
    result = subprocess.run(
        command,
        cwd=cwd or (ROOT if ROOT.exists() else Path("/content")),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        print(result.stdout, end="", flush=True)
        return result.stdout
    return ""


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
smi = run(["nvidia-smi"], capture=True)
match = re.search(r"NVIDIA A100-SXM4-(40|80)GB", smi)
if match is None:
    raise RuntimeError("Stage 2B-S reconciliation requires one NVIDIA A100-SXM4 runtime")
print(f"stage2bs_reconciliation_runtime_pinned=A100-SXM4-{match.group(1)}GB", flush=True)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git" if GH else f"https://github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(ROOT)], Path("/content"))
    run(["git", "sparse-checkout", "init", "--cone"])
    run(
        [
            "git",
            "sparse-checkout",
            "set",
            "colab",
            "eval",
            "training",
            "models",
            "tests",
            "configs",
            "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812",
        ]
    )
run(["git", "fetch", "origin", REF])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_stage2bs_reconciliation.py",
        "tests/test_paper2_stage2b_depth.py",
    ]
)
run([sys.executable, "-u", "-m", "colab.run_stage5_paper2_stage2bs_reconciliation"])
print("Stage 2B-S reconciliation landed; inspect the paired trace before releasing the A100.", flush=True)
