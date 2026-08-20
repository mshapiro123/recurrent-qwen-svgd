"""Colab A100 launcher for the signed Stage 2B-A score-only autopsy."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_STAGE2B_AUTOPSY_VERSION = "paper2_stage2b_autopsy_v2"
# Safety marker: signed score-only lock no optimizer no training
# Safety marker: CONFIRM and EVAL-E remain sealed
# Contract marker: zero-write full-logit identity precedes diagnostic cells
# Contract marker: amended six-arm autopsy both seeds one A100-SXM4-40GB session
# Contract marker: exact endpoints plus contemporaneous telemetry no onset interpolation
# Contract marker: Arm 6 read-only autograd no optimizer no parameter mutation
# Reliability marker: batch-resumable DEV-1 and sparse transport
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = os.environ.get("GH_TOKEN", "").strip()


def run(command: list[str], cwd: Path | None = None, *, capture: bool = False) -> str:
    rendered = " ".join(command)
    if GH:
        rendered = rendered.replace(GH, "****")
    print("$", rendered, flush=True)
    if capture:
        process = subprocess.run(
            command,
            cwd=cwd or (ROOT if ROOT.exists() else Path("/content")),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(process.stdout, end="", flush=True)
        return process.stdout
    subprocess.run(
        command,
        cwd=cwd or (ROOT if ROOT.exists() else Path("/content")),
        check=True,
    )
    return ""


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
smi = run(["nvidia-smi"], capture=True)
match = re.search(r"A100-SXM4-(\d+)GB", smi)
if match is None or int(match.group(1)) != 40:
    raise RuntimeError("Signed Stage 2B-A runtime requires NVIDIA A100-SXM4-40GB")
url = f"https://x-access-token:{GH}@github.com/{REPO}.git" if GH else f"https://github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(ROOT)],
        Path("/content"),
    )
    run(["git", "sparse-checkout", "init", "--cone"])
    run(
        [
            "git", "sparse-checkout", "set", "colab", "eval", "training", "models",
            "tests", "configs", "docs/receipts",
            "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812",
        ]
    )
run(["git", "fetch", "origin", REF])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_paper2_stage2b_autopsy.py", "tests/test_paper2_stage2b_depth.py",
])
os.environ["STAGE2B_AUTOPSY_MODE"] = "run"
run([sys.executable, "-u", "-m", "colab.run_stage5_paper2_stage2b_autopsy"])
print("Stage 2B-A score-only autopsy landed; inspect durable receipts before releasing the A100.", flush=True)
