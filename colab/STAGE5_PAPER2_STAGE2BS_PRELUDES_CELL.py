"""Colab A100 launcher for the signed Stage 2B-S preludes."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_STAGE2BS_PRELUDES_VERSION = "paper2_stage2bs_preludes_v1"
# Contract marker: mandatory bit-exact preflight relay precedes every probe cell
# Contract marker: both score-only preludes one pinned session no optimizer no training
# Contract marker: CONFIRM and EVAL-E remain sealed
# Reliability marker: resumable condition receipts and all artifacts SHA-256 banked
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = os.environ.get("GH_TOKEN", "").strip()
MODE = os.environ.get("STAGE2BS_PRELUDE_MODE", "preflight").strip().lower()


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
match = re.search(r"A100-SXM4-(\d+)GB", smi)
if match is None or int(match.group(1)) != 40:
    raise RuntimeError("Signed Stage 2B-S runtime requires NVIDIA A100-SXM4-40GB")
url = f"https://x-access-token:{GH}@github.com/{REPO}.git" if GH else f"https://github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(ROOT)], Path("/content"))
    run(["git", "sparse-checkout", "init", "--cone"])
    run([
        "git", "sparse-checkout", "set", "colab", "eval", "training", "models",
        "tests", "configs", "docs/receipts",
        "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812",
    ])
run(["git", "fetch", "origin", REF])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_paper2_stage2bs_preludes.py",
    "tests/test_paper2_stage2b_depth.py",
])
os.environ["STAGE2BS_PRELUDE_MODE"] = MODE
run([sys.executable, "-u", "-m", "colab.run_stage5_paper2_stage2bs_preludes"])
print(
    "Stage 2B-S preflight landed; relay before probes."
    if MODE == "preflight"
    else "Stage 2B-S prelude wave landed; inspect receipts before releasing the A100.",
    flush=True,
)
