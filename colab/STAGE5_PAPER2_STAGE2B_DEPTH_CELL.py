"""Colab launcher for the signed Stage 2B-D cache, preflight, or seed run."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_STAGE2B_DEPTH_VERSION = "paper2_stage2b_depth_campaign_v1"
# Safety marker: signed lock asserted immediately before optimizer construction
# Safety marker: first wave stops at step 5000 for strategy adjudication
# Safety marker: resumable sharded 14B cache and deterministic seed checkpoints
# Safety marker: DEV-1 floors DEV-2 margins CONFIRM and EVAL-E remain sealed
# Runner marker: colab.run_stage5_paper2_stage2b_depth
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = os.environ.get("GH_TOKEN", "").strip()


def run(command: list[str], cwd: Path | None = None) -> str:
    printable = " ".join(command).replace(GH, "****") if GH else " ".join(command)
    print("$", printable, flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd or (ROOT if ROOT.exists() else Path("/content")),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return "".join(lines)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
smi = run(["nvidia-smi"])
match = re.search(r"A100-SXM4-(\d+)GB", smi)
if match is None or int(match.group(1)) != 40:
    raise RuntimeError("Signed Stage 2B runtime requires NVIDIA A100-SXM4-40GB")
url = (
    f"https://x-access-token:{GH}@github.com/{REPO}.git"
    if GH
    else f"https://github.com/{REPO}.git"
)
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "fetch", "origin", "main"])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_stage2b_depth.py",
        "tests/test_paper2_stage2b_runtime.py",
        "tests/test_stage5_notebooks.py::test_stage2b_depth_campaign_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "-m", "colab.run_stage5_paper2_stage2b_depth"])
print("Stage 2B target returned; inspect the durable status before releasing this session.", flush=True)
