"""A100/L4 launcher for DEV-C, DC1-P, and the RG-4/RG-11 preconditions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_DC1_PREFLIGHT_VERSION = "paper2_dc1_preflight_v1"
# Safety marker: no training no optimizer no checkpoint writes and EVAL-C untouched
# Safety marker: DEV-C 500000 tokens 50 50 sources document disjoint from D0 and EVAL-B
# Safety marker: scale interpolation slot attention position id fragility proxy
# Safety marker: horizontal append k <= 3 asserted and vertical loops fixed L=1
# Safety marker: stage-C-ready control readout logged but never routes execution
# Safety marker: RG-4 adjacent epsilon stability and RG-11 k 1 2 3 precision policy
# Safety marker: DC1 preregistration remains required before any training
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main")
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH and HF, "Missing GH_TOKEN or HF_TOKEN."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd or ROOT,
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
    return int(code)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
memory = int(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
)
assert memory >= 22000, f"DC1-P requires an L4-class or larger GPU; observed {memory} MiB."
if memory >= 70000:
    os.environ.setdefault("STAGE5_DC1_APPEND_BATCH_SIZE", "24")
elif memory >= 35000:
    os.environ.setdefault("STAGE5_DC1_APPEND_BATCH_SIZE", "12")
else:
    os.environ.setdefault("STAGE5_DC1_APPEND_BATCH_SIZE", "8")
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "reset", "--hard", REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_composite_training_design.py",
        "tests/test_paper2_dc1_preflight.py",
        "tests/test_coconut_composite.py",
        "tests/test_coconut_composite_numerics.py",
    ]
)
code = run(
    [sys.executable, "-u", "colab/run_stage5_paper2_dc1_preflight.py"],
    allowed=(0, 2),
)
if code == 2:
    print("DC1-P landed, but RG-4/RG-11 require review before preregistration.", flush=True)
