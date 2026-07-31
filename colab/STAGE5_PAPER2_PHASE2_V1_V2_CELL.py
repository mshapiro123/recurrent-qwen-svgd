"""L4/A100 launcher for the no-training Phase-2 V1/V2 checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_V1_V2_VERSION = "paper2_phase2_v1_v2_v1"
# Safety marker: DEV-C only V1 V2 centered finite-difference JVP no optimizer no training
# Safety marker: sampled directional gain is not a certified Lipschitz upper bound
# Safety marker: gamma 0.05 rho 0.8 c 0.01 0.02 0.05
# Safety marker: pre-D0 and post-D0 block iterates 1 through 4
# Safety marker: colab/run_stage5_paper2_phase2_v1_v2.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH and HF, "Missing GH_TOKEN or HF_TOKEN."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
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
    if code:
        raise subprocess.CalledProcessError(code, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert shutil.which("nvidia-smi"), "Attach an L4 or larger NVIDIA GPU and rerun."
run(["nvidia-smi"], Path("/content"))
memory = int(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
)
assert memory >= 22000, f"V1/V2 require L4-class memory; observed {memory} MiB."
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
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_phase2_v1_v2.py"])
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_v1_v2.py"])
print("Phase-2 V1/V2 diagnostics landed; no training window opened automatically.", flush=True)
