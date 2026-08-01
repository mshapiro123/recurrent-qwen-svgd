"""A100 launcher for the authorized DEV-only Phase-2 V1b check."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_V1B_VERSION = "paper2_phase2_v1b_v1"
# Safety marker: DEV-only finite perturbation causal check no optimizer no training
# Safety marker: teacher flip pair crossing and collateral damage reported separately
# Safety marker: preserve controls use teacher-favoring strongest-competitor margins
# Safety marker: locked V1 append cache prefix ignores trailing historical batches
# Safety marker: colab/run_stage5_paper2_phase2_v1b.py
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
assert shutil.which("nvidia-smi"), "Attach an A100 GPU and rerun."
run(["nvidia-smi"], Path("/content"))
memory = int(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
)
assert memory >= 39000, f"V1b requires A100-class memory; observed {memory} MiB."
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
        "tests/test_paper2_phase2_v1b.py",
        "tests/test_paper2_phase2_v1b_launcher.py",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_v1b.py"])
print("Phase-2 V1b receipt landed; no training window opened automatically.", flush=True)
