"""Colab launcher for the locked Phase-2 Option B four-arm matrix."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_OPTION_B_VERSION = "paper2_phase2_option_b_v1"
# Safety marker: hash-only amendment locked before Option B training
# Safety marker: four A2 endpoint arms fresh AdamW state exact step 4000 splice
# Safety marker: 20000 steps eval checkpoint 1000 directional audit 2000
# Safety marker: identical full control sample schedule within seed
# Safety marker: fixed evaluation excluded from both training populations
# Safety marker: colab/run_stage5_paper2_phase2_option_b.py
# Safety marker: tests/test_paper2_phase2_option_b_training.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH = secret("GH_TOKEN", "GITHUB_TOKEN")
HF = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH and HF, "Missing GH_TOKEN or HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy(), check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert shutil.which("nvidia-smi"), "Attach an A100 runtime and rerun."
run(["nvidia-smi"], Path("/content"))
memory = max(
    int(value.strip())
    for value in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if value.strip()
)
assert memory >= 38000, f"Option B requires A100 40GB or larger; observed {memory} MiB."
print(f"option_b_gpu_memory_mib={memory}", flush=True)

url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
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
        "tests/test_paper2_phase2_option_b_training.py",
        "tests/test_paper2_phase2_option_b_lock.py",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_option_b.py"])
print("Option B four-arm matrix completed or reached a named locked cliff.", flush=True)
