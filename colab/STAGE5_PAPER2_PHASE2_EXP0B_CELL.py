"""L4-or-larger launcher for DEV-only Experiment 0B flow-path screening."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata

STAGE5_PAPER2_PHASE2_EXP0B_VERSION = "paper2_phase2_exp0b_v1"
# Safety marker: DEV-only interpolation and serial-flow geometry screening
# Safety marker: affine targets no persistent-state renormalization loop cap four
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy(), check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert subprocess.run(["bash", "-lc", "command -v nvidia-smi"], check=False).returncode == 0
memory = max(
    int(value.strip())
    for value in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if value.strip()
)
assert memory >= 22000, f"Experiment 0B requires an L4-or-larger GPU; observed {memory} MiB."
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
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_phase2_stage0ab.py"])
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_exp0b.py"])
print("Experiment 0B interpolation and serial-flow geometry screening landed.", flush=True)

