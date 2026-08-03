"""A100 launcher for DEV-only Experiment 0A canonicalizer screening."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata

STAGE5_PAPER2_PHASE2_EXP0A_VERSION = "paper2_phase2_exp0a_v1"
# Safety marker: DEV-only canonicalizer and whitening screening no backbone training
# Safety marker: shared PCA basis alpha 0 0.5 1 single eigenvalue floor no second epsilon
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
assert memory >= 70000, f"Experiment 0A requires an A100-80GB; observed {memory} MiB."
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
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_exp0a.py"])
print("Experiment 0A canonicalizer and whitening screening landed.", flush=True)

