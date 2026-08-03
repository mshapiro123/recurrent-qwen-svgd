"""High-RAM CPU launcher for Phase-2 arbitration and loss-free build receipts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil
from google.colab import drive, userdata

STAGE5_PAPER2_PHASE2_ARBITRATION_BUILD_VERSION = "paper2_phase2_arbitration_build_v2"
# Safety marker: CPU high RAM cached canonicalizer arbitration and loss-free student build
# Safety marker: three common SVD seeds paired mixture arbitration no optimizer no training
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."
assert HF, "Missing HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy(), check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
ram_gib = psutil.virtual_memory().total / 2**30
assert ram_gib >= 100, (
    f"Canonicalizer refit requires a Colab high-RAM CPU runtime (>=100 GiB); observed {ram_gib:.1f} GiB."
)
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
        "tests/test_paper2_phase2_arbitration.py",
        "tests/test_paper2_dc2_student.py",
        "tests/test_paper2_phase2_stage0ab.py",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_arbitration_build.py"])
print("Phase-2 canonicalizer arbitration and loss-free student build landed.", flush=True)
