"""L4 launcher for the read-only pre/post-D0 population parity ledger."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_DC1_PARITY_LEDGER_VERSION = "paper2_dc1_parity_ledger_v1"
# Safety marker: read-only pre post D0 population parity no optimizer no training
# Safety marker: legacy rejected-only floor forces exact same-row DEV-C fallback
# Safety marker: accepted rejected split is defined by each checkpoint depth-1 agreement
# Safety marker: EVAL-C and EVAL-B remain untouched
# Safety marker: fail clear before launch when Colab has no NVIDIA GPU runtime
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main")
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH and HF, "Missing GH_TOKEN or HF_TOKEN."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy(), check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert shutil.which("nvidia-smi"), (
    "No NVIDIA GPU runtime is attached. In Colab choose Runtime > Change runtime type > "
    "L4 GPU, reconnect, and rerun this cell. No experiment work has started."
)
run(["nvidia-smi"], Path("/content"))
memory = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
assert memory >= 22000, f"DC1 parity fallback requires L4-class memory; observed {memory} MiB."
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
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_dc1_followups.py"])
run([sys.executable, "-u", "colab/run_stage5_paper2_dc1_parity_ledger.py"])
