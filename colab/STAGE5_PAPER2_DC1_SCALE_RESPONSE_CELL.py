"""L4 launcher for the read-only DC1 scale-response probe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_DC1_SCALE_RESPONSE_VERSION = "paper2_dc1_scale_response_v1"
# Safety marker: read-only residual-stream scale response no optimizer no training
# Safety marker: same 113 DEV-C rows with raw 1.5x and 2x extensions
# Safety marker: final hidden cosine to fed state and registered k0 state
# Safety marker: optional layerwise residual cosine at raw and 10x
# Safety marker: EVAL-C and EVAL-B remain untouched
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
run(["nvidia-smi"], Path("/content"))
memory = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
assert memory >= 22000, f"DC1 scale probe requires L4-class memory; observed {memory} MiB."
if memory >= 70000:
    os.environ.setdefault("STAGE5_DC1_SCALE_BATCH_SIZE", "24")
elif memory >= 35000:
    os.environ.setdefault("STAGE5_DC1_SCALE_BATCH_SIZE", "12")
else:
    os.environ.setdefault("STAGE5_DC1_SCALE_BATCH_SIZE", "8")
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
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_dc1_followups.py", "tests/test_coconut_composite.py"])
run([sys.executable, "-u", "colab/run_stage5_paper2_dc1_scale_response.py"])
