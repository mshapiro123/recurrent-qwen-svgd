"""L4/A100 launcher for fresh DC0 EVAL-B freeze and 7B cache."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata

STAGE5_PAPER2_DC0_EVAL_B_PREPARE_VERSION = "paper2_dc0_eval_b_prepare_v1"
# Safety marker: fresh 200000 token EVAL-B 50 50 sources document disjoint from all prior D0 data
# Safety marker: one Qwen2.5 7B teacher pass only no 14B no optimizer no training
# Safety marker: read-once EVAL-B remains unspent until DC0 scoring
# Safety marker: resumable private teacher shards in Pharma Initiatives Drive
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
    subprocess.run(command, cwd=cwd or ROOT, check=True)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
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
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper2_dc0_eval_b.py"])
run([sys.executable, "-u", "colab/run_stage5_paper2_dc0_eval_b_prepare.py"])
