"""CPU launcher for the existing-record Phase-2 V1b RMS-tail audit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_V1B_RMS_AUDIT_VERSION = "paper2_phase2_v1b_rms_audit_v1"
# Safety marker: CPU only existing private V1b records no model inference no training
# Safety marker: p99 or fixed multiple median cap recommendation
# Safety marker: public token aggregates private per-row outlier identities
# Safety marker: colab/run_stage5_paper2_phase2_v1b_rms_audit.py
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
    subprocess.run(command, cwd=cwd or ROOT, env=os.environ.copy(), check=True)


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
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase2_v1b_rms_audit.py",
        "tests/test_paper2_phase2_v1b_rms_audit_launcher.py",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_v1b_rms_audit.py"])
print("Phase-2 V1b RMS audit landed; no model inference or training ran.", flush=True)
