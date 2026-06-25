"""Fresh Colab launcher for the current Stage 5 GPU action.

Paste/run this whole file in a new Colab notebook, or fetch it from GitHub and
``exec`` it. It is intentionally self-contained: it authenticates from Colab
secrets, clones or hard-resets the private repo, mounts Drive in the notebook
process, and then delegates to ``colab/CURRENT_A100_BOOTSTRAP_CELL.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from google.colab import drive, userdata


CURRENT_STAGE5_FRESH_LAUNCHER_VERSION = "fresh_launcher_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_TARGET = "traced_sft_competence_preserving_pipeline"
DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_debiased_benchmark_assessment_20260625_121302/summary.json"
DEFAULT_RUN_ID = "stage5_competence_recovery_from_reentry_benchmark"


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    if GH_TOKEN:
        printable = printable.replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "-C", str(ROOT), "remote", "set-url", "origin", clone_url])
    run(["git", "-C", str(ROOT), "fetch", "origin", "main"])
    run(["git", "-C", str(ROOT), "checkout", "main"])
    run(["git", "-C", str(ROOT), "reset", "--hard", "origin/main"])
else:
    run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))

run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)

os.chdir(ROOT)
print("launcher_version:", CURRENT_STAGE5_FRESH_LAUNCHER_VERSION, flush=True)
print(subprocess.check_output(["git", "log", "--oneline", "-6"], text=True), flush=True)

bootstrap = (ROOT / "colab" / "CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
competence_cell = (ROOT / "colab" / "STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL.py").read_text(
    encoding="utf-8"
)
assert "STAGE5_BOOTSTRAP_PREFER_LOCAL_HEAD" in bootstrap, "Missing local-HEAD bootstrap freshness guard."
assert "STAGE5_COMPETENCE_MOUNT_DRIVE_FIRST" in competence_cell, "Missing top-level Drive mount fix."

drive.mount("/content/drive", force_remount=env_bool("FORCE_DRIVE_REMOUNT", False))

os.environ.setdefault("STAGE5_CURRENT_A100_TARGET", DEFAULT_TARGET)
os.environ.setdefault("STAGE5_COMPETENCE_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
os.environ.setdefault("STAGE5_COMPETENCE_PIPELINE_RUN_ID", DEFAULT_RUN_ID)
os.environ.setdefault("STAGE5_COMPETENCE_PIPELINE_DISCONNECT", "0")
os.environ.setdefault("STAGE5_COMPETENCE_MOUNT_DRIVE_FIRST", "0")
os.environ.setdefault("STAGE5_BOOTSTRAP_PREFER_LOCAL_HEAD", "1")

exec(compile(bootstrap, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
