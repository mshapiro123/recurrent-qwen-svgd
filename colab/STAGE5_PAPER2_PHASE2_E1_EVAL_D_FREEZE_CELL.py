"""Colab launcher for the score-blind E1 EVAL-D infrastructure pass."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_E1_EVAL_D_FREEZE_VERSION = (
    "paper2_phase2_e1_eval_d_freeze_v1"
)
# Safety marker: 8000 anchors 4000 general 4000 code seed 20260808
# Safety marker: absent legacy EVAL-D materialized data-only with original seed 20260731
# Safety marker: EVAL-D infrastructure only no endpoint checkpoint no outcome score
# Safety marker: base student forward materializes cache tensors only no quality score
# Safety marker: no EAL no retention no acceptance no optimizer no training
# Safety marker: read-once scoring remains unspent and readiness only authorizes lock
# Safety marker: tests/test_paper2_phase2_e1_eval_d.py
# Safety marker: colab/run_stage5_paper2_phase2_e1_eval_d_freeze.py
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
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code:
        print("\nE1 EVAL-D launcher tail:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert shutil.which("nvidia-smi"), "Attach an A100-SXM4-80GB runtime and rerun."
run(["nvidia-smi"], Path("/content"))
memory = max(
    int(value.strip())
    for value in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if value.strip()
)
assert memory >= 70000, (
    "The pinned 32B score-blind teacher pass requires A100 80GB; "
    f"observed {memory} MiB."
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
        "tests/test_paper2_phase2_e1_confirmation.py",
        "tests/test_paper2_phase2_e1_eval_d.py",
        "tests/test_paper2_phase2_stage0a.py",
        "tests/test_paper2_phase2_eval_de.py",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_e1_eval_d_freeze.py"])
print("E1 EVAL-D cache frozen score-blind; read-once scoring remains unspent.", flush=True)
