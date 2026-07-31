"""L4/A100 launcher for the pre-lock EVAL-C freeze and sole teacher cache."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_DC1_EVAL_C_FREEZE_VERSION = "paper2_dc1_eval_c_freeze_v1"
# Safety marker: pre-lock EVAL-C freeze only no scoring no optimizer no training
# Safety marker: fresh 200000 token EVAL-C 50 50 sources disjoint from D0 EVAL-B DEV-C
# Safety marker: one Qwen2.5 7B teacher pass no 14B and hash-only public receipt
# Safety marker: read-once EVAL-C scoring remains unspent after cache construction
# Safety marker: tests/test_paper2_dc1_preflight.py
# Safety marker: colab/run_stage5_paper2_dc1_eval_c_freeze.py
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
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
memory = int(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
)
assert memory >= 22000, (
    f"EVAL-C cache construction requires an L4-class or larger GPU; observed {memory} MiB."
)

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
        "tests/test_paper2_dc1_preflight.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_dc1_eval_c_freeze_target",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_dc1_eval_c_freeze.py"])
print("EVAL-C frozen and hash receipt published; registered scoring remains unspent.", flush=True)
