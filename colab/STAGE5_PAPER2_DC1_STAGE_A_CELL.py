"""A100 launcher for locked DC1 Stage A training and the sole EVAL-C pass."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_DC1_STAGE_A_VERSION = "paper2_dc1_stage_a_v1"
# Safety marker: locked_before_training commit d25b3d0e before runner construction
# Safety marker: full fp32 bridge-only AdamW 2000 steps forced k1 L1 recompute only
# Safety marker: exact allowlist horizontal_bridge.delta.weight and frozen hash assertions
# Safety marker: 20-step throughput memory projection then passive checkpoints 500 1000 1500 2000
# Safety marker: single read-once EVAL-C pass immutable cache all five registered arms
# Safety marker: verdict script eval/eval_paper2_dc1_stage_a_verdict.py 10000 row bootstraps seed 20260730
# Safety marker: Drive checkpoint resume and training receipt publishes before EVAL-C
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
assert memory >= 70000, (
    f"Locked Stage A requires an A100-80GB-class GPU; observed {memory} MiB."
)
os.environ.setdefault("STAGE5_DC1_STAGE_A_EVAL_BATCH_SIZE", "24")

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
        "tests/test_paper2_dc1_stage_a_lock.py",
        "tests/test_paper2_dc1_stage_a.py",
        "tests/test_coconut_composite.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_dc1_stage_a_target",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_dc1_stage_a.py"])
print("DC1 Stage A training and registered EVAL-C verdict landed.", flush=True)
