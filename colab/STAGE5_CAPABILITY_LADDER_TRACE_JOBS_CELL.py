"""Colab cell: build capability-ladder trace-generation jobs on CPU.

This target follows the latest capability-ladder MCQ probe summary, restores
private scored rows from Drive if a fresh runtime lost them, and builds
provider-neutral strong-model trace jobs. It refuses visible GPU runtimes by
default because no model inference is needed here.

Bootstrap marker: training/build_capability_ladder_trace_jobs.py is the
underlying curriculum builder invoked by colab/run_stage5_capability_ladder_trace_jobs.py.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION = "capability_ladder_trace_jobs_cpu_v1"
STAGE5_CAPABILITY_LADDER_TRACE_JOBS_TARGET = "capability_ladder_trace_jobs_cpu"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


def secret(*names):
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


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."


def redact(text):
    return str(text).replace(GH_TOKEN, "****") if GH_TOKEN else str(text)


def run(cmd, cwd=None, env=None, check=True):
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(redact(proc.stdout), flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {redact(' '.join(map(str, cmd)))}")
    return proc


def attached_gpu_names():
    if shutil.which("nvidia-smi") is None:
        return []
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


gpus = attached_gpu_names()
if gpus and os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU", "0") != "1":
    raise RuntimeError(
        "Refusing CPU-only capability-ladder trace-job building on attached GPU runtime: "
        + "; ".join(gpus)
        + ". Switch to CPU runtime or set STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU=1 deliberately."
    )
print(f"STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION={STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION}", flush=True)

clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
else:
    run(["git", "clone", clone_url, str(ROOT)])

run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive", force_remount=True)

run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_capability_ladder_trace_jobs.py",
        "tests/test_stage5_next_plan.py::test_capability_ladder_mcq_probe_with_rows_recommends_trace_jobs_before_sft_gate",
    ],
    cwd=ROOT,
)

env = os.environ.copy()
env.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_MODELS", "opus-strong,glm-strong")
env.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_BACKUP_DRIVE", "1")
env.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_PUSH", "1")
env.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_REFUSE_GPU", "1")

run([sys.executable, "colab/run_stage5_capability_ladder_trace_jobs.py"], cwd=ROOT, env=env)

summary_pointer = ROOT / "config" / "stage5_current_source_summary.txt"
print("current_source_summary:", summary_pointer.read_text(encoding="utf-8").strip(), flush=True)
print("Capability-ladder trace jobs are ready for provider response generation.", flush=True)

if os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_DISCONNECT", "1") == "1":
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    runtime.unassign()
