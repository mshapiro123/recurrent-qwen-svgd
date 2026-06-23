"""Colab cell: run capability-ladder trace provider responses.

This CPU/network target follows the latest trace-job summary and runs
``training/run_curriculum_job_responses.py``. It refuses visible GPU runtimes by
default. It also refuses provider/API spend unless
``STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER=1`` is set.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL_VERSION = "capability_ladder_trace_responses_cpu_v1"
STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_TARGET = "capability_ladder_trace_responses_cpu"
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
API_KEY_ENV = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_API_KEY_ENV", "").strip()
if not API_KEY_ENV:
    if secret("OPENAI_API_KEY"):
        API_KEY_ENV = "OPENAI_API_KEY"
    elif secret("OPENROUTER_API_KEY"):
        API_KEY_ENV = "OPENROUTER_API_KEY"
    else:
        API_KEY_ENV = "OPENAI_API_KEY"
os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_API_KEY_ENV"] = API_KEY_ENV
if (
    API_KEY_ENV == "OPENROUTER_API_KEY"
    and "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BASE_URL" not in os.environ
):
    os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BASE_URL"] = "https://openrouter.ai/api/v1"
provider_token = secret(API_KEY_ENV)
if provider_token:
    os.environ[API_KEY_ENV] = provider_token


def redact(text):
    text = str(text).replace(GH_TOKEN, "****") if GH_TOKEN else str(text)
    if provider_token:
        text = text.replace(provider_token, "****")
    return text


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
if gpus and os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU", "0") != "1":
    raise RuntimeError(
        "Refusing CPU/network trace-response collection on attached GPU runtime: "
        + "; ".join(gpus)
        + ". Switch to CPU runtime or set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU=1 deliberately."
    )
print(f"STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL_VERSION={STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL_VERSION}", flush=True)
print(
    {
        "api_key_env": API_KEY_ENV,
        "has_provider_token": bool(provider_token),
        "base_url": os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BASE_URL", "https://api.openai.com/v1"),
        "model_override": os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE", ""),
        "run_provider": os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER", "0"),
    },
    flush=True,
)

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
        "tests/test_curriculum_job_responses.py",
        "tests/test_stage5_capability_ladder_trace_responses_runner.py",
    ],
    cwd=ROOT,
)

env = os.environ.copy()
env.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKUP_DRIVE", "1")
env.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_PUSH", "1")
env.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_REFUSE_GPU", "1")

run([sys.executable, "colab/run_stage5_capability_ladder_trace_responses.py"], cwd=ROOT, env=env)

summary_pointer = ROOT / "config" / "stage5_current_source_summary.txt"
print("current_source_summary:", summary_pointer.read_text(encoding="utf-8").strip(), flush=True)
print("Capability-ladder trace responses step is complete.", flush=True)

if os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_DISCONNECT", "1") == "1":
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    runtime.unassign()
