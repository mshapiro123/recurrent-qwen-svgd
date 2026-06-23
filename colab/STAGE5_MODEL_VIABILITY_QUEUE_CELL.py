"""Colab cell: run queued Qwen model-scale viability probes.

Default queue:
  * Qwen2.5-3B-Instruct on >=22 GB VRAM;
  * Qwen2.5-7B-Instruct on >=39 GB VRAM.

Override ``STAGE5_MODEL_QUEUE_MODELS`` to probe other Qwen-family checkpoints.
Entry format is semicolon-separated:

``label=model_name|layer_split|loops|min_vram_gb|arc_easy_limit|arc_challenge_limit|identity_dtype``
"""

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_MODEL_VIABILITY_QUEUE_CELL_VERSION = "model_viability_queue_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_MODEL_QUEUE_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


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
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def run(cmd, *, cwd=None, env=None, check=True):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def sync_repo():
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


def disconnect(reason):
    if not DISCONNECT_ON_FINISH:
        print(f"Leaving Colab runtime connected: {reason}", flush=True)
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    sync_repo()
    os.chdir(ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_model_viability_probe.py",
            "tests/test_stage5_notebooks.py",
        ],
        cwd=ROOT,
    )

    env = os.environ.copy()
    env.setdefault(
        "STAGE5_MODEL_QUEUE_MODELS",
        (
            "qwen_3b=Qwen/Qwen2.5-3B-Instruct|auto|1,2|22|32|32|float32;"
            "qwen_7b=Qwen/Qwen2.5-7B-Instruct|auto|1,2|39|24|24|float32"
        ),
    )
    env.setdefault("STAGE5_MODEL_QUEUE_SCORE_TARGETS", "label,content_question_only")
    env.setdefault("STAGE5_MODEL_QUEUE_EVAL_DTYPE", "bfloat16")
    env.setdefault("STAGE5_MODEL_QUEUE_ADAPTER_DTYPE", "float32")
    env.setdefault("STAGE5_MODEL_QUEUE_DEVICE", "cuda")
    env.setdefault("STAGE5_MODEL_QUEUE_CHILD_PUSH", "1")
    env.setdefault("STAGE5_MODEL_QUEUE_PUSH", "1")
    env.setdefault("STAGE5_MODEL_QUEUE_CONTINUE_ON_FAILURE", "1")

    print("model_viability_queue_config:", flush=True)
    for key in sorted(k for k in env if k.startswith("STAGE5_MODEL_QUEUE_")):
        print(f"{key}={env[key]}", flush=True)

    run([sys.executable, "colab/run_stage5_model_viability_queue.py"], cwd=ROOT, env=env)
    disconnect("model viability queue finished")
except Exception:
    disconnect("model viability queue errored")
    raise
