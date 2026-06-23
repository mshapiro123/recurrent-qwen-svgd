"""Colab cell: no-training Qwen model viability probe.

Default target is Qwen 1.5B, but this is intentionally generic. Override
``STAGE5_MODEL_PROBE_MODEL_NAME`` to probe Qwen 3B or larger checkpoints.
The probe runs identity, loop-1 preservation, and a tiny loop-depth sweep
without SFT or checkpoint loading.
"""

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_MODEL_VIABILITY_PROBE_CELL_VERSION = "model_viability_probe_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_MODEL_PROBE_DISCONNECT", "0").strip().lower() in {
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
    env.setdefault("STAGE5_MODEL_PROBE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    env.setdefault("STAGE5_MODEL_PROBE_MODEL_LABEL", "qwen_1_5b")
    env.setdefault("STAGE5_MODEL_PROBE_LAYER_SPLIT", "auto")
    env.setdefault("STAGE5_MODEL_PROBE_LOOPS", "1,2")
    env.setdefault("STAGE5_MODEL_PROBE_ARC_EASY_LIMIT", "32")
    env.setdefault("STAGE5_MODEL_PROBE_ARC_CHALLENGE_LIMIT", "32")
    env.setdefault("STAGE5_MODEL_PROBE_SCORE_TARGETS", "label,content_question_only")
    env.setdefault("STAGE5_MODEL_PROBE_IDENTITY_DTYPE", "float32")
    env.setdefault("STAGE5_MODEL_PROBE_IDENTITY_ATTN", "eager")
    env.setdefault("STAGE5_MODEL_PROBE_EVAL_DTYPE", "bfloat16")
    env.setdefault("STAGE5_MODEL_PROBE_ADAPTER_DTYPE", "float32")
    env.setdefault("STAGE5_MODEL_PROBE_DEVICE", "cuda")
    env.setdefault("STAGE5_MODEL_PROBE_PUSH", "1")

    print("model_viability_probe_config:", flush=True)
    for key in sorted(k for k in env if k.startswith("STAGE5_MODEL_PROBE_")):
        print(f"{key}={env[key]}", flush=True)

    run([sys.executable, "colab/run_stage5_model_viability_probe.py"], cwd=ROOT, env=env)
    disconnect("model viability probe finished")
except Exception:
    disconnect("model viability probe errored")
    raise
