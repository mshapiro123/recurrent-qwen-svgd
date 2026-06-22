"""Colab cell: confirm max_loops=1 direct-route preservation on larger ARC slices.

This intentionally bypasses the generic planner. It runs a broader benchmark
suite with recurrent ``max_loops=1`` on ARC-Easy and ARC-Challenge, comparing
against base Qwen with the same MCQ scoring path.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL_VERSION = "direct_preservation_confirm_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
SOURCE_SUMMARY = "outputs/stage5/stage5_direct_preservation_loop1_20260622_232720/summary.json"
CHECKPOINT = "outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt"


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
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    run_id = os.environ.get("STAGE5_DIRECT_CONFIRM_RUN_ID") or time.strftime(
        "stage5_direct_preservation_confirm_loop1_%Y%m%d_%H%M%S"
    )
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": run_id,
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": SOURCE_SUMMARY,
            "STAGE5_BENCHMARK_CHECKPOINT": CHECKPOINT,
            "STAGE5_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get("STAGE5_DIRECT_CONFIRM_ARC_EASY_LIMIT", "256"),
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                "STAGE5_DIRECT_CONFIRM_ARC_CHALLENGE_LIMIT", "256"
            ),
            "STAGE5_BENCHMARK_MAX_LOOPS": "1",
            "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
            "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get("STAGE5_DIRECT_CONFIRM_SCORE_TARGETS", "label"),
            "STAGE5_BENCHMARK_AGGREGATES": "mean",
            "STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS": "1",
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
        }
    )
    print("direct_preservation_confirmation_run_id:", run_id, flush=True)
    print("recommended_runtime: T4/L4 is sufficient; A100 is not required.", flush=True)
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=env)

    run_dir = ROOT / "outputs" / "stage5" / run_id
    if run_dir.exists():
        drive_dst = DRIVE_ARTIFACT_ROOT / "stage5" / run_id
        drive_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, drive_dst, dirs_exist_ok=True)
        print(f"backed_up_run_dir={run_dir} -> {drive_dst}", flush=True)
    disconnect("direct preservation confirmation finished")
except Exception:
    disconnect("direct preservation confirmation errored")
    raise
