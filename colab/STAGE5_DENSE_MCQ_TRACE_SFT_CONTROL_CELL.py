"""Colab cell: dense Qwen same-curriculum MCQ trace-SFT control.

This launches the standard-Qwen control arm for the current recurrent trace
SFT experiment. It trains dense LoRA on the same traced positive SFT rows, then
evaluates base-vs-dense on ARC-Easy/Challenge content and cyclic surfaces.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL_VERSION = "dense_mcq_trace_sft_control_v1"
STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_TARGET = "dense_mcq_trace_sft_control"
# Marker breadcrumbs for bootstrap stale-fetch checks. The runner invokes:
# - training/train_dense_lora.py
# - eval/eval_mcq.py --mode base --checkpoint
# - colab/run_stage5_mcq_dense_sft_control.py
# - colab/assess_stage5_mcq_recipe_control.py

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/"
    "stage5_local_hf_traced_capability_sft_20260623_194543/"
    "summary.json"
)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


DISCONNECT_WHEN_DONE = env_bool("STAGE5_DENSE_MCQ_DISCONNECT", True)


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


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def redact(text: str) -> str:
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            text = text.replace(token, "****")
    return text


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
):
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)


def disconnect(reason: str) -> None:
    if not DISCONNECT_WHEN_DONE:
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    print(
        "STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL_VERSION="
        f"{STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL_VERSION}",
        flush=True,
    )
    run(["nvidia-smi"], cwd=Path("/content"), check=False)
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
            "tests/test_eval_mcq_dense_lora.py",
            "tests/test_stage5_mcq_dense_sft_control.py",
            "tests/test_stage5_mcq_recipe_control_assessment.py",
        ],
        cwd=ROOT,
    )

    env = os.environ.copy()
    env.setdefault("STAGE5_DENSE_MCQ_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
    env.setdefault("STAGE5_DENSE_MCQ_RUN_ID", "stage5_dense_mcq_trace_sft_control_20260623")
    env.setdefault("STAGE5_DENSE_MCQ_BENCHMARKS", "arc_easy,arc_challenge")
    env.setdefault("STAGE5_DENSE_MCQ_ARC_EASY_LIMIT", "256")
    env.setdefault("STAGE5_DENSE_MCQ_ARC_CHALLENGE_LIMIT", "256")
    env.setdefault("STAGE5_DENSE_MCQ_SCORE_TARGETS", "content_question_only,cyclic_label_aggregated")
    env.setdefault("STAGE5_DENSE_MCQ_AGGREGATES", "mean")
    env.setdefault(
        "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY",
        "outputs/stage5/stage5_local_hf_traced_sft_scale64_benchmark_20260623_201923/summary.json",
    )
    env.setdefault("STAGE5_DENSE_MCQ_COMMIT_CHECKPOINT", "0")
    env.setdefault("STAGE5_DENSE_MCQ_PUSH", "1")
    print("dense_mcq_source:", env["STAGE5_DENSE_MCQ_SOURCE_SUMMARY"], flush=True)
    print("dense_mcq_run_id:", env["STAGE5_DENSE_MCQ_RUN_ID"], flush=True)
    run([sys.executable, "colab/run_stage5_mcq_dense_sft_control.py"], cwd=ROOT, env=env)
    disconnect("dense MCQ trace-SFT control finished")
except Exception:
    disconnect("dense MCQ trace-SFT control errored")
    raise
