"""Colab cell: run bounded MCQ content/cyclic surface-alignment repair.

This is the narrow follow-up to the direct-preservation confirmation. It
diagnoses the ARC-Easy content/cyclic surface mismatch, builds a small targeted
surface-alignment SFT shard, runs a bounded Phase 1 continuation, then
benchmarks and assesses the repaired checkpoint.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL_VERSION = "surface_alignment_repair_v1"
STAGE5_SURFACE_ALIGNMENT_REPAIR_TARGET = "traced_sft_surface_alignment_repair"
# Marker breadcrumbs for bootstrap stale-fetch checks. The runner invokes:
# - eval/analyze_mcq_order_sensitivity.py
# - eval/analyze_mcq_surface_mismatch.py
# - training/prepare_mcq_conditional_invariance_jsonl.py
# - training/prepare_mcq_surface_alignment_jsonl.py
# - colab/assess_stage5_surface_repair.py

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/"
    "stage5_traced_sft_direct_preservation_20260623_scale64_confirm_assessment/"
    "summary.json"
)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


DISCONNECT_WHEN_DONE = env_bool("STAGE5_SURFACE_ALIGN_DISCONNECT", True)


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
        "STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL_VERSION="
        f"{STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL_VERSION}",
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
            "tests/test_prepare_mcq_surface_alignment_jsonl.py",
            "tests/test_prepare_mcq_conditional_invariance_jsonl.py",
            "tests/test_analyze_mcq_order_sensitivity.py",
            "tests/test_analyze_mcq_surface_mismatch.py",
            "tests/test_stage5_surface_alignment_repair.py",
            "tests/test_stage5_surface_repair_assessment.py",
        ],
        cwd=ROOT,
    )

    env = os.environ.copy()
    env.setdefault("STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
    env.setdefault(
        "STAGE5_SURFACE_ALIGN_RUN_ID",
        "stage5_surface_alignment_repair_from_direct_confirm_20260623",
    )
    env.setdefault("STAGE5_SURFACE_ALIGN_MAX_STEPS", "50")
    env.setdefault("STAGE5_SURFACE_ALIGN_LR", "5e-7")
    env.setdefault("STAGE5_SURFACE_ALIGN_DISTILL_WEIGHT", "0.05")
    env.setdefault("STAGE5_SURFACE_ALIGN_PUSH", "1")
    print("surface_alignment_source:", env["STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY"], flush=True)
    print("surface_alignment_run_id:", env["STAGE5_SURFACE_ALIGN_RUN_ID"], flush=True)
    run([sys.executable, "colab/run_stage5_surface_alignment_repair.py"], cwd=ROOT, env=env)
    disconnect("surface-alignment repair finished")
except Exception:
    disconnect("surface-alignment repair errored")
    raise
