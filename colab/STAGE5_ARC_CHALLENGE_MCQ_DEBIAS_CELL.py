"""Colab cell: confirm ARC-Challenge MCQ option-bias with cyclic scoring.

This is a bounded no-training GPU diagnostic. It reuses the current MCQ debias
runner, but pins it to ARC-Challenge and enables quiet/resumable output so a
Colab notebook does not fill with per-example JSON.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL_VERSION = "arc_challenge_mcq_debias_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


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


def mask(text: str, token: str | None) -> str:
    return text.replace(token, "****") if token else text


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    shown = mask(" ".join(map(str, cmd)), GH_TOKEN)
    print("$", shown, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = mask(proc.stdout or "", GH_TOKEN)
    if output:
        print(output, flush=True)
    if proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(output.splitlines()[-120:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN/GITHUB_TOKEN in Colab secrets."

HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

try:
    gpu_check = shutil.which("nvidia-smi")
    assert gpu_check, "This bounded diagnostic should run on a GPU runtime; nvidia-smi was not found."
    run(["nvidia-smi"], cwd=Path("/content"))

    drive.mount("/content/drive", force_remount=False)
    authed = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", authed])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", authed, str(ROOT)], cwd=Path("/content"))
        run(["git", "remote", "set-url", "origin", authed])

    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_mcq_debias.py",
            "tests/test_stage5_next_plan.py",
        ]
    )

    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    source_summary = os.environ.get("STAGE5_MCQ_DEBIAS_SOURCE_SUMMARY", "").strip()
    if not source_summary:
        source_summary = pointer.read_text(encoding="utf-8").strip()
    assert source_summary, "Missing source summary for ARC-Challenge MCQ debias confirmation."
    print("source_summary:", source_summary, flush=True)

    env = os.environ.copy()
    env.setdefault(
        "STAGE5_MCQ_DEBIAS_RUN_ID",
        "stage5_arc_challenge_mcq_debias_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    env["STAGE5_MCQ_DEBIAS_SOURCE_SUMMARY"] = source_summary
    env["STAGE5_MCQ_DEBIAS_ARC_CONFIG"] = "ARC-Challenge"
    env["STAGE5_MCQ_DEBIAS_ARC_LIMIT"] = os.environ.get("STAGE5_MCQ_DEBIAS_ARC_LIMIT", "128")
    env["STAGE5_MCQ_DEBIAS_QUIET_EVAL"] = "1"
    env["STAGE5_MCQ_DEBIAS_RESUME_EXISTING"] = "1"
    env["STAGE5_MCQ_DEBIAS_PUSH"] = "1"
    env["MODEL_NAME"] = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    env["DTYPE"] = os.environ.get("DTYPE", "bfloat16")
    env["ADAPTER_DTYPE"] = os.environ.get("ADAPTER_DTYPE", "float32")
    env["DEVICE"] = os.environ.get("DEVICE", "cuda")

    run([sys.executable, "colab/run_stage5_mcq_debias_diagnostic.py"], env=env)

    latest_summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip()
    print("latest_summary:", latest_summary, flush=True)
    summary_md = ROOT / latest_summary.replace("summary.json", "summary.md")
    if summary_md.exists():
        print(summary_md.read_text(encoding="utf-8"), flush=True)

finally:
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    try:
        runtime.unassign()
    except Exception as exc:
        print("runtime.unassign failed:", repr(exc), flush=True)
