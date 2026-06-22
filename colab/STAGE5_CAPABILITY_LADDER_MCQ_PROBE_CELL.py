"""Colab cell: run a bounded capability-ladder MCQ scoring probe.

This no-training GPU action scores a small ARC-Train MCQ slice with Qwen
0.5B, 1.5B, and 3B, then builds a depth-targeted capability-ladder curriculum
candidate. It is designed to test the proposed "depth by model-scale gap"
signal before spending A100 time on recurrent SFT.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL_VERSION = "capability_ladder_mcq_probe_v1"
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
        print("\n".join(output.splitlines()[-160:]), flush=True)
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
    assert gpu_check, "Attach an A100/H100/L4/T4 GPU runtime before running this scoring probe."
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
            "tests/test_stage5_capability_ladder_mcq_probe.py",
            "tests/test_merge_capability_score_rows.py",
            "tests/test_capability_ladder_curriculum.py",
            "tests/test_curriculum_sft_gate.py",
        ]
    )

    env = os.environ.copy()
    env.setdefault(
        "STAGE5_CAPABILITY_LADDER_RUN_ID",
        "stage5_capability_ladder_mcq_probe_" + time.strftime("%Y%m%d_%H%M%S"),
    )
    env["STAGE5_CAPABILITY_LADDER_ARC_LIMIT"] = os.environ.get(
        "STAGE5_CAPABILITY_LADDER_ARC_LIMIT",
        "48",
    )
    env["STAGE5_CAPABILITY_LADDER_ARC_SPLIT"] = os.environ.get(
        "STAGE5_CAPABILITY_LADDER_ARC_SPLIT",
        "train",
    )
    env["STAGE5_CAPABILITY_LADDER_SCORE_MODE"] = os.environ.get(
        "STAGE5_CAPABILITY_LADDER_SCORE_MODE",
        "content_question_only",
    )
    env["STAGE5_CAPABILITY_LADDER_MODELS"] = os.environ.get(
        "STAGE5_CAPABILITY_LADDER_MODELS",
        ",".join(
            [
                "qwen_0_5b=Qwen/Qwen2.5-0.5B-Instruct",
                "qwen_1_5b=Qwen/Qwen2.5-1.5B-Instruct",
                "qwen_3b=Qwen/Qwen2.5-3B-Instruct",
            ]
        ),
    )
    env["STAGE5_CAPABILITY_LADDER_PUSH"] = "1"
    env["STAGE5_CAPABILITY_LADDER_BACKUP_DRIVE"] = "1"
    env["DTYPE"] = os.environ.get("DTYPE", "bfloat16")
    env["DEVICE"] = os.environ.get("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_capability_ladder_mcq_probe.py"], env=env)

    summary = (ROOT / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip()
    print("capability_ladder_probe_summary:", summary, flush=True)
    summary_md = ROOT / summary.replace("summary.json", "summary.md")
    if summary_md.exists():
        print(summary_md.read_text(encoding="utf-8"), flush=True)

finally:
    print("Disconnecting Colab runtime to conserve credits.", flush=True)
    try:
        runtime.unassign()
    except Exception as exc:
        print("runtime.unassign failed:", repr(exc), flush=True)
