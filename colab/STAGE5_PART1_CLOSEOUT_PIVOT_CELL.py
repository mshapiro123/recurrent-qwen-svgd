"""Colab cell for the Part 1 closeout micro-test and branching screen."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PART1_CLOSEOUT_PIVOT_CELL_VERSION = "part1_closeout_pivot_v1"
# Safety markers: training/continuation_policy.py disposable_measurement
# Safety markers: training/loop_position_transfer_task.py
# Safety markers: training/branching_relations_task.py
# Safety markers: eval/eval_branching_relations.py

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get("STAGE5_PART1_PIVOT_DISCONNECT", "0").strip().lower() in {
    "1", "true", "yes", "y"
}


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
assert GH_TOKEN, "Missing GH_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def sync_repo() -> None:
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


try:
    run(["nvidia-smi"])
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_continuation_policy.py",
            "tests/test_loop_position_transfer_task.py",
            "tests/test_branching_relations_task.py",
            "tests/test_eval_branching_relations.py",
            "tests/test_phase_g_alpha_spec.py",
        ],
        cwd=ROOT,
    )
    run([sys.executable, "colab/run_stage5_part1_closeout_pivot.py"], cwd=ROOT, env=os.environ.copy())
    print("Part 1 closeout pivot session finished.", flush=True)
    if DISCONNECT:
        runtime.unassign()
except Exception:
    print("Part 1 closeout pivot session errored; leaving runtime connected for diagnosis.", flush=True)
    raise
