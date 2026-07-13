"""Colab cell: read-only Phase G curriculum construction autopsy."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PHASE_G_CURRICULUM_AUTOPSY_CELL_VERSION = "phase_g_curriculum_autopsy_v1"
# Safety marker: phase_g_curriculum_autopsy
# Safety marker: eval/eval_abductive_curriculum_autopsy.py
# Safety marker: uniform_expected_coverage
# Safety marker: inverse_table_control_prepared_not_trained
# Safety marker: next_training_disabled_pending_strategy_review

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = os.environ.get(name) or userdata.get(name)
        except Exception:
            value = os.environ.get(name)
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


def redact(text: str) -> str:
    safe = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            safe = safe.replace(token, "****")
    return safe


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("$", redact(" ".join(cmd)), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        tail = (tail + [safe])[-200:]
    if process.wait():
        print("FAILED_COMMAND_TAIL_START\n" + "".join(tail) + "FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(process.returncode, cmd)


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


def main() -> None:
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_eval_abductive_curriculum_autopsy.py",
            "tests/test_eval_abductive_coverage.py",
            "tests/test_abductive_injective_task.py",
            "tests/test_stage5_phase_g_curriculum_autopsy.py",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_PHASE_G_AUTOPSY_RUN_ID", "stage5_phase_g_curriculum_autopsy_20260712")
    env.setdefault("STAGE5_PHASE_G_AUTOPSY_ROWS_PER_DEPTH", "16")
    env.setdefault("STAGE5_PHASE_G_AUTOPSY_STATE_QUERY_EXAMPLES", "8")
    env.setdefault("STAGE5_PHASE_G_AUTOPSY_DTYPE", "bfloat16")
    env.setdefault("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_phase_g_curriculum_autopsy.py"], env=env)
    if os.environ.get("STAGE5_PHASE_G_AUTOPSY_DISCONNECT", "0").lower() in {"1", "true", "yes", "y"}:
        runtime.unassign()
    else:
        print("Leaving runtime connected for strategy review; no training was launched.", flush=True)


try:
    main()
except Exception:
    print("Phase G curriculum autopsy errored; leaving runtime connected.", flush=True)
    raise
