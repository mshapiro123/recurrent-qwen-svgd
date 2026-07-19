"""Colab launcher for the descriptive Paper One wall-clock latency receipt."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_WALL_CLOCK_LATENCY_CELL_VERSION = "wall_clock_latency_v1"
# Safety markers: tests/test_wall_clock_latency.py, wall_clock_latency_descriptive
# Descriptive-only scope: single hardware configuration, batch size 1, registered evaluation paths.

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


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


def redact(text: str) -> str:
    value = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            value = value.replace(token, "****")
    return value


def run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    print("$", redact(" ".join(command)), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        lines.append(safe)
    result = subprocess.CompletedProcess(command, process.wait(), "".join(lines), None)
    if result.returncode:
        print("\n".join(result.stdout.splitlines()[-240:]), flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", SYNC_REF])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
        run(["git", "reset", "--hard", SYNC_REF])
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


try:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach one A100 or larger GPU runtime for the same-session five-arm receipt.")
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_WALL_CLOCK_LATENCY_CELL_VERSION={STAGE5_WALL_CLOCK_LATENCY_CELL_VERSION}", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_wall_clock_latency.py"])
    run([sys.executable, "colab/run_stage5_wall_clock_latency.py"])
    if os.environ.get("STAGE5_WALL_CLOCK_DISCONNECT", "0") == "1":
        runtime.unassign()
except Exception:
    print("Wall-clock latency receipt errored; leaving runtime connected for diagnosis.", flush=True)
    raise
