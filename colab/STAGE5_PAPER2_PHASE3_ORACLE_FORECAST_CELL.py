"""Colab launcher for the real P3.2 agreement-oracle and forecast cache."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE3_ORACLE_FORECAST_VERSION = "paper2_phase3_oracle_forecast_v1"
# Safety marker: actual cached 0.5B LM-head gradient path with batched one-row equivalence
# Safety marker: strict 14B/32B concurrent flips retained without selecting teachability threshold
# Safety marker: both migrated seed lineages and loops one through four feature-cached
# Safety marker: document-disjoint linear forecast neither upper nor lower bound
# Safety marker: GPU cache generation only no optimizer no training no CONFIRM scoring
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
GH = os.environ.get("GH_TOKEN") or userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=600)
    effective_cwd = cwd or (ROOT if ROOT.is_dir() else Path("/content"))
    process = subprocess.Popen(
        command,
        cwd=effective_cwd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode:
        print("phase3_oracle_forecast_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("phase3_oracle_forecast_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"])
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(
    [
        "git",
        "fetch",
        "origin",
        f"refs/heads/{SOURCE_BRANCH}:refs/remotes/origin/{SOURCE_BRANCH}",
    ]
)
run(["git", "reset", "--hard", REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase3_agreement_oracle.py",
        "tests/test_paper2_phase3_linear_forecast.py",
        "tests/test_paper2_phase3_p32.py",
        "tests/test_stage5_notebooks.py::test_phase3_oracle_forecast_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_oracle_forecast.py"])
print("P3.2 agreement oracle and forecast landed; P3.3 remains unauthorized.", flush=True)
