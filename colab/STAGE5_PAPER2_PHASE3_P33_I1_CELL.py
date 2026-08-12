"""Colab launcher for one registered P3.3 i1 aim-focused seed."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive


STAGE5_PAPER2_PHASE3_P33_I1_VERSION = "paper2_phase3_p33_i1_v1"
# Safety marker: e2 token-retention estimator only and no task capability scoring
# Safety marker: exact 1000 updates and 20 looks one every 50 updates
# Safety marker: A1 A2 A3 A4 A5 verified before optimizer construction
# Safety marker: gate and upstream selector frozen aim postclip share at least 70 percent
# Safety marker: canonical BF16 pi_dir pi_dep gate recall precision chi and Tier-1 observatory
# Safety marker: resumable one A100 session per seed and end-of-run A_state battery
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
SOURCE_BRANCH = "codex/phase3-opening-build"
SEED = int(os.environ.get("PAPER2_P33_I1_SEED", "0"))
GH = os.environ.get("GH_TOKEN", "").strip()
assert SEED in (0, 1), "PAPER2_P33_I1_SEED must be 0 or 1."


def run(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(command).replace(GH, "****") if GH else " ".join(command)
    print("$", printable, flush=True)
    tail: deque[str] = deque(maxlen=500)
    process = subprocess.Popen(
        command, cwd=cwd or ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    if process.wait():
        print("p33_failure_tail_begin\n" + "\n".join(tail) + "\np33_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(process.returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"])
url = (
    f"https://x-access-token:{GH}@github.com/{REPO}.git"
    if GH
    else f"https://github.com/{REPO}.git"
)
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "fetch", "origin", f"refs/heads/{SOURCE_BRANCH}:refs/remotes/origin/{SOURCE_BRANCH}"])
run(["git", "reset", "--hard", REF])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_paper2_phase3_p33_i1.py",
    "tests/test_paper2_phase3_p33_i1_runner.py",
    "tests/test_stage5_notebooks.py::test_phase3_p33_i1_target_is_wired_and_guarded",
])
run([sys.executable, "-u", "colab/run_stage5_paper2_phase3_p33_i1.py", "--seed", str(SEED)])
print(f"P3.3 i1 seed {SEED} landed; release this A100 session after artifact verification.", flush=True)
