"""Colab launcher for the read-only Option B document-bootstrap audit."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_OPTION_B_BOOTSTRAP_AUDIT_VERSION = (
    "paper2_phase2_option_b_bootstrap_audit_v1"
)
# Safety marker: CPU-only saved-row post-processing no model load no optimizer no training
# Safety marker: paired document cluster bootstrap 10000 replicates seed 20260808
# Safety marker: governing CI-qualified E1 reading preserved separately from source verdict
# Safety marker: fixed evaluation rows only and zero confirmatory partition contact
# Safety marker: source summary and every consumed row receipt hashed into output
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=600)
    process = subprocess.Popen(
        command,
        cwd=cwd or ROOT,
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
        print("option_b_bootstrap_audit_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("option_b_bootstrap_audit_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
memory = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
print("cpu_memory_preflight", " ".join(memory[:2]), flush=True)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
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
        "tests/test_paper2_phase2_option_b_bootstrap_audit.py",
        "tests/test_stage5_notebooks.py::test_phase2_option_b_bootstrap_audit_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_option_b_bootstrap_audit.py"])
print("Option B document-bootstrap audit landed; no model or optimizer was loaded.", flush=True)
