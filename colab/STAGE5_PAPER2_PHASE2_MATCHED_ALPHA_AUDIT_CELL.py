"""Colab L4 launcher for the read-only Phase-2 matched-alpha terminal audit."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch
from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_MATCHED_ALPHA_AUDIT_VERSION = (
    "paper2_phase2_matched_alpha_audit_v1"
)
# Safety marker: read-only exact abort checkpoint audit no optimizer no parameter updates
# Safety marker: DEV-only document-isolated rows no frozen E1 evaluation partition
# Safety marker: tripwires remain hard shapers observe until empirically grounded
# Safety marker: endpoint qualification is not relabeled as a catastrophe tripwire
# Safety marker: per-step trust magnitudes marked unrecoverable rather than reconstructed
# Safety marker: colab/run_stage5_paper2_phase2_matched_alpha_audit.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH, "Missing GH_TOKEN in Colab secrets."
assert HF, "Missing HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    tail: deque[str] = deque(maxlen=400)
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
        print("matched_alpha_audit_launcher_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("matched_alpha_audit_launcher_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
run(["nvidia-smi"], Path("/content"))
assert torch.cuda.is_available(), "Matched-alpha read-only audit requires CUDA."
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / 2**30
print(f"gpu_preflight name={properties.name} vram_gib={vram_gib:.1f}", flush=True)
assert properties.major >= 7 and vram_gib >= 20, (
    "Matched-alpha read-only audit requires an L4-class or larger GPU with at least "
    f"20 GiB visible VRAM; observed {properties.name} with {vram_gib:.1f} GiB."
)

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
        "tests/test_paper2_phase2_matched_alpha_audit.py",
        "tests/test_paper2_dc2_student.py",
        "tests/test_stage5_notebooks.py::test_phase2_matched_alpha_audit_target_is_wired_and_guarded",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_matched_alpha_audit.py"])
print("Phase-2 matched-alpha read-only audit landed; no model training ran.", flush=True)
