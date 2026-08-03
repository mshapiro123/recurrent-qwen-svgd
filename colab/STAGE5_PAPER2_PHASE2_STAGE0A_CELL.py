"""A100-80GB launcher for the resumable development-only Phase-2 Stage 0A job."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_STAGE0A_VERSION = "paper2_phase2_stage0a_v1"
# Safety marker: minimum_vram_mib=70000
# Safety marker: A100-SXM4-80GB sequential model loads and Drive-resumable shards
# Safety marker: DEV-C only sparse lattice and teacher states no optimizer no training
# Safety marker: 200000 boundary samples layers 16 32 44 one-based post-block
# Safety marker: one logical forward pass per pinned model completed shards never replayed
# Safety marker: automatic local-scratch cache with Drive-backed durable resume shards
# Safety marker: tests/test_paper2_phase2_stage0a.py
# Safety marker: colab/run_stage5_paper2_phase2_stage0a.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH = secret("GH_TOKEN", "GITHUB_TOKEN")
HF = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH and HF, "Missing GH_TOKEN or HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd or ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code:
        print("\nStage 0A launcher tail:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert shutil.which("nvidia-smi"), "Attach an A100-SXM4-80GB and rerun."
run(["nvidia-smi"], Path("/content"))
memory = max(
    int(value.strip())
    for value in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if value.strip()
)
assert memory >= 70000, (
    f"Stage 0A includes a pinned 32B bf16 pass and requires an A100-SXM4-80GB; "
    f"observed {memory} MiB. Do not use an L4."
)
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
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
        "tests/test_paper2_phase2_stage0a.py",
        "tests/test_paper2_phase2_v1d_launcher.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_phase2_stage0a_target",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_stage0a.py"])
print("Phase-2 Stage 0A cache and receipt landed; no training occurred.", flush=True)
