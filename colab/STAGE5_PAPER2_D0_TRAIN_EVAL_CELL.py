"""Colab launcher for registered Paper Two D0 training and final evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_D0_TRAIN_EVAL_VERSION = "paper2_d0_train_eval_v3_verbose_preflight"
# Safety marker: locked 4000-step 70/30 D0 pilot
# Safety marker: final-step EMA primary
# Safety marker: blocked outcomes exit 2 with tables written
# Safety marker: prelaunch receipts required before optimizer construction
# Safety marker: frozen q1-q4 binned target table
# Safety marker: deterministic fp32 argmax lowest token id ties counted
# Safety marker: teacher shift uses each teachers own rejection population
# Safety marker: Drive mount retry with explicit Pharma Initiatives account authorization
# Safety marker: preflight-only pass before model or optimizer construction
# Safety marker: evaluation partition restored only after training
# Safety marker: minimum_vram_mib=35000
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN and HF_TOKEN, "D0 train/eval requires GH_TOKEN and HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def run(command: list[str], *, cwd: Path | None = None, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command).replace(GH_TOKEN, "****"), flush=True)
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
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code not in allowed:
        raise subprocess.CalledProcessError(code, command)
    return code


def ensure_drive() -> None:
    my_drive = Path("/content/drive/MyDrive")
    if my_drive.is_dir():
        print("Drive already mounted at /content/drive", flush=True)
        return
    failures: list[str] = []
    for attempt, force_remount in enumerate((False, True), start=1):
        try:
            print(
                f"Drive mount attempt {attempt}/2 force_remount={int(force_remount)}. "
                "Authorize the Pharma Initiatives Google account when prompted.",
                flush=True,
            )
            drive.mount(
                "/content/drive",
                force_remount=force_remount,
                timeout_ms=240_000,
            )
        except (ValueError, TimeoutError) as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        if my_drive.is_dir():
            return
        time.sleep(3)
    raise RuntimeError(
        "Google Drive could not be mounted after two attempts. Authorize the Pharma Initiatives "
        f"account in Colab and rerun. Failures: {failures}"
    )


ensure_drive()
gpu_memory = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
)
available_mib = max(int(value.strip()) for value in gpu_memory.splitlines() if value.strip())
assert available_mib >= 35000, (
    f"Registered D0 train/eval is assigned to A100/H100; observed {available_mib} MiB. "
    "Use the L4 only for floor calibration."
)
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
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_speculative_depth_d0_postlock.py",
        "tests/test_speculative_depth_d0_launch_amendment.py",
        "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_paper2_d0_train_eval_target",
    ]
)
preflight_env = os.environ.copy()
preflight_env["STAGE5_PAPER2_D0_PREFLIGHT_ONLY"] = "1"
print("Running D0 preflight-only pass before model or optimizer construction.", flush=True)
process = subprocess.Popen(
    [sys.executable, "-u", "colab/run_stage5_paper2_d0_train_eval.py"],
    cwd=ROOT,
    env=preflight_env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
)
assert process.stdout is not None
for line in process.stdout:
    print(line, end="", flush=True)
preflight_code = process.wait()
if preflight_code:
    raise subprocess.CalledProcessError(
        preflight_code,
        [sys.executable, "-u", "colab/run_stage5_paper2_d0_train_eval.py"],
    )
code = run([sys.executable, "-u", "colab/run_stage5_paper2_d0_train_eval.py"], allowed=(0, 2))
print(f"D0 train/eval finished with registered exit code {code}.", flush=True)
