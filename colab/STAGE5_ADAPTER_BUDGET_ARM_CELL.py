"""Colab launcher for the matched R16 adapter-budget Arm E."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_ADAPTER_BUDGET_ARM_CELL_VERSION = "adapter_budget_arm_e_v1"
# Safety marker: fresh_base_qwen_surgery
# Safety marker: same_reader_final_rows.jsonl
# Safety marker: pretrained_base_hash_unchanged
# Safety marker: track_loop_dose
# Safety marker: adapter_budget_depth_profile
# Safety marker: tests/test_adapter_budget_arm.py

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
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(
    command: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    check: bool = True,
    accepted_returncodes: set[int] = {0},
) -> subprocess.CompletedProcess[str]:
    print("$", redact(" ".join(map(str, command))), flush=True)
    process = subprocess.Popen(
        list(map(str, command)),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    result = subprocess.CompletedProcess(command, process.wait(), "".join(chunks), None)
    if check and result.returncode not in accepted_returncodes:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(result.stdout.splitlines()[-240:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def require_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach an L4 or larger GPU runtime before launching Arm E.")
    run(["nvidia-smi"], cwd=Path("/content"), check=False)


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", SYNC_REF])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
        run(["git", "reset", "--hard", SYNC_REF])
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    print(f"pinned_checkout={SYNC_REF}", flush=True)
    run(["git", "log", "--oneline", "-5"], check=False)


try:
    require_gpu()
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    print(
        f"STAGE5_ADAPTER_BUDGET_ARM_CELL_VERSION={STAGE5_ADAPTER_BUDGET_ARM_CELL_VERSION}",
        flush=True,
    )
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_adapter_budget_arm.py",
            "tests/test_stage5_adapter_budget_arm.py",
            "tests/test_train_unfrozen_recurrent.py",
            "tests/test_bridge.py",
            "tests/test_lora.py",
            "tests/test_eval_synthetic_depth_final_symbol.py",
        ]
    )
    run(
        [sys.executable, "colab/run_stage5_adapter_budget_arm.py"],
        accepted_returncodes={0, 2},
    )
    if os.environ.get("STAGE5_ADAPTER_BUDGET_DISCONNECT", "0") == "1":
        runtime.unassign()
except Exception:
    print("Adapter-budget Arm E errored; leaving runtime connected.", flush=True)
    raise
