"""Colab launcher for Arm E adapter parity E3a, E2, and gated E4."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION = "adapter_parity_battery_v1"
# Safety marker: bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839
# Safety marker: stage5_adapter_parity_e3a
# Safety marker: stage5_adapter_parity_e2
# Safety marker: stage5_adapter_parity_e4
# Safety marker: tests/test_adapter_parity_battery.py

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
TARGET = os.environ["STAGE5_CURRENT_A100_TARGET"]
RUNNERS = {
    "adapter_parity_e3a": "colab/run_stage5_adapter_parity_e3a.py",
    "adapter_parity_e2": "colab/run_stage5_adapter_parity_e2.py",
    "adapter_parity_e4": "colab/run_stage5_adapter_parity_e4.py",
}
assert TARGET in RUNNERS, f"Unsupported adapter parity target: {TARGET}"


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


def run(
    command: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
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
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        lines.append(safe)
    result = subprocess.CompletedProcess(command, process.wait(), "".join(lines), None)
    if result.returncode not in accepted_returncodes:
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
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
        run(["git", "reset", "--hard", SYNC_REF])
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"], accepted_returncodes={0})


try:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach an L4 or larger GPU runtime.")
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    print(
        f"STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION={STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION}",
        flush=True,
    )
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_adapter_parity_battery.py",
            "tests/test_adapter_budget_arm.py",
            "tests/test_train_unfrozen_recurrent.py",
            "tests/test_stage5_inverse_table_rehearsal.py",
        ]
    )
    run([sys.executable, RUNNERS[TARGET]], accepted_returncodes={0, 2})
    if os.environ.get("STAGE5_ADAPTER_PARITY_DISCONNECT", "0") == "1":
        runtime.unassign()
except Exception:
    print(f"Arm E {TARGET} errored; leaving runtime connected.", flush=True)
    raise
