"""Colab cell: publish the durable Phase-A surpass receipt and figure."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_PHASE_A_SURPASS_RECEIPT_CELL_VERSION = "phase_a_surpass_receipt_v1"
# Safety marker: phase_a_surpass_receipt
# Safety marker: colab/run_stage5_phase_a_surpass_receipt.py
# Safety marker: tests/test_stage5_phase_a_surpass_receipt.py
# Safety marker: exact_paired_sign_mcnemar
# Safety marker: accepted_returncodes={0, 2}

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
PINNED_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


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
assert GH_TOKEN, "Missing GH_TOKEN in Colab secrets."


def redact(text: str) -> str:
    return str(text).replace(GH_TOKEN, "****")


def run(cmd: list[str], *, cwd: Path = ROOT, env=None, accepted_returncodes={0}) -> int:
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
        tail = (tail + [safe])[-240:]
    returncode = process.wait()
    if returncode not in accepted_returncodes:
        print("FAILED_COMMAND_TAIL_START\n" + "".join(tail) + "FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)
    return returncode


def sync_repo(clone_url: str) -> None:
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
    is_sha = len(PINNED_REF) == 40 and all(char in "0123456789abcdefABCDEF" for char in PINNED_REF)
    resolved_target = PINNED_REF if is_sha else "origin/main"
    run(["git", "reset", "--hard", resolved_target])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected = subprocess.check_output(["git", "rev-parse", resolved_target], cwd=ROOT, text=True).strip()
    assert head == expected, f"Pinned checkout mismatch: HEAD={head}, expected={expected}"
    print(f"Pinned checkout verified: {head}", flush=True)


def main() -> None:
    drive.mount("/content/drive", force_remount=False)
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    sync_repo(clone_url)
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_stage5_phase_a_surpass_receipt.py"])
    env = os.environ.copy()
    env.setdefault("STAGE5_PHASE_A_RECEIPT_RUN_ID", "stage5_phase_a_surpass_receipt_20260714")
    code = run(
        [sys.executable, "colab/run_stage5_phase_a_surpass_receipt.py"],
        env=env,
        accepted_returncodes={0, 2},
    )
    if code == 2:
        print("Phase-A analysis landed; one or more checkpoint-hash receipts remain pending.", flush=True)
    if os.environ.get("STAGE5_PHASE_A_RECEIPT_DISCONNECT", "0").lower() in {"1", "true", "yes", "y"}:
        runtime.unassign()
    else:
        print("Leaving runtime connected for Phase-A receipt review.", flush=True)


try:
    main()
except Exception:
    print("Phase-A surpass receipt errored; leaving runtime connected.", flush=True)
    raise
