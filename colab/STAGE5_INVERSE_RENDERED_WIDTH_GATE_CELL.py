"""Colab cell: run the inverse-rendered deterministic gate before G-alpha."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_INVERSE_RENDERED_WIDTH_GATE_CELL_VERSION = "inverse_rendered_width_gate_v1"
# Safety marker: inverse_rendered_width_gate
# Safety marker: colab/run_stage5_inverse_rendered_width_gate.py
# Safety marker: tests/test_stage5_inverse_rendered_width_gate.py
# Safety marker: data/phase_g_alpha_inverse_rendered
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
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)


def redact(text: str) -> str:
    safe = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            safe = safe.replace(token, "****")
    return safe


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
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    sync_repo(clone_url)
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_abductive_injective_task.py",
            "tests/test_eval_abductive_coverage.py",
            "tests/test_phase_g_coverage.py",
            "tests/test_stage5_inverse_rendered_width_gate.py",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_INVERSE_RENDERED_RUN_ID", "stage5_inverse_rendered_width_gate_20260714")
    env.setdefault("STAGE5_INVERSE_RENDERED_DTYPE", "bfloat16")
    env.setdefault("DEVICE", "cuda")
    code = run(
        [sys.executable, "colab/run_stage5_inverse_rendered_width_gate.py"],
        env=env,
        accepted_returncodes={0, 2},
    )
    if code == 2:
        print("Inverse-rendered gate reached its preregistered scientific stop; G-alpha remains closed.", flush=True)
    if os.environ.get("STAGE5_INVERSE_RENDERED_DISCONNECT", "0").lower() in {"1", "true", "yes", "y"}:
        runtime.unassign()
    else:
        print("Leaving runtime connected for inverse-rendered gate review.", flush=True)


try:
    main()
except Exception:
    print("Inverse-rendered width gate errored; leaving runtime connected.", flush=True)
    raise
