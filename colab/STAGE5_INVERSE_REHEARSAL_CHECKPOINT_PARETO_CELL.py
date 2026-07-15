"""Colab cell: eval-only Pareto sweep for saved cap-3 rehearsal checkpoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_INVERSE_REHEARSAL_CHECKPOINT_PARETO_CELL_VERSION = "inverse_rehearsal_checkpoint_pareto_v1"
# Safety marker: inverse_rehearsal_checkpoint_pareto
# Safety marker: colab/run_stage5_inverse_rehearsal_checkpoint_pareto.py
# Safety marker: tests/test_stage5_inverse_rehearsal_checkpoint_pareto.py
# Safety marker: candidate_requires_fresh_confirmation

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


def run(cmd: list[str], *, cwd: Path = ROOT, env=None) -> None:
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
    assert process.stdout is not None
    for line in process.stdout:
        print(redact(line), end="", flush=True)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, cmd)


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
    sync_repo(f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git")
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_inverse_rehearsal_checkpoint_pareto.py",
            "tests/test_stage5_inverse_rehearsal_attribution.py",
            "tests/test_stage5_inverse_table_rehearsal.py",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_REHEARSAL_PARETO_STEPS", "100,200,300,334")
    env.setdefault("STAGE5_STAIRCASE_DTYPE", "bfloat16")
    env.setdefault("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_inverse_rehearsal_checkpoint_pareto.py"], env=env)
    if os.environ.get("STAGE5_REHEARSAL_PARETO_DISCONNECT", "1").lower() in {"1", "true", "yes", "y"}:
        runtime.unassign()
    else:
        print("Leaving runtime connected for checkpoint Pareto review.", flush=True)


try:
    main()
except Exception:
    print("Checkpoint Pareto sweep errored; leaving runtime connected.", flush=True)
    raise
