"""Colab cell: run natural-surface receipts and robustness diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_NATURAL_SURFACE_RECEIPTS_CELL_VERSION = "natural_surface_receipts_v1"
# Safety marker: natural_surface_receipts
# Safety marker: pointer_holdout
# Safety marker: untouched_relay_d13_16
# Safety marker: untouched_pointer_d13_16
# Safety marker: robust_baton_default_d1_12
# Safety marker: robust_relay_unseen_names_d1_12
# Safety marker: robust_pointer_unseen_names_d1_12
# Safety marker: paired_relay_pointer_mcnemar
# Safety marker: eval/eval_synthetic_depth_final_symbol.py
# Safety marker: colab/run_stage5_natural_surface_receipts.py
# Safety marker: tests/test_natural_surface_transfer.py
# Safety marker: tests/test_eval_synthetic_depth_active_labels.py

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


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
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
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


def run(cmd: list[str | os.PathLike[str]], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
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
    rc = process.wait()
    if rc:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join("".join(chunks).splitlines()[-160:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(rc, cmd)


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


def main() -> None:
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_natural_surface_transfer.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_natural_surface_transfer.py",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_NATURAL_RECEIPTS_RUN_EVALS", "1")
    env.setdefault("STAGE5_NATURAL_RECEIPTS_RUN_FULL_SYNTHETIC", "1")
    env.setdefault("STAGE5_NATURAL_RECEIPTS_RUN_SAME_READER", "1")
    env.setdefault("STAGE5_NATURAL_RECEIPTS_DTYPE", "bfloat16")
    env.setdefault("STAGE5_NATURAL_RECEIPTS_CHECKPOINTS", "frozen_n24,step_2000,step_4000,step_6000")
    run([sys.executable, "colab/run_stage5_natural_surface_receipts.py"], env=env)
    if os.environ.get("STAGE5_NATURAL_RECEIPTS_DISCONNECT", "0").strip().lower() in {"1", "true", "yes", "y"}:
        print("Disconnecting Colab runtime after natural-surface receipts.", flush=True)
        runtime.unassign()
    else:
        print("Leaving Colab runtime connected after natural-surface receipts.", flush=True)


try:
    main()
except Exception:
    print("Natural-surface receipts errored; leaving runtime connected.", flush=True)
    raise
