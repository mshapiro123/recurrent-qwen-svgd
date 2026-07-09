"""Colab cell: run natural-surface transfer Experiments 0 and 1 on GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_NATURAL_SURFACE_TRANSFER_CELL_VERSION = "natural_surface_transfer_rung0_v1"
# Safety marker: natural_surface_transfer_rung0
# Safety marker: frozen_natural_surface_baseline
# Safety marker: verbal_rung_zero
# Safety marker: Experiment 0
# Safety marker: Experiment 1
# Safety marker: relay_test_chain_mcq
# Safety marker: pointer_test_chain_mcq
# Safety marker: synthetic_rehearsal_chain_symbol_sft
# Safety marker: rung0_train_mix_chain_symbol_sft
# Safety marker: value_prefix=name:
# Safety marker: value_prefix=letter:
# Safety marker: stage5_natural_surface_transfer_20260708_230229
# Safety marker: stage5_n24_support12_rung_20260707_140139
# Safety marker: STAGE5_NATURAL_TRANSFER_RUN_TRAIN
# Safety marker: STAGE5_NATURAL_TRANSFER_INIT_SOURCE_SUMMARY
# Safety marker: STAGE5_NATURAL_TRANSFER_DATA_SUMMARY
# Safety marker: STAGE5_NATURAL_TRANSFER_TRAIN_STEPS
# Safety marker: STAGE5_NATURAL_TRANSFER_REUSE_EXISTING
# Safety marker: STAGE5_NATURAL_TRANSFER_KEEP_FULL_ACTIVE_ROWS
# Safety marker: eval/eval_synthetic_depth_active_labels.py
# Safety marker: training/train_unfrozen_recurrent.py
# Safety marker: colab/run_stage5_natural_surface_transfer.py
# Safety marker: tests/test_stage5_natural_surface_transfer.py

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


def require_gpu_runtime() -> None:
    run(["nvidia-smi"], cwd=Path("/content"))


def main() -> None:
    require_gpu_runtime()
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_natural_surface_transfer.py",
            "tests/test_natural_surface_transfer.py",
            "tests/test_eval_synthetic_depth_active_labels.py::test_name_value_prefix_maps_full_symbol_space",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_natural_surface_transfer_target",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_NATURAL_TRANSFER_DATA_SUMMARY", "outputs/stage5/stage5_natural_surface_transfer_20260708_230229/summary.json")
    env.setdefault("STAGE5_NATURAL_TRANSFER_INIT_SOURCE_SUMMARY", "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json")
    env.setdefault("STAGE5_NATURAL_TRANSFER_RUN_TRAIN", "1")
    env.setdefault("STAGE5_NATURAL_TRANSFER_TRAIN_STEPS", "8000")
    env.setdefault("STAGE5_NATURAL_TRANSFER_EVAL_MAX_DEPTH", "12")
    env.setdefault("STAGE5_NATURAL_TRANSFER_TRAIN_MAX_DEPTH", "8")
    env.setdefault("STAGE5_NATURAL_TRANSFER_DTYPE", "bfloat16")
    env.setdefault("STAGE5_NATURAL_TRANSFER_KEEP_FULL_ACTIVE_ROWS", "0")
    env.setdefault("STAGE5_NATURAL_TRANSFER_REUSE_EXISTING", "1")
    env.setdefault("STAGE5_NATURAL_TRANSFER_BACKUP_CHECKPOINTS_TO_DRIVE", "1")
    run([sys.executable, "colab/run_stage5_natural_surface_transfer.py"], env=env)
    if os.environ.get("STAGE5_NATURAL_TRANSFER_DISCONNECT", "0").strip().lower() in {"1", "true", "yes", "y"}:
        print("Disconnecting Colab runtime after natural-surface transfer.", flush=True)
        runtime.unassign()
    else:
        print("Leaving Colab runtime connected after natural-surface transfer.", flush=True)


try:
    main()
except Exception:
    print("Natural-surface transfer errored; leaving runtime connected.", flush=True)
    raise
