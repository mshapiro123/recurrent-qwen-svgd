"""Colab cell: run natural-surface plan items 2-4 in sequence."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_NATURAL_SURFACE_FOLLOWUPS_CELL_VERSION = "natural_surface_followups_2_4_v1"
# Safety marker: natural_surface_followups_2_4
# Safety marker: CORRECTED_HELDOUT_SINGLE_TOKEN_NAMES
# Safety marker: robust_relay_fronted_d1_12
# Safety marker: eval/eval_synthetic_depth_probe.py
# Safety marker: colab/run_stage5_natural_surface_followups.py
# Safety marker: colab/run_stage5_natural_surface_replication_dose.py
# Safety marker: 1000,1500,2000,2500,3000,4000,6000
# Safety marker: untouched_depth_13_16_opened

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
    if process.wait():
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join("".join(chunks).splitlines()[-200:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(process.returncode, cmd)


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
            "tests/test_stage5_natural_surface_transfer.py",
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_natural_surface_followups.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_natural_surface_replication_dose_target",
        ]
    )

    env = os.environ.copy()
    env.setdefault("STAGE5_NATURAL_FOLLOWUP_RUN_ID", "stage5_natural_surface_followups_2_3_20260710")
    env.setdefault("STAGE5_NATURAL_FOLLOWUP_CHECKPOINTS", "frozen_n24,step_2000,step_4000,step_6000")
    env.setdefault("STAGE5_NATURAL_FOLLOWUP_DTYPE", "bfloat16")
    env.setdefault("STAGE5_NATURAL_FOLLOWUP_PROBE_PERMUTATIONS", "20")
    run([sys.executable, "colab/run_stage5_natural_surface_followups.py"], env=env)

    env.setdefault("STAGE5_NATURAL_REPLICATION_RUN_ID", "stage5_natural_surface_replication_dose_seed931337_20260710")
    env.setdefault("STAGE5_NATURAL_REPLICATION_SEED", "931337")
    env.setdefault("STAGE5_NATURAL_REPLICATION_SAVE_STEPS", "1000,1500,2000,2500,3000,4000,6000")
    # The registered comparison ends at step 6000. Stopping there avoids two
    # thousand unscored steps while preserving every locked decision point.
    env.setdefault("STAGE5_NATURAL_REPLICATION_TRAIN_STEPS", "6000")
    env.setdefault("STAGE5_NATURAL_REPLICATION_DTYPE", "bfloat16")
    env.setdefault("STAGE5_NATURAL_REPLICATION_BACKUP_CHECKPOINTS_TO_DRIVE", "1")
    run([sys.executable, "colab/run_stage5_natural_surface_replication_dose.py"], env=env)

    if os.environ.get("STAGE5_NATURAL_FOLLOWUP_DISCONNECT", "0").strip().lower() in {"1", "true", "yes", "y"}:
        runtime.unassign()
    else:
        print("Natural-surface items 2-4 finished; leaving runtime connected.", flush=True)


try:
    main()
except Exception:
    print("Natural-surface items 2-4 errored; leaving runtime connected for inspection.", flush=True)
    raise
