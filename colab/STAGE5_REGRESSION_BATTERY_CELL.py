"""Colab cell: run the loop-1 AI2 ARC regression battery.

This is a measurement-only GPU action to protect general capability before
narrow synthetic/depth training. It evaluates the selected recurrent checkpoint
against base Qwen on frozen AI2 ARC item sets at forced loop 1.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_REGRESSION_BATTERY_CELL_VERSION = "regression_battery_ai2_arc_v1"
# Safety marker: STAGE5_REGRESSION_SOURCE_SUMMARIES
# Safety marker: STAGE5_REGRESSION_ARC_SPLIT
# Safety marker: STAGE5_REGRESSION_MARGIN
# Safety marker: STAGE5_BENCHMARK_ARC_EASY_SPLIT
# Safety marker: STAGE5_BENCHMARK_ARC_CHALLENGE_SPLIT
# Safety marker: forced loop 1
# Safety marker: AI2 ARC, not ARC-AGI
# Safety marker: drive.mount
# Safety marker: eval/assess_regression_battery.py
# Safety marker: colab/run_stage5_regression_battery.py
# Safety marker: tests/test_regression_battery.py
# Safety marker: tests/test_stage5_benchmark_suite.py::test_benchmark_specs_supports_all_arc_splits
# Safety marker: tier1_canary_status
# Safety marker: hellaswag_winogrande_lambada_status

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
else:
    print("HF token missing; model downloads will use anonymous Hub access.", flush=True)


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(cmd: list[str | os.PathLike[str]], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    rc = proc.wait()
    if rc:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join("".join(chunks).splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(rc, cmd)


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def require_gpu_runtime() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach a GPU runtime first. L4/T4 is sufficient for the AI2 ARC regression battery.")
    run(["nvidia-smi"], cwd=Path("/content"))


def mount_drive_first() -> None:
    """Mount Drive in the notebook process before subprocess restore attempts.

    ``google.colab.drive.mount`` can fail from a plain Python subprocess because
    the notebook kernel object is unavailable there. Mounting here leaves
    ``/content/drive`` visible to the benchmark runner's checkpoint restore
    code.
    """

    if env_flag("STAGE5_REGRESSION_SKIP_DRIVE_MOUNT", "0"):
        print("Skipping upfront Drive mount by request.", flush=True)
        return
    force = env_flag("FORCE_DRIVE_REMOUNT", "0") or env_flag("STAGE5_REGRESSION_FORCE_DRIVE_REMOUNT", "0")
    drive.mount("/content/drive", force_remount=force)


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
    run(["git", "log", "--oneline", "-5"], cwd=ROOT)


def main() -> None:
    require_gpu_runtime()
    mount_drive_first()
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_regression_battery.py",
            "tests/test_stage5_benchmark_suite.py::test_benchmark_specs_supports_all_arc_splits",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_target_markers_exist_in_launcher_files",
        ]
    )
    env = os.environ.copy()
    env.setdefault(
        "STAGE5_REGRESSION_CURRENT_SOURCE_SUMMARY",
        "outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/summary.json",
    )
    env.setdefault("STAGE5_REGRESSION_ARC_SPLIT", "all")
    env.setdefault("STAGE5_REGRESSION_ARC_EASY_LIMIT", "all")
    env.setdefault("STAGE5_REGRESSION_ARC_CHALLENGE_LIMIT", "all")
    env.setdefault("STAGE5_REGRESSION_MARGIN", "0.03")
    env.setdefault("STAGE5_REGRESSION_YELLOW_MARGIN", "0.015")
    env.setdefault("STAGE5_REGRESSION_PUSH", "1")
    run([sys.executable, "colab/run_stage5_regression_battery.py"], env=env)
    if env_flag("STAGE5_REGRESSION_DISCONNECT", "0"):
        print("Disconnecting Colab runtime after regression battery.", flush=True)
        runtime.unassign()
    else:
        print("Leaving Colab runtime connected after regression battery.", flush=True)


try:
    main()
except Exception:
    print("Regression battery errored; leaving runtime connected.", flush=True)
    raise
