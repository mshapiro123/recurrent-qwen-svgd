"""Colab cell: corrected scaled synthetic-depth chain run.

This target runs the active-label evaluator on the existing split-bridge
micro-test, then trains the N=16 full-symbol chain supervision run with the
split bridge and a true prelude LR group.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CHAIN_SCALED_CORRECTED_CELL_VERSION = "chain_scaled_corrected_v1"
# Safety marker: active-label evaluator scores f^k(x) for k <= depth.
# Safety marker: full-symbol chain SFT avoids MCQ label bottleneck.
# Safety marker: bridge_projection_mode=split true bridge_prelude_lr_multiplier param group.
# Safety marker: eval/eval_synthetic_depth_active_labels.py
# Safety marker: STAGE5_CHAIN_CORRECTED_STAGE12_STEPS
# Safety marker: STAGE5_CHAIN_CORRECTED_STAGE1234_STEPS
# Safety marker: STAGE5_CHAIN_CORRECTED_PRELUDE_LR_MULTIPLIER
# Safety marker: tests/test_stage5_chain_scaled_corrected.py

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


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
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
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(list(map(str, cmd)), process.wait(), stdout, None)
    if check and proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout)
    return proc


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    run(["git", "log", "--oneline", "-5"], check=False)
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def require_gpu_runtime() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach a GPU runtime first. L4 is enough; T4 should work but will be slower.")
    run(["nvidia-smi"], cwd=Path("/content"), check=False)


try:
    require_gpu_runtime()
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    print(
        "STAGE5_CHAIN_SCALED_CORRECTED_CELL_VERSION="
        f"{STAGE5_CHAIN_SCALED_CORRECTED_CELL_VERSION}",
        flush=True,
    )
    run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_synthetic_depth_task.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_scaled_corrected.py",
            "tests/test_train_unfrozen_recurrent.py",
            "tests/test_stage5_notebooks.py::test_chain_scaled_corrected_target_is_wired_and_guarded",
        ]
    )
    run([sys.executable, "colab/run_stage5_chain_scaled_corrected.py"])
    if env_flag("STAGE5_CHAIN_CORRECTED_DISCONNECT", "0"):
        print("Disconnecting Colab runtime after corrected scaled chain run.", flush=True)
        runtime.unassign()
except Exception:
    print("Corrected scaled chain run errored; leaving runtime connected.", flush=True)
    raise
