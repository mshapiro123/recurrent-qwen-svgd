"""Colab cell: generate natural-surface transfer datasets on CPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_NATURAL_SURFACE_PREPARE_CELL_VERSION = "natural_surface_prepare_v1"
# Safety marker: natural_surface_prepare_cpu
# Safety marker: stage5_natural_surface_transfer_dataset
# Safety marker: stage5_natural_surface_transfer_prepare
# Safety marker: training/generate_natural_surface_transfer.py
# Safety marker: colab/run_stage5_natural_surface_prepare.py
# Safety marker: tests/test_natural_surface_transfer.py
# Safety marker: relay_test_chain_mcq
# Safety marker: pointer_test_chain_mcq
# Safety marker: rung0_train_mix_chain_symbol_sft
# Safety marker: value_prefix=name:
# Safety marker: STAGE5_NATURAL_VERIFY_TOKENIZER
# Safety marker: tokenizer single-token names

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
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    rc = proc.wait()
    if rc:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join("".join(chunks).splitlines()[-120:]), flush=True)
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


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    sync_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_natural_surface_transfer.py",
            "tests/test_eval_synthetic_depth_active_labels.py::test_name_value_prefix_maps_full_symbol_space",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_target_markers_exist_in_launcher_files",
        ]
    )
    env = os.environ.copy()
    env.setdefault("STAGE5_NATURAL_VERIFY_TOKENIZER", "1")
    run([sys.executable, "colab/run_stage5_natural_surface_prepare.py"], env=env)
    if env_flag("STAGE5_NATURAL_SURFACE_DISCONNECT", "0"):
        print("Disconnecting Colab runtime after natural-surface prep.", flush=True)
        runtime.unassign()
    else:
        print("Leaving Colab runtime connected after natural-surface prep.", flush=True)


try:
    main()
except Exception:
    print("Natural-surface prep errored; leaving runtime connected.", flush=True)
    raise
