"""Colab cell for post-positive synthetic-depth consolidation targets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION = "chain_consolidation_v1"
# Safety marker: depth_extrapolation_eval
# Safety marker: synthetic_probe_battery
# Safety marker: chain_anneal_to_outcome
# Safety marker: post_anneal_readouts
# Safety marker: eval/eval_synthetic_depth_artifact_check.py
# Safety marker: eval/eval_synthetic_depth_probe.py
# Safety marker: eval/analyze_synthetic_reader_alignment.py
# Safety marker: colab/run_stage5_depth_extrapolation_eval.py
# Safety marker: colab/run_stage5_synthetic_probe_battery.py
# Safety marker: colab/run_stage5_chain_anneal_to_outcome.py
# Safety marker: colab/run_stage5_post_anneal_readouts.py
# Safety marker: STAGE5_EXTRAP_DEPTHS
# Safety marker: STAGE5_EXTRAP_MAX_LOOPS
# Safety marker: STAGE5_EXTRAP_CHECKPOINT
# Safety marker: STAGE5_PROBE_CHECKPOINT
# Safety marker: STAGE5_POST_ANNEAL_SOURCE_SUMMARY
# Safety marker: STAGE5_ANNEAL_TOTAL_STEPS
# Safety marker: STAGE5_ANNEAL_PRELUDE_LR_MULT
# Safety marker: loop_index_probe
# Safety marker: router_leak_exclusion
# Safety marker: state_envelope
# Safety marker: loop_loss_mode='annealed_chain_to_outcome'
# Safety marker: tests/test_eval_synthetic_depth_probe.py
# Safety marker: tests/test_stage5_chain_consolidation.py
# Safety marker: tests/test_recurrent_wrapper_tiny.py::test_annealed_chain_to_outcome_loss_mixes_chain_and_target_ce_on_tiny_model

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "depth_extrapolation_eval")

TARGETS = {
    "depth_extrapolation_eval": {
        "script": "colab/run_stage5_depth_extrapolation_eval.py",
        "tests": [
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_EXTRAP_DISCONNECT",
    },
    "synthetic_probe_battery": {
        "script": "colab/run_stage5_synthetic_probe_battery.py",
        "tests": [
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_PROBE_DISCONNECT",
    },
    "chain_anneal_to_outcome": {
        "script": "colab/run_stage5_chain_anneal_to_outcome.py",
        "tests": [
            "tests/test_recurrent_wrapper_tiny.py::test_annealed_chain_to_outcome_loss_mixes_chain_and_target_ce_on_tiny_model",
            "tests/test_train_unfrozen_recurrent.py::test_chain_label_weight_ramps_then_holds_zero",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_ANNEAL_DISCONNECT",
    },
    "post_anneal_readouts": {
        "script": "colab/run_stage5_post_anneal_readouts.py",
        "tests": [
            "tests/test_analyze_synthetic_reader_alignment.py",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
            "tests/test_stage5_notebooks.py::test_chain_consolidation_targets_are_wired_and_guarded",
        ],
        "disconnect_env": "STAGE5_POST_ANNEAL_DISCONNECT",
    },
}


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


def run(cmd: list[str | os.PathLike[str]], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
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


def require_gpu_runtime() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach a GPU runtime first. L4/T4 is sufficient for these consolidation targets.")
    run(["nvidia-smi"], cwd=Path("/content"), check=False)


try:
    if TARGET not in TARGETS:
        raise ValueError(f"Unknown consolidation target: {TARGET}")
    require_gpu_runtime()
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    print(f"STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION={STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION}", flush=True)
    print(f"stage5_chain_consolidation_target={TARGET}", flush=True)
    run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    target = TARGETS[TARGET]
    run([sys.executable, "-m", "pytest", "-q", *target["tests"]])
    run([sys.executable, target["script"]])
    if env_flag(str(target["disconnect_env"]), "0"):
        print(f"Disconnecting Colab runtime after {TARGET}.", flush=True)
        runtime.unassign()
except Exception:
    print(f"Stage 5 chain consolidation target errored: {TARGET}; leaving runtime connected.", flush=True)
    raise
