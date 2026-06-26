"""Colab cell: latent criticality probe from a completed router-validation run."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_LATENT_CRITICALITY_CELL_VERSION = "latent_criticality_probe_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_LATENT_CRITICALITY_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
FORCE_DRIVE_REMOUNT = os.environ.get("FORCE_DRIVE_REMOUNT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def secret(*names):
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


def run(cmd, *, cwd=None, env=None, check=True):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def sync_repo():
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
    else:
        run(["git", "clone", clone_url, str(ROOT)])
    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)


def mount_drive():
    if Path("/content/drive/MyDrive").exists() and not FORCE_DRIVE_REMOUNT:
        print("Drive already mounted.", flush=True)
        return
    drive.mount("/content/drive", force_remount=FORCE_DRIVE_REMOUNT)


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def current_source_summary() -> Path:
    explicit = os.environ.get("STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY", "").strip()
    if explicit:
        return ROOT / explicit if not explicit.startswith("/") else Path(explicit)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    assert pointer.exists(), "Missing config/stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip()
    assert value, "Current source summary pointer is empty."
    return ROOT / value if not value.startswith("/") else Path(value)


def resolve_sweeps(source_summary: Path) -> tuple[str, str]:
    payload = read_json(source_summary)
    kind = payload.get("kind")
    if kind == "stage5_heldout_router_validation":
        primary = payload.get("primary_transfer") or {}
        discovery = primary.get("discovery_sweep_summary")
        heldout = payload.get("sweep_summary")
    elif kind == "stage5_heldout_router_validation_sweep":
        discovery = payload.get("discovery_sweep_summary")
        heldout = path_for_cli(source_summary)
    else:
        discovery = os.environ.get("STAGE5_LATENT_CRITICALITY_DISCOVERY_SWEEP", "").strip()
        heldout = os.environ.get("STAGE5_LATENT_CRITICALITY_HELDOUT_SWEEP", "").strip()
    if not discovery or not heldout:
        raise RuntimeError(f"Cannot resolve discovery/heldout sweeps from {source_summary} kind={kind!r}")
    return str(discovery), str(heldout)


def write_pointer(summary_path: Path) -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def publish(run_dir: Path) -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    run(["git", "add", "-f", path_for_cli(run_dir)], cwd=ROOT, check=False)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], cwd=ROOT, check=False)
    status = run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False)
    if status.returncode == 0:
        print("No latent criticality outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 latent criticality {run_dir.name} [skip ci]"], cwd=ROOT)
    push = run(["git", "push", "origin", "main"], cwd=ROOT, check=False)
    if push.returncode:
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT)
        run(["git", "push", "origin", "main"], cwd=ROOT)


def disconnect(reason: str) -> None:
    if not DISCONNECT_ON_FINISH:
        print(f"Leaving Colab runtime connected: {reason}", flush=True)
        return
    try:
        print(f"Disconnecting Colab runtime: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    mount_drive()
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_LATENT_CRITICALITY_CELL_VERSION={STAGE5_LATENT_CRITICALITY_CELL_VERSION}", flush=True)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_eval_latent_criticality.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_latent_criticality_target",
        ],
        cwd=ROOT,
    )
    source_summary = current_source_summary()
    discovery_sweep, heldout_sweep = resolve_sweeps(source_summary)
    run_id = os.environ.get("STAGE5_LATENT_CRITICALITY_RUN_ID") or time.strftime(
        "stage5_latent_criticality_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    print("latent_criticality_source_summary:", path_for_cli(source_summary), flush=True)
    print("latent_criticality_discovery_sweep:", discovery_sweep, flush=True)
    print("latent_criticality_heldout_sweep:", heldout_sweep, flush=True)
    run(
        [
            sys.executable,
            "eval/eval_latent_criticality.py",
            "--discovery_sweep_summary",
            discovery_sweep,
            "--heldout_sweep_summary",
            heldout_sweep,
            "--output_dir",
            path_for_cli(run_dir),
            "--max_examples_per_benchmark",
            os.environ.get("STAGE5_LATENT_CRITICALITY_MAX_EXAMPLES_PER_BENCHMARK", "64"),
            "--jacobian_examples_per_benchmark",
            os.environ.get("STAGE5_LATENT_CRITICALITY_JACOBIAN_EXAMPLES_PER_BENCHMARK", "8"),
            "--jacobian_random_probes",
            os.environ.get("STAGE5_LATENT_CRITICALITY_JACOBIAN_RANDOM_PROBES", "1"),
            "--jacobian_epsilon",
            os.environ.get("STAGE5_LATENT_CRITICALITY_JACOBIAN_EPSILON", "0.02"),
            "--dtype",
            os.environ.get("DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ],
        cwd=ROOT,
    )
    write_pointer(run_dir / "summary.json")
    publish(run_dir)
    disconnect("latent criticality probe finished")
except Exception:
    disconnect("latent criticality probe errored")
    raise
