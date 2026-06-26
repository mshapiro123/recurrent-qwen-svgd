"""Colab cell: latent criticality probe from a completed router-validation run."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_LATENT_CRITICALITY_CELL_VERSION = "latent_criticality_probe_v1"
# Bootstrap safety marker: finite_difference_random_gain.

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
        print(line, end="", flush=True)
    returncode = process.wait()
    proc = subprocess.CompletedProcess(cmd, returncode)
    if check and returncode:
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


def infer_artifact_run_id(path: str | Path) -> str | None:
    parts = Path(path).parts
    for marker in ("stage5", "stage4"):
        for idx, part in enumerate(parts):
            if part == marker and idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def drive_roots() -> list[Path]:
    roots = [
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"),
        Path("/content/drive/MyDrive/recurrent-qwen-svgd"),
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-fresh"),
    ]
    if os.environ.get("STAGE5_DRIVE_BACKUP_DIR"):
        roots.insert(0, Path(os.environ["STAGE5_DRIVE_BACKUP_DIR"]))
    seen = set()
    unique = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def candidate_drive_checkpoints(run_id: str, filename: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    for root in drive_roots():
        for candidate in [
            root / run_id / "run_dir" / "phase1" / filename,
            root / run_id / "phase1" / filename,
            root / "outputs" / "stage5" / run_id / "run_dir" / "phase1" / filename,
            root / "outputs" / "stage5" / run_id / "phase1" / filename,
            root / "stage5" / run_id / "run_dir" / "phase1" / filename,
            root / "stage5" / run_id / "phase1" / filename,
        ]:
            add(candidate)
        if not root.exists():
            continue
        for pattern in [
            f"{run_id}*/run_dir/phase1/{filename}",
            f"{run_id}*/run_dir/*/phase1/{filename}",
            f"{run_id}*/phase1/{filename}",
            f"**/{run_id}*/run_dir/phase1/{filename}",
            f"**/{run_id}*/phase1/{filename}",
        ]:
            for candidate in root.glob(pattern):
                add(candidate)
    return candidates


def restore_checkpoint(candidate: Path) -> Path | None:
    if candidate.exists():
        return candidate
    run_id = infer_artifact_run_id(candidate)
    if not run_id:
        return None
    mount_drive()
    for drive_candidate in candidate_drive_checkpoints(run_id, candidate.name):
        if drive_candidate.exists():
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_candidate, candidate)
            print(f"restored_latent_criticality_checkpoint={drive_candidate} -> {candidate}", flush=True)
            return candidate
    return None


def checkpoint_from_sweep(heldout_sweep: str) -> Path:
    sweep_path = ROOT / heldout_sweep if not heldout_sweep.startswith("/") else Path(heldout_sweep)
    payload = read_json(sweep_path)
    for run_id in payload.get("loop_run_ids", []):
        summary_path = ROOT / "outputs" / "stage5" / str(run_id) / "summary.json"
        if not summary_path.exists():
            continue
        loop_payload = read_json(summary_path)
        checkpoint = loop_payload.get("checkpoint")
        if checkpoint:
            candidate = ROOT / checkpoint if not str(checkpoint).startswith("/") else Path(str(checkpoint))
            restored = restore_checkpoint(candidate)
            if restored and restored.exists():
                return restored
    raise FileNotFoundError(f"Could not resolve checkpoint from heldout sweep {heldout_sweep}")


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
    checkpoint = checkpoint_from_sweep(heldout_sweep)
    run_id = os.environ.get("STAGE5_LATENT_CRITICALITY_RUN_ID") or time.strftime(
        "stage5_latent_criticality_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    print("latent_criticality_source_summary:", path_for_cli(source_summary), flush=True)
    print("latent_criticality_discovery_sweep:", discovery_sweep, flush=True)
    print("latent_criticality_heldout_sweep:", heldout_sweep, flush=True)
    print("latent_criticality_checkpoint:", path_for_cli(checkpoint), flush=True)
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
            "--checkpoint",
            path_for_cli(checkpoint),
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
