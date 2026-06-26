"""Colab launcher: Stage 5 re-entry covariance pre-build gate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_REENTRY_COVARIANCE_CHECK_CELL_VERSION = "reentry_covariance_prebuild_v1"
# Bootstrap safety markers: covariance_match_check, directional_prebuild_gate.

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
FORCE_DRIVE_REMOUNT = os.environ.get("FORCE_DRIVE_REMOUNT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_REENTRY_COVARIANCE_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
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
    print("HF token not found; downloads will be anonymous.", flush=True)


def run(cmd, *, cwd: Path = ROOT, env: dict[str, str] | None = None, check: bool = True):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
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
        raise subprocess.CalledProcessError(returncode, cmd)
    return proc


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])


def mount_drive() -> None:
    if Path("/content/drive/MyDrive").exists() and not FORCE_DRIVE_REMOUNT:
        print("Drive already mounted.", flush=True)
        return
    drive.mount("/content/drive", force_remount=FORCE_DRIVE_REMOUNT)


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def current_source_summary() -> Path:
    explicit = os.environ.get("STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY", "").strip()
    if explicit:
        return ROOT / explicit if not explicit.startswith("/") else Path(explicit)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    assert pointer.exists(), "Missing config/stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip()
    assert value, "Current source summary pointer is empty."
    return ROOT / value if not value.startswith("/") else Path(value)


def find_checkpoint_value(payload) -> str | None:
    if isinstance(payload, dict):
        for key in ("checkpoint", "phase1_checkpoint", "resume_from"):
            value = payload.get(key)
            if isinstance(value, str) and value.endswith((".pt", ".pth", ".safetensors")):
                return value
        for value in payload.values():
            found = find_checkpoint_value(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_checkpoint_value(value)
            if found:
                return found
    return None


def candidate_summary_paths(source_summary: Path, payload: dict) -> list[Path]:
    values: list[str] = []
    primary = payload.get("primary_transfer") if isinstance(payload.get("primary_transfer"), dict) else {}
    for value in [
        payload.get("sweep_summary"),
        payload.get("heldout_sweep_summary"),
        payload.get("discovery_sweep_summary"),
        primary.get("heldout_sweep_summary"),
        primary.get("discovery_sweep_summary"),
    ]:
        if isinstance(value, str):
            values.append(value)
    paths = [source_summary]
    for value in values:
        paths.append(ROOT / value if not value.startswith("/") else Path(value))
    return paths


def checkpoint_from_source(source_summary: Path) -> Path:
    explicit = os.environ.get("STAGE5_REENTRY_COVARIANCE_CHECKPOINT", "").strip()
    if explicit:
        return ROOT / explicit if not explicit.startswith("/") else Path(explicit)

    source_payload = read_json(source_summary)
    for summary_path in candidate_summary_paths(source_summary, source_payload):
        if not summary_path.exists():
            continue
        payload = read_json(summary_path)
        direct = find_checkpoint_value(payload)
        if direct:
            return ROOT / direct if not direct.startswith("/") else Path(direct)
        for run_id in payload.get("loop_run_ids", []):
            loop_summary = ROOT / "outputs" / "stage5" / str(run_id) / "summary.json"
            if not loop_summary.exists():
                continue
            loop_payload = read_json(loop_summary)
            direct = find_checkpoint_value(loop_payload)
            if direct:
                return ROOT / direct if not direct.startswith("/") else Path(direct)
    raise FileNotFoundError(f"Could not resolve checkpoint from {source_summary}")


def infer_artifact_run_id(path: Path) -> str | None:
    parts = path.parts
    for marker in ("stage5", "stage4"):
        for idx, part in enumerate(parts):
            if part == marker and idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def drive_roots() -> list[Path]:
    roots = [
        DRIVE_ARTIFACT_ROOT,
        Path("/content/drive/MyDrive/recurrent-qwen-svgd"),
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-fresh"),
    ]
    if os.environ.get("STAGE5_DRIVE_BACKUP_DIR"):
        roots.insert(0, Path(os.environ["STAGE5_DRIVE_BACKUP_DIR"]))
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        if str(root) not in seen:
            seen.add(str(root))
            out.append(root)
    return out


def restore_checkpoint(candidate: Path) -> Path:
    if candidate.exists():
        return candidate
    run_id = infer_artifact_run_id(candidate)
    if not run_id:
        raise FileNotFoundError(candidate)
    mount_drive()
    rel_after_run = None
    parts = candidate.parts
    for idx, part in enumerate(parts):
        if part == run_id and idx + 1 < len(parts):
            rel_after_run = Path(*parts[idx + 1 :])
            break
    candidates: list[Path] = []
    for root in drive_roots():
        candidates.extend(
            [
                root / run_id / rel_after_run if rel_after_run else root / run_id / candidate.name,
                root / "outputs" / "stage5" / run_id / rel_after_run if rel_after_run else root / candidate.name,
                root / "stage5" / run_id / rel_after_run if rel_after_run else root / candidate.name,
            ]
        )
        if root.exists():
            candidates.extend(
                path for path in root.glob(f"**/{run_id}*/**/{candidate.name}") if path.is_file()
            )
    for source in candidates:
        if source.exists():
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, candidate)
            print(f"restored_reentry_covariance_checkpoint={source} -> {candidate}", flush=True)
            return candidate
    preview = "\n".join(str(path) for path in candidates[:20])
    raise FileNotFoundError(f"Could not restore checkpoint {candidate}\nTried:\n{preview}")


def write_pointer(summary_path: Path) -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def publish(run_dir: Path) -> None:
    from colab.stage5_publish_utils import publishable_artifact_paths

    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
    publishable = publishable_artifact_paths(run_dir)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        publishable.append(pointer)
    for path in publishable:
        run(["git", "add", "-f", path_for_cli(path)])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT))
    if diff.returncode == 0:
        print("No covariance-check outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 re-entry covariance check {run_dir.name} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode:
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


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
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_REENTRY_COVARIANCE_CHECK_CELL_VERSION={STAGE5_REENTRY_COVARIANCE_CHECK_CELL_VERSION}", flush=True)
    run(["git", "log", "--oneline", "-5"], check=False)
    run(["nvidia-smi"], cwd=Path("/content"), check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_eval_reentry_covariance_check.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_reentry_covariance_check_target",
        ]
    )
    source_summary = current_source_summary()
    checkpoint = restore_checkpoint(checkpoint_from_source(source_summary))
    run_id = os.environ.get("STAGE5_REENTRY_COVARIANCE_RUN_ID") or time.strftime(
        "stage5_reentry_covariance_check_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print("reentry_covariance_source_summary:", path_for_cli(source_summary), flush=True)
    print("reentry_covariance_checkpoint:", path_for_cli(checkpoint), flush=True)
    run(
        [
            sys.executable,
            "eval/eval_reentry_covariance_check.py",
            "--checkpoint",
            path_for_cli(checkpoint),
            "--prompts_jsonl",
            os.environ.get("STAGE5_REENTRY_COVARIANCE_PROMPTS", "eval/smoke_exact_tasks_v2.jsonl"),
            "--limit",
            os.environ.get("STAGE5_REENTRY_COVARIANCE_LIMIT", "14"),
            "--max_length",
            os.environ.get("STAGE5_REENTRY_COVARIANCE_MAX_LENGTH", "256"),
            "--subspace_rank",
            os.environ.get("STAGE5_REENTRY_COVARIANCE_SUBSPACE_RANK", "8"),
            "--analysis_rank",
            os.environ.get("STAGE5_REENTRY_COVARIANCE_ANALYSIS_RANK", "8"),
            "--dtype",
            os.environ.get("DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--output_json",
            path_for_cli(run_dir / "summary.json"),
            "--output_md",
            path_for_cli(run_dir / "summary.md"),
        ]
    )
    write_pointer(run_dir / "summary.json")
    if Path("/content/drive/MyDrive").exists():
        backup_dir = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / run_id
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(run_dir, backup_dir)
        print(f"drive_backup={backup_dir}", flush=True)
    publish(run_dir)
    disconnect("re-entry covariance pre-build check finished")
except Exception:
    disconnect("re-entry covariance pre-build check errored")
    raise
