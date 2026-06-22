"""CPU-only Colab cell to recover a completed direct-preservation run from Drive.

Use this if the A100 run completed, backed up its run directory to Drive, then
failed while pushing lightweight metadata to GitHub.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

from google.colab import drive, runtime, userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ROOTS = [
    Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5"),
    Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"),
    Path("/content/drive/MyDrive/recurrent-qwen-svgd"),
]


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
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."


def run(cmd, *, cwd=None, check=True):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, flush=True)
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


def find_latest_run() -> Path:
    explicit = os.environ.get("STAGE5_RECOVER_DIRECT_RUN_DIR", "").strip()
    if explicit:
        path = Path(explicit)
        if (path / "summary.json").exists():
            return path
        raise FileNotFoundError(f"Explicit recovery dir has no summary.json: {path}")

    candidates: list[Path] = []
    for root in DRIVE_ROOTS:
        if not root.exists():
            continue
        candidates.extend(path.parent for path in root.rglob("summary.json") if "stage5_direct_preservation_loop1_" in str(path))
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No Drive-backed stage5_direct_preservation_loop1_* summary.json found.")
    return candidates[0]


def print_key_summary(summary):
    best = summary.get("best_checkpoint") or {}
    print("\n=== DIRECT PRESERVATION RESULT ===")
    print("run_id:", summary.get("run_id"))
    print("status:", summary.get("status"))
    print("passed:", summary.get("passed"))
    print("base_eval:", summary.get("base_eval", {}).get("correct"), "/", summary.get("base_eval", {}).get("total"))
    print("start_loop1:", summary.get("start_loop1_eval", {}).get("correct"), "/", summary.get("start_loop1_eval", {}).get("total"))
    print("start_loop4:", summary.get("start_loop4_eval", {}).get("correct"), "/", summary.get("start_loop4_eval", {}).get("total"))
    print("best_loop1:", best.get("loop1_eval", {}).get("correct"), "/", best.get("loop1_eval", {}).get("total"))
    print("best_loop4:", best.get("loop4_eval", {}).get("correct"), "/", best.get("loop4_eval", {}).get("total"))
    print("best_checkpoint:", best.get("checkpoint"))
    print("next_step:", summary.get("next_step"))


def publish_run(source_dir: Path):
    run_id = source_dir.name
    target_dir = ROOT / "outputs" / "stage5" / run_id
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    pointer = ROOT / "config" / "stage5_latest_direct_preservation_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"outputs/stage5/{run_id}/summary.json\n", encoding="utf-8")

    suffixes = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".log", ".csv"}
    files = [path for path in target_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    files.append(pointer)
    rels = [str(path.relative_to(ROOT)) for path in files]
    run(["git", "add", "-f", *rels], cwd=ROOT)
    status = run(["git", "status", "--porcelain"], cwd=ROOT, check=False).stdout
    if not status.strip():
        print("No GitHub metadata changes to publish.", flush=True)
        return
    run(["git", "commit", "-m", f"Recover Stage 5 direct preservation results {run_id}"], cwd=ROOT)
    run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT)
    run(["git", "push", "origin", "main"], cwd=ROOT)


try:
    drive.mount("/content/drive", force_remount=False)
    source_dir = find_latest_run()
    print("recovering_from:", source_dir, flush=True)
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    print_key_summary(summary)
    sync_repo()
    publish_run(source_dir)
    print("Recovery complete.", flush=True)
except Exception:
    raise
finally:
    try:
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)
