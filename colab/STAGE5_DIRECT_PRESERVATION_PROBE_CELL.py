import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION = "direct_preservation_probe_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")

DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/"
    "stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation/"
    "answer_prior_diagnosis.json"
)


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


def env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


DISCONNECT_WHEN_DONE = env_bool("STAGE5_DIRECT_PRESERVE_DISCONNECT", True)
RUN_ID = os.environ.get("STAGE5_DIRECT_PRESERVE_RUN_ID") or time.strftime(
    "stage5_direct_preservation_loop1_%Y%m%d_%H%M%S"
)
SOURCE_SUMMARY = os.environ.get("STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY") or DEFAULT_SOURCE_SUMMARY


def printable_cmd(cmd):
    return " ".join(map(str, cmd)).replace(GH_TOKEN, "****")


def run(cmd, *, cwd=None, env=None, check=True):
    print("$", printable_cmd(cmd), flush=True)
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


def safe_stage_and_push(run_dir):
    suffixes = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".log", ".csv"}
    files = [path for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    if not files:
        print("No lightweight output files to commit.", flush=True)
        return
    rels = [str(path.relative_to(ROOT)) for path in files]
    run(["git", "add", "-f", *rels], cwd=ROOT)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, stdout=subprocess.PIPE)
    if not status.stdout.strip():
        print("No git changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 direct preservation probe {RUN_ID}"], cwd=ROOT)
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "rebase", "origin/main"], cwd=ROOT)
    run(["git", "push", "origin", "main"], cwd=ROOT)


def disconnect(reason):
    if not DISCONNECT_WHEN_DONE:
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    assert shutil.which("nvidia-smi"), "This cell is intended for a GPU runtime; nvidia-smi was not found."
    run(["nvidia-smi"], check=False)

    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    env = os.environ.copy()
    env.update(
        {
            "STAGE5_DIRECT_PRESERVE_RUN_ID": RUN_ID,
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY": SOURCE_SUMMARY,
            "STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT", "512"
            ),
            "STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT", "128"
            ),
            "STAGE5_DIRECT_PRESERVE_MAX_STEPS": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_MAX_STEPS", "75"
            ),
            "STAGE5_DIRECT_PRESERVE_SAVE_EVERY": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_SAVE_EVERY", "25"
            ),
            "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN", "1.0"
            ),
            "STAGE5_DIRECT_PRESERVE_LR": os.environ.get("STAGE5_DIRECT_PRESERVE_LR", "5e-7"),
            "STAGE5_DIRECT_PRESERVE_BETA": os.environ.get("STAGE5_DIRECT_PRESERVE_BETA", "0.02"),
            "STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT", "1.0"
            ),
            "STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE": os.environ.get(
                "STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE", "2.0"
            ),
        }
    )
    print("direct_preservation_probe_run_id:", RUN_ID, flush=True)
    print("direct_preservation_source_summary:", SOURCE_SUMMARY, flush=True)
    run([sys.executable, "colab/run_stage5_direct_preservation_probe.py"], cwd=ROOT, env=env)

    run_dir = ROOT / "outputs" / "stage5" / RUN_ID
    assert run_dir.exists(), f"Expected run_dir was not created: {run_dir}"
    drive_dst = DRIVE_ARTIFACT_ROOT / "stage5" / RUN_ID
    drive_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, drive_dst, dirs_exist_ok=True)
    print(f"backed_up_run_dir={run_dir} -> {drive_dst}", flush=True)

    safe_stage_and_push(run_dir)
    disconnect("direct preservation probe finished")
except Exception:
    disconnect("direct preservation probe errored")
    raise
