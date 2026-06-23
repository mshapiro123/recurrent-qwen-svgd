import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_ARC_MIX_OFFSET_THEN_DEPTH_CELL_VERSION = "arc_mix_offset_then_depth_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT_WHEN_DONE = os.environ.get(
    "STAGE5_ARC_MIX_CHAIN_DISCONNECT", "1"
).strip().lower() in {"1", "true", "yes", "y"}
MOUNT_DRIVE_FIRST = os.environ.get(
    "STAGE5_ARC_MIX_CHAIN_MOUNT_DRIVE_FIRST", "0"
).strip().lower() in {"1", "true", "yes", "y"}


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


def mask(text: str) -> str:
    masked = text
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            masked = masked.replace(token, "****")
    return masked


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    shown = mask(" ".join(map(str, cmd)))
    print("$", shown, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(mask(proc.stdout), flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {shown}")
    return proc


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        try:
            run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
            run(["git", "fetch", "origin", "main"], cwd=ROOT)
            run(["git", "checkout", "main"], cwd=ROOT)
            run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
            return
        except Exception as exc:
            print(f"Existing clone refresh failed; recloning cleanly: {exc}", flush=True)
            shutil.rmtree(ROOT)
    run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))


def disconnect_runtime(reason: str) -> None:
    if not DISCONNECT_WHEN_DONE:
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"runtime.unassign skipped/failed: {exc}", flush=True)


try:
    gpu_check = shutil.which("nvidia-smi")
    assert gpu_check, "Attach an A100/H100/L4/T4 GPU runtime before running this chain."
    run(["nvidia-smi"], cwd=Path("/content"), check=False)
    if MOUNT_DRIVE_FIRST:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
    else:
        print(
            "Skipping upfront Drive mount; checkpoint restore/backup will request Drive only if needed.",
            flush=True,
        )
    sync_repo()
    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_offset_then_depth.py",
            "tests/test_stage5_balanced_arc_mix_gate.py",
            "tests/test_stage5_benchmark_suite.py",
        ],
        cwd=ROOT,
    )

    if HF_TOKEN:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN, add_to_git_credential=False)
        who = HfApi(token=HF_TOKEN).whoami()
        print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user", flush=True)
    else:
        print("HF auth skipped; Hub downloads will be anonymous.", flush=True)

    env = os.environ.copy()
    env.setdefault(
        "STAGE5_ARC_MIX_CHAIN_RUN_ID",
        "stage5_arc_mix_offset_then_depth_chain",
    )
    env.setdefault("STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH", "1")
    env.setdefault("STAGE5_ARC_MIX_CHAIN_ALLOWED_NEGATIVE_DELTA", "0")
    env.setdefault("STAGE5_ARC_MIX_CHAIN_MIN_EXAMPLES", "256")
    env.setdefault("STAGE5_ARC_MIX_CHAIN_RUN_POST_DEPTH_DEBIASED_GATE", "1")
    env.setdefault("STAGE5_ARC_MIX_CHAIN_POST_DEPTH_MIN_EXAMPLES", "128")
    env.setdefault("STAGE5_ARC_MIX_CHAIN_PUSH", "1")

    print("RUN_ID", env["STAGE5_ARC_MIX_CHAIN_RUN_ID"], flush=True)
    print("STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH", env["STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH"], flush=True)
    print("Offset gate: ARC-Easy and ARC-Challenge, offset=256, content + cyclic MCQ.", flush=True)
    print("Depth gate: target_loop_count ARC-Easy=1 ARC-Challenge=3 if offset passes.", flush=True)
    print("Post-depth gate: debiased cyclic scoring is primary; content is a leading indicator.", flush=True)
    run([sys.executable, "colab/run_stage5_arc_mix_offset_then_depth.py"], cwd=ROOT, env=env)
except Exception:
    disconnect_runtime("ARC-mix offset-depth chain failed")
    raise
else:
    disconnect_runtime("ARC-mix offset-depth chain complete")
