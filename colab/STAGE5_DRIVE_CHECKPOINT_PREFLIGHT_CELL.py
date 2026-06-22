import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import drive, runtime, userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SOURCE_SUMMARY = "outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json"
GO_NO_GO_RUN_ID = "stage5_drive_checkpoint_preflight"
DISCONNECT_RUNTIME_WHEN_DONE = True

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
assert GH_TOKEN, "Missing GH_TOKEN/GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

def run(cmd, cwd=None, check=True, env=None):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc

clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    try:
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        pull = run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT, check=False)
        if pull.returncode != 0:
            shutil.rmtree(ROOT)
            run(["git", "clone", clone_url, str(ROOT)])
    except Exception:
        shutil.rmtree(ROOT)
        run(["git", "clone", clone_url, str(ROOT)])
else:
    run(["git", "clone", clone_url, str(ROOT)])

run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)

if not Path("/content/drive/MyDrive").exists():
    print("Mounting Google Drive for checkpoint visibility. Approve the prompt if you want to continue.", flush=True)
    drive.mount("/content/drive", force_remount=True)
else:
    print("Drive already mounted.", flush=True)

env = os.environ.copy()
env["STAGE5_A100_GO_NO_GO_RUN_ID"] = GO_NO_GO_RUN_ID
run(
    [
        sys.executable,
        "colab/check_stage5_a100_go_no_go.py",
        "--source-summary",
        SOURCE_SUMMARY,
    ],
    cwd=ROOT,
    env=env,
)

summary_path = ROOT / "outputs" / "stage5" / GO_NO_GO_RUN_ID / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print("decision:", summary["decision"], flush=True)
print("checkpoint_preflight:", summary["checkpoint_preflight"], flush=True)

if DISCONNECT_RUNTIME_WHEN_DONE:
    print("Disconnecting preflight runtime; reconnect with A100 only after checkpoint_preflight.available is True.", flush=True)
    runtime.unassign()
