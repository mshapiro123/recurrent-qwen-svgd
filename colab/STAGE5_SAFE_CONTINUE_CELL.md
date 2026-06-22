# Stage 5 Safe Continue Cell

Use this as the default Colab entrypoint when credits are tight. It clones or
updates the private repo, authenticates GitHub/Hugging Face from Colab secrets,
runs the no-GPU A100 go/no-go check, and then **stops by default**.

Only set `RUN_A100_ACTION = True` when you intentionally want to execute the
guarded planner-selected action. The maintained execution path still runs the
`a100_guard` before launching paid-GPU runners. It also refuses long CPU/data
actions, such as dataset audits, while a GPU runtime is attached unless
`STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_LOCAL_ONLY_ON_GPU=1` is set deliberately.
By default the cell disconnects the Colab runtime after the dry run or guarded
action so an attached A100 does not sit idle.

```python
import os, shutil, subprocess, sys
from pathlib import Path
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")

# Deliberate opt-in. Leave False for a no-GPU dry run / status check.
RUN_A100_ACTION = False

# Credit-saver default. Set False only if you intentionally want to keep the
# runtime attached after the cell prints the next action.
DISCONNECT_RUNTIME_WHEN_DONE = True

SOURCE_SUMMARY = (
    "outputs/stage5/stage5_full_assessment_once_20260622_005522/summary.json"
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

def disconnect_runtime(reason):
    if not DISCONNECT_RUNTIME_WHEN_DONE:
        return
    try:
        from google.colab import runtime

        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped/failed: {exc}", flush=True)

def sync_repo():
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        try:
            run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
            run(["git", "fetch", "origin", "main"], cwd=ROOT)
            run(["git", "checkout", "main"], cwd=ROOT)
            pull = run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT, check=False)
            if pull.returncode == 0:
                return
            print("Existing clone could not fast-forward; recloning cleanly.", flush=True)
        except Exception as exc:
            print(f"Existing clone refresh failed; recloning cleanly: {exc}", flush=True)
        shutil.rmtree(ROOT)
    run(["git", "clone", clone_url, str(ROOT)])

sync_repo()
run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

if HF_TOKEN:
    from huggingface_hub import HfApi, login

    login(token=HF_TOKEN, add_to_git_credential=False)
    who = HfApi(token=HF_TOKEN).whoami()
    print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user", flush=True)
else:
    print("HF auth skipped; Hub downloads will be anonymous.", flush=True)

run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
run(["nvidia-smi"], cwd=ROOT, check=False)

run(
    [
        sys.executable,
        "colab/check_stage5_a100_go_no_go.py",
        "--source-summary",
        SOURCE_SUMMARY,
    ],
    cwd=ROOT,
)

env = os.environ.copy()
env["STAGE5_ARC_AGI_NEXT_ACTION_SOURCE_SUMMARY"] = SOURCE_SUMMARY
env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "1" if RUN_A100_ACTION else "0"
env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] = "1"
env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] = "0"

run([sys.executable, "colab/run_stage5_next_action.py"], cwd=ROOT, env=env)

if not RUN_A100_ACTION:
    print("Dry run complete. Set RUN_A100_ACTION = True only when you intentionally want to spend A100 credits.", flush=True)
else:
    print("Guarded next action completed or stopped by a100_guard. Review the emitted summary before continuing.", flush=True)

disconnect_runtime("safe continue cell finished")
```
