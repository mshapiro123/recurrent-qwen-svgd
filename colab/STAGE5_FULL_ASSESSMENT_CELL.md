# Stage 5 Full ARC Assessment Single Cell

Use this only when the next A100 spend is the full balanced ARC-Easy /
ARC-Challenge assessment for the latest passed ARC-mix proxy gate. It clones or
updates the private repo, authenticates GitHub/Hugging Face from Colab secrets,
runs exactly one assessment, pushes safe text artifacts, and disconnects the
runtime when finished.
It mounts Drive and runs go/no-go before installing dependencies, and it
disconnects on setup failure by default.

Do not use this for dataset audits, notebook repair, Phase 2/SVGD, or GPQA.

```python
import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SOURCE_SUMMARY = "outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/summary.json"
GO_NO_GO_RUN_ID = "stage5_full_assessment_once_go_no_go"
DISCONNECT_RUNTIME_ON_FAILURE = True

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
    if not DISCONNECT_RUNTIME_ON_FAILURE:
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

try:
    sync_repo()

    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
    try:
        from google.colab import drive

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    except Exception as exc:
        print(f"Drive mount skipped/failed before go/no-go: {exc}", flush=True)

    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run(["nvidia-smi"], cwd=ROOT, check=False)

    check_env = os.environ.copy()
    check_env["STAGE5_A100_GO_NO_GO_RUN_ID"] = GO_NO_GO_RUN_ID
    run(
        [
            sys.executable,
            "colab/check_stage5_a100_go_no_go.py",
            "--source-summary",
            SOURCE_SUMMARY,
        ],
        cwd=ROOT,
        env=check_env,
    )
    go_payload = json.loads((ROOT / "outputs" / "stage5" / GO_NO_GO_RUN_ID / "summary.json").read_text(encoding="utf-8"))
    go_decision = go_payload.get("decision", {})
    if not go_decision.get("go"):
        raise RuntimeError(f"A100 go/no-go blocked full ARC assessment: {go_decision}")

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    if HF_TOKEN:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN, add_to_git_credential=False)
        who = HfApi(token=HF_TOKEN).whoami()
        print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user", flush=True)
    else:
        print("HF auth skipped; Hub downloads will be anonymous.", flush=True)

    env = os.environ.copy()
    env["STAGE5_FULL_ASSESS_AUTO_DISCONNECT"] = "1"
    env.setdefault("STAGE5_FULL_ASSESS_SOURCE_SUMMARY", SOURCE_SUMMARY)
    run([sys.executable, "colab/run_stage5_full_assessment_once.py"], cwd=ROOT, env=env)
except Exception:
    print("Full assessment cell failed.", flush=True)
    disconnect_runtime("full assessment cell failed")
    raise
```
