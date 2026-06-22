import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
SOURCE_SUMMARY = "outputs/stage5/stage5_full_assessment_once_20260622_005522/summary.json"
GO_NO_GO_RUN_ID = "stage5_arc_mix_recovery_once_go_no_go"
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
    go_allowed = bool(go_decision.get("go"))
    if not go_allowed:
        raise RuntimeError(f"A100 go/no-go blocked ARC-mix recovery: {go_decision}")
    if go_decision.get("spend_class") != "single_arc_mix_proxy":
        raise RuntimeError(f"Unexpected A100 spend class for ARC-mix recovery: {go_decision}")

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    if HF_TOKEN:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN, add_to_git_credential=False)
        who = HfApi(token=HF_TOKEN).whoami()
        print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user", flush=True)
    else:
        print("HF auth skipped; Hub downloads will be anonymous.", flush=True)

    env = os.environ.copy()
    env["STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT"] = "1"
    env.setdefault("STAGE5_ARC_MIX_ONCE_SOURCE_SUMMARY", SOURCE_SUMMARY)
    env.setdefault("STAGE5_ARC_MIX_ARMS", "arc_mix_response_w01_lr2e6")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", "2")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_REPEAT", "4")
    env.setdefault("STAGE5_ARC_MIX_ARC_EVAL_LIMIT", "128")
    env.setdefault("STAGE5_ARC_MIX_OPUS_LIMIT", "3000")
    env.setdefault("STAGE5_ARC_MIX_MIN_MARGIN_DELTA", "-0.05")
    env.setdefault("STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT", "16")
    run([sys.executable, "colab/run_stage5_arc_mix_recovery_once.py"], cwd=ROOT, env=env)
except Exception:
    print("ARC-mix recovery cell failed.", flush=True)
    disconnect_runtime("ARC-mix recovery cell failed")
    raise
