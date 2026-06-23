import os, shutil, subprocess, sys, time
from pathlib import Path
from google.colab import userdata

STAGE5_ARC_MIX_DEPTH_ROUTING_CELL_VERSION = "arc_mix_depth_routing_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_SOURCE_SUMMARY = (
    "outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json"
)
DISCONNECT_WHEN_DONE = os.environ.get("STAGE5_ARC_MIX_DEPTH_DISCONNECT", "1").strip().lower() in {
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


def mask(text):
    masked = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            masked = masked.replace(token, "****")
    return masked


def run(cmd, *, cwd=None, env=None, check=True):
    printable = mask(" ".join(map(str, cmd)))
    print("$", printable, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(mask(proc.stdout), flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc


def sync_repo():
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
    run(["git", "clone", clone_url, str(ROOT)])


def disconnect_runtime(reason):
    if not DISCONNECT_WHEN_DONE:
        return
    try:
        from google.colab import runtime

        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped/failed: {exc}", flush=True)


try:
    sync_repo()
    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run(["nvidia-smi"], cwd=ROOT, check=False)

    try:
        from google.colab import drive

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    except Exception as exc:
        print(f"Drive mount skipped/failed before depth-routing run: {exc}", flush=True)

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_balanced_arc_mix_gate.py",
            "tests/test_causal_dataset_loop_targets.py",
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
    env.setdefault("STAGE5_ARC_MIX_RUN_ID", "stage5_arc_mix_depth_routing_probe_" + time.strftime("%Y%m%d_%H%M%S"))
    env.setdefault("STAGE5_ARC_MIX_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY)
    env.setdefault("STAGE5_ARC_MIX_OPUS_LIMIT", "0")
    env.setdefault("STAGE5_ARC_MIX_ARC_TRAIN_LIMIT", "0")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", "2")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_REPEAT", "6")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP", "3")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP", "1")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE", "deep_narrow")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE", "direct")
    env.setdefault("STAGE5_ARC_MIX_PROMPT_STYLE", "question_only")
    env.setdefault("STAGE5_ARC_MIX_SCORE_TARGET", "option_text")
    env.setdefault("STAGE5_ARC_MIX_ARC_EVAL_CONFIG", "ARC-Challenge")
    env.setdefault("STAGE5_ARC_MIX_ARC_EVAL_LIMIT", "128")
    env.setdefault("STAGE5_ARC_MIX_ARMS", "arc_mix_response_w02_lr2e6")
    env.setdefault("STAGE5_ARC_MIX_USE_LEARNED_LOOP_CONTROL", "1")
    env.setdefault("STAGE5_ARC_MIX_EVAL_USE_LEARNED_LOOP_CONTROL", "1")
    env.setdefault("STAGE5_ARC_MIX_LOOP_CONTROL_CE_WEIGHT", "0.05")
    env.setdefault("STAGE5_ARC_MIX_HALT_TARGET_NLL_WEIGHT", "0.03")
    env.setdefault("STAGE5_ARC_MIX_MIN_MARGIN_DELTA", "-0.05")
    env.setdefault("STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT", "16")
    env.setdefault("STAGE5_ARC_MIX_PUSH", "1")

    print("RUN_ID", env["STAGE5_ARC_MIX_RUN_ID"], flush=True)
    print("SOURCE_SUMMARY", env["STAGE5_ARC_MIX_SOURCE_SUMMARY"], flush=True)
    print("target_loop_count ARC-Easy=1 ARC-Challenge=3", flush=True)
    print("learned_loop_control", env["STAGE5_ARC_MIX_USE_LEARNED_LOOP_CONTROL"], flush=True)
    run([sys.executable, "colab/run_stage5_balanced_arc_mix_gate.py"], cwd=ROOT, env=env)
except Exception:
    disconnect_runtime("depth-routing probe failed")
    raise
else:
    disconnect_runtime("depth-routing probe complete")
