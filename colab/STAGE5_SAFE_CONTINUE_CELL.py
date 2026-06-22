import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")

def env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

# Deliberate opt-in. Leave False for a no-GPU dry run / status check.
RUN_A100_ACTION = env_bool("STAGE5_SAFE_CONTINUE_RUN_A100_ACTION", False)

# Credit-saver default. Set False only if you intentionally want to keep the
# runtime attached after the cell prints the next action.
DISCONNECT_RUNTIME_WHEN_DONE = env_bool("STAGE5_SAFE_CONTINUE_DISCONNECT", True)

SOURCE_SUMMARY = os.environ.get(
    "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
    "outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json",
)
GO_NO_GO_RUN_ID = "stage5_safe_continue_go_no_go"

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

def mount_drive_for_paid_action():
    if Path("/content/drive/MyDrive").exists():
        print("Drive already mounted.", flush=True)
        return
    from google.colab import drive

    print("Mounting Google Drive so checkpoint artifacts can be restored.", flush=True)
    drive.mount("/content/drive", force_remount=True)

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

run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
run(["nvidia-smi"], cwd=ROOT, check=False)

if RUN_A100_ACTION:
    mount_drive_for_paid_action()

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
checkpoint_available = bool((go_payload.get("checkpoint_preflight") or {}).get("available"))
print("a100_guard_decision:", go_decision, flush=True)
print("a100_checkpoint_preflight:", go_payload.get("checkpoint_preflight"), flush=True)
if not RUN_A100_ACTION and go_allowed:
    print(
        "DRY_RUN_GREEN: guarded action is currently allowed. Set "
        "STAGE5_CURRENT_A100_TARGET=safe_continue_execute only when you intentionally want to spend paid GPU.",
        flush=True,
    )
elif not RUN_A100_ACTION and not checkpoint_available:
    print(
        "DRY_RUN_RED: required checkpoint is not visible. Run the Drive/checkpoint preflight on a cheap runtime first.",
        flush=True,
    )
if RUN_A100_ACTION and not go_allowed:
    print(f"RUN_A100_ACTION requested, but a100_guard blocked spend: {go_decision}", flush=True)

execute_action = bool(RUN_A100_ACTION and go_allowed)
if execute_action:
    mount_drive_for_paid_action()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_a100_go_no_go.py",
            "tests/test_stage5_next_action.py",
            "tests/test_stage5_routing_repair.py",
            "tests/test_stage5_balanced_arc_mix_gate.py",
            "tests/test_curriculum_sft_gate.py",
            "tests/test_stage5_curriculum_sft.py",
            "tests/test_curriculum_pipeline_from_artifacts.py",
            "tests/test_curriculum_jsonl.py",
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
else:
    print("Skipping requirements install because no paid action will execute.", flush=True)

env = os.environ.copy()
env["STAGE5_ARC_AGI_NEXT_ACTION_SOURCE_SUMMARY"] = SOURCE_SUMMARY
env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "1" if execute_action else "0"
env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] = "1"
env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] = "0"

run([sys.executable, "colab/run_stage5_next_action.py"], cwd=ROOT, env=env)

if not execute_action:
    print("Dry run complete. Set RUN_A100_ACTION = True only when you intentionally want to spend A100 credits.", flush=True)
else:
    print("Guarded next action completed or stopped by a100_guard. Review the emitted summary before continuing.", flush=True)

disconnect_runtime("safe continue cell finished")
