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
action so an attached A100 does not sit idle. Dry-runs and blocked actions skip
`pip install -r requirements.txt`; dependency installation happens only after
the A100 guard allows an intentional paid action. When a paid action is
requested, the notebook mounts Drive in the top-level Colab process before the
go/no-go checkpoint preflight; Colab Drive authorization cannot reliably be
initiated from a child Python process. Paid actions also run a small focused
preflight over the A100 guard, next-action parser, routing repair, and
ARC-mix repair gate before launching the selected action.
The cell executes one planner action by default. To deliberately continue
through multiple guarded actions in one runtime lease, set
`STAGE5_SAFE_CONTINUE_MAX_ACTIONS` to a small integer such as `2`. Keep the
default at `1` unless you are intentionally trading more GPU time for less
manual review.
Set `STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY` to force a specific Stage 5 summary.
Without that override, the cell auto-resumes from
`config/stage5_current_source_summary.txt` when the pointer exists and targets
an existing summary. If no valid pointer is available, it falls back to the last
committed routing diagnostic.

```python
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
MAX_NEXT_ACTIONS = int(os.environ.get("STAGE5_SAFE_CONTINUE_MAX_ACTIONS", "1"))
ALLOW_REPEAT_NEXT_ACTION = env_bool("STAGE5_SAFE_CONTINUE_ALLOW_REPEAT", False)
A100_BUDGET_PROFILE = os.environ.get("STAGE5_A100_BUDGET_PROFILE", "credit_saver").strip()
PREFER_TRAINING_SOURCE = env_bool("STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE", False)

DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json"
SOURCE_SUMMARY_OVERRIDE = os.environ.get("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", "").strip()
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

def safe_read_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None

def training_source_priority(payload):
    kind = payload.get("kind")
    if kind == "stage5_capability_ladder_trace_collection":
        gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
        if payload.get("status") == "trace_curriculum_gate_ready" and gate.get("go") is True:
            return 100
    if kind == "curriculum_sft_gate" and payload.get("go") is True:
        return 90
    if kind == "stage5_curriculum_sft":
        checks = payload.get("validation_checks") if isinstance(payload.get("validation_checks"), dict) else {}
        if checks.get("status") == "validation_sane":
            return 80
    return 0

def latest_training_source_summary():
    candidates = []
    scan_roots = [ROOT / "outputs" / "stage5"]
    drive_stage5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd/stage5_capability_ladder_trace_collection")
    if drive_stage5.exists():
        scan_roots.append(drive_stage5)
    for scan_root in scan_roots:
        for path in scan_root.glob("**/summary.json"):
            payload = safe_read_json(path)
            if not payload:
                continue
            priority = training_source_priority(payload)
            if priority <= 0:
                continue
            candidates.append((priority, path.stat().st_mtime, path))
    if not candidates:
        return ""
    selected = sorted(candidates, reverse=True)[0][2]
    try:
        return str(selected.relative_to(ROOT))
    except ValueError:
        return str(selected)

def resolve_source_summary():
    if SOURCE_SUMMARY_OVERRIDE:
        print(f"Using explicit STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY={SOURCE_SUMMARY_OVERRIDE}", flush=True)
        return SOURCE_SUMMARY_OVERRIDE
    if PREFER_TRAINING_SOURCE:
        latest = latest_training_source_summary()
        if latest:
            print(f"Using latest training source summary: {latest}", flush=True)
            return latest
        print("No gate-ready training source summary found; falling back to current pointer.", flush=True)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            target = Path(value)
            target = target if target.is_absolute() else ROOT / target
            if target.exists():
                print(f"Using current source summary pointer: {value}", flush=True)
                return value
            print(f"Current source summary pointer target is missing, using fallback: {value}", flush=True)
    print(f"Using fallback source summary: {DEFAULT_SOURCE_SUMMARY}", flush=True)
    return DEFAULT_SOURCE_SUMMARY

if RUN_A100_ACTION and PREFER_TRAINING_SOURCE:
    mount_drive_for_paid_action()

SOURCE_SUMMARY = resolve_source_summary()

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
checkpoint_preflight = go_payload.get("checkpoint_preflight") or {}
checkpoint_available = bool(checkpoint_preflight.get("available"))
input_preflight = checkpoint_preflight.get("input_preflight") if isinstance(checkpoint_preflight, dict) else None
input_available = True if not input_preflight else bool(input_preflight.get("available"))
print("a100_guard_decision:", go_decision, flush=True)
print("a100_checkpoint_preflight:", checkpoint_preflight, flush=True)
if input_preflight:
    print("a100_input_preflight:", input_preflight, flush=True)
if not RUN_A100_ACTION and go_allowed:
    print(
        "DRY_RUN_GREEN: guarded action is currently allowed. Set "
        "STAGE5_CURRENT_A100_TARGET=safe_continue_execute only when you intentionally want to spend paid GPU.",
        flush=True,
    )
elif not RUN_A100_ACTION and not input_available:
    print(
        "DRY_RUN_RED: curriculum input artifacts are not visible locally or in the configured Drive backup. "
        "Run STAGE5_CURRENT_A100_TARGET=programmatic_curriculum_cpu on a CPU runtime first, or reauthorize Drive.",
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
            "tests/test_filter_mcq_sft_by_eval.py",
            "tests/test_mcq_debias.py",
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
env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] = str(MAX_NEXT_ACTIONS)
env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] = "1" if ALLOW_REPEAT_NEXT_ACTION else "0"
env["STAGE5_A100_BUDGET_PROFILE"] = A100_BUDGET_PROFILE

next_action_proc = run([sys.executable, "colab/run_stage5_next_action.py"], cwd=ROOT, env=env, check=False)
print(f"next_action_returncode={next_action_proc.returncode}", flush=True)

if not execute_action:
    print("Dry run complete. Set RUN_A100_ACTION = True only when you intentionally want to spend A100 credits.", flush=True)
elif next_action_proc.returncode == 0:
    print("Guarded next action completed or stopped by a100_guard. Review the emitted summary before continuing.", flush=True)
else:
    print(
        "Guarded next action returned nonzero after writing its summary/log. "
        "Review the emitted Stage 5 Next Action summary before continuing.",
        flush=True,
    )

disconnect_runtime("safe continue cell finished")
```
