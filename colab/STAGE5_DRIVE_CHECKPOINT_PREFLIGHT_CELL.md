# Stage 5 Drive / Checkpoint Preflight Cell

Run this in a CPU or cheap GPU Colab runtime before attaching an A100/H100.
It authorizes Google Drive, clones the private GitHub repo, checks that the
current recovered deterministic Phase 1 checkpoint is visible, runs the
A100 go/no-go guard, and disconnects. It does **not** train.

Use this when `colab/STAGE5_SAFE_CONTINUE_CELL.md` reports
`routing_checkpoint_missing_no_go`, or whenever the runtime has reset and
Drive may need reauthorization.
Set `STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY`, or the shared
`STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY`, to check a newer Stage 5 summary without
editing this cell.

```python
import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import drive, runtime, userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json"
SOURCE_SUMMARY_OVERRIDE = os.environ.get(
    "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY",
    os.environ.get("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", ""),
).strip()
GO_NO_GO_RUN_ID = "stage5_drive_checkpoint_preflight"
NEXT_ACTION_RUN_ID = "stage5_drive_checkpoint_preflight_next_action"
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

def resolve_source_summary():
    if SOURCE_SUMMARY_OVERRIDE:
        print(f"Using explicit Drive preflight source summary: {SOURCE_SUMMARY_OVERRIDE}", flush=True)
        return SOURCE_SUMMARY_OVERRIDE
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

SOURCE_SUMMARY = resolve_source_summary()

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

next_env = os.environ.copy()
next_env["STAGE5_ARC_AGI_NEXT_ACTION_RUN_ID"] = NEXT_ACTION_RUN_ID
next_env["STAGE5_ARC_AGI_NEXT_ACTION_SOURCE_SUMMARY"] = SOURCE_SUMMARY
next_env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "0"
next_env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] = "1"
run(
    [
        sys.executable,
        "colab/run_stage5_next_action.py",
    ],
    cwd=ROOT,
    env=next_env,
)

summary_path = ROOT / "outputs" / "stage5" / GO_NO_GO_RUN_ID / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
next_summary_path = ROOT / "outputs" / "stage5" / NEXT_ACTION_RUN_ID / "summary.json"
next_summary = json.loads(next_summary_path.read_text(encoding="utf-8"))
next_step = (next_summary.get("steps") or [{}])[0]
next_guard = next_step.get("a100_guard") or {}
print("decision:", summary["decision"], flush=True)
print("checkpoint_preflight:", summary["checkpoint_preflight"], flush=True)
print("next_action_guard:", next_guard, flush=True)
decision_go = bool((summary.get("decision") or {}).get("go"))
checkpoint_available = bool((summary.get("checkpoint_preflight") or {}).get("available"))
next_allowed = bool(next_guard.get("allowed"))
if decision_go and checkpoint_available and next_allowed:
    print(
        "PREFLIGHT_GREEN: checkpoint is visible and both guarded dry-runs are allowed. "
        "Reconnect with an A100/H100 and run this bootstrap with "
        "STAGE5_CURRENT_A100_TARGET=safe_continue_execute.",
        flush=True,
    )
elif not checkpoint_available:
    print(
        "PREFLIGHT_RED: checkpoint is not visible. Do not attach a paid GPU yet; "
        "reauthorize Drive or fix the Drive backup path, then rerun preflight.",
        flush=True,
    )
else:
    print(
        "PREFLIGHT_BLOCKED: checkpoint is visible but a guard blocked the paid action. "
        "Inspect the printed decision and next_action_guard before spending GPU.",
        flush=True,
    )

if DISCONNECT_RUNTIME_WHEN_DONE:
    print("Disconnecting preflight runtime; reconnect with A100 only after checkpoint_preflight.available is True.", flush=True)
    runtime.unassign()
```
