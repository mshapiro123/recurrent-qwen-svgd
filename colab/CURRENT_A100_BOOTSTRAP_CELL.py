import base64, json, os, time, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"

# Safe default: verify Drive/checkpoint visibility on a CPU/cheap runtime.
# Other options:
#   "programmatic_curriculum_cpu" - generate/publish the direct/deep curriculum gate on CPU.
#   "safe_continue_dry_run" - fetch safe-continue but do not spend GPU.
#   "safe_continue_execute" - fetch safe-continue and opt in to the guarded paid action.
TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "preflight")
SOURCE_SUMMARY_OVERRIDE = os.environ.get("STAGE5_CURRENT_A100_SOURCE_SUMMARY", "").strip()

TARGETS = {
    "preflight": {
        "path": "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py",
        "markers": [
            "stage5_drive_checkpoint_preflight",
            "checkpoint_preflight",
            "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "drive.mount",
            "runtime.unassign",
            "colab/check_stage5_a100_go_no_go.py",
            "stage5_current_source_summary.txt",
            "Using current source summary pointer",
        ],
        "env": {},
    },
    "safe_continue_dry_run": {
        "path": "colab/STAGE5_SAFE_CONTINUE_CELL.py",
        "markers": [
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "RUN_A100_ACTION",
            "colab/check_stage5_a100_go_no_go.py",
            "colab/run_stage5_next_action.py",
            "Skipping requirements install because no paid action will execute.",
        ],
        "env": {"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "0"},
    },
    "programmatic_curriculum_cpu": {
        "path": "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py",
        "markers": [
            "REFUSE_GPU_RUNTIME",
            "ALLOW_GPU_RUNTIME_FOR_CPU_WORK",
            "training/run_programmatic_curriculum_pipeline.py",
            "training/check_curriculum_sft_gate.py",
            "colab/publish_stage5_curriculum_gate.py",
            "REQUIRE_DRIVE_BACKUP_FOR_PUBLISH",
            "PUBLISH_GATE_TO_GITHUB",
            "stage5_current_source_summary",
            "Refusing to run CPU-only programmatic curriculum generation",
        ],
        "env": {},
    },
    "safe_continue_execute": {
        "path": "colab/STAGE5_SAFE_CONTINUE_CELL.py",
        "markers": [
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "RUN_A100_ACTION",
            "mount_drive_for_paid_action",
            "tests/test_stage5_routing_repair.py",
            "colab/run_stage5_next_action.py",
        ],
        "env": {"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "1"},
    },
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

if TARGET not in TARGETS:
    raise AssertionError(f"Unknown TARGET={TARGET!r}; expected one of {sorted(TARGETS)}")

GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."


def github_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


ref_payload = github_json(
    f"https://api.github.com/repos/{REPO}/git/ref/heads/{REF}?cache_bust={int(time.time())}"
)
RESOLVED_REF = ((ref_payload.get("object") or {}).get("sha") or REF).strip()

selected = TARGETS[TARGET]
if SOURCE_SUMMARY_OVERRIDE:
    os.environ["STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
else:
    # Avoid accidentally pinning a new session to an old target-specific source
    # summary. The safe-continue launcher will follow config/stage5_current_source_summary.txt.
    os.environ.pop("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY", None)
for key, value in selected["env"].items():
    os.environ[key] = value
os.environ.setdefault("STAGE5_SAFE_CONTINUE_DISCONNECT", "1")

launcher_path = selected["path"]
url = f"https://api.github.com/repos/{REPO}/contents/{launcher_path}?ref={RESOLVED_REF}&cache_bust={int(time.time())}"
payload = github_json(url)

code = base64.b64decode(payload["content"]).decode("utf-8")
missing = [marker for marker in selected["markers"] if marker not in code]
assert not missing, f"Fetched launcher is missing expected safety markers: {missing}"

print(f"Fetched {launcher_path} from {REPO}@{REF} ({RESOLVED_REF[:12]}) sha={payload.get('sha')} target={TARGET}", flush=True)
exec(compile(code, launcher_path, "exec"))
