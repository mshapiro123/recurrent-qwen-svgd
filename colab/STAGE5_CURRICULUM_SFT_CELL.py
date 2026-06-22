import os, subprocess, sys
from pathlib import Path

from google.colab import drive, userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")

# Edit these before running if your completed curriculum shard uses a different path.
WORK_DIR = "data/curriculum/run_001"
MIN_POSITIVE_ROWS = "16"
MIN_MODE_ROWS = ""  # Optional, e.g. "direct=64,deep_narrow=64" or "wide=64".
PHASE1_STEPS = "150"
MAX_LOOPS = "4"
DISCONNECT_RUNTIME_WHEN_DONE = False


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


def redact(text):
    text = str(text)
    for token in [GH_TOKEN, HF_TOKEN]:
        if token:
            text = text.replace(token, "****")
    return text


def run(cmd, cwd=None, env=None, check=True):
    printable = redact(" ".join(map(str, cmd)))
    print("$", printable, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(redact(proc.stdout), flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc


clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT)
else:
    run(["git", "clone", clone_url, str(ROOT)])

run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

if HF_TOKEN:
    from huggingface_hub import HfApi, login

    login(token=HF_TOKEN, add_to_git_credential=False)
    who = HfApi(token=HF_TOKEN).whoami()
    print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user")

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive", force_remount=True)

run(["nvidia-smi"], cwd=ROOT, check=False)
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_stage5_curriculum_sft.py",
        "tests/test_curriculum_sft_gate.py",
        "tests/test_stage5_a100_go_no_go.py",
        "tests/test_stage5_next_action.py",
    ],
    cwd=ROOT,
)

env = os.environ.copy()
env.update(
    {
        "STAGE5_CURRICULUM_WORK_DIR": WORK_DIR,
        "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS": MIN_POSITIVE_ROWS,
        "STAGE5_CURRICULUM_MIN_MODE_ROWS": MIN_MODE_ROWS,
        "STAGE5_CURRICULUM_PHASE1_STEPS": PHASE1_STEPS,
        "STAGE5_CURRICULUM_MAX_LOOPS": MAX_LOOPS,
        "DTYPE": "bfloat16",
        "ADAPTER_DTYPE": "float32",
        "DEVICE": "cuda",
    }
)
run([sys.executable, "colab/run_stage5_curriculum_sft.py"], cwd=ROOT, env=env)

if DISCONNECT_RUNTIME_WHEN_DONE:
    from google.colab import runtime

    runtime.unassign()
