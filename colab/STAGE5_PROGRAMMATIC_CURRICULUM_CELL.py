import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import drive, runtime, userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")

# CPU-only direct/deep shard. Run this before any generated-curriculum GPU SFT.
WORK_DIR = "data/curriculum/programmatic_direct_deep_001"
NUM_DIRECT = "1000"
NUM_DEEP_NARROW = "1000"
DIRECT_STEPS = "1,2"
DEEP_STEPS = "5,9"
SEED = "17"
MAX_TARGET_LOOPS = "4"

MIN_POSITIVE_ROWS = str(int(NUM_DIRECT) + int(NUM_DEEP_NARROW))
MIN_MODE_ROWS = f"direct={NUM_DIRECT},deep_narrow={NUM_DEEP_NARROW}"

MOUNT_DRIVE = True
BACKUP_TO_DRIVE = True
DRIVE_BACKUP_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd/curriculum_runs")
REQUIRE_DRIVE_BACKUP_FOR_PUBLISH = True
PUBLISH_GATE_TO_GITHUB = True
PUBLISHED_GATE_DIR = "outputs/stage5/programmatic_direct_deep_curriculum_gate"
REFUSE_GPU_RUNTIME = True
ALLOW_GPU_RUNTIME_FOR_CPU_WORK = False
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
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."


def redact(text):
    return str(text).replace(GH_TOKEN, "****") if GH_TOKEN else str(text)


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


def attached_gpu_names():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def refuse_gpu_runtime_for_cpu_work():
    gpus = attached_gpu_names()
    if REFUSE_GPU_RUNTIME and gpus and not ALLOW_GPU_RUNTIME_FOR_CPU_WORK:
        raise RuntimeError(
            "Refusing to run CPU-only programmatic curriculum generation on an attached GPU runtime: "
            + "; ".join(gpus)
            + ". Switch to CPU runtime, or set ALLOW_GPU_RUNTIME_FOR_CPU_WORK=True deliberately."
        )
    if gpus:
        print("GPU attached but CPU work override is enabled:", gpus, flush=True)
    else:
        print("No GPU attached; good for CPU curriculum generation.", flush=True)


def backup_work_dir():
    if not BACKUP_TO_DRIVE:
        return None
    if not Path("/content/drive/MyDrive").exists():
        print("Drive not mounted; skipping backup.", flush=True)
        return None
    src = ROOT / WORK_DIR
    dst = DRIVE_BACKUP_ROOT / Path(WORK_DIR).name
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"backed_up_work_dir={dst}", flush=True)
        return dst
    return None


refuse_gpu_runtime_for_cpu_work()

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

if MOUNT_DRIVE and not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive", force_remount=True)

run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_programmatic_curriculum_pipeline.py",
        "tests/test_curriculum_sft_gate.py",
    ],
    cwd=ROOT,
)

run(
    [
        sys.executable,
        "training/run_programmatic_curriculum_pipeline.py",
        "--work_dir",
        WORK_DIR,
        "--num_direct",
        NUM_DIRECT,
        "--num_deep_narrow",
        NUM_DEEP_NARROW,
        "--direct_steps",
        DIRECT_STEPS,
        "--deep_steps",
        DEEP_STEPS,
        "--seed",
        SEED,
        "--max_target_loops",
        MAX_TARGET_LOOPS,
    ],
    cwd=ROOT,
)

run(
    [
        sys.executable,
        "training/check_curriculum_sft_gate.py",
        "--work_dir",
        WORK_DIR,
        "--output_json",
        str(Path(WORK_DIR) / "curriculum_sft_gate.json"),
        "--output_md",
        str(Path(WORK_DIR) / "curriculum_sft_gate.md"),
        "--min_positive_rows",
        MIN_POSITIVE_ROWS,
        "--min_mode_rows",
        MIN_MODE_ROWS,
        "--max_loop_target",
        MAX_TARGET_LOOPS,
        "--fail_on_no_go",
    ],
    cwd=ROOT,
)

backup_path = backup_work_dir()

if PUBLISH_GATE_TO_GITHUB:
    if REQUIRE_DRIVE_BACKUP_FOR_PUBLISH and backup_path is None:
        raise RuntimeError(
            "Refusing to publish the green curriculum gate without a Drive backup of the work directory."
        )
    run(
        [
            sys.executable,
            "colab/publish_stage5_curriculum_gate.py",
            "--gate_json",
            str(Path(WORK_DIR) / "curriculum_sft_gate.json"),
            "--gate_md",
            str(Path(WORK_DIR) / "curriculum_sft_gate.md"),
            "--published_dir",
            PUBLISHED_GATE_DIR,
            "--push",
        ],
        cwd=ROOT,
    )

print("Programmatic direct/deep curriculum shard is gate-green.", flush=True)
print("WORK_DIR =", WORK_DIR, flush=True)
print("MIN_POSITIVE_ROWS =", MIN_POSITIVE_ROWS, flush=True)
print("MIN_MODE_ROWS =", MIN_MODE_ROWS, flush=True)
print("Published current-source gate =", PUBLISHED_GATE_DIR + "/curriculum_sft_gate.json", flush=True)
print("Next GPU step, only after choosing an A100 runtime:", flush=True)
print("  run colab/CURRENT_A100_BOOTSTRAP_CELL.py with STAGE5_CURRENT_A100_TARGET=safe_continue_execute", flush=True)

if DISCONNECT_RUNTIME_WHEN_DONE:
    runtime.unassign()
