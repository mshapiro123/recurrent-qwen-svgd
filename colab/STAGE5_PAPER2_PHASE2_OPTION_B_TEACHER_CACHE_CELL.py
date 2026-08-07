"""A100-80GB launcher for the locked Option B fresh-anchor teacher/cache pass."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER2_PHASE2_OPTION_B_TEACHER_CACHE_VERSION = (
    "paper2_phase2_option_b_teacher_cache_v5"
)
# Safety marker: locked fresh documents target 140000 floor 100000 anchors
# Safety marker: all-admitted-anchor 14B states and per-anchor label-tier admission
# Safety marker: A100 40GB uses pinned bf16 32B Accelerate offload on CUDA
# Safety marker: A100 40GB storage profile total 200 GiB free 150 GiB
# Safety marker: A100 40GB launch is preflight-only before full cache authorization
# Safety marker: derived exclusion receipts require hash-closed source JSONL lineage
# Safety marker: A100 80GB remains fully resident sequential model loads
# Safety marker: teacher cache only no model optimizer no training
# Safety marker: no optimizer no training
# Safety marker: tests/test_paper2_phase2_option_b_teacher_cache.py
# Safety marker: colab/run_stage5_paper2_phase2_option_b_teacher_cache.py
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH = secret("GH_TOKEN", "GITHUB_TOKEN")
HF = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH and HF, "Missing GH_TOKEN or HF_TOKEN in Colab secrets."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command).replace(GH, "****"), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd or ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code:
        print("\nOption B launcher tail:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
assert shutil.which("nvidia-smi"), "Attach an A100-SXM4-80GB and rerun."
run(["nvidia-smi"], Path("/content"))
memory = max(
    int(value.strip())
    for value in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if value.strip()
)
assert memory >= 38000, f"Option B requires an A100 40GB or larger; observed {memory} MiB."
if memory < 70000:
    os.environ["STAGE5_PHASE2_OPTION_B_OFFLOAD_32B"] = "1"
    os.environ["STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_TOTAL_GIB"] = "200"
    os.environ["STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_FREE_GIB"] = "150"
    os.environ["STAGE5_PHASE2_OPTION_B_PREFLIGHT_ONLY"] = "1"
    print(
        "hardware_mode=a100_40gb_32b_accelerate_cpu_disk_offload_cuda_execution",
        flush=True,
    )
else:
    os.environ["STAGE5_PHASE2_OPTION_B_OFFLOAD_32B"] = "0"
    os.environ["STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_TOTAL_GIB"] = "300"
    os.environ["STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_FREE_GIB"] = "250"
    os.environ["STAGE5_PHASE2_OPTION_B_PREFLIGHT_ONLY"] = "0"
    print("hardware_mode=a100_80gb_fully_resident", flush=True)
listing = subprocess.check_output(
    ["df", "-B1", "--output=target,size,avail"], text=True
)
minimum_total = int(os.environ["STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_TOTAL_GIB"]) * 1024**3
minimum_free = int(os.environ["STAGE5_PHASE2_OPTION_B_MIN_SCRATCH_FREE_GIB"]) * 1024**3
scratch_ok = False
for line in listing.splitlines()[1:]:
    fields = line.split()
    if len(fields) != 3:
        continue
    target, size_text, free_text = fields
    try:
        size = int(size_text)
        free = int(free_text)
    except ValueError:
        continue
    if (
        size >= minimum_total
        and free >= minimum_free
        and not target.startswith("/content/drive")
    ):
        scratch_ok = True
        break
assert scratch_ok, (
    "No local disk satisfies the active Option B profile: "
    f"total>={minimum_total / 1024**3:.0f} GiB, "
    f"free>={minimum_free / 1024**3:.0f} GiB."
)

url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
else:
    run(["git", "clone", url, str(ROOT)], Path("/content"))
run(["git", "reset", "--hard", REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_paper2_phase2_option_b_lock.py",
        "tests/test_paper2_phase2_option_b_teacher_cache.py",
    ]
)
run([sys.executable, "-u", "colab/run_stage5_paper2_phase2_option_b_teacher_cache.py"])
print("Option B teacher/cache receipt landed; no training occurred.", flush=True)
