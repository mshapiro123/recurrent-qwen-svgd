"""L4/A100 launcher for the no-training DC0 depth-by-append diagnostic."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata

STAGE5_PAPER2_DC0_DEPTH_BY_APPEND_VERSION = "paper2_dc0_depth_by_append_v2"
# Safety marker: forward-only DC0 no optimizer no backward no model mutation no training
# Safety marker: in-place 1 through 4 raw RMS neutral append 0 through 3 and read-at-t query arm
# Safety marker: transient eviction exact counters and post-eviction position identity assertion
# Safety marker: append k1 matched compute comparator is in-place depth3
# Safety marker: EVAL-B read once and spent after this scoring pass
# Safety marker: bridge adaptation unauthorized
# Safety marker: registered k0 anchor cached path drift disclosed resumable batch ETA
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main")
GH = userdata.get("GH_TOKEN")
HF = userdata.get("HF_TOKEN")
assert GH and HF, "Missing GH_TOKEN or HF_TOKEN."
os.environ["HF_TOKEN"] = HF
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF


def run(command: list[str], cwd: Path | None = None, allowed: tuple[int, ...] = (0,)) -> int:
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
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code not in allowed:
        raise subprocess.CalledProcessError(code, command)
    return code


if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", force_remount=False, timeout_ms=240_000)
memory = int(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
    ).splitlines()[0]
)
assert memory >= 22000, f"DC0 requires an L4-class or larger GPU; observed {memory} MiB."
if memory >= 70000:
    os.environ.setdefault("STAGE5_DC0_APPEND_BATCH_SIZE", "24")
elif memory >= 35000:
    os.environ.setdefault("STAGE5_DC0_APPEND_BATCH_SIZE", "12")
else:
    os.environ.setdefault("STAGE5_DC0_APPEND_BATCH_SIZE", "8")
url = f"https://x-access-token:{GH}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", url])
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
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
        "tests/test_coconut_composite.py",
        "tests/test_paper2_dc0_eval_b.py",
    ]
)
code = run([sys.executable, "-u", "colab/run_stage5_paper2_dc0_depth_by_append.py"], allowed=(0, 2))
if code == 2:
    print("DC0 stopped at the preregistered baseline-validity check; receipts were published.")
