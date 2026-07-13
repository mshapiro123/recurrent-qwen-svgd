"""Colab cell: continue the inverse-table substrate through caps 3 and 4."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_INVERSE_TABLE_REBASE_CELL_VERSION = "inverse_table_rebase_caps3_4_v1"
# Safety marker: inverse_table_rebase_caps3_4
# Safety marker: SOURCE_CAP2_SHA256
# Safety marker: tests/test_stage5_inverse_table_rebase.py
# Safety marker: rebase_caps3_4_green_pending_review
# Safety marker: accepted_returncodes={0, 2}

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


def secret(*names: str) -> str | None:
    for name in names:
        try:
            value = os.environ.get(name) or userdata.get(name)
        except Exception:
            value = os.environ.get(name)
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)


def redact(text: str) -> str:
    safe = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            safe = safe.replace(token, "****")
    return safe


def run(cmd: list[str], *, cwd: Path = ROOT, env=None, accepted_returncodes={0}) -> int:
    print("$", redact(" ".join(cmd)), flush=True)
    process = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        tail = (tail + [safe])[-240:]
    returncode = process.wait()
    if returncode not in accepted_returncodes:
        print("FAILED_COMMAND_TAIL_START\n" + "".join(tail) + "FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)
    return returncode


def main() -> None:
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_stage5_inverse_composition_staircase.py", "tests/test_stage5_inverse_table_rebase.py", "tests/test_eval_abductive_staircase.py", "tests/test_train_unfrozen_recurrent.py"])
    env = os.environ.copy()
    env.setdefault("STAGE5_REBASE_RUN_ID", "stage5_inverse_table_rebase_caps3_4_20260713")
    env.setdefault("STAGE5_REBASE_SOURCE_SUMMARY", "outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json")
    env.setdefault("STAGE5_STAIRCASE_DTYPE", "bfloat16")
    env.setdefault("DEVICE", "cuda")
    code = run([sys.executable, "colab/run_stage5_inverse_table_rebase.py"], env=env, accepted_returncodes={0, 2})
    if code == 2:
        print("Inverse-table rebase reached its preregistered scientific stop.", flush=True)
    if os.environ.get("STAGE5_REBASE_DISCONNECT", "0").lower() in {"1", "true", "yes", "y"}:
        runtime.unassign()
    else:
        print("Leaving runtime connected for rebase review.", flush=True)


try:
    main()
except Exception:
    print("Inverse-table rebase errored; leaving runtime connected.", flush=True)
    raise
