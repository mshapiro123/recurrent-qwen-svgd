"""Colab launcher for the single bounded inverse-rendered N24 continuation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_INVERSE_RENDERED_CONTINUATION_CELL_VERSION = "inverse_rendered_n24_continuation_v1"
# Safety marker: inverse_rendered_n24_continuation
# Safety marker: colab/run_stage5_inverse_rendered_continuation.py
# Safety marker: tests/test_stage5_inverse_rendered_continuation.py
# Safety marker: forward_rehearsal_fraction
# Safety marker: bounded_tune_review_required

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
PINNED_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


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


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(redact(line), end="", flush=True)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, cmd)


def main() -> None:
    run(["nvidia-smi"], cwd=Path("/content"))
    drive.mount("/content/drive", force_remount=False)
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
    else:
        run(["git", "clone", clone_url, str(ROOT)], cwd=Path("/content"))
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
    is_sha = len(PINNED_REF) == 40 and all(char in "0123456789abcdefABCDEF" for char in PINNED_REF)
    resolved = PINNED_REF if is_sha else "origin/main"
    run(["git", "reset", "--hard", resolved])
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_stage5_inverse_rendered_width_gate.py", "tests/test_stage5_inverse_rendered_continuation.py"])
    env = os.environ.copy()
    env.setdefault("STAGE5_INVERSE_RENDERED_CONTINUATION_RUN_ID", "stage5_inverse_rendered_n24_continuation_20260715")
    env.setdefault("STAGE5_INVERSE_RENDERED_CONTINUATION_DTYPE", "bfloat16")
    env.setdefault("DEVICE", "cuda")
    run([sys.executable, "colab/run_stage5_inverse_rendered_continuation.py"], env=env)
    if os.environ.get("STAGE5_INVERSE_RENDERED_CONTINUATION_DISCONNECT", "1").lower() in {"1", "true", "yes"}:
        runtime.unassign()


try:
    main()
except Exception:
    print("Inverse-rendered N24 continuation errored; leaving runtime connected.", flush=True)
    raise
