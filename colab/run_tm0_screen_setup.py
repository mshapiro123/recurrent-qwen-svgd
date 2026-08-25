"""Set up the CPU-only TM-0 hermetic screening job."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "6ea8af056abe307000bcac24ced69f6838cc01e4"
ROOT = Path("/content/recurrent-qwen-svgd")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = ["****" if "x-access-token:" in value else value for value in command]
    print("$", " ".join(printable), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("TM-0 screen setup requires HF_TOKEN in the attached kernel")
if not ROOT.exists():
    run(["git", "clone", f"https://github.com/{REPO}.git", str(ROOT)])
run(["git", "fetch", "origin", REF], cwd=ROOT)
run(["git", "reset", "--hard", REF], cwd=ROOT)
run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
run(["python", "-m", "pytest", "-q", "tests/test_paper2_tm0.py"], cwd=ROOT)
Path("/content/tm0_inputs").mkdir(parents=True, exist_ok=True)
Path("/content/tm0_results").mkdir(parents=True, exist_ok=True)
print("tm0_screen_setup_complete=true", flush=True)
