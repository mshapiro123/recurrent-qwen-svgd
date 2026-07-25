"""Run and publish the bounded COCONUT RG-4/RG-11 numerical follow-up."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get(
    "STAGE5_COCONUT_NUMERICS_RUN_ID",
    "stage5_coconut_composite_numerics_20260725",
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE = ROOT / "outputs/stage5/stage5_coconut_composite_rg1_rg11_20260725/summary.json"


def run(command: list[str], *, accepted: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode not in accepted:
        raise subprocess.CalledProcessError(result.returncode, command)
    return int(result.returncode)


def publish() -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in sorted(RUN_DIR.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".md", ".log"}:
            subprocess.run(["git", "add", "-f", path.relative_to(ROOT).as_posix()], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", f"Record COCONUT numerical follow-up {RUN_ID} [skip ci]"])
        if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
            run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
            run(["git", "push", "origin", "main"])


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(f"missing landed COCONUT preflight: {SOURCE}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("status") != "failed_integrity_contract":
        raise RuntimeError("COCONUT numerical follow-up source verdict drifted")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    code = run(
        [
            sys.executable,
            "eval/eval_coconut_composite_numerics.py",
            "--output_dir",
            str(RUN_DIR.relative_to(ROOT)),
            "--device",
            "cuda",
        ],
        accepted=(0, 2),
    )
    summary_path = RUN_DIR / "summary.json"
    if not summary_path.exists():
        raise AssertionError("COCONUT numerical follow-up exited without summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("training_performed") is not False or summary.get("checkpoint_written") is not False:
        raise AssertionError("COCONUT numerical follow-up must be read-only")
    if summary.get("rg12", {}).get("authorized") is not False or summary.get("rg12", {}).get("run") is not False:
        raise AssertionError("COCONUT numerical follow-up cannot authorize RG-12")
    publish()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

