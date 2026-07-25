"""Run, validate, and publish the no-training COCONUT composite battery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get(
    "STAGE5_COCONUT_PREFLIGHT_RUN_ID",
    "stage5_coconut_composite_rg1_rg11_20260725",
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID


def run(command: list[str], *, accepted: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode not in accepted:
        raise subprocess.CalledProcessError(result.returncode, command)
    return int(result.returncode)


def publish() -> None:
    subprocess.run(
        ["git", "pull", "--rebase", "--autostash", "origin", "main"],
        cwd=ROOT,
        check=False,
    )
    for path in sorted(RUN_DIR.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".md", ".log"}:
            subprocess.run(
                ["git", "add", "-f", path.relative_to(ROOT).as_posix()],
                cwd=ROOT,
                check=True,
            )
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", f"Record COCONUT composite preflight {RUN_ID} [skip ci]"])
        if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
            run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
            run(["git", "push", "origin", "main"])


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    exit_code = run(
        [
            sys.executable,
            "eval/eval_coconut_composite_integrity.py",
            "--output_dir",
            str(RUN_DIR.relative_to(ROOT)),
            "--device",
            "cuda",
        ],
        accepted=(0, 2),
    )
    summary_path = RUN_DIR / "summary.json"
    if not summary_path.exists():
        raise AssertionError("Composite preflight exited without a summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("training_performed") is not False:
        raise AssertionError("Composite integrity preflight must not train")
    if summary.get("rg12", {}).get("run") is not False:
        raise AssertionError("RG-12 is not authorized by this launcher")
    publish()
    if exit_code == 2 or summary.get("status") != "passed_rg1_through_rg11":
        print("Composite battery landed a red contract for diagnosis.", flush=True)
        return 2
    print("Composite RG-1 through RG-11 passed; RG-12 remains unrun.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
