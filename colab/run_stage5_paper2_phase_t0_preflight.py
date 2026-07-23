"""Record and publish the executable Paper Two Phase T0 preflight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get(
    "STAGE5_PAPER2_T0_RUN_ID",
    "stage5_paper2_internal_token_t0_preflight_20260722",
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


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
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"Record Paper Two T0 preflight {RUN_ID} [skip ci]",
            ],
            cwd=ROOT,
            check=True,
        )
        if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
            run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
            run(["git", "push", "origin", "main"])


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "colab/run_paper2_phase_t0_preflight.py",
            "--output_dir",
            str(RUN_DIR.relative_to(ROOT)),
            "--dtype",
            os.environ.get("STAGE5_PAPER2_T0_DTYPE", "bfloat16"),
            "--device",
            "cuda",
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "passed_all_five_contracts":
        raise AssertionError("Paper Two T0 did not pass all five contracts")
    if summary.get("training_performed") is not False:
        raise AssertionError("Paper Two T0 must not perform training")
    publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
