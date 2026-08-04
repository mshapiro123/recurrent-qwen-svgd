"""Compute and publish Phase-2 oracle-selector headroom from banked private rows."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_oracle_selector_headroom_20260805"
AUDIT_ID = "stage5_paper2_phase2_matched_alpha_audit_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
PRIVATE_ROWS = DRIVE_ROOT / AUDIT_ID / "private/exact_abort_rows"
DRIVE_RECEIPTS = DRIVE_ROOT / RUN_ID / "receipts"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def markdown(summary: dict) -> str:
    lines = [
        "# Phase-2 Oracle-Selector Headroom Receipt",
        "",
        "CPU-only post-processing of banked DEV row tensors; no model inference or training.",
        "",
        "| Alpha | Seed | Always-on delta | Oracle delta | Safe-oracle delta | Selected |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        lines.append(
            "| {alpha:.1f} | {seed} | {always:.6f} | {oracle:.6f} | {safe:.6f} | {selected:.2%} |".format(
                alpha=arm["alpha"],
                seed=arm["seed"],
                always=arm["always_on_acceptance_delta"],
                oracle=arm["oracle_acceptance_delta"],
                safe=arm["quality_safe_oracle_acceptance_delta"],
                selected=arm["oracle_selected_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "This is a perfect-hindsight ceiling on cached teacher-forced accepted length, not a deployable selector result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not PRIVATE_ROWS.is_dir():
        raise FileNotFoundError(f"missing banked private rows: {PRIVATE_ROWS}")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_oracle_selector_headroom",
            "--private_dir",
            str(PRIVATE_ROWS),
            "--output_dir",
            str(RUN_DIR),
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    (RUN_DIR / "receipt.md").write_text(markdown(summary), encoding="utf-8")
    DRIVE_RECEIPTS.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.iterdir():
        if path.is_file():
            shutil.copy2(path, DRIVE_RECEIPTS / path.name)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 oracle selector headroom [skip ci]"])
        run(["git", "push", "origin", "main"])
    print("Phase-2 oracle-selector headroom landed.", flush=True)


if __name__ == "__main__":
    main()
