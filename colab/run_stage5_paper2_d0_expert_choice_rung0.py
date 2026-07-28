"""Restore and publish the CPU-only D0 expert-choice Rung 0 receipt."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_d0_expert_choice_rung0_20260728"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
D0 = ROOT / "outputs/stage5/stage5_paper2_d0_20260726"
D1 = ROOT / "outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727"
DRIVE_D0 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d0_20260726")
DRIVE_D1 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d1_causal_allocation_audit_20260727")
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record D0 expert-choice Rung 0 [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    feature_candidates = [
        DRIVE_D1 / "private/evaluation_feature_cache.pt",
        DRIVE_D1 / "private/evaluation/evaluation_feature_cache.pt",
    ]
    feature = next((path for path in feature_candidates if path.exists()), None)
    floor = DRIVE_D0 / "private/floor/floor_rows.json"
    if feature is None:
        raise FileNotFoundError(f"missing D1 private feature cache: {feature_candidates}")
    if not floor.exists():
        raise FileNotFoundError(f"missing pre-D0 floor rows: {floor}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "eval/rescore_d0_expert_choice.py",
            "--feature_cache",
            str(feature),
            "--audit_summary",
            str(D1 / "summary.json"),
            "--floor_private_rows",
            str(floor),
            "--output_summary",
            str(RUN_DIR / "summary.json"),
        ]
    )
    paths = [RUN_DIR / name for name in ("summary.json", "summary.md", "summary.png", "summary.svg")]
    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path, DRIVE_RUN / path.name)
    commit = publish(paths)
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
