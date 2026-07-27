"""Build and publish the zero-GPU D0 prelaunch receipts."""

# Safety marker: read-only prelaunch post-processing no model no optimizer no evaluation partition

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import D0_RUN_ID, validate_cache_summary


LOCK_RUN = ROOT / "outputs/stage5/stage5_paper2_d0_preregistration_20260726"
RUN_DIR = ROOT / "outputs" / "stage5" / D0_RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_RUN_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{D0_RUN_ID}",
    )
)
ADDENDUM = ROOT / "docs/STRATEGY_ADDENDUM_D0_FIGURE_REVIEW_20260727.md"
ADDENDUM_SHA256 = "ff93c5011872e91d64dcdc380169b93beedd3097b6cee2987722d28af06a36b8"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
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
    if code:
        raise subprocess.CalledProcessError(code, command)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 prelaunch produced no aggregate receipt changes")
    run(["git", "commit", "-m", "Record Paper Two D0 prelaunch receipts [skip ci]"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return commit


def main() -> int:
    if sha256_file(ADDENDUM) != ADDENDUM_SHA256:
        raise RuntimeError("D0 figure-review addendum does not match the authenticated Drive copy")
    cache_summary_path = RUN_DIR / "labeling" / "summary.json"
    floor_summary_path = RUN_DIR / "floor" / "summary.json"
    cache_summary = read_json(cache_summary_path)
    validate_cache_summary(cache_summary)
    floor = read_json(floor_summary_path)
    if floor.get("status") != "complete" or floor.get("training_started") is not False:
        raise RuntimeError("D0 prelaunch requires the completed pre-training floor")

    manifest = read_json(LOCK_RUN / "data_manifest.json")
    label_train = Path(manifest["artifacts"]["label_train"]["drive_path"])
    if not label_train.exists() or sha256_file(label_train) != manifest["artifacts"]["label_train"]["sha256"]:
        raise RuntimeError("D0 locked label-train partition is missing or corrupt")
    floor_resume_dir = DRIVE_ROOT / "private" / "floor" / "row_cache_pretraining"
    if not floor_resume_dir.exists():
        raise FileNotFoundError(f"D0 pretraining floor row cache is missing: {floor_resume_dir}")

    output_dir = RUN_DIR / "prelaunch"
    summary_path = output_dir / "summary.json"
    target_path = output_dir / "target_policy_receipt.json"
    run(
        [
            sys.executable,
            "eval/build_speculative_depth_d0_prelaunch.py",
            "--cache_summary",
            str(cache_summary_path),
            "--floor_summary",
            str(floor_summary_path),
            "--floor_resume_dir",
            str(floor_resume_dir),
            "--label_train_jsonl",
            str(label_train),
            "--output_summary",
            str(summary_path),
            "--target_policy_output",
            str(target_path),
        ]
    )
    summary = read_json(summary_path)
    target = read_json(target_path)
    if summary.get("status") != "complete" or target.get("status") != "verified_before_training":
        raise RuntimeError("D0 prelaunch receipts did not complete")
    seven = summary["teacher_demand"]["teacher_7b_own_rejections"]
    fourteen = summary["teacher_demand"]["teacher_14b_own_rejections"]
    overlap = summary["teacher_demand"]["teacher_overlap_on_7b_rejections"]
    cached_overlap = summary["teacher_demand"]["cached_7b_rejection_overlap"]
    markdown = output_dir / "summary.md"
    markdown.write_text(
        "# Paper Two D0 Prelaunch Receipts\n\n"
        f"- Target table: `{target['registered_target_table']}`\n"
        f"- Scheduled target counts: `{target['target_depth_counts']}`\n"
        f"- 7B own-rejection median first-correct depth: `{seven['median_first_correct_depth_recoverable']}`\n"
        f"- 14B own-rejection median first-correct depth: `{fourteen['median_first_correct_depth_recoverable']}`\n"
        f"- 14B endorsement share on floor-defined 7B loop-1 rejections: `{overlap['share']}`\n"
        f"- 14B endorsement share on cached 7B rejection labels: `{cached_overlap['share']}`\n"
        "- Evaluation partition touched: `false`\n"
        "- Model loaded: `false`\n"
        "- Optimizer steps: `0`\n",
        encoding="utf-8",
    )
    amendment = output_dir / "teacher_shift_amendment.md"
    amendment.write_text(
        "# D0 Teacher-Shift Measurement Amendment Receipt\n\n"
        "The primary statistic is the median first-correct depth among recoverable positions, "
        "with the full distribution reported for each teacher on that teacher's own loop-1 "
        "rejection population. The 14B-on-7B-rejections result is descriptive teacher overlap.\n\n"
        "This clarification was made after the floor data landed and before any trained model exists. "
        "The floor layer is therefore partially observed at amendment time (the 7B distribution was "
        "visible; the 14B distribution had not been computed). The trained-model layer, which carries "
        "the registered test, is untouched. The trained-model comparison carries one bias both "
        "directions share and the receipt must state: binary depth-2 targets compress trained demand "
        "toward 2 for both teachers, so the trained layer measures demand under the trained policy's "
        "regime, and the floor layer is the less confounded of the two.\n",
        encoding="utf-8",
    )
    receipt_dir = DRIVE_ROOT / "receipts" / "prelaunch"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for path in (summary_path, target_path, markdown, amendment):
        (receipt_dir / path.name).write_bytes(path.read_bytes())
    commit = publish([summary_path, target_path, markdown, amendment])
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
