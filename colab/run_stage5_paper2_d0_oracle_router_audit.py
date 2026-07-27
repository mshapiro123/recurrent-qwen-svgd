"""Run and publish the CPU-only D0 oracle-router ceiling audit."""

# Safety marker: read-only calibration receipt no checkpoint no optimizer no training

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
from training.speculative_depth_d0_postlock import D0_RUN_ID


RUN_DIR = ROOT / "outputs" / "stage5" / D0_RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_D0_RUN_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{D0_RUN_ID}",
    )
)


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
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    run(["git", "add", "-f", "--", *relative])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        raise RuntimeError("D0 oracle-router audit produced no aggregate receipt changes")
    run(["git", "commit", "-m", "Record Paper Two D0 oracle router audit [skip ci]"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(["git", "push", "origin", "main"])
    return commit


def main() -> int:
    floor_summary_path = RUN_DIR / "floor" / "summary.json"
    floor_summary = read_json(floor_summary_path)
    if floor_summary.get("status") != "complete" or floor_summary.get("training_started") is not False:
        raise RuntimeError("D0 oracle-router audit requires the completed pre-training floor")
    private_rows = DRIVE_ROOT / "private" / "floor" / "floor_rows.json"
    print(f"oracle_router_private_rows={private_rows} exists={private_rows.exists()}", flush=True)
    if not private_rows.exists():
        raise FileNotFoundError(f"D0 floor private rows are missing: {private_rows}")
    observed = sha256_file(private_rows)
    expected = str(floor_summary["private_rows_sha256"])
    print(f"oracle_router_private_sha observed={observed} expected={expected}", flush=True)
    if observed != expected:
        raise RuntimeError("D0 floor private rows hash mismatch")

    output_dir = RUN_DIR / "router_oracle_audit"
    summary_path = output_dir / "summary.json"
    run(
        [
            sys.executable,
            "eval/eval_speculative_depth_router_feasibility.py",
            "--floor_private_rows",
            str(private_rows),
            "--output_summary",
            str(summary_path),
            "--expected_private_rows_sha256",
            observed,
        ]
    )
    summary = read_json(summary_path)
    seven = summary["teacher_7b"]
    markdown = output_dir / "summary.md"
    markdown.write_text(
        "# Paper Two D0 Oracle Router Audit\n\n"
        f"- Positions: `{seven['positions']}`\n"
        f"- Best fixed depth: `{seven['best_fixed_depth']}`\n"
        f"- Best fixed accuracy: `{seven['best_fixed_accuracy']:.6f}`\n"
        f"- Oracle any-depth accuracy: `{seven['oracle_any_depth']['accuracy']:.6f}`\n"
        f"- Oracle uplift over best fixed: `{seven['oracle_any_depth']['uplift_over_best_fixed']:.6f}`\n"
        f"- Oracle mean loops: `{seven['oracle_any_depth']['mean_loops_first_correct_else_one']:.6f}`\n"
        "- Evaluation partition touched: `false`\n"
        "- Training: `none`\n",
        encoding="utf-8",
    )
    commit = publish([summary_path, markdown])
    receipt = DRIVE_ROOT / "receipts" / "oracle_router_audit_summary.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
