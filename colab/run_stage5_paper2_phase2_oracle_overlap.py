"""Publish the cache-only Phase-2 oracle and hurt-overlap receipt."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402


RUN_ID = "stage5_paper2_phase2_prewindow_20260731"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")
STAGE_A_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_stage_a_20260730"
)
TRAINED_SHA = "5f2f2d89d26642e16c0e4640ea01fa79a408c25bdf71794e6235948ed96ce0cb"
IMMUTABLE_CACHE_SHA = "bd40c743b1f9ec28c44a9f4d49c483e888bcb2b66de746c2277ffc9583f56502"


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


def publish(path: Path) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", path.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 oracle overlap receipt [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Phase-2 oracle prerequisite missing: {path}")
    return path


def main() -> int:
    private_eval = STAGE_A_DRIVE / "private/eval_c"
    private_result = STAGE_A_DRIVE / "private/eval_c_result"
    data = require(private_eval / "eval_c.jsonl")
    teacher = require(private_eval / "teacher_cache_summary.json")
    immutable_receipt = require(
        private_result / "immutable_scoring_cache_receipt.json"
    )
    inplace = require(private_result / "arm_batches/inplace_locked_init")
    trained = require(
        private_result
        / f"arm_batches/trained_{TRAINED_SHA[:16]}/trained_append_k1"
    )
    untrained = require(
        private_result / "arm_batches/untrained_identity/untrained_append_k1"
    )
    output = RUN_DIR / "oracle_overlap/summary.json"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_oracle_overlap",
            "--data_jsonl",
            str(data),
            "--teacher_cache_summary",
            str(teacher),
            "--immutable_cache_receipt",
            str(immutable_receipt),
            "--expected_immutable_cache_sha256",
            IMMUTABLE_CACHE_SHA,
            "--inplace_cache_dir",
            str(inplace),
            "--trained_cache_dir",
            str(trained),
            "--untrained_cache_dir",
            str(untrained),
            "--output_summary",
            str(output),
        ]
    )
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, receipt_dir / "oracle_overlap_summary.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["scoring_reexecuted"] or payload["model_loaded"]:
        raise RuntimeError("oracle-overlap receipt violated cache-only policy")
    commit = publish(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary_sha256": sha256_file(output),
                "publish_commit": commit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
