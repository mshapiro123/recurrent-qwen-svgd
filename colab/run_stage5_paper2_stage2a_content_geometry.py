"""Run the resumable score-blind Stage 2A content and geometry pass."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

from training.paper2_phase3_p31_completion import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_stage2a_content_geometry_20260817"
P31_ID = "stage5_paper2_phase3_p31_completion_20260810"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
LOCAL_DIR = ROOT / "outputs/stage5" / RUN_ID
PANEL = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=800)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode:
        print("stage2a_content_geometry_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("stage2a_content_geometry_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 120 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2A content pass requires 120 GiB of local scratch")


def main() -> int:
    for path in (PRIVATE_DIR, RECEIPT_DIR, LOCAL_DIR):
        path.mkdir(parents=True, exist_ok=True)
    status_path = RECEIPT_DIR / "status.json"

    def status(value: str, **details: object) -> None:
        write_json(
            status_path,
            {
                "kind": "paper2_stage2a_content_geometry_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                "training_authorized": False,
                "optimizer_steps": 0,
                **details,
            },
        )
        print(f"stage2a_content_geometry_status={value} details={details}", flush=True)

    try:
        scratch = scratch_root()
        p31_private = DRIVE_STAGE5 / P31_ID / "private"
        source_rows = p31_private / "p31_partitioned_rows.jsonl"
        merged_rows = p31_private / "p31_merged_dev_verified_scores.jsonl"
        teacher_scores = p31_private / "model_scores/teacher_14b_scores.jsonl"
        for path in (source_rows, merged_rows, teacher_scores, PANEL):
            if not path.is_file():
                raise FileNotFoundError(f"missing Stage 2A source: {path}")
        status(
            "running_score_blind_content_and_geometry",
            scratch=str(scratch),
            verifier_cache_exists=(PRIVATE_DIR / "verifier_32b_scores.jsonl").is_file(),
        )
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_stage2a_content_geometry",
                "--source_rows",
                str(source_rows),
                "--merged_rows",
                str(merged_rows),
                "--teacher_14b_scores",
                str(teacher_scores),
                "--panel",
                str(PANEL),
                "--output_dir",
                str(LOCAL_DIR),
                "--private_dir",
                str(PRIVATE_DIR),
                "--model_cache",
                str(scratch / "hf_model_cache"),
                "--verifier_mcq_batch_size",
                os.environ.get("STAGE2A_VERIFIER_MCQ_BATCH_SIZE", "4"),
                "--verifier_generation_batch_size",
                os.environ.get("STAGE2A_VERIFIER_GENERATION_BATCH_SIZE", "2"),
                "--student_state_batch_size",
                os.environ.get("STAGE2A_STUDENT_STATE_BATCH_SIZE", "32"),
                "--teacher_state_batch_size",
                os.environ.get("STAGE2A_TEACHER_STATE_BATCH_SIZE", "8"),
            ]
        )
        summary_path = LOCAL_DIR / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Stage 2A evaluator omitted summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary["training_authorized"]
            or summary["optimizer_constructed"]
            or summary["optimizer_steps"] != 0
            or summary["confirm_scored"]
            or summary["eval_e_scored"]
        ):
            raise RuntimeError("Stage 2A content pass crossed its score-blind boundary")
        for path in LOCAL_DIR.glob("*.json"):
            shutil.copy2(path, RECEIPT_DIR / path.name)
        status(
            "complete_score_blind_pre_signature",
            summary_sha256=sha256_file(summary_path),
            memory_slots=summary["memory"]["slots"],
            admitted=summary["firm_knowledge"]["admitted"],
            geometry_artifact_sha256=summary["geometry"]["artifact_sha256"],
        )
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return 0
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
