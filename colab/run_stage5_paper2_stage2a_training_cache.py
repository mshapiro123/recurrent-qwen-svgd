"""Build the score-blind Stage 2A training population and 14B lattice."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.paper2_phase3_p31_completion import sha256_file


RUN_ID = "stage5_paper2_stage2a_training_cache_20260817"
CONTENT_ID = "stage5_paper2_stage2a_content_geometry_20260817"
P31_ID = "stage5_paper2_phase3_p31_completion_20260810"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
LOCAL_DIR = ROOT / "outputs/stage5" / RUN_ID


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        print("stage2a_training_cache_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("stage2a_training_cache_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 80 * 1024**3:
            target = root / "recurrent-qwen-svgd-stage" / RUN_ID
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise RuntimeError("Stage 2A training-cache pass requires 80 GiB of local scratch")


def model_cache(scratch: Path) -> Path:
    prior = (
        Path("/mnt/local-scratch/recurrent-qwen-svgd-stage")
        / CONTENT_ID
        / "hf_model_cache"
    )
    if prior.is_dir():
        print(f"stage2a_training_cache_reuse_model_cache={prior}", flush=True)
        return prior
    return scratch / "hf_model_cache"


def main() -> int:
    for path in (PRIVATE_DIR, RECEIPT_DIR, LOCAL_DIR):
        path.mkdir(parents=True, exist_ok=True)
    status_path = RECEIPT_DIR / "status.json"

    def status(value: str, **details: object) -> None:
        write_json(
            status_path,
            {
                "kind": "paper2_stage2a_training_cache_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                "training_authorized": False,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "confirm_scored": False,
                "eval_e_scored": False,
                **details,
            },
        )
        print(f"stage2a_training_cache_status={value} details={details}", flush=True)

    try:
        scratch = scratch_root()
        content_private = DRIVE_STAGE5 / CONTENT_ID / "private"
        content_receipts = DRIVE_STAGE5 / CONTENT_ID / "receipts"
        p31_private = DRIVE_STAGE5 / P31_ID / "private"
        inputs = {
            "source_rows": p31_private / "p31_partitioned_rows.jsonl",
            "teacher_14b_scores": p31_private / "model_scores/teacher_14b_scores.jsonl",
            "firm_manifest": content_private / "stage2a_firm_knowledge_manifest.jsonl",
            "memory_manifest": content_private / "stage2a_memory_manifest.jsonl",
            "content_summary": content_receipts / "summary.json",
        }
        for label, path in inputs.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing Stage 2A cache input {label}: {path}")
        content_summary = json.loads(
            inputs["content_summary"].read_text(encoding="utf-8")
        )
        if content_summary.get("status") != "complete_score_blind_pre_signature":
            raise RuntimeError("Stage 2A content/geometry prerequisite is incomplete")
        if any(
            bool(content_summary.get(field))
            for field in ("training_authorized", "optimizer_constructed", "confirm_scored", "eval_e_scored")
        ) or int(content_summary.get("optimizer_steps", -1)) != 0:
            raise RuntimeError("Stage 2A content prerequisite crossed a sealed boundary")

        status(
            "running_score_blind_teacher_lattice",
            scratch=str(scratch),
            source_hashes={label: sha256_file(path) for label, path in inputs.items()},
        )
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.cache_paper2_stage2a_training",
                "--source_rows",
                str(inputs["source_rows"]),
                "--teacher_14b_scores",
                str(inputs["teacher_14b_scores"]),
                "--firm_manifest",
                str(inputs["firm_manifest"]),
                "--memory_manifest",
                str(inputs["memory_manifest"]),
                "--output_dir",
                str(LOCAL_DIR),
                "--private_dir",
                str(PRIVATE_DIR),
                "--model_cache",
                str(model_cache(scratch)),
                "--device",
                "cuda",
                "--batch_size",
                os.environ.get("STAGE2A_TRAINING_CACHE_BATCH_SIZE", "8"),
            ]
        )
        summary_path = LOCAL_DIR / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete_score_blind_pre_training":
            raise RuntimeError("Stage 2A teacher lattice did not complete")
        if any(
            bool(summary.get(field))
            for field in ("training_started", "optimizer_constructed", "confirm_scored", "eval_e_scored")
        ) or int(summary.get("optimizer_steps", -1)) != 0:
            raise RuntimeError("Stage 2A teacher lattice crossed a sealed boundary")
        for path in LOCAL_DIR.glob("*.json"):
            shutil.copy2(path, RECEIPT_DIR / path.name)
        status(
            "complete_score_blind_pre_training",
            summary_sha256=sha256_file(summary_path),
            cached_rows=summary["cached_rows"],
            cached_positions=summary["cached_positions"],
            population_manifest_sha256=summary["population_manifest_sha256"],
            teacher_lattice_artifact_sha256=summary["teacher_lattice_artifact_sha256"],
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
