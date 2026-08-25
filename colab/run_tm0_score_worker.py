"""Run the resumable TM-0 7B correctness phase on Colab."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


ROOT = Path("/content/tm0_repo")
PANEL = Path("/content/tm0_panel.jsonl")
OUTPUT = Path("/content/tm0_score_output")
STATUS = Path("/content/tm0_score_status.json")
EXPECTED_PANEL_SHA = "e108b0a92fdc69b9cb27274ac420908b65303213307f9d8dfc1f4ba73d58b5ca"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(status: str, **details: object) -> None:
    payload = {
        "kind": "paper2_tm0_score_phase_status_v1",
        "status": status,
        "updated_at_unix": time.time(),
        **details,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)
    print(f"tm0_score_status status={status} details={details}", flush=True)


def main() -> int:
    if sha256_file(PANEL) != EXPECTED_PANEL_SHA:
        raise RuntimeError("TM-0 frozen panel SHA mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seal = OUTPUT / "tm0_correctness_seal.json"
    if not seal.exists():
        seal.write_text(
            json.dumps(
                {
                    "kind": "paper2_tm0_correctness_seal_v1",
                    "status": "sealed_before_model_scoring",
                    "panel_sha256": EXPECTED_PANEL_SHA,
                    "assertions": {
                        "confirm_membership_sealed": True,
                        "confirm_scored": False,
                        "eval_e_scored": False,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    existing = OUTPUT / "teacher_7b_scores.jsonl"
    existing_rows = sum(1 for _ in existing.open()) if existing.exists() else 0
    write_status(
        "running",
        panel_sha256=EXPECTED_PANEL_SHA,
        resumed_rows=existing_rows,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_p31_references",
            "--rows_jsonl",
            str(PANEL),
            "--output_dir",
            str(OUTPUT),
            "--model_key",
            "teacher_7b",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--teacher_mcq_candidate_batch_size",
            "32",
            "--teacher_generation_batch_size",
            "8",
            "--confirm_seal_ledger",
            str(seal),
        ],
        cwd=ROOT,
        check=True,
    )
    scores = OUTPUT / "teacher_7b_scores.jsonl"
    rows = sum(1 for _ in scores.open())
    if rows != 6144:
        raise RuntimeError(f"TM-0 7B score coverage mismatch: {rows}/6144")
    receipt = OUTPUT / "model_score_receipts.json"
    bundle = Path("/content/tm0_teacher_7b_scores_bundle.tar")
    with tarfile.open(bundle, "w") as archive:
        archive.add(OUTPUT, arcname="scores")
    write_status(
        "complete",
        rows=rows,
        scores_sha256=sha256_file(scores),
        receipt_sha256=sha256_file(receipt),
        bundle_path=str(bundle),
        bundle_sha256=sha256_file(bundle),
        bundle_bytes=bundle.stat().st_size,
        optimizer_steps=0,
        confirm_scored=False,
        eval_e_scored=False,
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        write_status(
            "failed",
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    raise SystemExit(exit_code)
