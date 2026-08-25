"""Score the frozen 40-row TM-0 MBPP transport partition."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


ROOT = Path("/content/tm0_repo")
PANEL = Path("/content/tm0_score_missing_mbpp_40.jsonl")
OUTPUT = Path("/content/tm0_score_mbpp_output")
STATUS = Path("/content/tm0_score_mbpp_status.json")
EXPECTED_PANEL_SHA = "27ca9e8f703d1937607d71927c222803ce7a08656b985ef2372ae11bafd676b0"
EXPECTED_ROWS = 40


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(status: str, **details: object) -> None:
    payload = {
        "kind": "paper2_tm0_score_mbpp_partition_status_v1",
        "status": status,
        "updated_at_unix": time.time(),
        **details,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)
    print(f"tm0_score_mbpp_status status={status} details={details}", flush=True)


def main() -> int:
    if sha256_file(PANEL) != EXPECTED_PANEL_SHA:
        raise RuntimeError("TM-0 frozen MBPP-partition SHA mismatch")
    if sum(1 for _ in PANEL.open()) != EXPECTED_ROWS:
        raise RuntimeError("TM-0 frozen MBPP-partition row-count mismatch")
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
        transport_partition="forty_non_tail_mbpp_rows",
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
            "--mcq_candidate_batch_size",
            "32",
            "--generation_batch_size",
            "8",
            "--confirm_seal_ledger",
            str(seal),
        ],
        cwd=ROOT,
        check=True,
    )
    scores = OUTPUT / "teacher_7b_scores.jsonl"
    rows = sum(1 for _ in scores.open())
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"TM-0 7B MBPP coverage mismatch: {rows}/{EXPECTED_ROWS}")
    bundle = Path("/content/tm0_teacher_7b_scores_mbpp_bundle.tar")
    with tarfile.open(bundle, "w") as archive:
        archive.add(OUTPUT, arcname="scores_mbpp")
    write_status(
        "complete",
        rows=rows,
        scores_sha256=sha256_file(scores),
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
