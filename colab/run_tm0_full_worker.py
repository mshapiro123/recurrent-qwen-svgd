"""Resumable full GPU caching and 7B correctness pass for TM-0."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


ROOT = Path("/content/tm0_repo")
OUTPUT = Path("/content/tm0_full")
STATUS = OUTPUT / "status.json"
PANEL = Path("/content/tm0_panel.jsonl")
PROBE = Path("/content/tm0_cost_probe.jsonl")
AMENDMENT = Path("/content/tm0_spend_amendment.json")
EXPECTED_PANEL_SHA = "e108b0a92fdc69b9cb27274ac420908b65303213307f9d8dfc1f4ba73d58b5ca"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def set_status(status: str, **details) -> None:
    payload = {
        "kind": "paper2_tm0_gpu_status_v1",
        "status": status,
        "updated_at_unix": time.time(),
        **details,
    }
    atomic_json(STATUS, payload)
    print(f"tm0_status status={status} details={details}", flush=True)


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sha256_file(PANEL) != EXPECTED_PANEL_SHA:
        raise RuntimeError("TM-0 frozen panel SHA mismatch")
    amendment = json.loads(AMENDMENT.read_text())
    if amendment.get("status") != "RATIFIED_BY_PROGRAM_OWNER":
        raise RuntimeError("TM-0 spend amendment is not ratified")
    if not amendment.get("scope", {}).get("spend_cap_only"):
        raise RuntimeError("TM-0 amendment changed more than spend")
    set_status("running_state_caches", panel_sha256=EXPECTED_PANEL_SHA)
    cache_receipts = []
    for model_key in ("student", "teacher_7b", "teacher_14b"):
        target = OUTPUT / "state_cache" / model_key
        index = target / f"{model_key}_cache_index.json"
        if index.is_file():
            receipt = json.loads(index.read_text())
            if receipt.get("rows") == 6144 and not receipt.get("dry_run"):
                print(f"tm0_resume_skip model={model_key}", flush=True)
                cache_receipts.append(receipt)
                continue
        run(
            [
                sys.executable, "-u", "-m", "eval.cache_paper2_tm0",
                "--panel", str(PANEL), "--probe_manifest", str(PROBE),
                "--output_dir", str(target), "--model_cache", "/content/model-cache",
                "--model_key", model_key, "--shard_rows", "64",
            ]
        )
        cache_receipts.append(json.loads(index.read_text()))
    set_status("running_7b_correctness", cached_models=len(cache_receipts))
    seal = OUTPUT / "tm0_correctness_seal.json"
    if not seal.exists():
        atomic_json(
            seal,
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
        )
    score_dir = OUTPUT / "scores"
    run(
        [
            sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p31_references",
            "--rows_jsonl", str(PANEL), "--output_dir", str(score_dir),
            "--model_key", "teacher_7b", "--device", "cuda", "--dtype", "bfloat16",
            "--teacher_mcq_candidate_batch_size", "32",
            "--teacher_generation_batch_size", "8",
            "--confirm_seal_ledger", str(seal),
        ]
    )
    summary = {
        "kind": "paper2_tm0_gpu_cache_summary_v1",
        "status": "complete",
        "panel_sha256": EXPECTED_PANEL_SHA,
        "spend_amendment": amendment,
        "state_caches": cache_receipts,
        "teacher_7b_score_receipt": json.loads(
            (score_dir / "model_score_receipts.json").read_text()
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "injection_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(OUTPUT / "summary.json", summary)
    set_status("packing", summary_sha256=sha256_file(OUTPUT / "summary.json"))
    with tarfile.open("/content/tm0_full_bundle.tar", "w") as archive:
        archive.add(OUTPUT, arcname="tm0_full")
    set_status(
        "complete",
        summary_sha256=sha256_file(OUTPUT / "summary.json"),
        bundle_sha256=sha256_file(Path("/content/tm0_full_bundle.tar")),
        bundle_bytes=Path("/content/tm0_full_bundle.tar").stat().st_size,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        set_status(
            "failed",
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
