"""Build the authorized P3.1/P3.2 receipts and migrate both E1 seed endpoints."""

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
RUN_ID = "stage5_paper2_phase3_p31_p32_receipts_20260810"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / RUN_ID
PRIVATE_DIR = DRIVE_RUN / "private"
RECEIPT_DIR = DRIVE_RUN / "receipts"
SOURCE_MANIFEST = ROOT / "training/paper2_phase3_p31_source_manifest.json"
E1_REGISTRATION = ROOT / "training/paper2_phase2_e1_confirmation_preregistration.json"
MIGRATION_SOURCES = ROOT / "training/paper2_phase3_migration_sources.json"
TIER1 = ROOT / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/data/base_capability_canary_64.jsonl"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=600)
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
        print("phase3_receipts_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("phase3_receipts_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def write_status(status: str, **details: object) -> None:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPT_DIR / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase3_p31_p32_receipts_status_v1",
                "status": status,
                "updated_at_unix": time.time(),
                "p33_training_authorized": False,
                "optimizer_steps": 0,
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"phase3_receipts_status status={status} details={details}", flush=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_metadata() -> tuple[dict[str, str], dict[str, str]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    revisions = {
        battery: str(sources[battery].get("revision") or sources[battery]["sha256"])
        for battery in sources
    }
    readers = {
        "arc_easy": "mcq_choice_text_same_reader_v1",
        "arc_challenge": "mcq_choice_text_same_reader_v1",
        "gsm8k": "final_number_after_hash_delimiter_v1",
        "mbpp": "sandboxed_unit_test_execution_v1",
        "mmlu": "mcq_choice_text_same_reader_v1",
        "tier1": "paper_one_same_reader_v1",
    }
    return revisions, readers


def main() -> int:
    for path in (SOURCE_MANIFEST, E1_REGISTRATION, MIGRATION_SOURCES, TIER1):
        if not path.exists():
            raise FileNotFoundError(f"missing Phase 3 receipt input: {path}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    write_status("materializing_score_blind_sources")

    source_summary = RUN_DIR / "p31_source_summary.json"
    private_rows = PRIVATE_DIR / "p31_rows.jsonl"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_phase3_p31_sources",
            "--source_manifest",
            str(SOURCE_MANIFEST),
            "--tier1_path",
            str(TIER1),
            "--private_rows",
            str(private_rows),
            "--output_summary",
            str(source_summary),
            "--mmlu_slice_size",
            os.environ.get("STAGE5_PHASE3_MMLU_SLICE_SIZE", "512"),
        ]
    )

    revisions, readers = source_metadata()
    revisions_path = PRIVATE_DIR / "dataset_revisions.json"
    readers_path = PRIVATE_DIR / "reader_versions.json"
    write_json(revisions_path, revisions)
    write_json(readers_path, readers)
    calibration_summary = RUN_DIR / "p31_calibration_summary.json"
    calibration_command = [
        sys.executable,
        "-u",
        "-m",
        "eval.prepare_paper2_phase3_p31",
        "--rows_jsonl",
        str(private_rows),
        "--dataset_revisions_json",
        str(revisions_path),
        "--reader_versions_json",
        str(readers_path),
        "--output_ledger",
        str(RUN_DIR / "p31_split_ledger.json"),
        "--output_simulation",
        str(calibration_summary),
        "--candidate_rows",
        os.environ.get("STAGE5_PHASE3_CALIBRATION_ROWS", "512"),
        "--candidate_alphas",
        os.environ.get("STAGE5_PHASE3_CALIBRATION_ALPHA", "0.00005"),
        "--campaigns",
        os.environ.get("STAGE5_PHASE3_CALIBRATION_CAMPAIGNS", "100000"),
    ]
    empirical_path = os.environ.get("STAGE5_PHASE3_EMPIRICAL_DIFFERENCES_JSON", "").strip()
    if empirical_path:
        calibration_command.extend(["--empirical_differences_json", empirical_path])
    run(calibration_command)

    write_status("migrating_e1_endpoints_no_optimizer")
    migration_summary = RUN_DIR / "checkpoint_migration_summary.json"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.prepare_paper2_phase3_checkpoint_migration",
            "--registration",
            str(E1_REGISTRATION),
            "--output_dir",
            str(PRIVATE_DIR / "migrated_checkpoints"),
            "--output_summary",
            str(migration_summary),
            "--migration_sources",
            str(MIGRATION_SOURCES),
        ]
    )

    p32_summary = RUN_DIR / "p32_preflight_summary.json"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_p32_preflight",
            "--output_summary",
            str(p32_summary),
        ]
    )

    summaries = {
        "p31_sources": json.loads(source_summary.read_text(encoding="utf-8")),
        "p31_calibration": json.loads(calibration_summary.read_text(encoding="utf-8")),
        "checkpoint_migration": json.loads(migration_summary.read_text(encoding="utf-8")),
        "p32_preflight": json.loads(p32_summary.read_text(encoding="utf-8")),
    }
    final = {
        "kind": "paper2_phase3_p31_p32_receipts_summary_v1",
        "status": "complete_p33_still_unauthorized",
        "components": summaries,
        "assertions": {
            "confirm_unscored": not summaries["p31_sources"]["confirm_scoring_spent"],
            "no_models_loaded_for_source_assembly": not summaries["p31_sources"]["models_loaded"],
            "both_e1_seed_migrations_bit_exact": summaries["checkpoint_migration"][
                "assertions"
            ]["all_migrations_bit_exact_steps_0_through_4"],
            "p32_schema_preflight_green": all(
                summaries["p32_preflight"]["assertions"].values()
            ),
            "p33_training_unauthorized": True,
            "optimizer_steps_zero": True,
        },
        "next_required_before_p33_lock": [
            "empirical DEV noise-model calibration if not supplied to this run",
            "full P3.2 cache coverage arithmetic",
            "document-disjoint linear-decodability forecast",
        ],
    }
    if not all(final["assertions"].values()):
        raise RuntimeError(f"Phase 3 receipt assertions failed: {final['assertions']}")
    write_json(RUN_DIR / "summary.json", final)
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.glob("*.json"):
        shutil.copy2(path, RECEIPT_DIR / path.name)
    write_status("complete", summary=str(RECEIPT_DIR / "summary.json"))
    print(json.dumps({"status": final["status"], "drive": str(DRIVE_RUN)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise
