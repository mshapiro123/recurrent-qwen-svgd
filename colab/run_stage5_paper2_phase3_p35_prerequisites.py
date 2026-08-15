"""Stage and run the exact-reader repair and seed-0 persistence probe."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from colab.run_stage5_paper2_phase3_p34 import (
    DRIVE_STAGE5,
    I1_ID,
    MIGRATED_SHA,
    MIGRATION_ID,
    NEW_ID,
    OLD_ID,
    ORACLE_ID,
    P33_ID,
    P33_SHA,
    resolve_preflight,
    rsync,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p35_prerequisites_20260815"
P34_ID = "stage5_paper2_phase3_p34_a2_20260814"
P34_SHA = "381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 80 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("P3.5 prerequisites require at least 80 GiB local scratch")


def main() -> int:
    drive_run = DRIVE_STAGE5 / RUN_ID
    private = drive_run / "private"
    receipts = drive_run / "receipts"
    status_path = receipts / "status.json"

    def status(value: str, **details: object) -> None:
        write_json(
            status_path,
            {
                "kind": "paper2_phase3_p35_prerequisites_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                **details,
            },
        )
        print(f"p35_prerequisites_status={value} details={details}", flush=True)

    try:
        status("staging_inputs")
        scratch = scratch_root()
        old = scratch / "old"
        new = scratch / "new"
        preflight = resolve_preflight(scratch / "preflight")
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/sample_manifest.jsonl", old / "sample_manifest.jsonl")
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/lattice", old / "lattice")
        rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/model_cache/student_0p5b", old / "model_cache/student_0p5b")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/sample_manifest.jsonl", new / "sample_manifest.jsonl")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/lattice", new / "lattice")
        rsync(DRIVE_STAGE5 / NEW_ID / "private/full/model_cache/student_0p5b", new / "model_cache/student_0p5b")
        prior_cache = scratch / "agreement_oracle_directions_v1.pt"
        rsync(DRIVE_STAGE5 / ORACLE_ID / "private/oracle_cache/agreement_oracle_directions.pt", prior_cache)

        repaired = private / "serving_oracle/agreement_oracle_directions_v2.pt"
        repaired_summary = repaired.with_suffix(".summary.json")
        if not repaired_summary.is_file():
            status("repairing_serving_oracle")
            run([
                sys.executable, "-u", "-m", "eval.repair_paper2_phase3_serving_oracle_cache",
                "--old_summary", str(ROOT / "outputs/stage5" / OLD_ID / "summary.json"),
                "--old_private", str(old),
                "--new_summary", str(DRIVE_STAGE5 / NEW_ID / "receipts/full_cache_summary.json"),
                "--new_private", str(new),
                "--positive_audit", str(preflight / "private/p33_prep/p33_audit_slice.jsonl"),
                "--prior_cache", str(prior_cache),
                "--output_cache", str(repaired),
                "--device", "cuda",
            ])
        repair = json.loads(repaired_summary.read_text(encoding="utf-8"))
        if repair["source_anchor_identity"]["identity_rate"] != 1.0:
            raise RuntimeError("P3.5 serving-reader repair failed 100% identity")

        seed = 0
        migrated = DRIVE_STAGE5 / MIGRATION_ID / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt"
        p33 = DRIVE_STAGE5 / P33_ID / f"private/seed_{seed}/checkpoint_step_1000.pt"
        i1 = DRIVE_STAGE5 / I1_ID / f"private/seed_{seed}/resume.pt"
        p34 = DRIVE_STAGE5 / P34_ID / f"private/main_seed_{seed}/checkpoint_step_4000.pt"
        p34_lock = json.loads((ROOT / "training/paper2_phase3_p34_preregistration.json").read_text(encoding="utf-8"))
        i1_sha = p34_lock["initialization"]["seed_0"]["sha256"]
        for path, expected in (
            (migrated, MIGRATED_SHA[seed]),
            (p33, P33_SHA[seed]),
            (i1, i1_sha),
            (p34, P34_SHA),
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"P3.5 persistence source mismatch: {path}")
        persistence_dir = private / "persistence_seed_0"
        if not (persistence_dir / "summary.json").is_file():
            status("running_persistence_probe")
            run([
                sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p35_persistence",
                "--panel", str(ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"),
                "--migrated", str(migrated), "--migrated_sha256", MIGRATED_SHA[seed],
                "--p33", str(p33), "--p33_sha256", P33_SHA[seed],
                "--i1", str(i1), "--i1_sha256", i1_sha,
                "--p34", str(p34), "--p34_sha256", P34_SHA,
                "--output_dir", str(persistence_dir), "--device", "cuda",
            ])
        persistence = json.loads((persistence_dir / "summary.json").read_text(encoding="utf-8"))
        binding = {
            "kind": "paper2_phase3_p35_prerequisite_bindings_v1",
            "status": "complete_no_training",
            "serving_oracle_cache": {
                "path": str(repaired),
                "sha256": sha256_file(repaired),
                "summary_path": str(repaired_summary),
                "summary_sha256": sha256_file(repaired_summary),
                "source_anchor_identity": repair["source_anchor_identity"],
            },
            "persistence_probe": {
                "path": str(persistence_dir / "summary.json"),
                "sha256": sha256_file(persistence_dir / "summary.json"),
                "rows_sha256": persistence["rows_sha256"],
                "by_battery": persistence["by_battery"],
            },
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        write_json(receipts / "bindings.json", binding)
        status("complete", bindings_sha256=sha256_file(receipts / "bindings.json"))
        print(json.dumps(binding, indent=2, sort_keys=True))
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
