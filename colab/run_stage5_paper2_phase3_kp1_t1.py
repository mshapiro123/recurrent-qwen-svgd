"""Run the authorized DEV-only KP-1 and amended T1 score-only wave."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from colab.run_stage5_paper2_phase3_p34_a2 import (
    DRIVE_STAGE5,
    I1_ID,
    MIGRATED_SHA,
    MIGRATION_ID,
    P33_ID,
    P33_SHA,
    rsync,
    sha256_file,
    write_json,
)
from colab.run_stage5_paper2_phase3_p35 import I1_SHA, stage_chain
from colab.run_stage5_paper2_phase3_p35_amplitude_t1_preflight import P34_SHA, P35_SHA
from training.paper2_phase3_kp1_t1 import (
    canonical_sha256,
    knowledge_gap_rows,
    stratified_probe_split,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_kp1_t1_20260816"
P31_ID = "stage5_paper2_phase3_p31_completion_20260810"
P35_ID = "stage5_paper2_phase3_p35_20260815"
LOCK_PATH = ROOT / "training/paper2_phase3_kp1_t1_lock.json"
AUTHORITY_PATH = ROOT / "docs/STRATEGY_DIAGNOSTIC_WAVE_ANALYSIS_20260816.md"
PANEL_PATH = (
    ROOT
    / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
)
BASE_SCORES_PATH = (
    ROOT
    / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
)
EXPECTED_GAP_BY_BATTERY = {
    "arc_challenge": 31,
    "arc_easy": 52,
    "gsm8k": 122,
    "mbpp": 25,
    "mmlu": 96,
    "tier1": 3,
}


def stage_chain_with_verified_p34(
    scratch: Path, *, seed: int, expected_p34: str
) -> dict[str, Path]:
    """Reuse an independently staged P3.4 endpoint after exact SHA verification."""
    p34 = scratch / f"seed_{seed}_p34_step_4000.pt"
    if not p34.is_file():
        return stage_chain(scratch, seed=seed, expected_p34=expected_p34)
    observed_p34 = sha256_file(p34)
    if observed_p34 != expected_p34:
        raise RuntimeError(
            "KP-1/T1 pre-staged P3.4 SHA mismatch: "
            f"seed={seed} expected={expected_p34} observed={observed_p34}"
        )

    paths = {
        "migrated": scratch / f"seed_{seed}_migrated.pt",
        "p33": scratch / f"seed_{seed}_p33_step_1000.pt",
        "i1": scratch / f"seed_{seed}_i1.pt",
        "p34": p34,
    }
    rsync(
        DRIVE_STAGE5
        / MIGRATION_ID
        / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt",
        paths["migrated"],
    )
    rsync(
        DRIVE_STAGE5 / P33_ID / f"private/seed_{seed}/checkpoint_step_1000.pt",
        paths["p33"],
    )
    rsync(DRIVE_STAGE5 / I1_ID / f"private/seed_{seed}/resume.pt", paths["i1"])
    expected = {
        "migrated": MIGRATED_SHA[seed],
        "p33": P33_SHA[seed],
        "i1": I1_SHA[seed],
        "p34": expected_p34,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != expected[name]:
            raise RuntimeError(
                "KP-1/T1 staged chain SHA mismatch: "
                f"seed={seed} name={name} expected={expected[name]} observed={observed}"
            )
    print(f"kp1_t1_reused_verified_p34 seed={seed} sha256={observed_p34}", flush=True)
    return paths


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 20 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("KP-1/T1 requires at least 20 GiB local scratch")


def build_pre_model_manifest(*, references: Path, receipts: Path) -> dict[str, object]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    authority = lock["authority"]
    if AUTHORITY_PATH.stat().st_size != int(authority["bytes"]):
        raise RuntimeError("KP-1/T1 authority byte count changed")
    if sha256_file(AUTHORITY_PATH) != authority["sha256"]:
        raise RuntimeError("KP-1/T1 authority SHA changed")
    for key, path in (
        ("panel_sha256", PANEL_PATH),
        ("base_scores_sha256", BASE_SCORES_PATH),
        ("reference_scores_sha256", references),
    ):
        if sha256_file(path) != lock["source_files"][key]:
            raise RuntimeError(f"KP-1/T1 source identity changed: {key}")

    panel = read_jsonl(PANEL_PATH)
    base_scores = read_jsonl(BASE_SCORES_PATH)
    reference_rows = read_jsonl(references)
    panel_ids = [str(row["item_id"]) for row in panel]
    if len(panel_ids) != len(set(panel_ids)):
        raise RuntimeError("KP-1/T1 panel item ids are not unique")
    if any(str(row.get("partition")) != "dev" for row in panel):
        raise RuntimeError("KP-1/T1 panel contains a non-DEV row")

    base_by_id = {str(row["item_id"]): bool(row["correct"]) for row in base_scores}
    reference_by_id = {
        str(row["item_id"]): bool(row["base_correct"]) for row in reference_rows
    }
    if set(base_by_id) != set(panel_ids):
        raise RuntimeError("KP-1/T1 panel and base-score table do not have identical row ids")
    missing_references = set(panel_ids) - set(reference_by_id)
    if missing_references:
        raise RuntimeError(
            f"KP-1/T1 merged reference table lacks {len(missing_references)} panel rows"
        )
    mismatches = [item_id for item_id in panel_ids if base_by_id[item_id] != reference_by_id[item_id]]
    if mismatches:
        raise RuntimeError(f"KP-1 base-reader identity mismatch on {len(mismatches)} rows")

    gap = knowledge_gap_rows(panel, reference_rows)
    gap_ids = [str(row["item_id"]) for row in gap]
    gap_counts = dict(sorted(Counter(str(row["battery"]) for row in gap).items()))
    if len(gap) != 329 or gap_counts != EXPECTED_GAP_BY_BATTERY:
        raise RuntimeError(
            f"KP-1 gap population changed: rows={len(gap)} counts={gap_counts}"
        )
    split = stratified_probe_split(
        gap,
        seed=int(lock["kp1"]["split_seed"]),
        eval_fraction=float(lock["kp1"]["probe_eval_fraction"]),
    )
    split_counts = dict(sorted(Counter(split.values()).items()))
    payload: dict[str, object] = {
        "kind": "paper2_phase3_kp1_t1_pre_model_manifest_v1",
        "status": "locked_before_model_access",
        "authority": authority,
        "lock_sha256": sha256_file(LOCK_PATH),
        "source_sha256": {
            "panel": sha256_file(PANEL_PATH),
            "base_scores": sha256_file(BASE_SCORES_PATH),
            "references": sha256_file(references),
        },
        "panel_rows": len(panel),
        "reference_table_rows": len(reference_rows),
        "reference_rows_outside_panel": len(set(reference_by_id) - set(panel_ids)),
        "panel_item_ids_sha256": canonical_sha256(panel_ids),
        "kp1_gap_rows": len(gap),
        "kp1_gap_rows_by_battery": gap_counts,
        "kp1_gap_item_ids": gap_ids,
        "kp1_gap_item_ids_sha256": canonical_sha256(gap_ids),
        "probe_split": split,
        "probe_split_counts": split_counts,
        "base_reader_identity_mismatches": 0,
        "dev_only": True,
        "confirm_scored": False,
        "eval_e_scored": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "created_at_unix": time.time(),
    }
    path = receipts / "pre_model_manifest.json"
    write_json(path, payload)
    print(
        "kp1_t1_manifest_locked "
        f"gap_rows={len(gap)} split={split_counts} sha256={sha256_file(path)}",
        flush=True,
    )
    return payload


def stage_inputs(*, scratch: Path, references: Path, private: Path) -> Path:
    model_cache = scratch / "hf_model_cache"
    model_cache.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, object] = {}
    for seed in (0, 1):
        chain = stage_chain_with_verified_p34(
            scratch / f"chain_seed_{seed}", seed=seed, expected_p34=P34_SHA[seed]
        )
        p35 = scratch / f"seed_{seed}_p35_ema_step_4400.pt"
        rsync(DRIVE_STAGE5 / P35_ID / f"private/arm_s_seed_{seed}/ema_step_4400.pt", p35)
        if sha256_file(p35) != P35_SHA[seed]:
            raise RuntimeError(f"KP-1/T1 P3.5 endpoint SHA mismatch seed={seed}")
        common_sha = {
            "migrated": MIGRATED_SHA[seed],
            "p33": P33_SHA[seed],
            "i1": I1_SHA[seed],
            "p34": P34_SHA[seed],
        }
        checkpoints[f"p34_seed_{seed}_step_4000"] = {
            "paths": {name: str(path) for name, path in chain.items()} | {"p35": None},
            "sha256": common_sha | {"p35": None},
        }
        checkpoints[f"p35_seed_{seed}_ema_step_4400"] = {
            "paths": {name: str(path) for name, path in chain.items()} | {"p35": str(p35)},
            "sha256": common_sha | {"p35": P35_SHA[seed]},
        }

    chain_manifest = {
        "kind": "paper2_phase3_kp1_t1_chain_manifest_v1",
        "checkpoints": checkpoints,
        "model_cache": str(model_cache),
        "references": str(references),
        "all_checkpoint_sha256_verified": True,
        "model_loaded_during_staging": False,
        "optimizer_constructed": False,
    }
    path = private / "chain_manifest.json"
    write_json(path, chain_manifest)
    return path


def main() -> int:
    drive_run = DRIVE_STAGE5 / RUN_ID
    receipts = drive_run / "receipts"
    private = drive_run / "private"
    status_path = receipts / "status.json"
    local = ROOT / "outputs/stage5" / RUN_ID

    def status(value: str, **details: object) -> None:
        write_json(
            status_path,
            {
                "kind": "paper2_phase3_kp1_t1_status_v1",
                "status": value,
                "updated_at_unix": time.time(),
                **details,
            },
        )
        print(f"kp1_t1_status={value} details={details}", flush=True)

    try:
        receipts.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        scratch = scratch_root()
        references = scratch / "p31_merged_dev_verified_scores.jsonl"
        status("staging_score_tables_before_model_access", scratch=str(scratch))
        rsync(
            DRIVE_STAGE5 / P31_ID / "private/p31_merged_dev_verified_scores.jsonl",
            references,
        )
        manifest = build_pre_model_manifest(references=references, receipts=receipts)
        manifest_path = receipts / "pre_model_manifest.json"
        status(
            "staging_frozen_models_after_manifest_lock",
            manifest_sha256=sha256_file(manifest_path),
            gap_rows=manifest["kp1_gap_rows"],
        )
        chain_manifest = stage_inputs(scratch=scratch, references=references, private=private)
        status("extracting_dev_states_score_only", chain_manifest_sha256=sha256_file(chain_manifest))
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.eval_paper2_phase3_kp1_t1",
                "--lock",
                str(LOCK_PATH),
                "--manifest",
                str(manifest_path),
                "--panel",
                str(PANEL_PATH),
                "--base_scores",
                str(BASE_SCORES_PATH),
                "--references",
                str(references),
                "--model_cache",
                str(scratch / "hf_model_cache"),
                "--chain_manifest",
                str(chain_manifest),
                "--output_dir",
                str(local),
                "--private_dir",
                str(private),
                "--batch_size",
                "8",
            ]
        )
        summary = local / "summary.json"
        if not summary.is_file():
            raise RuntimeError("KP-1/T1 evaluator did not produce summary.json")
        shutil.copy2(summary, receipts / "summary.json")
        status(
            "complete",
            summary_sha256=sha256_file(summary),
            state_cache_sha256=json.loads(summary.read_text(encoding="utf-8"))["t1"][
                "state_cache"
            ]["sha256"],
            confirm_scored=False,
            eval_e_scored=False,
            optimizer_steps=0,
        )
        print(summary.read_text(encoding="utf-8"), flush=True)
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
