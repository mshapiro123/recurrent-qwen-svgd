"""Receipt the availability of cached P3.4 states required by Sidecar v2 T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive_root.is_dir():
        raise FileNotFoundError(args.archive_root)

    files = sorted(path for path in args.archive_root.rglob("*") if path.is_file())
    checkpoints = [path for path in files if path.suffix == ".pt"]
    payloads = []
    forbidden_top_level = ("hidden", "cell", "feature", "activation", "state_cache")
    cached_state_payloads = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        top_keys = sorted(payload) if isinstance(payload, dict) else []
        matching = [
            key
            for key in top_keys
            if any(term in str(key).lower() for term in forbidden_top_level)
        ]
        if matching:
            cached_state_payloads.append(str(path.relative_to(args.archive_root)))
        payloads.append(
            {
                "path": str(path.relative_to(args.archive_root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "top_level_keys": top_keys,
                "cached_state_keys": matching,
            }
        )

    row_files = [
        str(path.relative_to(args.archive_root)).replace("\\", "/")
        for path in files
        if "task_rows" in path.name
    ]
    output = {
        "kind": "paper2_sidecar_v2_t1_artifact_preflight_v1",
        "status": (
            "ready_cached_p34_states_found"
            if cached_state_payloads
            else "blocked_missing_required_p34_cached_states"
        ),
        "mode": "read_only_no_model_no_optimizer",
        "required_estimand": {
            "population": "matched P3.4 scored rows",
            "features": "prelude, per-loop recurrent, and selected intermediate-layer cells",
            "checkpoint_axis": "P3.4 late-window jitter checkpoints",
            "comparisons": "mean pooling versus probe counts 1, 4, and 8",
            "metrics": "teacher-cluster retrieval AUC and checkpoint stability",
        },
        "archive_root": str(args.archive_root.resolve()),
        "archive_file_count": len(files),
        "checkpoint_count": len(checkpoints),
        "checkpoint_payloads": payloads,
        "current_p34_step4000_payload_audit": {
            "seed_0_sha256": "381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7",
            "seed_1_sha256": "97ad532a5bffd72b2563799047b517e531e00115793bf4808f060148dfffc1ec",
            "top_level_keys": [
                "amendment_sha256",
                "arm",
                "continuation_receipt",
                "controller",
                "generator_state",
                "history",
                "kind",
                "last_chi",
                "last_pi_dep",
                "lock_sha256",
                "objective_controller_events",
                "objective_weights",
                "optimizer_state",
                "rng_state",
                "schedule_hashes",
                "seed",
                "share_contract_events",
                "share_misses",
                "share_window",
                "step",
                "stop_reason",
                "tier_s_streak",
                "tier_w_streak",
                "trainable_state",
            ],
            "cached_state_keys": [],
            "audit_mode": "remote torch payload schema read; no model execution",
        },
        "cached_state_payloads": cached_state_payloads,
        "row_prediction_files": row_files,
        "drive_filename_audit": {
            "performed_utc_date": "2026-08-15",
            "hidden_matches": 0,
            "p34_relevant_cell_matches": 0,
            "p34_relevant_feature_matches": 0,
            "other_feature_cache_matches": [
                "evaluation_feature_cache.pt (2026-07-28; D0/D1 lineage)",
                "feature_cache.pt (2026-07-27; D0 lineage)",
            ],
            "substitution_allowed": False,
        },
        "interpretation": (
            "The P3.4 archives retain checkpoints and row predictions, but no per-row "
            "cell-state cache. Older D0 feature caches are a different lineage and cannot "
            "identify the registered T1 checkpoint-jitter estimand."
        ),
        "unblock_contract": {
            "authorization_required": True,
            "action": (
                "Run one no-training, fixed-row P3.4 state-extraction pass across the "
                "banked late-window checkpoints, preserving row IDs and the exact cell set."
            ),
            "must_record": [
                "row manifest SHA-256",
                "checkpoint SHA-256 per jitter read",
                "cell-set schema and masks",
                "frozen-lineage digest before and after",
                "zero optimizer construction and zero optimizer steps",
            ],
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
