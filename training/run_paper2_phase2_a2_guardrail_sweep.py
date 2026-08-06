"""Run the CPU-only A2 guardrail inventory and step-237 lock sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.paper2_phase2_a2 import validate_guardrail_inventory
from training.run_paper2_phase2_matched_alpha import sha256_file, write_json


PROTOCOL = Path("training/paper2_phase2_staged_repilot_preregistration.json")
LOCK_KEY = "a2_step237_tripwire_amendment_20260806"
STATUS = "locked_before_a2_step237_continuation"
EXPECTED_ARMS = {
    "seed_0_full_a2",
    "seed_0_draft_only_control",
    "seed_1_full_a2",
    "seed_1_draft_only_control",
}


def sha256_lf_text(path: Path) -> str:
    """Hash UTF-8 text after Git-independent newline normalization."""
    normalized = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _assert_file(root: Path, row: dict[str, Any], *, path_key: str = "document") -> Path:
    path = root / row[path_key]
    if not path.is_file():
        raise RuntimeError(f"locked file is absent: {path}")
    if "bytes" in row and path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"locked byte count differs: {path}")
    if "document_bytes" in row and path.stat().st_size != int(row["document_bytes"]):
        raise RuntimeError(f"locked document byte count differs: {path}")
    expected = row.get("sha256") or row.get("document_sha256")
    if expected and sha256_file(path) != expected:
        raise RuntimeError(f"locked SHA differs: {path}")
    return path


def run(*, root: Path, output: Path) -> dict[str, Any]:
    registration = json.loads((root / PROTOCOL).read_text(encoding="utf-8"))
    lock = registration[LOCK_KEY]
    if lock["status"] != STATUS:
        raise RuntimeError("A2 step-237 continuation is not locked")
    documents = {
        "amendment": _assert_file(root, lock),
        "strategy_resolution": _assert_file(root, lock["strategy_resolution"]),
        "guardrail_doctrine": _assert_file(root, lock["guardrail_doctrine"]),
        "hash_portability_erratum": _assert_file(root, lock["technical_erratum"]),
    }
    source_summary = root / lock["source_result"]["summary"]
    audit_summary = root / lock["source_result"]["audit_summary"]
    if lock["source_result"].get("text_hash_mode") != "utf8_lf_normalized_sha256":
        raise RuntimeError("step-237 text receipt hash mode is not locked")
    if sha256_lf_text(source_summary) != lock["source_result"]["summary_sha256"]:
        raise RuntimeError("step-237 source summary SHA mismatch")
    if sha256_lf_text(audit_summary) != lock["source_result"]["audit_summary_sha256"]:
        raise RuntimeError("tripwire audit summary SHA mismatch")
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    if audit["status"] != "complete_descriptive" or audit["optimizer_updates_persisted"] != 0:
        raise RuntimeError("tripwire audit is not read-only complete")
    observed_arms = {f"seed_{row['seed']}_{row['arm']}": row for row in source["arms"]}
    if set(observed_arms) != EXPECTED_ARMS:
        raise RuntimeError("step-237 source does not contain the exact four-arm matrix")
    source_checks = {}
    expected_schedule = None
    for name, row in observed_arms.items():
        expected_sha = lock["source_resume_sha256_by_arm"][name]
        if int(row["step"]) != int(lock["resume_step"]):
            raise RuntimeError(f"{name} is not at the locked resume step")
        if row["checkpoint"]["sha256"] != expected_sha:
            raise RuntimeError(f"{name} checkpoint SHA differs from the lock")
        schedule = row["training_row_schedule"]
        if int(schedule["batches"]) != int(lock["resume_step"]):
            raise RuntimeError(f"{name} schedule does not contain 237 updates")
        if expected_schedule is None:
            expected_schedule = schedule
        elif schedule != expected_schedule:
            raise RuntimeError("the four source schedules are not identical")
        source_checks[name] = {
            "step": row["step"],
            "status": row["status"],
            "abort_reason": row["abort_reason"],
            "checkpoint_sha256": expected_sha,
            "schedule_sha256": schedule["sha256"],
        }
    inventory_validation = validate_guardrail_inventory(lock["rule_inventory"])
    if not inventory_validation["valid"]:
        raise RuntimeError(f"guardrail inventory is invalid: {inventory_validation['errors']}")
    stop_without_cliff = [
        row["name"]
        for row in lock["rule_inventory"]
        if row["disposition"] == "stop" and not row["named_cliff"]
    ]
    if stop_without_cliff:
        raise RuntimeError(f"stop rules lack named cliffs: {stop_without_cliff}")
    if lock["generator_reconstruction"]["next_row_hash"] != audit["stopping_row_hash"]:
        raise RuntimeError("locked attempt-238 hash differs from the audit")
    result = {
        "kind": "paper2_phase2_a2_guardrail_grounding_sweep_v1",
        "status": "complete_lock_valid",
        "training_authorized_by_this_job": False,
        "gpu_used": False,
        "lock_key": LOCK_KEY,
        "lock_status": lock["status"],
        "classification": lock["source_result"]["classification"],
        "source_receipt_hash_contract": {
            "mode": lock["source_result"]["text_hash_mode"],
            "source_summary_sha256": sha256_lf_text(source_summary),
            "audit_summary_sha256": sha256_lf_text(audit_summary),
        },
        "documents": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in documents.items()
        },
        "source_checks": source_checks,
        "generator_contract": lock["generator_reconstruction"],
        "relative_explosion_contract": lock["relative_explosion"],
        "inventory_validation": inventory_validation,
        "rule_inventory": lock["rule_inventory"],
        "demoted_rules": [
            row["name"] for row in lock["rule_inventory"] if row["disposition"] == "log"
        ],
        "warnings": [
            row["name"] for row in lock["rule_inventory"] if row["disposition"] == "warn"
        ],
        "stop_rules": [
            {"name": row["name"], "named_cliff": row["named_cliff"]}
            for row in lock["rule_inventory"]
            if row["disposition"] == "stop"
        ],
        "endpoint_only_rules": [
            row["name"]
            for row in lock["rule_inventory"]
            if row["disposition"] == "endpoint_verdict"
        ],
        "implementation_note": (
            "The rolling guard arms after 100 newly observed post-resume norms; "
            "cross-sectional audit norms are not backfilled as trajectory history."
        ),
    }
    write_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run(root=root, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
