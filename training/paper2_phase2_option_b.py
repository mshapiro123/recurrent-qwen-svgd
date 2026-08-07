"""Pure contracts for the locked Phase-2 Option B teacher/cache pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from training.paper2_phase2_stage0a import sha256_file


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = ROOT / "training/paper2_phase2_option_b_preregistration.json"
RULE_INVENTORY_PATH = ROOT / "training/paper2_phase2_option_b_rule_inventory.json"


def load_locked_registration(path: str | Path = REGISTRATION_PATH) -> dict[str, Any]:
    registration = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed_statuses = {
        "locked_teacher_pass_authorized_training_prohibited",
        "locked_post_generation_hash_amendment_training_authorized",
    }
    if registration.get("status") not in allowed_statuses:
        raise RuntimeError("Option B registration is not locked for the teacher pass")
    if not registration.get("locked_before_teacher_pass"):
        raise RuntimeError("Option B lock does not precede teacher generation")
    if not registration.get("teacher_pass_authorized"):
        raise RuntimeError("Option B teacher pass is not authorized")
    post_generation = registration.get("status") == (
        "locked_post_generation_hash_amendment_training_authorized"
    )
    if post_generation:
        if registration.get("training_authorized") is not True:
            raise RuntimeError("Option B post-generation training authorization is absent")
        if registration["teacher_pass"].get("status") != "banked_complete_at_target":
            raise RuntimeError("Option B teacher pass is not banked")
    elif registration.get("training_authorized"):
        raise RuntimeError("Option B training must remain prohibited before the amendment")
    if registration.get("lock_blockers"):
        raise RuntimeError("Option B registration still has lock blockers")
    teacher = registration["teacher_pass"]
    if teacher.get("teacher_14b_state_coverage_policy") != "all_admitted_anchors":
        raise RuntimeError("Option B requires all-admitted-anchor 14B states")
    if not teacher.get("per_anchor_label_tier_admission_required"):
        raise RuntimeError("Option B requires per-anchor label-tier receipts")
    return registration


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_digest(*parts: object, seed: int) -> str:
    return hashlib.sha256(
        ":".join([str(seed), *[str(part) for part in parts]]).encode("utf-8")
    ).hexdigest()


def build_cache_config(
    *,
    registration: dict[str, Any],
    data_path: str | Path,
    anchor_count: int,
    run_id: str,
    pilot: bool = False,
) -> dict[str, Any]:
    teacher = registration["teacher_pass"]
    if anchor_count % 2:
        raise ValueError("Option B anchor count must split evenly across strata")
    minimum = int(teacher["new_training_anchor_minimum"])
    target = int(teacher["new_training_anchor_target"])
    if pilot and not 0 < anchor_count < minimum:
        raise ValueError("Option B pilot must be smaller than the locked floor")
    if not pilot and anchor_count not in {minimum, target}:
        raise ValueError("Option B full cache must use the locked floor or target")
    data_path = Path(data_path)
    models = teacher["models"]
    return {
        "kind": "paper2_phase2_option_b_teacher_cache_config",
        "version": "option_b_teacher_cache_v1_20260806",
        "run_id": run_id,
        "execution_scope": "hardware_preflight_pilot" if pilot else "locked_full_cache",
        "data_partition": "OPTION_B_NEW_DOCUMENTS_TRAINING_ONLY",
        "data_sha256": sha256_file(data_path),
        "seed": int(teacher["selection_seed"]),
        "anchor_count": int(anchor_count),
        "anchors_per_stratum": {
            "general": int(anchor_count // 2),
            "code": int(anchor_count // 2),
        },
        "boundary_sample_count": int(anchor_count * len(teacher["horizons"])),
        "horizons": list(teacher["horizons"]),
        "top_k": int(teacher["top_k"]),
        "full_logit_audit_fraction": float(teacher["full_logit_audit_fraction"]),
        "selected_layer_ordinals_one_based": list(
            teacher["selected_layer_ordinals_one_based"]
        ),
        "teacher_state_model": {
            "key": "teacher_14b",
            "model": models["teacher_14b"]["model"],
            "revision": models["teacher_14b"]["revision"],
            "hidden_size": 5120,
            "num_hidden_layers": 48,
        },
        "models": models,
        "cascade": {
            "query_32b_on_7b_14b_argmax_disagreement": True,
            "query_32b_on_verifier_available": True,
            "stable_audit_fraction": float(teacher["full_logit_audit_fraction"]),
        },
        "teacher_14b_state_coverage_policy": "all_admitted_anchors",
        "per_anchor_label_tier_admission_required": True,
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
    }


def choose_full_anchor_count(
    *,
    target: int,
    floor: int,
    pilot_anchors: int,
    pilot_total_bytes: int,
    pilot_fixed_bytes: int,
    scratch_free_bytes: int,
    drive_free_bytes: int,
    reserve_fraction: float = 0.25,
) -> dict[str, Any]:
    if not 0 <= pilot_fixed_bytes <= pilot_total_bytes:
        raise ValueError("Option B pilot fixed-byte accounting is invalid")
    variable_per_anchor = (pilot_total_bytes - pilot_fixed_bytes) / pilot_anchors

    def projection(count: int) -> int:
        return int((pilot_fixed_bytes + variable_per_anchor * count) * (1 + reserve_fraction))

    target_bytes = projection(target)
    floor_bytes = projection(floor)
    available = min(int(scratch_free_bytes), int(drive_free_bytes))
    if available >= target_bytes:
        selected = target
        reason = "target_fits_with_25_percent_reserve"
    elif available >= floor_bytes:
        selected = floor
        reason = "target_does_not_fit_floor_fits_with_25_percent_reserve"
    else:
        raise RuntimeError(
            "Option B storage preflight cannot fit the locked 100,000-anchor floor"
        )
    return {
        "selected_anchor_count": selected,
        "reason": reason,
        "pilot_anchors": pilot_anchors,
        "pilot_total_bytes": pilot_total_bytes,
        "pilot_fixed_bytes": pilot_fixed_bytes,
        "variable_bytes_per_anchor": variable_per_anchor,
        "target_projected_bytes_with_reserve": target_bytes,
        "floor_projected_bytes_with_reserve": floor_bytes,
        "scratch_free_bytes": int(scratch_free_bytes),
        "drive_free_bytes": int(drive_free_bytes),
        "reserve_fraction": reserve_fraction,
    }


def build_anchor_admission_rows(
    samples: Iterable[dict[str, Any]], cascade_indices: set[int]
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for sample in samples:
        anchor_index = int(sample["anchor_index"])
        row = grouped.setdefault(
            anchor_index,
            {
                "anchor_index": anchor_index,
                "document_id": str(sample["document_id"]),
                "stratum": str(sample["stratum"]),
                "label_tier_admission": {
                    "teacher_7b_by_horizon": {},
                    "teacher_14b_by_horizon": {},
                    "teacher_32b_by_horizon": {},
                },
                "teacher_14b_states_by_horizon": {},
            },
        )
        horizon = str(int(sample["horizon"]))
        sample_index = int(sample["sample_index"])
        row["label_tier_admission"]["teacher_7b_by_horizon"][horizon] = True
        row["label_tier_admission"]["teacher_14b_by_horizon"][horizon] = True
        row["label_tier_admission"]["teacher_32b_by_horizon"][horizon] = (
            sample_index in cascade_indices
        )
        row["teacher_14b_states_by_horizon"][horizon] = True
    return [grouped[index] for index in sorted(grouped)]


def fixed_anchor_subset(
    admission_rows: Iterable[dict[str, Any]], *, count: int, seed: int
) -> list[int]:
    rows = list(admission_rows)
    if len(rows) < count:
        raise ValueError("Option B fixed new subset exceeds landed anchors")
    ranked = sorted(
        rows,
        key=lambda row: stable_digest(
            row["document_id"], row["anchor_index"], seed=seed
        ),
    )
    return sorted(int(row["anchor_index"]) for row in ranked[:count])
