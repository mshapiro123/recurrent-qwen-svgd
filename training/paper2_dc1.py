"""Locked no-training DC1-P constants and receipt helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from training.composite_training_design import COMPOSITE_TRAINING_POLICY


DEV_C_SEED = 20260729
DEV_C_TOKENS = 500_000
EVAL_C_SEED = 20260730
EVAL_C_TOKENS = 200_000
PREFLIGHT_POSITION_BUDGET = 50_000
STRATUM_FRACTIONS = {"general": 0.5, "code": 0.5}
SCALE_MULTIPLIERS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def document_manifest(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ids = sorted({str(row["document_id"]) for row in rows})
    payload = ("\n".join(ids) + "\n").encode("utf-8")
    return {
        "documents": len(ids),
        "document_id_list_sha256": hashlib.sha256(payload).hexdigest(),
    }


def assert_dc1_document_disjoint(
    rows: Iterable[dict[str, Any]],
    *,
    prior_document_ids: set[str],
    partition: str,
) -> dict[str, Any]:
    current = {str(row["document_id"]) for row in rows}
    overlap = sorted(current & set(prior_document_ids))
    if overlap:
        raise RuntimeError(f"{partition} overlaps prior documents: {overlap[:10]}")
    return {
        "partition": str(partition),
        "document_disjoint": True,
        "documents": len(current),
        "prior_documents": len(prior_document_ids),
        "overlap_count": 0,
    }


def eval_c_freeze_receipt(
    *,
    source_revisions: dict[str, Any],
    data_jsonl_sha256: str,
    private_manifest_sha256: str,
    teacher_cache_sha256: str,
    disjointness: dict[str, Any],
    teacher_model: str,
    teacher_revision: str,
) -> dict[str, Any]:
    """Build the hash-only public receipt without exposing EVAL-C outcomes."""

    return {
        "kind": "paper2_dc1_eval_c_freeze",
        "status": "complete_unread_unscored",
        "seed": EVAL_C_SEED,
        "tokens": EVAL_C_TOKENS,
        "source_revisions": dict(source_revisions),
        "mix": dict(STRATUM_FRACTIONS),
        "data_jsonl_sha256": str(data_jsonl_sha256),
        "manifest_sha256": str(private_manifest_sha256),
        "teacher_cache_sha256": str(teacher_cache_sha256),
        "document_disjointness": dict(disjointness),
        "teacher": {
            "model": str(teacher_model),
            "revision": str(teacher_revision),
            "passes": 1,
        },
        "read_log": [
            {
                "purpose": "single-pass teacher cache construction before scoring",
                "interpretive_scoring": False,
            }
        ],
        "scores_exposed": False,
        "read_once_scoring_spent": False,
        "training_started": False,
        "optimizer_steps": 0,
    }


def scale_interpolation_schedule(
    *,
    embedding_rms: float,
    raw_rms: float,
) -> list[dict[str, Any]]:
    if not 0 < float(embedding_rms) < float(raw_rms):
        raise ValueError("scale sweep requires 0 < embedding_rms < raw_rms")
    rows = [
        {
            "label": "matched" if value == 1.0 else f"{int(value)}x",
            "feedback_mode": "scaled",
            "feedback_scale": value,
            "target_rms": float(embedding_rms) * value,
        }
        for value in SCALE_MULTIPLIERS
        if float(embedding_rms) * value < float(raw_rms)
    ]
    rows.append(
        {
            "label": "raw",
            "feedback_mode": "raw",
            "feedback_scale": None,
            "target_rms": float(raw_rms),
        }
    )
    return rows


def dc1_preflight_spec() -> dict[str, Any]:
    return {
        "kind": "paper2_dc1_preflight_spec",
        "status": "authorized_no_training_preflight_only",
        "training_authorized": False,
        "evaluation_c_touched": False,
        "checkpoint": {
            "role": "post_d0_ema",
            "sha256": "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf",
        },
        "dev_c": {
            "tokens": DEV_C_TOKENS,
            "seed": DEV_C_SEED,
            "mix": dict(STRATUM_FRACTIONS),
            "document_disjoint_from_all_prior_partitions": True,
        },
        "horizontal_append_cap": COMPOSITE_TRAINING_POLICY["horizontal_append_cap"],
        "vertical_loops": 1,
        "preflight": {
            "position_budget_per_probe": PREFLIGHT_POSITION_BUDGET,
            "findings_are_gates": False,
            "probes": [
                "scale_interpolation",
                "slot_attention_profile",
                "position_id_ablation",
                "fragility_stratification_saved_outputs_only",
            ],
        },
        "rg_numerics": {
            "rg4": "two adjacent epsilon values pass the original ten-percent criterion",
            "rg11": "one declared precision policy passes cosine >= 0.99 at k=1,2,3",
            "rg12_authorized": False,
        },
        "control_readout": {
            "stage_c_ready": True,
            "execution_control_enabled": False,
            "visible_control_tokens": False,
        },
        "preconditions_before_training": [
            "dc1_p_packet_banked",
            "rg4_green",
            "rg11_green_with_declared_precision_policy",
            "dc1_preregistration_locked_to_drive_with_sha256",
        ],
        "open_for_markup": list(COMPOSITE_TRAINING_POLICY["open_for_markup"]),
    }
