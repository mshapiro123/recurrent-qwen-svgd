"""Pure contracts for score-blind E1 EVAL-D cache generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.paper2_phase2_e1_confirmation import (
    E1_EVAL_D_FREEZE_KIND,
    OPTION_B_CACHE_KIND,
    REQUIRED_CACHE_FIELDS,
    sha256_file,
)
from training.paper2_phase2_matched_alpha import document_partition


RUN_ID = "stage5_paper2_phase2_e1_eval_d_20260808"
SELECTION_SEED = 20260808
ANCHORS_PER_STRATUM = {"general": 4_000, "code": 4_000}
HORIZONS = [1, 2, 3, 4]
SELECTION_RULE = (
    "stable_sha256_rank_within_stratum_then_greedy_nonoverlapping_"
    "four_position_spans_then_row_major_execution_order"
)
OPTION_B_FRESH_CASCADE_FRACTION = 0.16748046875


def build_score_blind_config(
    *, registration: Mapping[str, Any], data_path: str | Path
) -> dict[str, Any]:
    """Copy the locked Option B teacher stack while changing only population."""

    teacher = registration["teacher_pass"]
    models = teacher["models"]
    anchor_count = sum(ANCHORS_PER_STRATUM.values())
    return {
        "kind": "paper2_phase2_e1_eval_d_cache_config",
        "version": "paper2_phase2_e1_eval_d_cache_v1_20260808",
        "run_id": RUN_ID,
        "execution_scope": "frozen_eval_d_score_blind_infrastructure_only",
        "data_partition": "EVAL-D_CACHE_ONLY",
        "data_sha256": sha256_file(data_path),
        "seed": SELECTION_SEED,
        "selection_rule": SELECTION_RULE,
        "anchor_count": anchor_count,
        "anchors_per_stratum": dict(ANCHORS_PER_STRATUM),
        "boundary_sample_count": anchor_count * len(HORIZONS),
        "horizons": list(HORIZONS),
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
        "score_blind": True,
        "endpoint_checkpoints_loaded": False,
        "model_quality_scores_computed": False,
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": ["EVAL-D_CACHE_ONLY"],
    }


def dev_mixture_weights(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Recover the immutable DEV evaluation mixture used by Option B."""

    anchors: dict[int, tuple[str, str]] = {}
    for sample in samples:
        anchor = int(sample["anchor_index"])
        anchors.setdefault(
            anchor, (str(sample["document_id"]), str(sample["stratum"]))
        )
    ordered = [anchors[index] for index in sorted(anchors)]
    mask = document_partition(
        [document for document, _stratum in ordered],
        evaluation_fraction=0.2,
        seed=20260804,
    )
    counts = Counter(
        stratum for selected, (_document, stratum) in zip(mask.tolist(), ordered) if selected
    )
    total = sum(counts.values())
    if total <= 0 or set(counts) != {"general", "code"}:
        raise RuntimeError("immutable DEV mixture is missing a registered stratum")
    return {
        "source": "immutable_stage0a_dev_manifest_under_document_partition_seed_20260804",
        "anchor_count": total,
        "counts": dict(sorted(counts.items())),
        "weights": {key: counts[key] / total for key in ("general", "code")},
    }


def build_freeze_receipt(
    *,
    cache: Mapping[str, Any],
    private_cache_path: Path,
    data_sha256: str,
    document_count: int,
    canonicalizer_sha256: str,
    sample_manifest_sha256: str,
    position_key_sha256_value: str,
    admission_ledger_sha256: str,
    cascade_count: int,
    dev_mixture: Mapping[str, Any],
    model_revisions: Mapping[str, str],
    cross_partition_document_overlap: list[str],
) -> dict[str, Any]:
    anchor_count = len(cache["documents"])
    strata = Counter(str(value) for value in cache["strata"])
    if anchor_count != 8_000 or strata != Counter(ANCHORS_PER_STRATUM):
        raise RuntimeError(f"EVAL-D anchor contract mismatch: {anchor_count=} {strata=}")
    fields = sorted(set(cache) & REQUIRED_CACHE_FIELDS)
    if set(fields) != REQUIRED_CACHE_FIELDS:
        raise RuntimeError("EVAL-D cache is missing evaluator-required fields")
    if cross_partition_document_overlap:
        raise RuntimeError("EVAL-D overlaps a training or development partition")
    cascade_fraction = cascade_count / (anchor_count * len(HORIZONS))
    return {
        "kind": E1_EVAL_D_FREEZE_KIND,
        "version": "paper2_phase2_e1_eval_d_freeze_v1_20260808",
        "status": "complete_frozen_unscored",
        "partition": "eval_d",
        "scores_exposed": False,
        "read_once_scoring_spent": False,
        "training_started": False,
        "optimizer_steps": 0,
        "endpoint_checkpoints_loaded": False,
        "model_quality_scores_computed": False,
        "eal_computed": False,
        "retention_computed": False,
        "acceptance_computed": False,
        "student_teacher_quality_aggregates_emitted": False,
        "base_student_forward_role": (
            "tensor_cache_materialization_only_no_quality_or_outcome_scoring"
        ),
        "cross_partition_document_overlap": [],
        "selection": {
            "seed": SELECTION_SEED,
            "rule": SELECTION_RULE,
            "anchors_per_stratum": dict(ANCHORS_PER_STRATUM),
        },
        "estimators": {
            "primary": {
                "population": "balanced_eval_d",
                "weights": {"general": 0.5, "code": 0.5},
            },
            "dev_mixture_reweighted_secondary": dict(dev_mixture),
        },
        "teacher_stack": {
            "model_revisions": dict(model_revisions),
            "teacher_14b_states_for_all_anchors": True,
            "admission_ledger_sha256": admission_ledger_sha256,
            "cascade_count": int(cascade_count),
            "cascade_fraction": cascade_fraction,
            "option_b_fresh_document_cascade_fraction": OPTION_B_FRESH_CASCADE_FRACTION,
            "cascade_fraction_delta": cascade_fraction - OPTION_B_FRESH_CASCADE_FRACTION,
            "role": "population_note_not_outcome_score",
        },
        "option_b_cache": {
            "kind": OPTION_B_CACHE_KIND,
            "fields": fields,
            "anchor_count": anchor_count,
            "anchors_per_stratum": dict(sorted(strata.items())),
            "document_count": int(document_count),
            "data_sha256": str(data_sha256),
            "position_key_sha256": str(position_key_sha256_value),
            "sample_manifest_sha256": str(sample_manifest_sha256),
            "private_cache_sha256": sha256_file(private_cache_path),
            "canonicalizer_sha256": str(canonicalizer_sha256),
        },
        "allowed_public_outputs_only": [
            "hashes",
            "counts",
            "model_revisions",
            "cascade_fraction",
            "integrity_telemetry",
        ],
    }
