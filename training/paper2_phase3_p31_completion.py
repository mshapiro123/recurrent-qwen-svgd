"""Completion contracts for the Phase 3.1 currency assembly.

This module is intentionally model-free.  It seals CONFIRM membership, builds
the six-cohort sentinel panel, and summarizes already-produced DEV/verified
predictions without providing any route that can score CONFIRM.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from training.paper2_phase3_p31 import ALL_BATTERIES, canonical_sha256


SENTINEL_SEED = 20260810
SENTINEL_SIZE = 2_048
SENTINEL_COHORTS = (
    "consensus_no_op",
    "stable_missing_knowledge",
    "procedural_reasoning",
    "mixed",
    "paired_counterfactuals",
    "paraphrase_ood",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if exclusive:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            raise RuntimeError(f"Phase 3 immutable artifact already exists: {path}") from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def seal_confirm_membership(
    ledger: Mapping[str, Any],
    *,
    output_dir: Path,
    source_rows_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Write immutable, score-blind CONFIRM seals before any model loads."""

    if bool(ledger.get("scores_computed")) or bool(ledger.get("confirm_scoring_spent")):
        raise RuntimeError("CONFIRM must be sealed before any score is computed")
    seals = []
    for battery in ALL_BATTERIES:
        payload = {
            "kind": "paper2_phase3_p31_confirm_membership_seal_v1",
            "battery": battery,
            "partition": "confirm",
            "row_count": int(ledger["counts"][battery]["confirm"]),
            "membership_sha256": str(ledger["partition_hashes"][battery]["confirm"]),
            "complete_ledger_sha256": str(ledger["complete_ledger_sha256"]),
            "source_rows_sha256": source_rows_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "split_seed": int(ledger["split_seed"]),
            "scoring_authorized": False,
            "scoring_spent": False,
            "atomic_lease_required_for_future_scoring": True,
        }
        path = output_dir / f"confirm_{battery}.seal.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError(f"existing CONFIRM seal changed: {path}")
        else:
            _atomic_json(path, payload, exclusive=True)
        seals.append({"battery": battery, "path": str(path), "sha256": sha256_file(path), **payload})
    receipt = {
        "kind": "paper2_phase3_p31_confirm_seal_ledger_v1",
        "status": "sealed_before_model_scoring",
        "seals": seals,
        "seal_set_sha256": canonical_sha256(
            [{"battery": row["battery"], "sha256": row["sha256"]} for row in seals]
        ),
        "models_loaded": False,
        "scores_computed": False,
        "confirm_scoring_spent": False,
        "assertions": {
            "confirm_membership_sealed": len(seals) == len(ALL_BATTERIES),
            "models_not_loaded": True,
            "scores_not_computed": True,
            "confirm_scoring_unspent": True,
        },
    }
    _atomic_json(output_dir / "confirm_seal_ledger.json", receipt)
    return receipt


def verified_label_class(*, student_right: bool, teacher_right: bool) -> str:
    if teacher_right and not student_right:
        return "teacher_right_student_wrong"
    if not teacher_right:
        return "teacher_wrong"
    return "both_correct"


def verified_stratum_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()
    seen = set()
    for row in rows:
        if row.get("partition") != "verified_train":
            raise ValueError("verified-stratum receipt contains a non-training row")
        key = (str(row["battery"]), str(row["item_id"]))
        if key in seen:
            raise ValueError(f"duplicate verified result: {key}")
        seen.add(key)
        label = verified_label_class(
            student_right=bool(row["base_correct"]),
            teacher_right=bool(row["teacher_14b_correct"]),
        )
        by_family[key[0]][label] += 1
        total[label] += 1
    labels = ("teacher_right_student_wrong", "teacher_wrong", "both_correct")
    return {
        "kind": "paper2_phase3_p31_verified_stratum_counts_v1",
        "counts_by_family": {
            family: {label: int(counter[label]) for label in labels} | {"total": sum(counter.values())}
            for family, counter in sorted(by_family.items())
        },
        "counts_all": {label: int(total[label]) for label in labels} | {"total": sum(total.values())},
        "label_classes": list(labels),
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("document bootstrap requires a non-empty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 512):
        stop = min(replicates, start + 512)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        estimates[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def reference_score_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260810,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Summarize paired base/14B DEV scores with document bootstrap intervals."""

    by_battery: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("partition") != "dev":
            raise ValueError("reference table is DEV-only; CONFIRM contact is prohibited")
        by_battery[str(row["battery"])].append(row)
    table = {}
    for offset, battery in enumerate(ALL_BATTERIES):
        selected = by_battery.get(battery, [])
        if not selected:
            raise ValueError(f"DEV reference rows missing battery {battery}")
        documents: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            documents[str(row["document_id"])].append(row)
        document_rows = []
        for document_id, group in sorted(documents.items()):
            base = float(np.mean([bool(row["base_correct"]) for row in group]))
            teacher = float(np.mean([bool(row["teacher_14b_correct"]) for row in group]))
            document_rows.append((document_id, base, teacher))
        base_values = np.asarray([row[1] for row in document_rows], dtype=np.float64)
        teacher_values = np.asarray([row[2] for row in document_rows], dtype=np.float64)
        delta = teacher_values - base_values
        delta_ci = _bootstrap_mean_interval(
            delta,
            seed=bootstrap_seed + offset,
            replicates=bootstrap_replicates,
        )
        denominator_touches_zero = delta_ci[0] <= 0.0 <= delta_ci[1]
        table[battery] = {
            "role": str(selected[0]["battery_role"]),
            "items": len(selected),
            "documents": len(document_rows),
            "base_correct": int(sum(bool(row["base_correct"]) for row in selected)),
            "teacher_14b_correct": int(
                sum(bool(row["teacher_14b_correct"]) for row in selected)
            ),
            "base_accuracy": float(np.mean([bool(row["base_correct"]) for row in selected])),
            "teacher_14b_accuracy": float(
                np.mean([bool(row["teacher_14b_correct"]) for row in selected])
            ),
            "teacher_minus_base_delta": float(delta.mean()),
            "teacher_minus_base_document_bootstrap_95_ci": list(delta_ci),
            "augmented_accuracy": None,
            "augmented_minus_base_delta": None,
            "gap_closed": None,
            "gap_closed_formula": "(augmented_accuracy-base_accuracy)/(teacher_14b_accuracy-base_accuracy)",
            "gap_closed_pending_augmented_model": True,
            "gap_closed_denominator_stable": not denominator_touches_zero,
            "reporting_rule": (
                "delta_and_gap_closed"
                if not denominator_touches_zero
                else "delta_only_denominator_ci_touches_zero"
            ),
        }
    return {
        "kind": "paper2_phase3_p31_reference_score_table_v1",
        "status": "dev_references_complete_confirm_unscored",
        "models": {
            "base": "Qwen/Qwen2.5-0.5B-Instruct",
            "teacher_14b": "Qwen/Qwen2.5-14B-Instruct",
        },
        "bootstrap": {
            "unit": "document",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
        },
        "batteries": table,
        "headline_batteries": [
            battery
            for battery in ALL_BATTERIES
            if table[battery]["role"] != "floor_retention_only"
        ],
        "floor_batteries_excluded_from_headline_numerator": [
            battery
            for battery in ALL_BATTERIES
            if table[battery]["role"] == "floor_retention_only"
        ],
        "confirm_scoring_spent": False,
    }


def _stable_rank(row: Mapping[str, Any], cohort: str, seed: int) -> bytes:
    return hashlib.sha256(
        f"{seed}:{cohort}:{row['battery']}:{row['item_id']}".encode("utf-8")
    ).digest()


def _alternate_variant(row: Mapping[str, Any], cohort: str) -> dict[str, Any]:
    variant = "source"
    if cohort == "paired_counterfactuals":
        variant = "deterministic_choice_order_counterfactual"
    elif cohort == "paraphrase_ood":
        variant = "deterministic_alternate_layout_ood"
    return {
        "sentinel_id": canonical_sha256(
            {"cohort": cohort, "battery": row["battery"], "item_id": row["item_id"]}
        ),
        "cohort": cohort,
        "battery": str(row["battery"]),
        "battery_role": str(row["battery_role"]),
        "source_partition": str(row["partition"]),
        "source_item_id": str(row["item_id"]),
        "source_document_id": str(row["document_id"]),
        "source_content_sha256": str(row["content_sha256"]),
        "variant_contract": variant,
        "deployment_visible": False,
        "audit_only": True,
    }


def build_sentinel_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    size: int = SENTINEL_SIZE,
    seed: int = SENTINEL_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic six-cohort panel from verified-train and DEV only."""

    if size < len(SENTINEL_COHORTS):
        raise ValueError("sentinel panel is too small for six cohorts")
    allowed = [row for row in rows if row["partition"] in {"verified_train", "dev"}]
    if any(row["partition"] == "confirm" for row in allowed):
        raise RuntimeError("sentinel assembly touched CONFIRM")
    score_lookup = {
        (str(row["battery"]), str(row["item_id"])): row for row in scored_rows
    }
    candidates: dict[str, list[Mapping[str, Any]]] = {}
    candidates["consensus_no_op"] = [
        row
        for row in allowed
        if (score := score_lookup.get((str(row["battery"]), str(row["item_id"]))))
        and bool(score["base_correct"])
        and bool(score["teacher_14b_correct"])
    ]
    candidates["stable_missing_knowledge"] = [
        row
        for row in allowed
        if (score := score_lookup.get((str(row["battery"]), str(row["item_id"]))))
        and not bool(score["base_correct"])
        and bool(score["teacher_14b_correct"])
    ]
    candidates["procedural_reasoning"] = [
        row for row in allowed if row["battery"] in {"gsm8k", "mbpp"}
    ]
    candidates["mixed"] = [
        row
        for row in allowed
        if (score := score_lookup.get((str(row["battery"]), str(row["item_id"]))))
        and bool(score["base_correct"]) != bool(score["teacher_14b_correct"])
    ]
    candidates["paired_counterfactuals"] = [
        row for row in allowed if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}
    ]
    candidates["paraphrase_ood"] = [
        row for row in allowed if row["battery"] in {"arc_easy", "arc_challenge", "mmlu", "gsm8k"}
    ]

    base_quota, remainder = divmod(size, len(SENTINEL_COHORTS))
    quotas = {
        cohort: base_quota + (index < remainder)
        for index, cohort in enumerate(SENTINEL_COHORTS)
    }
    panel = []
    for cohort in SENTINEL_COHORTS:
        ranked = sorted(candidates[cohort], key=lambda row: _stable_rank(row, cohort, seed))
        if len(ranked) < quotas[cohort]:
            raise RuntimeError(
                f"sentinel cohort {cohort} is too thin: {len(ranked)} < {quotas[cohort]}"
            )
        panel.extend(_alternate_variant(row, cohort) for row in ranked[: quotas[cohort]])
    ids = [row["sentinel_id"] for row in panel]
    if len(ids) != len(set(ids)):
        raise RuntimeError("sentinel variant ids are not unique")
    protocol = {
        "kind": "paper2_phase3_per_loop_diagnostic_coda_protocol_v1",
        "scope": "sentinel_panel_only",
        "execution": "after_each_loop_k_at_each_saved_checkpoint",
        "audit_only": True,
        "deployment_path_unchanged": True,
        "per_loop_outputs": [
            "accuracy_by_battery_and_cohort",
            "per_example_marginal_gain_delta_i_k",
            "compute_normalized_utility_eta_k",
            "state_and_write_sketch_receipts",
        ],
        "randomness_contract": "instrumentation_must_not_perturb_rng_precision_or_kernels",
    }
    receipt = {
        "kind": "paper2_phase3_p31_sentinel_panel_receipt_v1",
        "status": "frozen_train_and_dev_only",
        "seed": seed,
        "rows": len(panel),
        "cohort_counts": dict(Counter(row["cohort"] for row in panel)),
        "panel_sha256": canonical_sha256(panel),
        "source_partitions": sorted({row["source_partition"] for row in panel}),
        "confirm_rows": sum(row["source_partition"] == "confirm" for row in panel),
        "diagnostic_coda": protocol,
    }
    return panel, receipt


def gap_closed(*, augmented: float, base: float, teacher: float) -> dict[str, Any]:
    delta = float(augmented - base)
    denominator = float(teacher - base)
    return {
        "augmented_minus_base": delta,
        "teacher_minus_base": denominator,
        "gap_closed": None if math.isclose(denominator, 0.0) else delta / denominator,
    }
