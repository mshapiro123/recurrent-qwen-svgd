"""Phase 3.1 battery, split, lease, and paired sequential-stop contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t


SPLIT_SEED = 20260809
TARGET_PRIMARY = ("gsm8k", "mbpp")
TARGET_SECONDARY = ("arc_challenge",)
FLOOR_ONLY = ("arc_easy", "mmlu", "tier1")
ALL_BATTERIES = (*TARGET_PRIMARY, *TARGET_SECONDARY, *FLOOR_ONLY)
DEFAULT_MACRO_WEIGHTS = {name: 1.0 / len(ALL_BATTERIES) for name in ALL_BATTERIES}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_sha256(row: Mapping[str, Any]) -> str:
    fields = {
        key: row.get(key)
        for key in ("battery", "item_id", "prompt", "answer", "tests", "reader")
    }
    return canonical_sha256(fields)


def eval_half(document_id: str, *, seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}:eval-half:{document_id}".encode("utf-8")).digest()
    return "dev" if int.from_bytes(digest[:8], "big") % 2 == 0 else "confirm"


def battery_role(name: str) -> str:
    if name in TARGET_PRIMARY:
        return "target_primary"
    if name in TARGET_SECONDARY:
        return "target_secondary"
    if name in FLOOR_ONLY:
        return "floor_retention_only"
    raise ValueError(f"unregistered Phase 3 battery: {name}")


def _partition_row(row: Mapping[str, Any]) -> dict[str, Any]:
    battery = str(row["battery"])
    native_split = str(row["native_split"])
    document_id = str(row["document_id"])
    if not document_id:
        raise ValueError("document_id must be non-empty")
    if native_split == "train":
        if battery_role(battery) == "floor_retention_only":
            raise ValueError(f"floor-only battery cannot supply Phase 3 training rows: {battery}")
        partition = "verified_train"
        if not bool(row.get("programmatic_verifier_available")):
            raise ValueError(f"verified training row lacks verifier: {battery}/{document_id}")
    elif native_split == "evaluation":
        partition = eval_half(document_id)
    else:
        raise ValueError("native_split must be train or evaluation")
    return {
        **dict(row),
        "battery": battery,
        "battery_role": battery_role(battery),
        "document_id": document_id,
        "partition": partition,
        "content_sha256": content_sha256(row),
    }


def build_split_ledger(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_revisions: Mapping[str, str],
    reader_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Build a score-blind, document-disjoint ledger from source manifests."""

    missing_revisions = sorted(set(ALL_BATTERIES) - set(dataset_revisions))
    missing_readers = sorted(set(ALL_BATTERIES) - set(reader_versions))
    if missing_revisions or missing_readers:
        raise ValueError(
            f"P3.1 lock metadata incomplete revisions={missing_revisions} readers={missing_readers}"
        )
    partitioned = [_partition_row(row) for row in rows]
    if not partitioned:
        raise ValueError("P3.1 split ledger requires rows")

    seen_items: set[tuple[str, str]] = set()
    documents: dict[str, set[str]] = {"verified_train": set(), "dev": set(), "confirm": set()}
    for row in partitioned:
        key = (row["battery"], str(row["item_id"]))
        if key in seen_items:
            raise ValueError(f"duplicate battery item: {key}")
        seen_items.add(key)
        documents[row["partition"]].add(row["document_id"])
    overlaps = {
        "train_dev": sorted(documents["verified_train"] & documents["dev"]),
        "train_confirm": sorted(documents["verified_train"] & documents["confirm"]),
        "dev_confirm": sorted(documents["dev"] & documents["confirm"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"P3.1 document partition overlap: {overlaps}")

    counts: dict[str, dict[str, int]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for battery in ALL_BATTERIES:
        counts[battery] = {}
        hashes[battery] = {}
        for partition in ("verified_train", "dev", "confirm"):
            selected = [
                {
                    "item_id": str(row["item_id"]),
                    "document_id": row["document_id"],
                    "content_sha256": row["content_sha256"],
                }
                for row in partitioned
                if row["battery"] == battery and row["partition"] == partition
            ]
            selected.sort(key=lambda item: (item["document_id"], item["item_id"]))
            counts[battery][partition] = len(selected)
            hashes[battery][partition] = canonical_sha256(selected)

    rows_for_hash = [
        {
            "battery": row["battery"],
            "item_id": str(row["item_id"]),
            "document_id": row["document_id"],
            "partition": row["partition"],
            "content_sha256": row["content_sha256"],
        }
        for row in partitioned
    ]
    rows_for_hash.sort(
        key=lambda item: (item["battery"], item["partition"], item["document_id"], item["item_id"])
    )
    return {
        "kind": "paper2_phase3_p31_split_ledger_v1",
        "status": "score_blind_split_frozen",
        "split_seed": SPLIT_SEED,
        "dataset_revisions": dict(sorted(dataset_revisions.items())),
        "reader_versions": dict(sorted(reader_versions.items())),
        "battery_roles": {name: battery_role(name) for name in ALL_BATTERIES},
        "macro_weights": DEFAULT_MACRO_WEIGHTS,
        "counts": counts,
        "partition_hashes": hashes,
        "complete_ledger_sha256": canonical_sha256(rows_for_hash),
        "document_overlap": overlaps,
        "confirm_scoring_spent": False,
        "scores_computed": False,
        "limitations": {
            "public_benchmark_pretraining_contamination_excluded": False,
            "paired_augmented_minus_base_delta_is_primary": True,
        },
    }


def claim_partition_lease(
    path: Path,
    *,
    partition: str,
    run_id: str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    """Atomically claim a scoring lease; CONFIRM files are deliberately permanent."""

    if partition not in {"dev", "confirm"}:
        raise ValueError("lease partition must be dev or confirm")
    payload = {
        "kind": "paper2_phase3_partition_scoring_lease_v1",
        "partition": partition,
        "run_id": run_id,
        "preregistration_sha256": preregistration_sha256,
        "claimed_at_unix": time.time(),
        "scoring_started": False,
        "scoring_finished": False,
        "confirm_scoring_spent": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError(f"Phase 3 {partition} scoring lease already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def update_partition_lease(path: Path, payload: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("run_id") != payload.get("run_id"):
        raise RuntimeError("Phase 3 lease ownership changed")
    updated = {**current, **updates, "updated_at_unix": time.time()}
    if current["partition"] == "confirm" and bool(updated.get("scoring_started")):
        updated["confirm_scoring_spent"] = True
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return updated


def paired_upper_confidence_bound(
    differences: Sequence[float] | np.ndarray,
    *,
    alpha: float,
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired confidence bound requires at least two paired rows")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be a one-sided tail probability")
    standard_error = values.std(ddof=1) / math.sqrt(values.size)
    critical = float(student_t.ppf(1.0 - alpha, df=values.size - 1))
    return float(values.mean() + critical * standard_error)


def paired_false_stop(
    differences_by_look: Sequence[Sequence[float] | np.ndarray],
    *,
    alpha: float,
    margin: float = -0.03,
    consecutive: int = 2,
) -> dict[str, Any]:
    upper_bounds = [
        paired_upper_confidence_bound(values, alpha=alpha)
        for values in differences_by_look
    ]
    below = [bound < margin for bound in upper_bounds]
    run = 0
    stop_look = None
    for index, crossed in enumerate(below, start=1):
        run = run + 1 if crossed else 0
        if run >= consecutive:
            stop_look = index
            break
    return {
        "stopped": stop_look is not None,
        "stop_look": stop_look,
        "upper_bounds": upper_bounds,
        "below_margin": below,
    }


@dataclass(frozen=True)
class PairedNullDesign:
    rows: int
    looks: int = 20
    margin: float = -0.03
    discordant_probability: float = 0.20
    adjacent_correlation: float = 0.80

    def validate(self) -> None:
        if self.rows < 2 or self.looks < 2:
            raise ValueError("paired null design requires rows and looks")
        if not abs(self.margin) < self.discordant_probability < 1.0:
            raise ValueError("discordance must exceed the absolute null mean")
        if not 0.0 <= self.adjacent_correlation < 1.0:
            raise ValueError("adjacent correlation must be in [0, 1)")


def simulate_false_stop_probability(
    design: PairedNullDesign,
    *,
    alpha: float,
    campaigns: int,
    seed: int,
    batch_campaigns: int = 128,
) -> dict[str, Any]:
    """Simulate the registered paired looks at the non-inferiority boundary."""

    design.validate()
    if campaigns <= 0 or batch_campaigns <= 0:
        raise ValueError("simulation campaign counts must be positive")
    rng = np.random.default_rng(seed)
    negative_probability = (design.discordant_probability - design.margin) / 2.0
    positive_probability = (design.discordant_probability + design.margin) / 2.0
    if min(negative_probability, positive_probability) < 0:
        raise ValueError("paired null probabilities are invalid")
    negative_cut = float(_normal_ppf(negative_probability))
    positive_cut = float(_normal_ppf(1.0 - positive_probability))
    stopped = 0
    completed = 0
    while completed < campaigns:
        batch = min(batch_campaigns, campaigns - completed)
        latent = rng.standard_normal((batch, design.rows))
        decisions = np.zeros((batch, design.looks), dtype=bool)
        scale = math.sqrt(1.0 - design.adjacent_correlation**2)
        for look in range(design.looks):
            if look:
                latent = design.adjacent_correlation * latent + scale * rng.standard_normal(
                    latent.shape
                )
            differences = np.where(latent < negative_cut, -1.0, 0.0)
            differences = np.where(latent > positive_cut, 1.0, differences)
            means = differences.mean(axis=1)
            standard_errors = differences.std(axis=1, ddof=1) / math.sqrt(design.rows)
            critical = float(student_t.ppf(1.0 - alpha, df=design.rows - 1))
            decisions[:, look] = means + critical * standard_errors < design.margin
        stopped += int(np.any(decisions[:, 1:] & decisions[:, :-1], axis=1).sum())
        completed += batch
    probability = stopped / campaigns
    upper_95 = (
        1.0
        if stopped == campaigns
        else float(beta_distribution.ppf(0.95, stopped + 1, campaigns - stopped))
    )
    return {
        "kind": "paper2_phase3_p31_false_stop_simulation_v1",
        "seed": seed,
        "campaigns": campaigns,
        "rows": design.rows,
        "looks": design.looks,
        "null_mean_difference": design.margin,
        "discordant_probability": design.discordant_probability,
        "adjacent_correlation": design.adjacent_correlation,
        "one_sided_alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "consecutive_looks_required": 2,
        "false_stops": stopped,
        "estimated_familywise_false_stop_probability": probability,
        "conservative_upper_95_probability": upper_95,
        "target_probability": 1e-4,
        "target_met_by_conservative_upper": upper_95 < 1e-4,
    }


def _normal_ppf(probability: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(probability)
