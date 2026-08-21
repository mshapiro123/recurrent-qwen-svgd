"""Locked decision rules and CPU audit helpers for the Stage 2B-S preludes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


LOCK_KIND = "paper2_stage2bs_preludes_lock_v1"
EXPECTED_K_SWEEP = {0: [162, 10, 2, 160], 1: [162, 9, 5, 161]}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lock(lock: Mapping[str, Any], *, require_signed: bool = True) -> None:
    if lock.get("kind") != LOCK_KIND:
        raise RuntimeError("wrong Stage 2B-S prelude lock kind")
    if require_signed and (
        lock.get("status") != "SIGNED" or lock.get("mark_signed") is not True
    ):
        raise RuntimeError("Stage 2B-S prelude lock is not signed")
    if lock.get("training_authorized") is not False:
        raise RuntimeError("Stage 2B-S preludes must not authorize training")
    if int(lock.get("optimizer_steps_allowed", -1)) != 0:
        raise RuntimeError("Stage 2B-S prelude optimizer-step allowance changed")
    authority = lock.get("authority", {})
    if (
        authority.get("drive_id") != "17nYHGA1dzY-G-aC614ynkSspt7_lrPbk"
        or authority.get("bytes") != 7558
        or authority.get("sha256")
        != "0c738c28d5c759530f6c6ce9e35afcb78bc38af14e6d06e289688a0c584be566"
    ):
        raise RuntimeError("Stage 2B-S prelude authority changed")
    runtime = lock.get("runtime", {})
    if runtime.get("weights_dtype") != "bfloat16" or runtime.get("attention_backend") != "sdpa":
        raise RuntimeError("Stage 2B-S runtime semantics changed")
    preflight = lock.get("preflight", {})
    expected = {str(seed): values for seed, values in EXPECTED_K_SWEEP.items()}
    if preflight.get("expected_correct_by_seed_and_k") != expected:
        raise RuntimeError("Stage 2B-S preflight expectation changed")
    probes = lock.get("prelude_1", {})
    if probes.get("noise_epsilons") != [0.001, 0.003, 0.01, 0.03, 0.1]:
        raise RuntimeError("Stage 2B-S noise grid changed")
    if probes.get("zero_inherited_flow_reentry_indices") != [1, 2]:
        raise RuntimeError("Stage 2B-S K2/K3 zero-write indices changed")
    if probes.get("transplant_pairs_per_seed") != 64:
        raise RuntimeError("Stage 2B-S transplant pair count changed")


def load_lock(path: str | Path, *, require_signed: bool = True) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_lock(lock, require_signed=require_signed)
    return lock


def noise_verdict(retained_fraction: Mapping[float, float]) -> str:
    ordered = sorted((float(k), float(v)) for k, v in retained_fraction.items())
    low = [value for epsilon, value in ordered if epsilon <= 0.003]
    through = [value for epsilon, value in ordered if epsilon <= 0.03]
    monotone = all(right <= left + 1e-12 for (_, left), (_, right) in zip(ordered, ordered[1:]))
    if low and min(low) <= 0.5:
        return "SHATTERS"
    if through and min(through) >= 0.5 and monotone:
        return "SMOOTH"
    return "MIXED"


def dependency_verdict(native_correct: int, ablated_correct: int) -> str:
    fraction = ablated_correct / max(native_correct, 1)
    if fraction >= 0.8:
        return "SURVIVES"
    if fraction <= 0.5:
        return "DEPENDENT"
    return "MIXED"


def transplant_verdict(native_pair_correct: int, transplanted_pair_correct: int) -> str:
    fraction = transplanted_pair_correct / max(native_pair_correct, 1)
    if fraction >= 0.5:
        return "GRACEFUL"
    if fraction <= 0.15:
        return "CATASTROPHIC"
    return "MIXED"


def prelude1_decision(seed_verdicts: Sequence[Mapping[str, str]]) -> str:
    tuples = [
        (row["noise"], row["dependency"], row["transplant"])
        for row in seed_verdicts
    ]
    if len(set(tuples)) != 1 or any("MIXED" in values for values in tuples):
        return "ESCALATE_STRATEGY"
    values = tuples[0]
    if values == ("SMOOTH", "SURVIVES", "GRACEFUL"):
        return "REUSABLE_COMPUTATION"
    if values == ("SHATTERS", "DEPENDENT", "CATASTROPHIC"):
        return "PHASE_CANCELLATION"
    return "ESCALATE_STRATEGY"


def relative_frobenius(initial: torch.Tensor, final: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(initial.float()))
    return float(torch.linalg.vector_norm(final.float() - initial.float())) / max(
        denominator, 1e-12
    )


def top_singular_receipt(delta: torch.Tensor, rank: int = 3) -> dict[str, Any]:
    matrix = delta.float().reshape(delta.shape[0], -1)
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    energy = singular.square()
    fractions = energy / energy.sum().clamp_min(1e-12)
    value_fractions = singular / singular.sum().clamp_min(1e-12)
    count = min(rank, singular.numel())
    return {
        "singular_values": singular[:count].tolist(),
        "singular_value_fractions": value_fractions[:count].tolist(),
        "singular_energy_fractions": fractions[:count].tolist(),
        "right_singular_vectors": vh[:count].cpu(),
    }


def correction_references(
    artifact: Mapping[str, Any], *, seed: int, clusters: int = 2, iterations: int = 50
) -> dict[str, torch.Tensor]:
    from training.paper2_stage2b_autopsy import spherical_kmeans

    corrections = artifact["corrections"][4].float()
    directions = F.normalize(corrections, dim=-1, eps=1e-12)
    common = F.normalize(directions.mean(dim=0), dim=0, eps=1e-12)
    labels, _silhouette = spherical_kmeans(
        directions,
        clusters=clusters,
        restarts=8,
        iterations=iterations,
        seed=seed + clusters,
    )
    centers = torch.stack(
        [
            F.normalize(directions[labels == index].mean(dim=0), dim=0, eps=1e-12)
            for index in range(clusters)
        ]
    )
    return {"common_mode": common, "cluster_centroids": centers}


def alignment_receipt(vectors: torch.Tensor, references: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, reference in references.items():
        ref = reference.float()
        if ref.ndim == 1:
            ref = ref.unsqueeze(0)
        cosine = F.normalize(vectors.float(), dim=-1) @ F.normalize(ref, dim=-1).T
        result[name] = cosine.tolist()
    return result


def starvation_verdict(ratios: Sequence[float]) -> str:
    values = [float(value) for value in ratios]
    if len(values) != 2:
        raise ValueError("Prelude-2 requires exactly two seed ratios")
    if all(value <= 0.25 for value in values):
        return "STARVED"
    if any(value >= 0.75 for value in values):
        return "NOT_STARVED"
    return "PARTIAL"
