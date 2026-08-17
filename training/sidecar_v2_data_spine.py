"""Deterministic, loss-free Stage 2A functional-fingerprint utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch.nn import functional as F


STAGE2A_TEACHER_14B = {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
}
STAGE2A_VERIFIER_32B = {
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["battery"]), str(row["item_id"])


def _seeded_row_rank(row: Mapping[str, Any], *, seed: int) -> tuple[bytes, str, str]:
    identity = f"{seed}:{row['battery']}:{row['item_id']}:{row['content_sha256']}"
    return (
        hashlib.sha256(identity.encode("utf-8")).digest(),
        str(row["battery"]),
        str(row["item_id"]),
    )


def _battery_stratified_select(
    rows: Iterable[Mapping[str, Any]], *, count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select an exact count with deterministic largest-remainder quotas."""

    records = [dict(row) for row in rows]
    if not 1 <= int(count) <= len(records):
        raise ValueError("stratified selection count must be in [1, population]")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row["battery"]), []).append(row)
    total = len(records)
    exact = {battery: count * len(group) / total for battery, group in groups.items()}
    quotas = {battery: int(value) for battery, value in exact.items()}
    remaining = int(count) - sum(quotas.values())
    priority = sorted(groups, key=lambda battery: (-(exact[battery] - quotas[battery]), battery))
    for battery in priority[:remaining]:
        quotas[battery] += 1
    selected: list[dict[str, Any]] = []
    for battery in sorted(groups):
        ranked = sorted(groups[battery], key=lambda row: _seeded_row_rank(row, seed=seed))
        selected.extend(ranked[: quotas[battery]])
    selected.sort(key=lambda row: _seeded_row_rank(row, seed=seed))
    if len(selected) != int(count):
        raise RuntimeError("battery-stratified selection did not produce the requested count")
    return selected, quotas


def largest_power_of_two_memory_size(admitted_count: int, *, cap: int = 4_096) -> int:
    """Apply the registered automatic Stage 2A memory-size fallback ladder."""

    available = min(int(admitted_count), int(cap))
    if available < 1:
        raise RuntimeError("no post-concurrence rows are available for Stage 2A memory")
    return 1 << (available.bit_length() - 1)


def select_stage2a_validation_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    panel_item_ids: set[tuple[str, str]],
    correct_field: str = "teacher_14b_correct",
    count: int = 512,
    seed: int = 20_260_817,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the ratified battery-stratified pre-concurrence validation set."""

    records = [dict(row) for row in rows]
    candidates = [
        row
        for row in records
        if bool(row.get(correct_field))
        and _row_identity(row) not in panel_item_ids
        and str(row.get("partition")) == "verified_train"
    ]
    selected, quotas = _battery_stratified_select(candidates, count=count, seed=seed)
    identities = [
        {
            "battery": str(row["battery"]),
            "item_id": str(row["item_id"]),
            "content_sha256": str(row["content_sha256"]),
        }
        for row in selected
    ]
    receipt = {
        "kind": "paper2_stage2a_nondev_validation_manifest_v1",
        "status": "selected_before_family_concurrence",
        "seed": int(seed),
        "rows": int(count),
        "eligible_14b_correct_nondev_rows": len(candidates),
        "battery_quotas": dict(sorted(quotas.items())),
        "selection_rule": (
            "battery-proportional Hamilton quotas; within-battery "
            "SHA256(seed:battery:item_id:content_sha256)"
        ),
        "selected_identities_sha256": _canonical_sha256(identities),
        "memory_overlap": 0,
        "dev_overlap": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    return selected, receipt


def select_stage2a_geometry_population(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int = 1_024,
    seed: int = 20_260_817,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the bounded non-DEV population used to fit and audit PCA geometry."""

    records = [dict(row) for row in rows]
    if any(str(row.get("partition")) != "verified_train" for row in records):
        raise RuntimeError("Stage 2A geometry population must be verified-train only")
    selected, quotas = _battery_stratified_select(records, count=count, seed=seed)
    identities = [
        {
            "battery": str(row["battery"]),
            "item_id": str(row["item_id"]),
            "content_sha256": str(row["content_sha256"]),
        }
        for row in selected
    ]
    receipt = {
        "kind": "paper2_stage2a_geometry_population_v1",
        "status": "selected_nondev_before_state_extraction",
        "seed": int(seed),
        "rows": int(count),
        "source_rows": len(records),
        "battery_quotas": dict(sorted(quotas.items())),
        "selection_rule": (
            "battery-proportional Hamilton quotas; within-battery "
            "SHA256(seed:battery:item_id:content_sha256)"
        ),
        "selected_identities_sha256": _canonical_sha256(identities),
        "dev_rows_used": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    return selected, receipt


def apply_stage2a_firm_knowledge_rule(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize V(x): correct 14B output plus 14B/32B concurrence.

    Model inference and answer normalization remain upstream registered reader
    operations. This boundary requires their row-level outputs and hashes, then
    applies the binary admission rule without confidence thresholds.
    """

    required = {
        "battery",
        "item_id",
        "teacher_14b_correct",
        "teacher_14b_normalized_answer",
        "teacher_32b_normalized_answer",
        "teacher_14b_output_sha256",
        "teacher_32b_output_sha256",
        "correctness_reader",
    }
    materialized: list[dict[str, Any]] = []
    excluded_by_battery: dict[str, dict[str, int]] = {}
    for source in rows:
        row = dict(source)
        missing = required.difference(row)
        if missing:
            raise ValueError(f"V(x) row missing required fields: {sorted(missing)}")
        for field in ("teacher_14b_output_sha256", "teacher_32b_output_sha256"):
            value = str(row[field])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"V(x) row has invalid {field}")
        teacher_correct = bool(row["teacher_14b_correct"])
        concurrence = (
            str(row["teacher_14b_normalized_answer"])
            == str(row["teacher_32b_normalized_answer"])
        )
        admitted = teacher_correct and concurrence
        row["stage2a_firm_knowledge_admitted"] = admitted
        row["stage2a_teacher_correct"] = teacher_correct
        row["stage2a_family_concurrent"] = concurrence
        materialized.append(row)
        battery = str(row["battery"])
        counts = excluded_by_battery.setdefault(
            battery,
            {
                "rows": 0,
                "admitted": 0,
                "teacher_incorrect": 0,
                "family_disagreement": 0,
            },
        )
        counts["rows"] += 1
        counts["admitted"] += int(admitted)
        counts["teacher_incorrect"] += int(not teacher_correct)
        counts["family_disagreement"] += int(not concurrence)
    if not materialized:
        raise ValueError("V(x) requires at least one row")
    verdicts = [
        {
            "battery": str(row["battery"]),
            "item_id": str(row["item_id"]),
            "admitted": bool(row["stage2a_firm_knowledge_admitted"]),
            "teacher_correct": bool(row["stage2a_teacher_correct"]),
            "family_concurrent": bool(row["stage2a_family_concurrent"]),
            "teacher_14b_output_sha256": str(row["teacher_14b_output_sha256"]),
            "teacher_32b_output_sha256": str(row["teacher_32b_output_sha256"]),
            "correctness_reader": str(row["correctness_reader"]),
        }
        for row in materialized
    ]
    receipt = {
        "kind": "paper2_stage2a_firm_knowledge_v1",
        "status": "materialized_binary_correctness_and_family_concurrence",
        "teacher_14b": STAGE2A_TEACHER_14B,
        "verifier_32b": STAGE2A_VERIFIER_32B,
        "probability_thresholds": None,
        "rows": len(materialized),
        "admitted": sum(bool(row["stage2a_firm_knowledge_admitted"]) for row in materialized),
        "counts_by_battery": excluded_by_battery,
        "verdicts_sha256": _canonical_sha256(verdicts),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "dev_scores_computed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return materialized, receipt


@dataclass(frozen=True)
class NonDevFingerprintGeometry:
    student_mean: torch.Tensor
    student_basis: torch.Tensor
    teacher_mean: torch.Tensor
    teacher_basis: torch.Tensor
    diagnostic_rotation: torch.Tensor

    def student_keys(self, states: torch.Tensor) -> torch.Tensor:
        return (states.float() - self.student_mean) @ self.student_basis

    def teacher_values(self, states: torch.Tensor) -> torch.Tensor:
        return (states.float() - self.teacher_mean) @ self.teacher_basis


def fit_nondev_fingerprint_geometry(
    *,
    student_fit: torch.Tensor,
    teacher_fit: torch.Tensor,
    student_holdout: torch.Tensor,
    teacher_holdout: torch.Tensor,
    rank: int = 128,
) -> tuple[NonDevFingerprintGeometry, dict[str, Any]]:
    """Fit live PCA bases on non-DEV rows and score diagnostic transport.

    Teacher values remain in teacher PCA coordinates. The orthogonal map is
    retained only for the required held-out addressing receipt and is not used
    by the memory forward path.
    """

    matrices = (student_fit, teacher_fit, student_holdout, teacher_holdout)
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("fingerprint geometry inputs must be rank-two matrices")
    if student_fit.shape[0] != teacher_fit.shape[0]:
        raise ValueError("fit student and teacher rows must align")
    if student_holdout.shape[0] != teacher_holdout.shape[0]:
        raise ValueError("holdout student and teacher rows must align")
    if student_fit.shape[0] <= int(rank) or teacher_fit.shape[0] <= int(rank):
        raise ValueError("PCA fit requires more fit rows than the registered rank")
    if student_holdout.shape[0] < 2:
        raise ValueError("held-out retrieval requires at least two rows")

    def pca(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = matrix.float().mean(dim=0, keepdim=True)
        _u, _s, vh = torch.linalg.svd(matrix.float() - mean, full_matrices=False)
        if vh.shape[0] < int(rank):
            raise ValueError("state width is smaller than the registered PCA rank")
        return mean, vh[: int(rank)].T.contiguous()

    student_mean, student_basis = pca(student_fit)
    teacher_mean, teacher_basis = pca(teacher_fit)
    student_fit_key = (student_fit.float() - student_mean) @ student_basis
    teacher_fit_value = (teacher_fit.float() - teacher_mean) @ teacher_basis
    cross = student_fit_key.double().T @ teacher_fit_value.double()
    left, _singular, right_h = torch.linalg.svd(cross, full_matrices=False)
    rotation = (left @ right_h).float()

    student_holdout_key = (student_holdout.float() - student_mean) @ student_basis
    teacher_holdout_value = (teacher_holdout.float() - teacher_mean) @ teacher_basis
    transported = student_holdout_key @ rotation
    similarity = F.normalize(transported, dim=-1) @ F.normalize(
        teacher_holdout_value, dim=-1
    ).T
    order = similarity.argsort(dim=-1, descending=True)
    target = torch.arange(similarity.shape[0], device=similarity.device)[:, None]
    ranks = (order == target).nonzero(as_tuple=False)[:, 1] + 1
    geometry = NonDevFingerprintGeometry(
        student_mean=student_mean,
        student_basis=student_basis,
        teacher_mean=teacher_mean,
        teacher_basis=teacher_basis,
        diagnostic_rotation=rotation,
    )
    receipt = {
        "kind": "paper2_stage2a_nondev_fingerprint_geometry_v1",
        "status": "fit_nondev_heldout_retrieval_scored_nondev",
        "rank": int(rank),
        "fit_rows": int(student_fit.shape[0]),
        "holdout_rows": int(student_holdout.shape[0]),
        "top1_retrieval": float(ranks.eq(1).float().mean()),
        "top10_retrieval": float(ranks.le(10).float().mean()),
        "median_rank": float(ranks.float().median()),
        "teacher_values_coordinate_system": "teacher_pca",
        "diagnostic_rotation_live_path": False,
        "dev_rows_used": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    return geometry, receipt


def build_fingerprint_memory_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    panel_item_ids: set[tuple[str, str]],
    reserved_item_ids: set[tuple[str, str]] | None = None,
    admitted_field: str,
    slots: int | None = None,
    seed: int = 20_260_817,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic non-panel memory population after external V(x).

    This function does not define the firm-knowledge rule. It requires the
    registered data preparation pass to materialize that decision in
    ``admitted_field`` and records the field name so the lock can bind its
    upstream definition and hash separately.
    """

    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("memory manifest requires source rows")
    required = {"battery", "item_id", "document_id", "content_sha256", admitted_field}
    for row in records:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"memory row missing required fields: {sorted(missing)}")
    identities = [(str(row["battery"]), str(row["item_id"])) for row in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("memory source contains duplicate battery/item identities")
    reserved = set() if reserved_item_ids is None else set(reserved_item_ids)
    if panel_item_ids.intersection(reserved):
        raise RuntimeError("panel and reserved validation identities overlap")
    candidates = [
        row
        for row, identity in zip(records, identities, strict=True)
        if bool(row[admitted_field])
        and identity not in panel_item_ids
        and identity not in reserved
        and str(row.get("partition", "verified_train")) != "dev"
    ]
    realized_slots = largest_power_of_two_memory_size(len(candidates)) if slots is None else int(slots)
    if realized_slots < 1 or realized_slots & (realized_slots - 1) or realized_slots > 4_096:
        raise ValueError("slots must be a positive power of two no greater than 4096")
    if len(candidates) < realized_slots:
        raise RuntimeError(
            f"firm-knowledge non-panel population has {len(candidates)} rows; "
            f"{realized_slots} required"
        )
    selected, quotas = _battery_stratified_select(
        candidates, count=realized_slots, seed=seed
    )
    selected_keys = {_row_identity(row) for row in selected}
    excluded = sorted(
        (
            {
                "battery": str(row["battery"]),
                "item_id": str(row["item_id"]),
                "content_sha256": str(row["content_sha256"]),
            }
            for row in candidates
            if _row_identity(row) not in selected_keys
        ),
        key=lambda row: (row["battery"], row["item_id"]),
    )
    selected_identities = [
        {
            "battery": str(row["battery"]),
            "item_id": str(row["item_id"]),
            "document_id": str(row["document_id"]),
            "content_sha256": str(row["content_sha256"]),
        }
        for row in selected
    ]
    receipt = {
        "kind": "paper2_stage2a_fingerprint_memory_manifest_v1",
        "status": "selected_score_blind_after_external_firm_knowledge_admission",
        "seed": int(seed),
        "slots": realized_slots,
        "source_rows": len(records),
        "admitted_nonpanel_rows": len(candidates),
        "admitted_field": str(admitted_field),
        "panel_rows_excluded": sum(identity in panel_item_ids for identity in identities),
        "reserved_rows_excluded": sum(identity in reserved for identity in identities),
        "panel_overlap": 0,
        "reserved_overlap": 0,
        "battery_quotas": dict(sorted(quotas.items())),
        "selection_rule": (
            "battery-proportional Hamilton quotas; within-battery "
            "SHA256(seed:battery:item_id:content_sha256)"
        ),
        "selected_identities_sha256": _canonical_sha256(selected_identities),
        "subselection_excluded_identities": excluded,
        "subselection_excluded_identities_sha256": _canonical_sha256(excluded),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "dev_scores_computed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return selected, receipt


@dataclass(frozen=True)
class FunctionalClusterResult:
    assignments: torch.Tensor
    medoids: torch.Tensor
    iterations: int


def canonicalize_expert_outputs(
    expert_outputs: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    """Project expert outputs into one frozen canonical functional space.

    Args:
        expert_outputs: Tensor shaped ``[examples, experts, output_dim]``.
        projection: Frozen tensor shaped ``[output_dim, canonical_dim]``.
    """

    if expert_outputs.ndim != 3:
        raise ValueError("expert_outputs must have shape [examples, experts, output_dim]")
    if projection.ndim != 2:
        raise ValueError("projection must have shape [output_dim, canonical_dim]")
    if expert_outputs.shape[-1] != projection.shape[0]:
        raise ValueError("expert output width and projection input width differ")
    if not expert_outputs.is_floating_point() or not projection.is_floating_point():
        raise TypeError("functional projection requires floating-point tensors")
    return expert_outputs @ projection.to(
        device=expert_outputs.device, dtype=expert_outputs.dtype
    )


def relevance_weighted_distance_matrix(
    canonical_outputs: torch.Tensor,
    routing_probabilities: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the registered output-space expert distance matrix.

    The pair weight on example ``i`` is ``a[i,e] + a[i,e'] + epsilon``;
    distances are the example mean of weighted squared Euclidean differences.
    """

    if canonical_outputs.ndim != 3:
        raise ValueError("canonical_outputs must have shape [examples, experts, dim]")
    if routing_probabilities.ndim != 2:
        raise ValueError("routing_probabilities must have shape [examples, experts]")
    if canonical_outputs.shape[:2] != routing_probabilities.shape:
        raise ValueError("canonical outputs and routing probabilities disagree")
    if canonical_outputs.shape[0] < 1 or canonical_outputs.shape[1] < 1:
        raise ValueError("functional distances require examples and experts")
    if float(epsilon) < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if not bool(torch.isfinite(canonical_outputs).all()):
        raise ValueError("canonical_outputs contain non-finite values")
    if not bool(torch.isfinite(routing_probabilities).all()):
        raise ValueError("routing_probabilities contain non-finite values")
    if bool((routing_probabilities < 0).any()):
        raise ValueError("routing probabilities must be nonnegative")

    outputs = canonical_outputs.float()
    routing = routing_probabilities.float()
    differences = outputs[:, :, None, :] - outputs[:, None, :, :]
    squared = differences.square().sum(dim=-1)
    weights = routing[:, :, None] + routing[:, None, :] + float(epsilon)
    distances = (weights * squared).mean(dim=0)
    distances = 0.5 * (distances + distances.T)
    distances.fill_diagonal_(0.0)
    return distances


def deterministic_k_medoids(
    distances: torch.Tensor,
    *,
    n_clusters: int,
    max_iterations: int = 100,
) -> FunctionalClusterResult:
    """Cluster experts with deterministic farthest-first k-medoids."""

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    n_experts = int(distances.shape[0])
    if not 1 <= int(n_clusters) <= n_experts:
        raise ValueError("n_clusters must be in [1, n_experts]")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("distances contain non-finite values")
    if bool((distances < 0).any()):
        raise ValueError("distances must be nonnegative")
    if not bool(torch.allclose(distances, distances.T, atol=1e-6, rtol=1e-6)):
        raise ValueError("distances must be symmetric")

    work = distances.detach().float().cpu()
    medoids = [int(torch.argmin(work.sum(dim=1)).item())]
    while len(medoids) < int(n_clusters):
        nearest = work[:, medoids].amin(dim=1)
        nearest[medoids] = -1.0
        medoids.append(int(torch.argmax(nearest).item()))

    assignments = torch.empty(n_experts, dtype=torch.long)
    for iteration in range(1, int(max_iterations) + 1):
        prior = torch.tensor(medoids, dtype=torch.long)
        assignments = torch.argmin(work[:, prior], dim=1)
        updated: list[int] = []
        for cluster in range(int(n_clusters)):
            members = torch.nonzero(assignments == cluster, as_tuple=False).flatten()
            if not len(members):
                nearest = work[:, prior].amin(dim=1)
                nearest[prior] = -1.0
                updated.append(int(torch.argmax(nearest).item()))
                continue
            within = work.index_select(0, members).index_select(1, members)
            updated.append(int(members[torch.argmin(within.sum(dim=1))].item()))
        if updated == medoids:
            return FunctionalClusterResult(
                assignments=assignments,
                medoids=torch.tensor(medoids, dtype=torch.long),
                iterations=iteration,
            )
        medoids = updated
    assignments = torch.argmin(work[:, torch.tensor(medoids)], dim=1)
    return FunctionalClusterResult(
        assignments=assignments,
        medoids=torch.tensor(medoids, dtype=torch.long),
        iterations=int(max_iterations),
    )


def aggregate_cluster_routing(
    routing_probabilities: torch.Tensor,
    assignments: torch.Tensor,
    *,
    n_clusters: int,
) -> torch.Tensor:
    """Aggregate native teacher routing mass into student expert clusters."""

    if routing_probabilities.ndim != 2:
        raise ValueError("routing_probabilities must have shape [examples, experts]")
    if assignments.ndim != 1 or assignments.numel() != routing_probabilities.shape[1]:
        raise ValueError("assignments must provide one cluster per expert")
    if bool((assignments < 0).any()) or bool((assignments >= int(n_clusters)).any()):
        raise ValueError("cluster assignment is out of range")
    index = assignments.to(device=routing_probabilities.device).expand(
        routing_probabilities.shape[0], -1
    )
    result = routing_probabilities.new_zeros(
        (routing_probabilities.shape[0], int(n_clusters))
    )
    return result.scatter_add(1, index, routing_probabilities)


def ridge_low_rank_initialization(
    queries: torch.Tensor,
    targets: torch.Tensor,
    *,
    rank: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a ridge map and return ``W``, ``B``, ``A`` with ``B @ A`` rank-limited."""

    if queries.ndim != 2 or targets.ndim != 2:
        raise ValueError("queries and targets must be rank-two matrices")
    if queries.shape[0] != targets.shape[0] or queries.shape[0] < 1:
        raise ValueError("queries and targets require the same nonzero row count")
    maximum_rank = min(int(queries.shape[1]), int(targets.shape[1]))
    if not 1 <= int(rank) <= maximum_rank:
        raise ValueError("rank exceeds the fitted map dimensions")
    if float(ridge) < 0.0:
        raise ValueError("ridge must be nonnegative")
    work_dtype = torch.float64
    x = queries.detach().to(dtype=work_dtype)
    y = targets.detach().to(dtype=work_dtype)
    gram = x.T @ x
    gram.diagonal().add_(float(ridge))
    fitted = torch.linalg.solve(gram, x.T @ y)
    u, singular, vh = torch.linalg.svd(fitted, full_matrices=False)
    root = singular[: int(rank)].sqrt()
    b = u[:, : int(rank)] * root.unsqueeze(0)
    a = root.unsqueeze(1) * vh[: int(rank), :]
    dtype = queries.dtype
    device = queries.device
    return (
        fitted.to(device=device, dtype=dtype),
        b.to(device=device, dtype=dtype),
        a.to(device=device, dtype=dtype),
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values together with dtype and shape for receipt identity."""

    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()
