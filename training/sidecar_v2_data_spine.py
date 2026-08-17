"""Deterministic, loss-free Stage 2A functional-fingerprint utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fingerprint_memory_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    panel_item_ids: set[tuple[str, str]],
    admitted_field: str,
    slots: int = 8_192,
    seed: int = 20_260_816,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic non-panel memory population after external V(x).

    This function does not define the firm-knowledge rule. It requires the
    registered data preparation pass to materialize that decision in
    ``admitted_field`` and records the field name so the lock can bind its
    upstream definition and hash separately.
    """

    if int(slots) < 1:
        raise ValueError("slots must be positive")
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
    candidates = [
        row
        for row, identity in zip(records, identities, strict=True)
        if bool(row[admitted_field]) and identity not in panel_item_ids
    ]
    if len(candidates) < int(slots):
        raise RuntimeError(
            f"firm-knowledge non-panel population has {len(candidates)} rows; "
            f"{int(slots)} required"
        )

    def rank(row: Mapping[str, Any]) -> tuple[bytes, str, str]:
        identity = f"{seed}:{row['battery']}:{row['item_id']}:{row['content_sha256']}"
        return (
            hashlib.sha256(identity.encode("utf-8")).digest(),
            str(row["battery"]),
            str(row["item_id"]),
        )

    selected = sorted(candidates, key=rank)[: int(slots)]
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
        "slots": int(slots),
        "source_rows": len(records),
        "admitted_nonpanel_rows": len(candidates),
        "admitted_field": str(admitted_field),
        "panel_rows_excluded": sum(identity in panel_item_ids for identity in identities),
        "panel_overlap": 0,
        "selection_rule": "SHA256(seed:battery:item_id:content_sha256), ascending",
        "selected_identities_sha256": _canonical_sha256(selected_identities),
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
