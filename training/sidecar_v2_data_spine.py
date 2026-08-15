"""Deterministic, loss-free Stage 2A functional-fingerprint utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


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
