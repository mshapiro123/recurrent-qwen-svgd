"""Deterministic contracts for the score-only Bicameral W1 target ladder."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


GAMMA = 0.05
STATE_RMS_CAP = 0.550893
SHUFFLE_SEED = 20260824
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_DRAWS = 10_000
CLUSTER_EXTENSION_MIN_FRACTION = 0.05
ORACLE_TARGET_ASSISTED = "oracle-target-assisted"
POPULATION_TARGET = "population-target"
RS0A_SPLITS = 20
RS0A_SEED_BASE = 20260823
RS0A_NUISANCE_RANK = 8


def rms(value: torch.Tensor, *, dim: int | tuple[int, ...] = -1) -> torch.Tensor:
    return value.float().square().mean(dim=dim).sqrt()


def scale_external_write(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    gamma: float = GAMMA,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the registered rowwise RMS-matched write without a learned bridge."""

    if hidden.ndim != 3 or direction.ndim != 2:
        raise ValueError("external write expects [B,T,D] hidden and [B,D] direction")
    if hidden.shape[0] != direction.shape[0] or hidden.shape[-1] != direction.shape[-1]:
        raise ValueError("external write batch or hidden width changed")
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError("external write attention mask changed")
    if float(gamma) != GAMMA:
        raise ValueError("W1 amplitude exploration is prohibited")

    active = attention_mask.to(hidden.device).bool()
    denominator = active.sum(dim=1).clamp_min(1).float() * hidden.shape[-1]
    hidden_rms = (
        (hidden.float().square().sum(dim=-1) * active.float()).sum(dim=1) / denominator
    ).sqrt()
    direction_rms = rms(direction).clamp_min(1e-12)
    scale = float(gamma) * hidden_rms / direction_rms
    write = direction.float() * scale[:, None]
    deployed = hidden.float() + active.unsqueeze(-1).float() * write[:, None, :]
    ratio = rms(write) / hidden_rms.clamp_min(1e-12)
    return deployed.to(hidden.dtype), {
        "hidden_rms": hidden_rms,
        "direction_rms": direction_rms,
        "write_rms": rms(write),
        "write_ratio": ratio,
        "state_rms_cap": hidden.new_full((hidden.shape[0],), STATE_RMS_CAP).float(),
    }


def deterministic_permutation(size: int, *, family: str, seed: int = SHUFFLE_SEED) -> list[int]:
    if size < 2:
        raise ValueError("shuffle control requires at least two rows")
    digest = hashlib.sha256(f"{seed}:{family}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(size))
    # Sattolo's algorithm produces one cycle, hence a guaranteed derangement.
    for index in range(size - 1, 0, -1):
        other = generator.randrange(index)
        order[index], order[other] = order[other], order[index]
    if any(index == source for index, source in enumerate(order)):
        raise RuntimeError("registered derangement construction failed")
    return order


def build_phase_b_granularity_targets(
    row_targets: torch.Tensor,
    cluster_assignments: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Construct the registered L1/L2/L3 targets from frozen assignments."""

    if row_targets.ndim != 2:
        raise ValueError("Phase B row targets must have shape [N,D]")
    assignments = torch.as_tensor(cluster_assignments, dtype=torch.long).cpu()
    if assignments.ndim != 1 or assignments.numel() != row_targets.shape[0]:
        raise ValueError("Phase B assignments must have shape [N]")
    labels = torch.unique(assignments, sorted=True)
    if labels.numel() != 2 or not torch.equal(labels, torch.tensor([0, 1])):
        raise ValueError("Phase B requires exactly the frozen k=2 labels {0,1}")

    targets = row_targets.float()
    means = torch.stack([targets[assignments == label].mean(dim=0) for label in labels])
    if not torch.isfinite(means).all():
        raise ValueError("Phase B cluster means must be finite")

    own_cluster = means[assignments]
    other_cluster = means[1 - assignments]
    global_mean = targets.mean(dim=0, keepdim=True).expand_as(targets)
    return {
        "l1": own_cluster.to(row_targets.dtype),
        "l2": global_mean.to(row_targets.dtype),
        "l3": other_cluster.to(row_targets.dtype),
        "cluster_means": means.to(row_targets.dtype),
    }


def validate_cluster_extension(
    cluster_assignments: torch.Tensor,
    *,
    minimum_fraction: float = CLUSTER_EXTENSION_MIN_FRACTION,
) -> dict[str, Any]:
    """Apply the registered frozen-centroid extension degeneracy gate."""

    assignments = torch.as_tensor(cluster_assignments, dtype=torch.long).cpu()
    if assignments.ndim != 1 or assignments.numel() == 0:
        raise ValueError("cluster extension assignments must have shape [N]")
    labels = torch.unique(assignments, sorted=True)
    if labels.numel() != 2 or not torch.equal(labels, torch.tensor([0, 1])):
        raise ValueError("cluster extension requires exactly labels {0,1}")
    counts = torch.bincount(assignments, minlength=2)
    fractions = counts.double() / assignments.numel()
    passed = bool(torch.all(fractions >= float(minimum_fraction)))
    receipt = {
        "rows": int(assignments.numel()),
        "counts": [int(value) for value in counts],
        "fractions": [float(value) for value in fractions],
        "minimum_fraction": float(minimum_fraction),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"W1 frozen-centroid extension gate failed: {receipt}")
    return receipt


def orient_residual_directions(
    directions: torch.Tensor,
    correction_mean: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Orient each L6 eigenvector against the seed-specific correction mean."""

    if directions.ndim != 2 or correction_mean.ndim != 1:
        raise ValueError("L6 orientation expects [K,D] directions and [D] mean")
    if directions.shape[1] != correction_mean.shape[0]:
        raise ValueError("L6 orientation width changed")
    dots = directions.float() @ correction_mean.float()
    signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    oriented = directions.float() * signs[:, None]
    oriented_dots = oriented @ correction_mean.float()
    if torch.any(oriented_dots < 0):
        raise RuntimeError("L6 orientation convention failed")
    return oriented.to(directions.dtype), {
        "original_inner_products": [float(value) for value in dots],
        "orientation_signs": [int(value) for value in signs],
        "oriented_inner_products": [float(value) for value in oriented_dots],
    }


def extend_frozen_centroids(
    features: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assign DEV-2 features to the frozen Stage-0 spherical centroids."""

    if features.ndim != 2 or centroids.ndim != 2:
        raise ValueError("frozen-centroid extension expects [N,D] and [2,D]")
    if centroids.shape[0] != 2 or features.shape[1] != centroids.shape[1]:
        raise ValueError("frozen-centroid extension geometry changed")
    unit_features = torch.nn.functional.normalize(features.float(), dim=-1, eps=1e-12)
    unit_centroids = torch.nn.functional.normalize(centroids.float(), dim=-1, eps=1e-12)
    similarities = unit_features @ unit_centroids.T
    assignments = similarities.argmax(dim=1).cpu()
    receipt = validate_cluster_extension(assignments)
    receipt.update(
        {
            "assignment": "nearest_frozen_stage0_centroid_no_refit",
            "maximum_similarity_mean": float(similarities.max(dim=1).values.mean()),
            "margin_mean": float(
                (similarities.max(dim=1).values - similarities.min(dim=1).values).mean()
            ),
        }
    )
    return assignments, receipt


def _feature_basis(values: torch.Tensor, rank: int) -> torch.Tensor:
    if values.ndim != 2 or not 1 <= rank <= min(values.shape):
        raise ValueError("invalid R-S0-A nuisance-basis geometry")
    _u, _s, vh = torch.linalg.svd(values.float(), full_matrices=False)
    return vh[:rank]


def _project_off(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return values
    return values - (values @ basis.T) @ basis


def build_crossfitted_residual_directions(
    corrections: torch.Tensor,
    labels: torch.Tensor,
    *,
    directions: int = 3,
    splits: int = RS0A_SPLITS,
    seed_base: int = RS0A_SEED_BASE,
    nuisance_rank: int = RS0A_NUISANCE_RANK,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pool held-out R-S0-A residual covariance and return its leading axes.

    Every residual row is projected by a nuisance basis and cluster direction
    estimated on the opposite split. Pooling only those held-out residual outer
    products preserves the cross-fitted estimand while yielding one deterministic
    per-seed eigenbasis for W1-L6.
    """

    x = torch.nn.functional.normalize(corrections.float(), dim=-1, eps=1e-12)
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    if x.ndim != 2 or labels.shape != (x.shape[0],):
        raise ValueError("R-S0-A residual directions require paired [N,D] rows")
    if not torch.equal(torch.unique(labels, sorted=True), torch.tensor([0, 1])):
        raise ValueError("R-S0-A residual directions require frozen labels {0,1}")
    if not 1 <= directions <= x.shape[1]:
        raise ValueError("invalid number of residual directions")

    covariance = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float64)
    heldout_rows = 0
    fold_receipts = []
    for cluster in (0, 1):
        members = torch.where(labels == cluster)[0]
        other = torch.where(labels != cluster)[0]
        for split in range(int(splits)):
            generator = torch.Generator().manual_seed(int(seed_base) + split)
            order = members[torch.randperm(members.numel(), generator=generator)]
            midpoint = order.numel() // 2
            for fit, evaluate, direction_name in (
                (order[:midpoint], order[midpoint:], "A_to_B"),
                (order[midpoint:], order[:midpoint], "B_to_A"),
            ):
                nuisance_rows = torch.cat([x[fit], x[other]], dim=0)
                nuisance = _feature_basis(nuisance_rows, int(nuisance_rank))
                fit_projected = _project_off(x[fit], nuisance)
                cluster_direction = torch.nn.functional.normalize(
                    fit_projected.mean(dim=0), dim=0, eps=1e-12
                )
                combined = torch.cat([nuisance, cluster_direction.unsqueeze(0)], dim=0)
                combined = torch.linalg.qr(combined.T, mode="reduced").Q.T
                residual = _project_off(x[evaluate], combined).double()
                covariance += residual.T @ residual
                heldout_rows += int(residual.shape[0])
                fold_receipts.append(
                    {
                        "cluster": cluster,
                        "split": split,
                        "direction": direction_name,
                        "fit_rows": int(fit.numel()),
                        "evaluation_rows": int(evaluate.numel()),
                    }
                )
    covariance /= max(heldout_rows, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    selected_values = eigenvalues[order[:directions]]
    selected_vectors = eigenvectors[:, order[:directions]].T.float()
    oriented, orientation = orient_residual_directions(selected_vectors, x.mean(dim=0))
    total = eigenvalues.clamp_min(0).sum().clamp_min(1e-30)
    receipt = {
        "kind": "paper2_bicameral_w1_crossfitted_residual_directions_v1",
        "rows": int(x.shape[0]),
        "dimensions": int(x.shape[1]),
        "splits": int(splits),
        "directions_per_split": 2,
        "seed_base": int(seed_base),
        "nuisance_rank": int(nuisance_rank),
        "heldout_row_appearances": heldout_rows,
        "eigenvalues": [float(value) for value in selected_values],
        "energy_fractions": [float(value / total) for value in selected_values],
        "orientation": orientation,
        "folds": fold_receipts,
    }
    return oriented, receipt


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap requires observations")
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    means = []
    chunk = min(512, int(draws))
    remaining = int(draws)
    while remaining:
        count = min(chunk, remaining)
        indices = torch.randint(
            tensor.numel(), (count, tensor.numel()), generator=generator
        )
        means.append(tensor[indices].mean(dim=1))
        remaining -= count
    samples = torch.cat(means)
    return {
        "rows": int(tensor.numel()),
        "draws": int(draws),
        "mean": float(tensor.mean()),
        "ci_low": float(torch.quantile(samples, 0.025)),
        "ci_high": float(torch.quantile(samples, 0.975)),
    }


def summarize_margin_deltas(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["margin_delta"]) for row in rows]
    by_battery: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_battery[str(row["battery"])].append(float(row["margin_delta"]))
    return {
        **bootstrap_mean_ci(values),
        "by_battery": {
            battery: bootstrap_mean_ci(items, seed=BOOTSTRAP_SEED + index + 1)
            for index, (battery, items) in enumerate(sorted(by_battery.items()))
        },
    }


def project_cost_hours(
    *,
    target_seconds_per_row: Mapping[str, float],
    margin_seconds_per_row: float,
    rows: int,
    seeds: int,
    phase_a_cells_per_seed: int,
    phase_b_cells_per_seed: int,
    generation_hours: float = 0.0,
) -> dict[str, Any]:
    target_seconds = sum(float(value) for value in target_seconds_per_row.values()) * rows * seeds
    margin_seconds = (
        float(margin_seconds_per_row)
        * rows
        * seeds
        * (phase_a_cells_per_seed + phase_b_cells_per_seed)
    )
    total = (target_seconds + margin_seconds) / 3600.0 + float(generation_hours)
    return {
        "target_hours": target_seconds / 3600.0,
        "margin_hours": margin_seconds / 3600.0,
        "generation_hours": float(generation_hours),
        "projected_total_a100_hours": total,
        "cap_a100_hours": 8.0,
        "within_cap": total <= 8.0,
    }


def resolve_phase_a(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the registered both-seed, own-shuffle, and null winner rule."""

    by_arm_seed = {(str(cell["arm"]), int(cell["seed"])): cell for cell in cells}
    families = ("l0a", "l0b", "l0c", "l0d", "l0g")
    eligible = []
    for family in families:
        arm_cells = [by_arm_seed.get((family, seed)) for seed in (0, 1)]
        shuffle_cells = [by_arm_seed.get((f"l5_{family[-1]}", seed)) for seed in (0, 1)]
        random_cells = [by_arm_seed.get(("l4", seed)) for seed in (0, 1)]
        if any(cell is None for cell in arm_cells + shuffle_cells + random_cells):
            continue
        clears = all(
            float(arm["ci_low"]) > 0.0
            and float(arm["mean"]) > float(shuffle["mean"])
            and float(arm["mean"]) > float(null["mean"])
            for arm, shuffle, null in zip(arm_cells, shuffle_cells, random_cells)
        )
        if clears:
            pooled = sum(float(cell["mean"]) for cell in arm_cells) / 2.0
            eligible.append((pooled, family))
    if not eligible:
        return {"key": "TARGETS-NOT-ANSWER-GRADE", "winner": None, "eligible": []}
    maximum = max(value for value, _family in eligible)
    tied = [family for value, family in eligible if math.isclose(value, maximum, abs_tol=1e-12)]
    winner = "l0d" if "l0d" in tied else sorted(tied)[0]
    return {
        "key": "PHASE-A-WINNER",
        "winner": winner,
        "eligible": [{"family": family, "pooled_mean": value} for value, family in eligible],
    }
