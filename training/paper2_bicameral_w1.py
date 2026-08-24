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
