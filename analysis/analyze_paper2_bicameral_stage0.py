"""Run the locked, CPU-only Bicameral Stage-0 geometry analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from models.sidecar_v2 import fast_wht
from training.paper2_stage2b_autopsy import (
    discrete_mutual_information,
    spherical_kmeans,
)


RUN_KIND = "paper2_bicameral_stage0_geometry_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _unit(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=-1, eps=1e-12)


def blockwise_wht(values: torch.Tensor, *, block_size: int) -> torch.Tensor:
    if values.ndim < 1 or values.shape[-1] % block_size:
        raise ValueError("hidden width must be divisible by the WHT block size")
    shape = values.shape
    blocks = values.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
    return fast_wht(blocks).reshape(shape)


def reachable_fraction(
    corrections: torch.Tensor,
    states: torch.Tensor,
    *,
    block_size: int,
) -> dict[str, Any]:
    """Energy captured by the optimal shared per-sequency state rescaling.

    For each sequency s, rows and WHT blocks form one vector. The closed-form
    scalar coefficient projects the correction vector onto the state vector.
    """

    if corrections.shape != states.shape or corrections.ndim != 2:
        raise ValueError("rho_reach requires paired [rows, hidden] tensors")
    correction_wht = blockwise_wht(corrections, block_size=block_size)
    state_wht = blockwise_wht(states, block_size=block_size)
    blocks = corrections.shape[1] // block_size
    correction_by_s = correction_wht.reshape(-1, blocks, block_size).permute(2, 0, 1)
    state_by_s = state_wht.reshape(-1, blocks, block_size).permute(2, 0, 1)
    numerator_by_s = (correction_by_s * state_by_s).sum(dim=(1, 2)).square()
    state_energy_by_s = state_by_s.square().sum(dim=(1, 2))
    captured_by_s = torch.where(
        state_energy_by_s > 0.0,
        numerator_by_s / state_energy_by_s,
        torch.zeros_like(numerator_by_s),
    )
    total_energy = correction_by_s.square().sum()
    rho = captured_by_s.sum() / total_energy.clamp_min(1e-12)
    coefficients = torch.where(
        state_energy_by_s > 0.0,
        (correction_by_s * state_by_s).sum(dim=(1, 2)) / state_energy_by_s,
        torch.zeros_like(state_energy_by_s),
    )
    return {
        "rho_reach": float(rho),
        "correction_energy": float(total_energy),
        "captured_energy": float(captured_by_s.sum()),
        "nonzero_state_bands": int((state_energy_by_s > 0.0).sum()),
        "coefficient_l2": float(coefficients.norm()),
        "coefficients": [float(value) for value in coefficients],
        "estimator": (
            "sum_s <delta_h_tilde_s,h_tilde_s>^2/||h_tilde_s||^2 "
            "/ ||delta_h_tilde||^2; each s pools rows and seven WHT128 blocks"
        ),
    }


def _offdiagonal_cosines(values: torch.Tensor) -> torch.Tensor:
    unit = _unit(values)
    matrix = unit @ unit.T
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def common_mode_receipt(corrections: torch.Tensor) -> dict[str, Any]:
    x = _unit(corrections)
    mean = x.mean(dim=0)
    direction = _unit(mean)
    scores = x @ direction
    residual = x - scores.unsqueeze(1) * direction.unsqueeze(0)
    before = _offdiagonal_cosines(x)
    after = _offdiagonal_cosines(residual)
    return {
        "rows": int(x.shape[0]),
        "common_mode_fraction": float(scores.square().sum() / x.square().sum()),
        "common_mode_mean_norm": float(mean.norm()),
        "pre_projection_cross_row_correlation": {
            "mean": float(before.mean()),
            "median": float(before.median()),
        },
        "rho_res": float(after.mean()),
        "post_projection_cross_row_correlation": {
            "mean": float(after.mean()),
            "median": float(after.median()),
            "p95": float(torch.quantile(after, 0.95)),
        },
        "common_mode_estimator": (
            "energy fraction after projection onto the unit global mean correction"
        ),
        "rho_res_estimator": "mean offdiagonal cosine of projected residual rows",
    }


def cluster_receipt(
    corrections: torch.Tensor,
    states: torch.Tensor,
    *,
    item_ids: list[str],
    batteries: list[str],
    clusters: int,
    seed: int,
    restarts: int,
    iterations: int,
    block_size: int,
    global_common_mode_direction: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    labels, silhouette = spherical_kmeans(
        corrections,
        clusters=clusters,
        restarts=restarts,
        iterations=iterations,
        seed=seed + clusters,
    )
    assignments = []
    cells = {}
    tensors: dict[str, torch.Tensor] = {"labels": labels.cpu()}
    for cluster in range(clusters):
        mask = labels == cluster
        correction_rows = corrections[mask]
        state_rows = states[mask]
        unit_rows = _unit(correction_rows)
        residual_rows = unit_rows - (
            unit_rows @ global_common_mode_direction
        ).unsqueeze(1) * global_common_mode_direction.unsqueeze(0)
        residual_cosines = _offdiagonal_cosines(residual_rows)
        correction_mean = correction_rows.mean(dim=0)
        state_mean = state_rows.mean(dim=0)
        key = str(cluster)
        tensors[f"cluster_{cluster}_correction_mean"] = correction_mean.cpu()
        tensors[f"cluster_{cluster}_correction_mean_unit"] = _unit(correction_mean).cpu()
        tensors[f"cluster_{cluster}_state_loop4_mean"] = state_mean.cpu()
        reachability = reachable_fraction(
            correction_rows, state_rows, block_size=block_size
        )
        tensors[f"cluster_{cluster}_bank_gain"] = torch.tensor(
            reachability.pop("coefficients"), dtype=torch.float32
        )
        battery_counts = Counter(b for b, keep in zip(batteries, mask.tolist()) if keep)
        cells[key] = {
            "rows": int(mask.sum()),
            "battery_counts": dict(sorted(battery_counts.items())),
            "post_global_projection_rho_res": float(residual_cosines.mean()),
            "post_global_projection_rho_res_median": float(residual_cosines.median()),
            "rho_reach": reachability,
        }
    for item_id, battery, label in zip(item_ids, batteries, labels.tolist()):
        assignments.append({"item_id": item_id, "battery": battery, "cluster": int(label)})
    association = discrete_mutual_information(labels.tolist(), batteries)
    return (
        {
            "clusters": clusters,
            "silhouette": silhouette,
            "cluster_cells": cells,
            "battery_mutual_information": association,
            "assignments": assignments,
        },
        tensors,
    )


def _project_off(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return values
    return values - (values @ basis.T) @ basis


def _feature_basis(values: torch.Tensor, rank: int) -> torch.Tensor:
    """Return feature-space singular vectors for an uncentered row matrix."""

    if values.ndim != 2 or not 1 <= rank <= min(values.shape):
        raise ValueError("invalid nuisance-basis geometry")
    _u, _s, vh = torch.linalg.svd(values.float(), full_matrices=False)
    return vh[:rank]


def _gram_ratio(residuals: torch.Tensor) -> float:
    rows = int(residuals.shape[0])
    if rows < 2:
        return float("nan")
    total = residuals.sum(dim=0).square().sum() - residuals.square().sum()
    mean_offdiagonal = total / (rows * (rows - 1))
    mean_energy = residuals.square().sum(dim=1).mean().clamp_min(1e-12)
    return float(mean_offdiagonal / mean_energy)


def _mp_spike_receipt(residuals: torch.Tensor) -> dict[str, Any]:
    rows, dimensions = residuals.shape
    covariance_eigenvalues = torch.linalg.eigvalsh(
        residuals @ residuals.T / max(rows, 1)
    ).clamp_min(0.0)
    sigma2 = residuals.square().sum() / max(rows * dimensions, 1)
    aspect = dimensions / max(rows, 1)
    edge = sigma2 * (1.0 + math.sqrt(aspect)) ** 2
    return {
        "rows": rows,
        "dimensions": dimensions,
        "aspect_d_over_n": aspect,
        "noise_variance": float(sigma2),
        "mp_upper_edge": float(edge),
        "spikes_above_edge": int((covariance_eigenvalues > edge).sum()),
        "largest_eigenvalue": float(covariance_eigenvalues[-1]),
    }


def _crossfit_fold(
    corrections: torch.Tensor,
    labels: torch.Tensor,
    *,
    cluster: int,
    fit_indices: torch.Tensor,
    evaluation_indices: torch.Tensor,
    nuisance_rank: int,
) -> dict[str, Any]:
    other = torch.where(labels != cluster)[0]
    nuisance_rows = torch.cat([corrections[fit_indices], corrections[other]], dim=0)
    nuisance_basis = _feature_basis(nuisance_rows, nuisance_rank)
    fit_projected = _project_off(corrections[fit_indices], nuisance_basis)
    cluster_direction = _unit(fit_projected.mean(dim=0))
    combined_basis = torch.cat([nuisance_basis, cluster_direction.unsqueeze(0)], dim=0)
    combined_basis = torch.linalg.qr(combined_basis.T, mode="reduced").Q.T
    residuals = _project_off(corrections[evaluation_indices], combined_basis)
    return {
        "rho_res": _gram_ratio(residuals),
        "mp": _mp_spike_receipt(residuals),
        "fit_rows": int(fit_indices.numel()),
        "evaluation_rows": int(evaluation_indices.numel()),
    }


def cross_fitted_residual_correlation(
    corrections: torch.Tensor,
    labels: torch.Tensor,
    *,
    splits: int,
    seed_base: int,
    rank_start: int,
    rank_max: int,
    persistence_fraction: float,
) -> dict[str, Any]:
    """Implement registered ruling R-S0-A without same-sample deflation."""

    x = _unit(corrections)
    clusters = [int(value) for value in labels.unique(sorted=True)]
    permutations: dict[tuple[int, int], torch.Tensor] = {}
    for cluster in clusters:
        members = torch.where(labels == cluster)[0]
        for split in range(splits):
            generator = torch.Generator().manual_seed(seed_base + split)
            permutations[(cluster, split)] = members[
                torch.randperm(members.numel(), generator=generator)
            ]

    by_rank: dict[str, Any] = {}
    terminal_by_cluster: dict[int, int] = {cluster: rank_max for cluster in clusters}
    unresolved = set(clusters)
    fold_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rank in range(rank_start, rank_max + 1):
        rank_cells: dict[str, Any] = {}
        for cluster in clusters:
            folds = []
            for split in range(splits):
                permutation = permutations[(cluster, split)]
                midpoint = permutation.numel() // 2
                left, right = permutation[:midpoint], permutation[midpoint:]
                for fit, evaluation, direction in (
                    (left, right, "A_to_B"),
                    (right, left, "B_to_A"),
                ):
                    cell = _crossfit_fold(
                        x,
                        labels,
                        cluster=cluster,
                        fit_indices=fit,
                        evaluation_indices=evaluation,
                        nuisance_rank=rank,
                    )
                    cell["split"] = split
                    cell["direction"] = direction
                    folds.append(cell)
            fold_cache[(cluster, rank)] = folds
            rhos = [cell["rho_res"] for cell in folds]
            spike_fraction = sum(
                cell["mp"]["spikes_above_edge"] > 0 for cell in folds
            ) / len(folds)
            rank_cells[str(cluster)] = {
                "rho_res_mean": statistics.fmean(rhos),
                "rho_res_sd": statistics.stdev(rhos),
                "spike_persistence_fraction": spike_fraction,
                "folds": folds,
            }
            if cluster in unresolved and spike_fraction < persistence_fraction:
                terminal_by_cluster[cluster] = rank
                unresolved.remove(cluster)
        by_rank[str(rank)] = rank_cells
    terminal_rank = max(terminal_by_cluster.values())
    terminal_clusters = {}
    pooled_folds = []
    for cluster in clusters:
        folds = fold_cache[(cluster, terminal_rank)]
        rhos = [cell["rho_res"] for cell in folds]
        rows = int((labels == cluster).sum())
        terminal_clusters[str(cluster)] = {
            "rows": rows,
            "rho_res_mean": statistics.fmean(rhos),
            "rho_res_sd": statistics.stdev(rhos),
            "cluster_terminal_rank": terminal_by_cluster[cluster],
            "residual_spike_count_mean": statistics.fmean(
                cell["mp"]["spikes_above_edge"] for cell in folds
            ),
        }
    total_rows = sum(cell["rows"] for cell in terminal_clusters.values())
    for fold_index in range(2 * splits):
        pooled_folds.append(
            sum(
                terminal_clusters[str(cluster)]["rows"]
                * fold_cache[(cluster, terminal_rank)][fold_index]["rho_res"]
                for cluster in clusters
            )
            / total_rows
        )
    pooled_mean = statistics.fmean(pooled_folds)
    if pooled_mean <= 3e-4:
        decision = "RHO_SIZING_STANDS"
    elif pooled_mean <= 1e-3:
        decision = "RHO_SIZING_INFLATES"
    elif terminal_rank == rank_max:
        decision = "RHO_ESCALATE_AT_RANK_CAP"
    else:
        decision = "RHO_UNMAPPED_STRATEGY_REVIEW"
    return {
        "kind": "cross_fitted_within_cluster_residual_correlation_v1",
        "splits": splits,
        "directions_per_split": 2,
        "split_seed_rule": "20260823 + split_index",
        "nuisance_basis": (
            "top-m feature-space right singular vectors of fit-cluster rows plus "
            "all rows outside the evaluated cluster"
        ),
        "same_sample_projection_prohibited": True,
        "rank_start": rank_start,
        "rank_max": rank_max,
        "mp_persistence_fraction": persistence_fraction,
        "terminal_rank": terminal_rank,
        "terminal_clusters": terminal_clusters,
        "pooled": {
            "rho_res_mean": pooled_mean,
            "rho_res_sd": statistics.stdev(pooled_folds),
            "weights": "cluster row counts",
            "decision": decision,
        },
        "by_rank": by_rank,
    }


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("kind") != "paper2_bicameral_stage0_lock_v1":
        raise RuntimeError("wrong Bicameral Stage-0 lock kind")
    if lock.get("status") != "LOCKED_STAGE0_ONLY" or not lock.get("mark_ratified"):
        raise RuntimeError("Bicameral Stage-0 is not locked")
    if lock.get("training_authorized") or lock.get("gpu_authorized"):
        raise RuntimeError("Bicameral Stage-0 lock improperly authorizes compute")
    if lock.get("optimizer_steps_allowed") != 0:
        raise RuntimeError("Bicameral Stage-0 optimizer allowance changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    validate_lock(lock)
    analysis = lock["analysis"]
    output: dict[str, Any] = {
        "kind": RUN_KIND,
        "status": "running",
        "lock_sha256": sha256_file(args.lock),
        "training_performed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "gpu_used": False,
        "confirm_scored": False,
        "eval_e_scored": False,
        "seeds": {},
    }
    atomic_json(args.output_dir / "status.json", output)
    for seed in (0, 1):
        correction_path = args.input_dir / f"seed_{seed}" / "correction_field__initialization.pt"
        state_path = args.input_dir / f"seed_{seed}" / "state_features__initialization.pt"
        expected = lock["inputs"][str(seed)]
        if sha256_file(correction_path) != expected["correction_field_sha256"]:
            raise RuntimeError(f"seed {seed} correction-field SHA changed")
        if sha256_file(state_path) != expected["state_features_sha256"]:
            raise RuntimeError(f"seed {seed} state-feature SHA changed")
        correction_artifact = torch.load(
            correction_path, map_location="cpu", weights_only=False
        )
        state_artifact = torch.load(state_path, map_location="cpu", weights_only=False)
        corrections = correction_artifact["corrections"][analysis["correction_loop"]].float()
        states = state_artifact[analysis["rho_reach_state"]].float()
        item_ids = [str(value) for value in correction_artifact["item_ids"]]
        batteries = [str(value) for value in correction_artifact["batteries"]]
        if corrections.shape != states.shape or len(item_ids) != corrections.shape[0]:
            raise RuntimeError(f"seed {seed} paired Arm-6 artifacts changed shape")
        common_mode = common_mode_receipt(corrections)
        common_mode_direction = _unit(_unit(corrections).mean(dim=0))
        common_mode_path = args.output_dir / f"seed_{seed}_common_mode.pt"
        torch.save(common_mode_direction, common_mode_path)
        common_mode["direction_tensor"] = {
            "path": common_mode_path.name,
            "sha256": sha256_file(common_mode_path),
        }
        seed_receipt: dict[str, Any] = {
            "rows": int(corrections.shape[0]),
            "hidden_size": int(corrections.shape[1]),
            "input_sha256": {
                "correction_field": sha256_file(correction_path),
                "state_features": sha256_file(state_path),
            },
            "common_mode": common_mode,
            "forced_clusterings": {},
        }
        forced_tensors: dict[int, dict[str, torch.Tensor]] = {}
        for clusters in analysis["forced_cluster_counts"]:
            receipt, tensors = cluster_receipt(
                corrections,
                states,
                item_ids=item_ids,
                batteries=batteries,
                clusters=clusters,
                seed=analysis["seed_base"] + seed,
                restarts=analysis["spherical_kmeans_restarts"],
                iterations=analysis["spherical_kmeans_iterations"],
                block_size=analysis["wht_block_size"],
                global_common_mode_direction=common_mode_direction,
            )
            tensor_path = args.output_dir / f"seed_{seed}_k{clusters}_initializers.pt"
            torch.save(tensors, tensor_path)
            receipt["initializer_tensor"] = {
                "path": tensor_path.name,
                "sha256": sha256_file(tensor_path),
            }
            seed_receipt["forced_clusterings"][str(clusters)] = receipt
            forced_tensors[clusters] = tensors
        seed_receipt["cross_fitted_residual_correlation"] = (
            cross_fitted_residual_correlation(
                corrections,
                forced_tensors[2]["labels"],
                splits=analysis["crossfit_splits"],
                seed_base=analysis["crossfit_seed_base"],
                rank_start=analysis["nuisance_rank_start"],
                rank_max=analysis["nuisance_rank_max"],
                persistence_fraction=analysis["mp_persistence_fraction"],
            )
        )
        output["seeds"][str(seed)] = seed_receipt
        atomic_json(args.output_dir / "status.json", output)
    output["status"] = "complete"
    output["registered_predictions"] = {
        "p5_rho_reach_below_0p5": all(
            cell["rho_reach"]["rho_reach"] < 0.5
            for seed in output["seeds"].values()
            for cell in seed["forced_clusterings"]["2"]["cluster_cells"].values()
        ),
        "p6_forced_k2_silhouette_at_least_0p70": all(
            seed["forced_clusterings"]["2"]["silhouette"] >= 0.70
            for seed in output["seeds"].values()
        ),
    }
    atomic_json(args.output_dir / "summary.json", output)
    atomic_json(args.output_dir / "status.json", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("training/paper2_bicameral_stage0_lock.json"),
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
