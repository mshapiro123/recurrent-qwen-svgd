"""Run the locked, CPU-only Bicameral Stage-0 geometry analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        battery_counts = Counter(b for b, keep in zip(batteries, mask.tolist()) if keep)
        cells[key] = {
            "rows": int(mask.sum()),
            "battery_counts": dict(sorted(battery_counts.items())),
            "post_global_projection_rho_res": float(residual_cosines.mean()),
            "post_global_projection_rho_res_median": float(residual_cosines.median()),
            "rho_reach": reachable_fraction(
                correction_rows, state_rows, block_size=block_size
            ),
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
