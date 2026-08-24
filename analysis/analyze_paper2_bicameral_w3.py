"""CPU-only X-6 metadata alignment and D-M5 conditional-map power audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_rank(matrix: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(matrix.double())
    probabilities = singular.square() / singular.square().sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def eta_squared(values: torch.Tensor, labels: Sequence[str]) -> float:
    total = values.double().var(unbiased=False) * values.numel()
    if float(total) == 0.0:
        return 0.0
    grand = values.double().mean()
    between = values.new_tensor(0.0, dtype=torch.float64)
    for label in sorted(set(labels)):
        selected = values[[item == label for item in labels]].double()
        between += selected.numel() * (selected.mean() - grand).square()
    return float(between / total)


def permutation_eta(
    values: torch.Tensor, labels: Sequence[str], *, seed: int, draws: int = 1000
) -> dict[str, Any]:
    observed = eta_squared(values, labels)
    generator = torch.Generator().manual_seed(seed)
    null = []
    label_tensor = torch.arange(len(labels))
    for _ in range(draws):
        order = torch.randperm(len(labels), generator=generator)
        shuffled = [labels[index] for index in label_tensor[order].tolist()]
        null.append(eta_squared(values, shuffled))
    null_tensor = torch.tensor(null)
    return {
        "eta_squared": observed,
        "permutation_draws": draws,
        "null_p95": float(torch.quantile(null_tensor, 0.95)),
        "empirical_p": (1 + sum(value >= observed for value in null)) / (draws + 1),
    }


def x6_alignment(
    correction: torch.Tensor,
    batteries: Sequence[str],
    cluster_labels: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    unit = F.normalize(correction.float(), dim=-1)
    common = F.normalize(unit.mean(dim=0), dim=0)
    residual = unit - (unit @ common)[:, None] * common[None]
    _u, singular, vh = torch.linalg.svd(residual.double(), full_matrices=False)
    total_energy = singular.square().sum().clamp_min(1e-30)
    directions = vh[:3].float()
    coefficients = residual @ directions.T
    cells = []
    cluster_names = [str(int(value)) for value in cluster_labels]
    for index in range(3):
        values = coefficients[:, index]
        by_battery = {}
        for battery in sorted(set(batteries)):
            selected = values[[item == battery for item in batteries]]
            by_battery[battery] = {
                "rows": int(selected.numel()),
                "signed_mean": float(selected.mean()),
                "absolute_mean": float(selected.abs().mean()),
            }
        by_cluster = {}
        for cluster in sorted(set(cluster_names)):
            selected = values[[item == cluster for item in cluster_names]]
            by_cluster[cluster] = {
                "rows": int(selected.numel()),
                "signed_mean": float(selected.mean()),
                "absolute_mean": float(selected.abs().mean()),
            }
        cells.append(
            {
                "direction": index + 1,
                "singular_value": float(singular[index]),
                "residual_energy_fraction": float(singular[index].square() / total_energy),
                "battery_association": permutation_eta(
                    values, batteries, seed=seed + index * 10_000
                ),
                "cluster_association": permutation_eta(
                    values, cluster_names, seed=seed + index * 10_000 + 1
                ),
                "by_battery": by_battery,
                "by_cluster": by_cluster,
            }
        )
    return {
        "kind": "paper2_bicameral_x6_metadata_alignment_v1",
        "common_mode_energy_fraction": float((unit @ common).square().sum() / unit.square().sum()),
        "residual_effective_rank": effective_rank(residual),
        "directions": cells,
    }


def projected_ridge(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    rank: int,
    ridge: float,
) -> torch.Tensor:
    x_mean = x_train.double().mean(dim=0, keepdim=True)
    y_mean = y_train.double().mean(dim=0, keepdim=True)
    _ux, _sx, vx = torch.linalg.svd(x_train.double() - x_mean, full_matrices=False)
    _uy, _sy, vy = torch.linalg.svd(y_train.double() - y_mean, full_matrices=False)
    width = min(rank, vx.shape[0], vy.shape[0])
    x_basis = vx[:width].T
    y_basis = vy[:width].T
    z_x = (x_train.double() - x_mean) @ x_basis
    z_y = (y_train.double() - y_mean) @ y_basis
    gram = z_x.T @ z_x
    weights = torch.linalg.solve(
        gram + ridge * torch.eye(width, dtype=gram.dtype), z_x.T @ z_y
    )
    return ((x_eval.double() - x_mean) @ x_basis @ weights @ y_basis.T + y_mean).float()


def map_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    target_energy = target.float().square().sum(dim=-1).clamp_min(1e-12)
    residual_energy = (prediction.float() - target.float()).square().sum(dim=-1)
    return {
        "mean_cosine": float(cosine.mean()),
        "cosine_sd": float(cosine.std(unbiased=True)),
        "positive_cosine_fraction": float((cosine > 0).float().mean()),
        "mean_explained_energy": float((1.0 - residual_energy / target_energy).mean()),
    }


def dm5_power(
    states: Mapping[str, torch.Tensor],
    correction: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    x = torch.cat([states["loop1"].float(), states["direction"].float()], dim=-1)
    y = F.normalize(correction.float(), dim=-1)
    generator = torch.Generator().manual_seed(seed)
    ranks = (4, 8, 16, 32, 64)
    cells = []
    for rank in ranks:
        folds = []
        for split in range(20):
            order = torch.randperm(x.shape[0], generator=generator)
            train = order[:192]
            evaluate = order[192:]
            prediction = projected_ridge(
                x[train], y[train], x[evaluate], rank=rank, ridge=1e-3
            )
            folds.append(map_metrics(prediction, y[evaluate]))
        mean_cosines = torch.tensor([fold["mean_cosine"] for fold in folds])
        sd = float(mean_cosines.std(unbiased=True))
        effect = float(mean_cosines.mean())
        # Normal approximation: two-sided alpha=.05, 80% power, split-level variance.
        required = math.inf if effect <= 0 else math.ceil(((1.96 + 0.8416) * sd / effect) ** 2)
        cells.append(
            {
                "rank": rank,
                "projected_map_parameters": rank * rank,
                "fit_rows_per_split": 192,
                "eval_rows_per_split": 64,
                "splits": 20,
                "mean_cosine": effect,
                "split_mean_cosine_sd": sd,
                "mean_explained_energy": sum(
                    fold["mean_explained_energy"] for fold in folds
                )
                / len(folds),
                "positive_cosine_fraction": sum(
                    fold["positive_cosine_fraction"] for fold in folds
                )
                / len(folds),
                "normal_approx_independent_eval_splits_for_80pct_power": required,
            }
        )
    return {
        "kind": "paper2_bicameral_dm5_map_estimand_power_v1",
        "rows": int(x.shape[0]),
        "input_width": int(x.shape[1]),
        "output_width": int(y.shape[1]),
        "input_effective_rank": effective_rank(x - x.mean(dim=0)),
        "output_effective_rank": effective_rank(y - y.mean(dim=0)),
        "cells": cells,
        "scope": "desk feasibility and power planning only; no Step-2 gate",
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    correction_path = args.input_root / f"private/seed_{seed}/correction_field__initialization.pt"
    state_path = args.input_root / f"private/seed_{seed}/state_features__initialization.pt"
    initializer_path = args.input_root / f"outputs_r3/seed_{seed}_k2_initializers.pt"
    correction_artifact = torch.load(correction_path, map_location="cpu", weights_only=False)
    states = torch.load(state_path, map_location="cpu", weights_only=False)
    initializers = torch.load(initializer_path, map_location="cpu", weights_only=False)
    correction = correction_artifact["corrections"][4].float()
    result = {
        "seed": seed,
        "inputs": {
            "correction_field_sha256": sha256_file(correction_path),
            "state_features_sha256": sha256_file(state_path),
            "k2_initializers_sha256": sha256_file(initializer_path),
        },
        "x6": x6_alignment(
            correction,
            correction_artifact["batteries"],
            initializers["labels"],
            seed=20260824 + seed,
        ),
        "dm5": dm5_power(states, correction, seed=20260824 + seed),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = {str(seed): run_seed(args, seed) for seed in (0, 1)}
    result = {
        "kind": "paper2_bicameral_w3_desk_wave_v1",
        "status": "complete_cpu_only",
        "seeds": seeds,
        "gpu_used": False,
        "training_performed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
