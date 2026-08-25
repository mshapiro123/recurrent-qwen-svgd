"""Closed-form contracts for the W2-prime conditional-mixer desk wave."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


RANK_GRID = (2, 4, 8, 16, 32)
RIDGE_MULTIPLIER_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
OUTER_FOLDS = 4
INNER_FOLDS = 3
FOLD_SEED = 20260825
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_DRAWS = 5000
CONDITIONAL_COSINE_GATE = 0.30
HEMISPHERIC_RISK_REDUCTION_GATE = 0.05


def validate_deployment_features(payload: Mapping[str, Any]) -> None:
    """Reject any feature cache that used gold or teacher information."""

    required = {
        "input_provenance": "student_prompt_only",
        "gold_answer_used": False,
        "teacher_forward_used": False,
        "oracle_routing_used": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"W2-prime leak-boundary violation: {mismatches}")


def deterministic_stratified_folds(
    labels: Sequence[str], *, folds: int = OUTER_FOLDS, seed: int = FOLD_SEED
) -> torch.Tensor:
    if folds < 2 or len(labels) < folds:
        raise ValueError("cross-fitting requires at least two folds and one row per fold")
    assignments = torch.full((len(labels),), -1, dtype=torch.long)
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(str(label), []).append(index)
    for label in sorted(grouped):
        digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
        generator = random.Random(int.from_bytes(digest[:8], "big"))
        indices = grouped[label]
        generator.shuffle(indices)
        for offset, index in enumerate(indices):
            assignments[index] = offset % folds
    if torch.any(assignments < 0) or any(int((assignments == fold).sum()) == 0 for fold in range(folds)):
        raise RuntimeError("deterministic stratified fold construction failed")
    return assignments


def deterministic_derangement(size: int, *, tag: str, seed: int = FOLD_SEED) -> torch.Tensor:
    if size < 2:
        raise ValueError("derangement requires at least two rows")
    digest = hashlib.sha256(f"{seed}:{tag}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(size))
    for index in range(size - 1, 0, -1):
        other = generator.randrange(index)
        order[index], order[other] = order[other], order[index]
    if any(index == source for index, source in enumerate(order)):
        raise RuntimeError("registered W2-prime derangement failed")
    return torch.tensor(order, dtype=torch.long)


def _ridge_scale(centered: torch.Tensor) -> float:
    return max(float(centered.double().square().sum() / centered.shape[1]), 1e-12)


def _supervised_projection(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    rank: int,
    ridge_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return a supervised ridge-whitened input basis.

    The SVD is evaluated in sample space. The resulting basis spans the leading
    left singular vectors of the ridge-whitened input-to-target cross-covariance.
    """

    x_mean = x_train.double().mean(dim=0, keepdim=True)
    y_mean = y_train.double().mean(dim=0, keepdim=True)
    x_centered = x_train.double() - x_mean
    y_centered = y_train.double() - y_mean
    u, singular, vh = torch.linalg.svd(x_centered, full_matrices=False)
    ridge = float(ridge_multiplier) * _ridge_scale(x_centered)
    denominator = (singular.square() + ridge).sqrt().clamp_min(1e-15)
    cross = (singular / denominator)[:, None] * (u.T @ y_centered)
    left, _values, _right = torch.linalg.svd(cross, full_matrices=False)
    width = min(int(rank), left.shape[1], vh.shape[0])
    projection = vh.T @ ((1.0 / denominator)[:, None] * left[:, :width])
    return x_mean.float(), y_mean.float(), projection.float(), ridge


@dataclass
class BlockMap:
    x_means: tuple[torch.Tensor, ...]
    projections: tuple[torch.Tensor, ...]
    y_mean: torch.Tensor
    theta: torch.Tensor
    ridge_effective: tuple[float, ...]

    def predict(self, blocks: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(blocks) != len(self.projections):
            raise ValueError("conditional-map block count changed")
        scores = [
            (block.float() - mean) @ projection
            for block, mean, projection in zip(blocks, self.x_means, self.projections)
        ]
        return torch.cat(scores, dim=-1) @ self.theta + self.y_mean


def fit_block_map(
    blocks: Sequence[torch.Tensor],
    target: torch.Tensor,
    *,
    ranks: Sequence[int],
    ridge_multipliers: Sequence[float],
) -> BlockMap:
    if not blocks or len(blocks) != len(ranks) or len(blocks) != len(ridge_multipliers):
        raise ValueError("conditional-map block hyperparameters changed")
    if any(block.ndim != 2 or block.shape[0] != target.shape[0] for block in blocks):
        raise ValueError("conditional-map block geometry changed")
    x_means = []
    projections = []
    effective = []
    y_mean = target.float().mean(dim=0, keepdim=True)
    for block, rank, ridge in zip(blocks, ranks, ridge_multipliers):
        x_mean, _unused_y_mean, projection, ridge_value = _supervised_projection(
            block, target, rank=int(rank), ridge_multiplier=float(ridge)
        )
        x_means.append(x_mean)
        projections.append(projection)
        effective.append(ridge_value)
    scores = torch.cat(
        [(block.float() - mean) @ projection for block, mean, projection in zip(blocks, x_means, projections)],
        dim=-1,
    ).double()
    centered_target = target.double() - y_mean.double()
    final_ridge = max(float(scores.square().sum() / max(scores.shape[1], 1)) * 1e-8, 1e-12)
    gram = scores.T @ scores + final_ridge * torch.eye(scores.shape[1], dtype=torch.float64)
    theta = torch.linalg.solve(gram, scores.T @ centered_target).float()
    return BlockMap(tuple(x_means), tuple(projections), y_mean, theta, tuple(effective))


def normalized_row_risk(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    baseline = target.float() - target.float().mean(dim=0, keepdim=True)
    denominator = baseline.square().sum(dim=-1).mean().clamp_min(1e-12)
    return (prediction.float() - target.float()).square().sum(dim=-1) / denominator


def conditional_row_cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean = target.float().mean(dim=0, keepdim=True)
    return F.cosine_similarity(prediction.float() - mean, target.float() - mean, dim=-1, eps=1e-12)


def _candidate_grid(
    rank_options: Sequence[Sequence[int]],
    ridge_options: Sequence[Sequence[float]],
) -> list[tuple[tuple[int, float], ...]]:
    if len(rank_options) != len(ridge_options):
        raise ValueError("rank and ridge option blocks changed")
    per_block = [tuple(itertools.product(ranks, ridges)) for ranks, ridges in zip(rank_options, ridge_options)]
    return list(itertools.product(*per_block))


def _crossfit_prediction(
    blocks: Sequence[torch.Tensor],
    target: torch.Tensor,
    folds: torch.Tensor,
    candidate: tuple[tuple[int, float], ...],
) -> torch.Tensor:
    prediction = torch.empty_like(target, dtype=torch.float32)
    ranks = tuple(item[0] for item in candidate)
    ridges = tuple(item[1] for item in candidate)
    for fold in sorted(set(int(value) for value in folds.tolist())):
        evaluate = folds == fold
        train = ~evaluate
        model = fit_block_map(
            [block[train] for block in blocks],
            target[train],
            ranks=ranks,
            ridge_multipliers=ridges,
        )
        prediction[evaluate] = model.predict([block[evaluate] for block in blocks])
    return prediction


def _prepare_fold_cache(
    blocks: Sequence[torch.Tensor],
    target: torch.Tensor,
    folds: torch.Tensor,
    *,
    rank_options: Sequence[Sequence[int]],
    ridge_options: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    caches = []
    for fold in sorted(set(int(value) for value in folds.tolist())):
        evaluate = folds == fold
        train = ~evaluate
        y_train = target[train].float()
        block_cache = []
        for block_index, block in enumerate(blocks):
            ridge_cache = {}
            for ridge in ridge_options[block_index]:
                mean, _y_mean, projection, effective = _supervised_projection(
                    block[train],
                    y_train,
                    rank=max(rank_options[block_index]),
                    ridge_multiplier=ridge,
                )
                ridge_cache[float(ridge)] = {
                    "train": (block[train].float() - mean) @ projection,
                    "evaluate": (block[evaluate].float() - mean) @ projection,
                    "ridge_effective": effective,
                }
            block_cache.append(ridge_cache)
        caches.append(
            {
                "evaluate": evaluate,
                "target_train": y_train,
                "blocks": block_cache,
            }
        )
    return caches


def _predict_from_fold_cache(
    caches: Sequence[Mapping[str, Any]],
    target: torch.Tensor,
    candidate: tuple[tuple[int, float], ...],
) -> torch.Tensor:
    prediction = torch.empty_like(target, dtype=torch.float32)
    for cache in caches:
        train_scores = []
        eval_scores = []
        for block_index, (rank, ridge) in enumerate(candidate):
            item = cache["blocks"][block_index][float(ridge)]
            train_scores.append(item["train"][:, : int(rank)])
            eval_scores.append(item["evaluate"][:, : int(rank)])
        z_train = torch.cat(train_scores, dim=-1).double()
        z_eval = torch.cat(eval_scores, dim=-1).double()
        y_train = cache["target_train"].double()
        y_mean = y_train.mean(dim=0, keepdim=True)
        centered = y_train - y_mean
        final_ridge = max(float(z_train.square().sum() / max(z_train.shape[1], 1)) * 1e-8, 1e-12)
        gram = z_train.T @ z_train + final_ridge * torch.eye(z_train.shape[1], dtype=torch.float64)
        theta = torch.linalg.solve(gram, z_train.T @ centered)
        prediction[cache["evaluate"]] = (z_eval @ theta + y_mean).float()
    return prediction


def _select_single_block(
    block: torch.Tensor,
    target: torch.Tensor,
    labels: Sequence[str],
    *,
    seed: int,
    ranks: Sequence[int],
    ridges: Sequence[float],
    folds: int,
) -> dict[str, Any]:
    assignments = deterministic_stratified_folds(labels, folds=folds, seed=seed)
    caches = _prepare_fold_cache(
        [block],
        target,
        assignments,
        rank_options=(tuple(ranks),),
        ridge_options=(tuple(ridges),),
    )
    table = []
    best: tuple[int, float] | None = None
    best_risk = math.inf
    for rank, ridge in itertools.product(ranks, ridges):
        candidate = ((int(rank), float(ridge)),)
        prediction = _predict_from_fold_cache(caches, target, candidate)
        risk = float(normalized_row_risk(prediction, target).mean())
        table.append(
            {
                "rank": int(rank),
                "ridge_multiplier": float(ridge),
                "normalized_risk": risk,
            }
        )
        key = (risk, int(rank), float(ridge))
        if best is None or key < (best_risk, best[0], best[1]):
            best = (int(rank), float(ridge))
            best_risk = risk
    assert best is not None
    return {
        "selected": {
            "rank": best[0],
            "ridge_multiplier": best[1],
            "normalized_risk": best_risk,
        },
        "grid": table,
    }


def select_crossfitted_map(
    blocks: Sequence[torch.Tensor],
    target: torch.Tensor,
    labels: Sequence[str],
    *,
    seed: int,
    rank_options: Sequence[Sequence[int]] | None = None,
    ridge_options: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    if rank_options is None:
        rank_options = tuple(RANK_GRID for _ in blocks)
    if ridge_options is None:
        ridge_options = tuple(RIDGE_MULTIPLIER_GRID for _ in blocks)
    if len(rank_options) != len(blocks) or len(ridge_options) != len(blocks):
        raise ValueError("cross-fitted map option blocks changed")
    folds = deterministic_stratified_folds(labels, seed=seed)
    prediction = torch.empty_like(target, dtype=torch.float32)
    outer_receipts = []
    for outer_fold in range(OUTER_FOLDS):
        evaluate = folds == outer_fold
        train = ~evaluate
        train_labels = [str(labels[index]) for index in torch.where(train)[0].tolist()]
        selected_blocks = []
        for block_index, block in enumerate(blocks):
            result = _select_single_block(
                block[train],
                target[train],
                train_labels,
                seed=seed + 1000 + outer_fold * 100 + block_index,
                ranks=rank_options[block_index],
                ridges=ridge_options[block_index],
                folds=INNER_FOLDS,
            )
            selected_blocks.append(result["selected"])
        model = fit_block_map(
            [block[train] for block in blocks],
            target[train],
            ranks=[item["rank"] for item in selected_blocks],
            ridge_multipliers=[item["ridge_multiplier"] for item in selected_blocks],
        )
        prediction[evaluate] = model.predict([block[evaluate] for block in blocks])
        outer_receipts.append(
            {
                "outer_fold": outer_fold,
                "train_rows": int(train.sum()),
                "evaluate_rows": int(evaluate.sum()),
                "selected_blocks": selected_blocks,
            }
        )

    final_blocks = []
    final_grids = []
    for block_index, block in enumerate(blocks):
        result = _select_single_block(
            block,
            target,
            labels,
            seed=seed + 9000 + block_index,
            ranks=rank_options[block_index],
            ridges=ridge_options[block_index],
            folds=OUTER_FOLDS,
        )
        final_blocks.append(result["selected"])
        final_grids.append(result["grid"])
    best_risk = float(normalized_row_risk(prediction, target).mean())
    return {
        "prediction": prediction,
        "folds": folds,
        "selected": {
            "ranks": [int(item["rank"]) for item in final_blocks],
            "ridge_multipliers": [float(item["ridge_multiplier"]) for item in final_blocks],
            "normalized_risk": best_risk,
        },
        "grid": {
            "selection_rule": "nested_blockwise_inner_cv_then_joint_refit",
            "outer_folds": outer_receipts,
            "full_data_block_grids": final_grids,
        },
    }


def bootstrap_mean_interval(
    values: torch.Tensor,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, float | int]:
    values = values.double().cpu()
    generator = torch.Generator().manual_seed(seed)
    means = torch.empty(draws, dtype=torch.float64)
    for start in range(0, draws, 250):
        width = min(250, draws - start)
        indices = torch.randint(values.numel(), (width, values.numel()), generator=generator)
        means[start : start + width] = values[indices].mean(dim=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(torch.quantile(means, 0.025)),
        "ci_high": float(torch.quantile(means, 0.975)),
        "draws": int(draws),
    }


def paired_relative_risk_reduction(
    reference_risk: torch.Tensor,
    candidate_risk: torch.Tensor,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, float | int]:
    reference = reference_risk.double().cpu()
    candidate = candidate_risk.double().cpu()
    if reference.shape != candidate.shape:
        raise ValueError("paired risk vectors changed shape")
    observed = 1.0 - candidate.mean() / reference.mean().clamp_min(1e-12)
    generator = torch.Generator().manual_seed(seed)
    samples = torch.empty(draws, dtype=torch.float64)
    for start in range(0, draws, 250):
        width = min(250, draws - start)
        indices = torch.randint(reference.numel(), (width, reference.numel()), generator=generator)
        ref = reference[indices].mean(dim=1).clamp_min(1e-12)
        cand = candidate[indices].mean(dim=1)
        samples[start : start + width] = 1.0 - cand / ref
    return {
        "relative_risk_reduction": float(observed),
        "ci_low": float(torch.quantile(samples, 0.025)),
        "ci_high": float(torch.quantile(samples, 0.975)),
        "draws": int(draws),
    }
