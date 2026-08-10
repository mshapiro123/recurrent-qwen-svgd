"""Document-disjoint ridge forecast from cached Phase 3 features to d*."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class DocumentSplit:
    train: torch.Tensor
    calibration: torch.Tensor
    holdout: torch.Tensor


def stable_fraction(value: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def document_split(
    documents: Sequence[str],
    *,
    seed: int,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> DocumentSplit:
    if not 0 < train_fraction < 1:
        raise ValueError("train fraction must be inside (0,1)")
    if not 0 < calibration_fraction < 1 - train_fraction:
        raise ValueError("calibration fraction must leave a holdout")
    assignments: dict[str, str] = {}
    for document in sorted(set(documents)):
        fraction = stable_fraction(document, seed=seed)
        if fraction < train_fraction:
            assignments[document] = "train"
        elif fraction < train_fraction + calibration_fraction:
            assignments[document] = "calibration"
        else:
            assignments[document] = "holdout"
    indices = {
        key: torch.tensor(
            [index for index, document in enumerate(documents) if assignments[document] == key],
            dtype=torch.long,
        )
        for key in ("train", "calibration", "holdout")
    }
    if any(value.numel() == 0 for value in indices.values()):
        raise ValueError("document split produced an empty partition")
    sets = {
        key: {documents[index] for index in value.tolist()}
        for key, value in indices.items()
    }
    if sets["train"] & sets["calibration"] or sets["train"] & sets["holdout"] or sets["calibration"] & sets["holdout"]:
        raise RuntimeError("document leakage across linear-forecast partitions")
    return DocumentSplit(**indices)


def cosine_rows(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    numerator = (prediction * target).sum(dim=-1)
    denominator = prediction.norm(dim=-1) * target.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-8)


def fit_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, torch.Tensor]:
    x = train_x.float()
    y = train_y.float()
    x_mean = x.mean(dim=0)
    x_scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y.mean(dim=0)
    normalized = (x - x_mean) / x_scale
    centered_y = y - y_mean
    gram = normalized.T @ normalized
    gram.diagonal().add_(float(ridge))
    weight = torch.linalg.solve(gram, normalized.T @ centered_y)
    return {
        "weight": weight,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
    }


def fit_ridge_family(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    ridges: Sequence[float],
) -> list[dict[str, torch.Tensor]]:
    """Fit a ridge family from one shared eigensystem."""

    x = train_x.float()
    y = train_y.float()
    x_mean = x.mean(dim=0)
    x_scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y.mean(dim=0)
    normalized = (x - x_mean) / x_scale
    centered_y = y - y_mean
    gram = normalized.T @ normalized
    right = normalized.T @ centered_y
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    projected = eigenvectors.T @ right
    models = []
    for ridge in ridges:
        if float(ridge) <= 0:
            raise ValueError("ridge candidates must be positive")
        weight = eigenvectors @ (projected / (eigenvalues + float(ridge)).unsqueeze(1))
        models.append(
            {
                "weight": weight,
                "x_mean": x_mean,
                "x_scale": x_scale,
                "y_mean": y_mean,
            }
        )
    return models


def predict_ridge(model: dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    normalized = (features.float() - model["x_mean"]) / model["x_scale"]
    return normalized @ model["weight"] + model["y_mean"]


def document_bootstrap_ci(
    values: torch.Tensor,
    documents: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    values_np = values.detach().cpu().numpy().astype(np.float64, copy=False)
    unique = sorted(set(documents))
    grouped = {
        document: values_np[np.asarray([value == document for value in documents])]
        for document in unique
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([grouped[document] for document in selected])
        estimates[index] = sampled.mean()
    return {
        "mean": float(values_np.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_unit": "document",
    }


def run_forecast(
    *,
    features: torch.Tensor,
    directions: torch.Tensor,
    documents: Sequence[str],
    ridge_candidates: Sequence[float],
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if features.ndim != 2 or directions.ndim != 2:
        raise ValueError("forecast features and directions must be rank two")
    if features.shape[0] != directions.shape[0] or features.shape[0] != len(documents):
        raise ValueError("forecast rows do not align")
    if not torch.isfinite(features).all() or not torch.isfinite(directions).all():
        raise ValueError("forecast inputs contain non-finite values")
    split = document_split(documents, seed=seed)
    candidates = []
    models = fit_ridge_family(
        features.index_select(0, split.train),
        directions.index_select(0, split.train),
        ridges=ridge_candidates,
    )
    for ridge, model in zip(ridge_candidates, models):
        prediction = predict_ridge(model, features.index_select(0, split.calibration))
        cosine = cosine_rows(prediction, directions.index_select(0, split.calibration))
        candidates.append(
            {
                "ridge": float(ridge),
                "calibration_mean_cosine": float(cosine.mean()),
            }
        )
    best_index = max(
        range(len(candidates)),
        key=lambda index: (candidates[index]["calibration_mean_cosine"], -candidates[index]["ridge"]),
    )
    best = candidates[best_index]
    holdout_prediction = predict_ridge(
        models[best_index], features.index_select(0, split.holdout)
    )
    holdout_target = directions.index_select(0, split.holdout)
    holdout_cosine = cosine_rows(holdout_prediction, holdout_target)
    holdout_documents = [documents[index] for index in split.holdout.tolist()]
    ci = document_bootstrap_ci(
        holdout_cosine,
        holdout_documents,
        seed=seed + 1,
        replicates=bootstrap_replicates,
    )
    partition_documents = {
        name: len({documents[index] for index in indices.tolist()})
        for name, indices in (
            ("train", split.train),
            ("calibration", split.calibration),
            ("holdout", split.holdout),
        )
    }
    return {
        "kind": "paper2_phase3_linear_decodability_forecast_v1",
        "status": "complete_descriptive_forecast_not_bound",
        "rows": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "direction_dimension": int(directions.shape[1]),
        "split": {
            "seed": int(seed),
            "train_rows": int(split.train.numel()),
            "calibration_rows": int(split.calibration.numel()),
            "holdout_rows": int(split.holdout.numel()),
            "documents": partition_documents,
            "document_disjoint": True,
        },
        "ridge_candidates": candidates,
        "selected_ridge": float(best["ridge"]),
        "holdout_cosine": ci,
        "interpretation": (
            "linear-decodability forecast only; it is neither an upper nor a lower bound "
            "on nonlinear bridge aim capture"
        ),
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_cache", type=Path, required=True)
    parser.add_argument("--direction_cache", type=Path)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument(
        "--ridge_candidates",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap_replicates", type=int, default=2_000)
    parser.add_argument("--min_teachability", type=float)
    args = parser.parse_args()
    payload = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    if args.direction_cache is None:
        required = {"features", "directions", "documents"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"linear-forecast cache missing fields: {missing}")
        directions = payload["directions"]
        documents = list(payload["documents"])
        selected = torch.arange(payload["features"].shape[0])
    else:
        direction_payload = torch.load(
            args.direction_cache, map_location="cpu", weights_only=False
        )
        if list(payload.get("record_ids", [])) != list(
            direction_payload.get("record_ids", [])
        ):
            raise ValueError("feature and direction cache record ids do not align")
        directions = direction_payload["directions"]
        documents = list(direction_payload["documents"])
        selected = torch.arange(payload["features"].shape[0])
        if args.min_teachability is not None:
            selected = torch.where(
                direction_payload["teachability"].float()
                >= float(args.min_teachability)
            )[0]
        if selected.numel() == 0:
            raise ValueError("teachability filter removed every forecast row")
    result = run_forecast(
        features=payload["features"].index_select(0, selected),
        directions=directions.index_select(0, selected),
        documents=[documents[index] for index in selected.tolist()],
        ridge_candidates=args.ridge_candidates,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    result["source"] = {
        "feature_cache": str(args.feature_cache),
        "direction_cache": str(args.direction_cache) if args.direction_cache else None,
        "min_teachability": args.min_teachability,
        "threshold_selected_for_p33": False,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
