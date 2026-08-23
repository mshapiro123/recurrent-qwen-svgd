"""CPU-only Stage 2B-S desk-math wave over banked artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from eval.eval_paper2_stage2b_autopsy import _checkpoint_state
from training.paper2_stage2b_autopsy import spherical_kmeans


WH = "stage2b_depth_attachment.flow.hidden_innovation.weight"
WP = "stage2b_depth_attachment.flow.prompt_gate.weight"
BL = "stage2b_depth_attachment.bridge.output_projection.weight"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-12)


def mp_median(aspect: float, points: int = 200_001) -> float:
    """Numerically return the unit-variance Marchenko-Pastur median."""

    if not 0.0 < aspect <= 1.0:
        raise ValueError("MP aspect must be in (0, 1]")
    lower = (1.0 - math.sqrt(aspect)) ** 2
    upper = (1.0 + math.sqrt(aspect)) ** 2
    grid = np.linspace(lower, upper, points, dtype=np.float64)
    density = np.sqrt(np.maximum((upper - grid) * (grid - lower), 0.0))
    density /= 2.0 * math.pi * aspect * np.maximum(grid, np.finfo(float).tiny)
    dx = grid[1] - grid[0]
    cdf = np.cumsum((density[:-1] + density[1:]) * 0.5 * dx)
    cdf /= cdf[-1]
    return float(np.interp(0.5, cdf, grid[1:]))


def mp_spectrum(matrix: torch.Tensor) -> dict[str, Any]:
    matrix = matrix.float().reshape(matrix.shape[0], -1)
    rows, columns = matrix.shape
    if rows > columns:
        matrix = matrix.T
        rows, columns = matrix.shape
    singular = torch.linalg.svdvals(matrix).double().cpu().numpy()
    eigenvalues = singular**2 / columns
    aspect = rows / columns
    median = mp_median(aspect)
    sigma2 = float(np.median(eigenvalues) / median)
    lower = sigma2 * (1.0 - math.sqrt(aspect)) ** 2
    upper = sigma2 * (1.0 + math.sqrt(aspect)) ** 2
    outliers = eigenvalues > upper
    return {
        "shape": [rows, columns],
        "aspect_ratio": aspect,
        "noise_variance_median_mp": sigma2,
        "mp_lower_edge": lower,
        "mp_upper_edge": upper,
        "outlier_spikes": int(outliers.sum()),
        "leading_singular_values": singular[:10].tolist(),
        "leading_normalized_eigenvalues": eigenvalues[:10].tolist(),
    }


def hidden_singular_direction(matrix: torch.Tensor, hidden_size: int = 896) -> torch.Tensor:
    matrix = matrix.float().reshape(matrix.shape[0], -1)
    u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
    if matrix.shape[1] == hidden_size:
        return unit(vh[0])
    if matrix.shape[0] == hidden_size:
        return unit(u[:, 0])
    raise RuntimeError(f"No hidden-width singular direction for shape {tuple(matrix.shape)}")


def correction_references(
    initial: Mapping[str, Any], stop: Mapping[str, Any], *, clusters: int, seed: int
) -> tuple[dict[str, torch.Tensor], list[int]]:
    initial_directions = unit(initial["corrections"][4])
    stop_directions = unit(stop["corrections"][4])
    labels, silhouette = spherical_kmeans(
        initial_directions,
        clusters=clusters,
        restarts=8,
        iterations=50,
        seed=20260819 + seed + clusters,
    )
    centers = torch.stack(
        [unit(initial_directions[labels == index].mean(dim=0)) for index in range(clusters)]
    )
    references = {
        "corpus_ce_descent_estimate_stop": unit(stop_directions.mean(dim=0)),
        "global_correction_mean_initialization": unit(initial_directions.mean(dim=0)),
        **{f"cluster_{index}_mean_initialization": centers[index] for index in range(clusters)},
    }
    counts = [int((labels == index).sum()) for index in range(clusters)]
    references["cluster_silhouette"] = torch.tensor(float(silhouette))
    return references, counts


def d_m1_seed(
    *,
    seed: int,
    initialization_state: Path,
    stop_checkpoint: Path,
    correction_initialization: Path,
    correction_stop: Path,
    clusters: int,
) -> tuple[dict[str, Any], dict[str, float], list[int]]:
    initial_payload = torch.load(initialization_state, map_location="cpu", weights_only=False)
    initial_state = initial_payload["state"]
    stop_state = _checkpoint_state(stop_checkpoint)
    correction_init = torch.load(correction_initialization, map_location="cpu", weights_only=False)
    correction_end = torch.load(correction_stop, map_location="cpu", weights_only=False)
    references, counts = correction_references(
        correction_init, correction_end, clusters=clusters, seed=seed
    )
    silhouette = float(references.pop("cluster_silhouette"))
    matrices = {
        "delta_W_H": stop_state[WH].float() - initial_state[WH].float(),
        "delta_W_P": stop_state[WP].float() - initial_state[WP].float(),
        "delta_bridge_B_L": stop_state[BL].float() - initial_state[BL].float(),
    }
    geometry = {}
    cluster_alignment: dict[str, float] = {}
    for name, matrix in matrices.items():
        direction = hidden_singular_direction(matrix)
        alignments = {
            reference_name: float(torch.dot(direction, reference.float()).abs())
            for reference_name, reference in references.items()
        }
        geometry[name] = {
            "frobenius_norm": float(torch.linalg.vector_norm(matrix)),
            "mp_bulk_fit": mp_spectrum(matrix),
            "top_hidden_direction_absolute_cosines": alignments,
        }
        for reference_name, value in alignments.items():
            if reference_name.startswith("cluster_"):
                cluster_alignment[reference_name] = max(
                    cluster_alignment.get(reference_name, 0.0), value
                )
    return (
        {
            "seed": seed,
            "clusters": clusters,
            "cluster_counts_in_256_row_audit": counts,
            "cluster_silhouette": silhouette,
            "matrices": geometry,
            "ce_descent_estimator": (
                "normalized mean of per-row negative CE gradients at the trained stop endpoint"
            ),
            "global_correction_estimator": (
                "normalized mean of per-row negative CE gradients at initialization"
            ),
        },
        cluster_alignment,
        counts,
    )


def fit_common_c(margins: np.ndarray, transitions: Sequence[int]) -> tuple[np.ndarray, float]:
    x = margins[:, list(transitions)]
    y = margins[:, [index + 1 for index in transitions]]
    c = 0.0
    for _ in range(10_000):
        r = (x * (y - c)).sum(axis=1) / np.maximum((x * x).sum(axis=1), 1e-12)
        updated = float((y - r[:, None] * x).mean())
        if abs(updated - c) < 1e-12:
            c = updated
            break
        c = updated
    r = (x * (y - c)).sum(axis=1) / np.maximum((x * x).sum(axis=1), 1e-12)
    return r, c


def fit_row_specific(margins: np.ndarray, transitions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x = margins[:, list(transitions)]
    y = margins[:, [index + 1 for index in transitions]]
    design = np.stack([x, np.ones_like(x)], axis=-1)
    coefficients = np.stack(
        [np.linalg.lstsq(design[index], y[index], rcond=None)[0] for index in range(len(x))]
    )
    return coefficients[:, 0], coefficients[:, 1]


def recurrence_cv(margins: np.ndarray) -> dict[str, Any]:
    errors = {"common_c": [], "row_specific_c": []}
    for held_out in range(3):
        training = [index for index in range(3) if index != held_out]
        r_common, c_common = fit_common_c(margins, training)
        r_row, c_row = fit_row_specific(margins, training)
        x = margins[:, held_out]
        y = margins[:, held_out + 1]
        errors["common_c"].extend((y - (r_common * x + c_common)).tolist())
        errors["row_specific_c"].extend((y - (r_row * x + c_row)).tolist())
    return {
        name: {
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "mae": float(np.mean(np.abs(values))),
            "predictions": len(values),
        }
        for name, values in errors.items()
    }


def decomposition(margins: np.ndarray, *, common_c: bool) -> dict[str, Any]:
    if common_c:
        r, shared = fit_common_c(margins, (0, 1, 2))
        c = np.full(len(r), shared)
    else:
        r, c = fit_row_specific(margins, (0, 1, 2))
        shared = None
    m1 = margins[:, 0]
    signal = (r**3 - 1.0) * m1
    bias = c * (1.0 + r + r**2)
    prediction = m1 + signal + bias
    total = signal + bias
    variance = float(np.var(total))
    covariance = float(np.cov(signal, bias, ddof=0)[0, 1])
    share_signal = (float(np.var(signal)) + covariance) / variance if variance > 0 else None
    share_bias = (float(np.var(bias)) + covariance) / variance if variance > 0 else None
    observed_flips = (m1 > 0.0) & (margins[:, 3] <= 0.0)
    no_bias = m1 + signal
    no_attenuation = m1 + bias
    return {
        "shared_c": shared,
        "r": {
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "p05": float(np.quantile(r, 0.05)),
            "p95": float(np.quantile(r, 0.95)),
        },
        "c": {
            "mean": float(np.mean(c)),
            "median": float(np.median(c)),
            "p05": float(np.quantile(c, 0.05)),
            "p95": float(np.quantile(c, 0.95)),
        },
        "k4_prediction_rmse": float(np.sqrt(np.mean((margins[:, 3] - prediction) ** 2))),
        "mean_absolute_signal_attenuation_contribution": float(np.mean(np.abs(signal))),
        "mean_absolute_bias_accumulation_contribution": float(np.mean(np.abs(bias))),
        "variance_share_signal_covariance_split": share_signal,
        "variance_share_bias_covariance_split": share_bias,
        "observed_positive_to_nonpositive_flips": int(observed_flips.sum()),
        "flips_prevented_without_bias": int((observed_flips & (no_bias > 0.0)).sum()),
        "flips_prevented_without_attenuation": int(
            (observed_flips & (no_attenuation > 0.0)).sum()
        ),
    }


def d_m2_condition(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    margins = np.asarray(
        [row["per_loop_mean_teacher_token_margin"] for row in rows], dtype=np.float64
    )
    if margins.ndim != 2 or margins.shape[1] != 4:
        raise RuntimeError(f"D-M2 requires four per-loop margins: {path}")
    cv = recurrence_cv(margins)
    winner = min(cv, key=lambda name: cv[name]["rmse"])
    return {
        "path": str(path),
        "rows": len(rows),
        "mean_margin_by_loop": margins.mean(axis=0).tolist(),
        "cross_validated_model_comparison": cv,
        "cv_winner": winner,
        "common_c": decomposition(margins, common_c=True),
        "row_specific_c": decomposition(margins, common_c=False),
    }


def infer_theta(aspect: float, cosine: float) -> float | None:
    if cosine <= 0.0 or not 0.0 < aspect:
        return None
    target = cosine**2
    lower = aspect**0.25 * (1.0 + 1e-9)
    upper = max(10.0, lower * 10.0)

    def overlap(theta: float) -> float:
        return (1.0 - aspect / theta**4) / (1.0 + aspect / theta**2)

    if target >= overlap(upper):
        return None
    for _ in range(200):
        middle = (lower + upper) / 2.0
        if overlap(middle) < target:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def required_n(hidden: int, theta: float, target_cosine: float) -> int:
    target = target_cosine**2
    q = (1.0 - target) / (theta**-4 + target * theta**-2)
    return int(math.ceil(hidden / max(q, 1e-12)))


def d_m4_seed(
    *,
    seed: int,
    alignments: Mapping[str, float],
    audit_counts: Sequence[int],
    training_rows: int,
    hidden_size: int,
) -> dict[str, Any]:
    output = []
    total = sum(audit_counts)
    for index, audit_count in enumerate(audit_counts):
        estimated_n = max(1, int(round(training_rows * audit_count / total)))
        cosine = float(alignments.get(f"cluster_{index}_mean_initialization", 0.0))
        aspect = hidden_size / estimated_n
        theta = infer_theta(aspect, cosine)
        row = {
            "cluster": index,
            "audit_rows": audit_count,
            "estimated_raw_training_rows": estimated_n,
            "aspect_d_over_n": aspect,
            "maximum_absolute_alignment_across_W_H_W_P_B_L": cosine,
            "inferred_spike_strength_theta": theta,
        }
        if theta is not None:
            row["minimum_rows_above_bbp_threshold"] = int(math.floor(hidden_size / theta**4) + 1)
            row["rows_for_target_alignment"] = {
                str(target): required_n(hidden_size, theta, target)
                for target in (0.25, 0.5, 0.75)
            }
        output.append(row)
    return {
        "seed": seed,
        "hidden_size": hidden_size,
        "raw_training_rows": training_rows,
        "clusters": output,
        "caution": (
            "Raw row counts ignore document correlation and repeat dose; inferred theta and "
            "required rows are planning estimates, not certified sample-complexity bounds."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization_states", type=Path, nargs=2, required=True)
    parser.add_argument("--stop_checkpoints", type=Path, nargs=2, required=True)
    parser.add_argument("--correction_initialization", type=Path, nargs=2, required=True)
    parser.add_argument("--correction_stop", type=Path, nargs=2, required=True)
    parser.add_argument("--margin_initialization", type=Path, nargs=2, required=True)
    parser.add_argument("--margin_stop", type=Path, nargs=2, required=True)
    parser.add_argument("--clusters", type=int, nargs=2, default=(2, 3))
    parser.add_argument("--training_rows", type=int, default=2920)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    d_m1 = []
    d_m4 = []
    for seed in (0, 1):
        result, alignments, counts = d_m1_seed(
            seed=seed,
            initialization_state=args.initialization_states[seed],
            stop_checkpoint=args.stop_checkpoints[seed],
            correction_initialization=args.correction_initialization[seed],
            correction_stop=args.correction_stop[seed],
            clusters=args.clusters[seed],
        )
        d_m1.append(result)
        d_m4.append(
            d_m4_seed(
                seed=seed,
                alignments=alignments,
                audit_counts=counts,
                training_rows=args.training_rows,
                hidden_size=896,
            )
        )
    d_m2 = []
    for seed in (0, 1):
        d_m2.append(
            {
                "seed": seed,
                "initialization": d_m2_condition(args.margin_initialization[seed]),
                "stop": d_m2_condition(args.margin_stop[seed]),
            }
        )
    payload = {
        "kind": "paper2_stage2bs_desk_math_wave_v1",
        "status": "complete_cpu_only",
        "d_m1_spectral_rmt": d_m1,
        "d_m2_margin_recursion": d_m2,
        "d_m3_jvp": {
            "status": "DEFERRED",
            "reason": "requires model/JVP execution; authorization says to defer when non-trivial",
        },
        "d_m4_bbp_feasibility": d_m4,
        "input_artifacts": {
            name: [
                {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in paths
            ]
            for name, paths in {
                "initialization_states": args.initialization_states,
                "stop_checkpoints": args.stop_checkpoints,
                "correction_initialization": args.correction_initialization,
                "correction_stop": args.correction_stop,
                "margin_initialization": args.margin_initialization,
                "margin_stop": args.margin_stop,
            }.items()
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
