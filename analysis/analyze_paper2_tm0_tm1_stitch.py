"""Fit and gate the ratified cross-fitted TM-1 teacher-to-student stitches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

from analysis.analyze_paper2_tm0_tm1_cka import cache_shards
from training.paper2_tm0 import (
    atomic_json,
    load_lock,
    read_jsonl,
    sha256_file,
    window_boundaries,
)


def full_cache(
    model_dir: Path, *, key: str, layer_indices: list[int] | None = None
) -> tuple[list[str], torch.Tensor]:
    item_ids: list[str] = []
    parts: list[torch.Tensor] = []
    for shard in cache_shards(model_dir):
        item_ids.extend(str(value) for value in shard["item_ids"])
        tensor = shard[key]
        if layer_indices is not None:
            tensor = tensor[:, layer_indices]
        parts.append(tensor.float())
    if len(item_ids) != len(set(item_ids)):
        raise RuntimeError(f"TM-1 cache has duplicate item ids: {model_dir}")
    return item_ids, torch.cat(parts, dim=0)


def mp_median(aspect: float) -> float:
    if not 0.0 < aspect < 1.0:
        raise ValueError("TM-1 MP shrinker expects 0 < d/n < 1")
    lower = (1.0 - math.sqrt(aspect)) ** 2
    upper = (1.0 + math.sqrt(aspect)) ** 2
    grid = np.linspace(lower + 1e-10, upper - 1e-10, 200_001)
    density = np.sqrt((upper - grid) * (grid - lower)) / (
        2.0 * math.pi * aspect * grid
    )
    cdf = np.cumsum((density[:-1] + density[1:]) * np.diff(grid) * 0.5)
    cdf /= cdf[-1]
    return float(np.interp(0.5, cdf, grid[1:]))


def rmt_whitener(samples: torch.Tensor) -> dict[str, torch.Tensor | float | int]:
    values = samples.double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(1e-12)
    aspect = values.shape[1] / values.shape[0]
    noise = float(torch.median(eigenvalues)) / mp_median(aspect)
    edge = noise * (1.0 + math.sqrt(aspect)) ** 2
    bulk = eigenvalues <= edge
    shrunk = eigenvalues.clone()
    if bool(bulk.any()):
        shrunk[bulk] = eigenvalues[bulk].mean()
    transform = eigenvectors @ torch.diag(shrunk.rsqrt()) @ eigenvectors.T
    return {
        "mean": mean.float(),
        "transform": transform.float(),
        "aspect": float(aspect),
        "noise_variance": float(noise),
        "mp_upper_edge": float(edge),
        "bulk_eigenvalues": int(bulk.sum()),
        "dimension": int(values.shape[1]),
        "samples": int(values.shape[0]),
    }


def deterministic_half(item_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:tm1-half:{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2


def _gcv_ridge(
    teacher: torch.Tensor,
    student: torch.Tensor,
    multipliers: list[float],
) -> dict[str, Any]:
    y = teacher.numpy().astype(np.float64, copy=False)
    x = student.numpy().astype(np.float64, copy=False)
    y_mean = y.mean(axis=0)
    x_mean = x.mean(axis=0)
    yc = y - y_mean
    xc = x - x_mean
    gram = yc.T @ yc
    cross = yc.T @ xc
    scale = float(np.trace(gram) / gram.shape[0])
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        gram, overwrite_a=True, check_finite=False, driver="evd"
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_cross = eigenvectors.T @ cross
    target_energy = float(np.square(xc).sum())
    rows = yc.shape[0]
    grid = []
    best = None
    for multiplier in multipliers:
        ridge = max(scale * float(multiplier), 1e-12)
        coefficients = projected_cross / (eigenvalues[:, None] + ridge)
        cross_term = float(np.sum(coefficients * projected_cross))
        quadratic = float(np.sum(eigenvalues[:, None] * np.square(coefficients)))
        residual = max(target_energy - 2.0 * cross_term + quadratic, 0.0)
        degrees = float(np.sum(eigenvalues / (eigenvalues + ridge)))
        gcv = residual / max((rows - degrees) ** 2, 1e-30)
        record = {
            "multiplier": float(multiplier),
            "ridge": float(ridge),
            "gcv": float(gcv),
            "residual_sum_squares": float(residual),
            "effective_degrees_of_freedom": degrees,
        }
        grid.append(record)
        if best is None or record["gcv"] < best["gcv"]:
            best = record
    assert best is not None
    ridge = best["ridge"]
    coefficients = projected_cross / (eigenvalues[:, None] + ridge)
    weight = eigenvectors @ coefficients
    return {
        "weight": torch.from_numpy(weight.astype(np.float32)),
        "teacher_mean": torch.from_numpy(y_mean.astype(np.float32)),
        "student_mean": torch.from_numpy(x_mean.astype(np.float32)),
        "selected": best,
        "grid": grid,
        "trace_scale": scale,
    }


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
) -> dict[str, float]:
    mse = float((prediction - target).square().mean())
    baseline_mse = float((baseline - target).square().mean())
    cosine = float(F.cosine_similarity(prediction, target, dim=-1, eps=1e-12).mean())
    return {
        "mse": mse,
        "mean_baseline_mse": baseline_mse,
        "relative_mse_reduction_vs_mean": 1.0 - mse / max(baseline_mse, 1e-30),
        "cosine": cosine,
    }


def _random_control(
    teacher_centered: torch.Tensor,
    target_centered: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randint(
        0,
        2,
        (teacher_centered.shape[1], target_centered.shape[1]),
        generator=generator,
        dtype=torch.int8,
    ).float()
    projection = projection.mul_(2).sub_(1).div_(math.sqrt(teacher_centered.shape[1]))
    value = teacher_centered @ projection
    target_norm = target_centered.norm(dim=-1, keepdim=True)
    return F.normalize(value, dim=-1, eps=1e-12) * target_norm


def fit_half(
    teacher: torch.Tensor,
    student: torch.Tensor,
    row_halves: torch.Tensor,
    *,
    heldout_half: int,
    multipliers: list[float],
    whitening: dict[str, Any],
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    train_rows = row_halves.ne(heldout_half)
    test_rows = row_halves.eq(heldout_half)
    y_train = teacher[train_rows].reshape(-1, teacher.shape[-1])
    x_train = student[train_rows].reshape(-1, student.shape[-1])
    y_test = teacher[test_rows].reshape(-1, teacher.shape[-1])
    x_test = student[test_rows].reshape(-1, student.shape[-1])
    fitted = _gcv_ridge(y_train, x_train, multipliers)
    prediction = (
        y_test - fitted["teacher_mean"]
    ) @ fitted["weight"] + fitted["student_mean"]
    baseline = fitted["student_mean"].expand_as(x_test)
    random_centered = _random_control(
        y_test - fitted["teacher_mean"],
        x_test - fitted["student_mean"],
        seed=random_seed,
    )
    random_prediction = random_centered + fitted["student_mean"]
    raw = _metrics(prediction, x_test, baseline)
    raw_random = _metrics(random_prediction, x_test, baseline)
    broad_mean = whitening["mean"]
    transform = whitening["transform"]
    whiten = lambda value: (value - broad_mean) @ transform
    white_target = whiten(x_test)
    white_prediction = whiten(prediction)
    white_baseline = whiten(baseline)
    white_random = whiten(random_prediction)
    whitened = _metrics(white_prediction, white_target, white_baseline)
    whitened_random = _metrics(white_random, white_target, white_baseline)
    metrics = {
        "heldout_half": heldout_half,
        "train_rows": int(train_rows.sum()),
        "heldout_rows": int(test_rows.sum()),
        "train_samples_with_two_views": int(y_train.shape[0]),
        "heldout_samples_with_two_views": int(y_test.shape[0]),
        "ridge": fitted["selected"],
        "ridge_grid": fitted["grid"],
        "raw": raw,
        "raw_random_control": raw_random,
        "whitened": whitened,
        "whitened_random_control": whitened_random,
        "primary_mse_gate_pass": whitened["relative_mse_reduction_vs_mean"] >= 0.20,
        "primary_cosine_gap": whitened["cosine"] - whitened_random["cosine"],
        "primary_cosine_gate_pass": (
            whitened["cosine"] - whitened_random["cosine"] >= 0.10
        ),
    }
    state = {
        "weight": fitted["weight"],
        "teacher_mean": fitted["teacher_mean"],
        "student_mean": fitted["student_mean"],
    }
    return metrics, state


def write_figure(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"teacher_7b": "#176B87", "teacher_14b": "#C54F3D"}
    markers = ("o", "s")
    for teacher, payload in summary["teachers"].items():
        layers = sorted(int(value) for value in payload["layers"])
        for half in (0, 1):
            rows = [payload["layers"][str(layer)]["halves"][half] for layer in layers]
            label = f"{teacher.replace('teacher_', '')}, half {half}"
            axes[0].plot(
                [layer + 1 for layer in layers],
                [row["whitened"]["relative_mse_reduction_vs_mean"] for row in rows],
                marker=markers[half],
                color=colors[teacher],
                alpha=0.75,
                label=label,
            )
            axes[1].plot(
                [layer + 1 for layer in layers],
                [row["primary_cosine_gap"] for row in rows],
                marker=markers[half],
                color=colors[teacher],
                alpha=0.75,
                label=label,
            )
    axes[0].axhline(0.20, color="#222222", linestyle="--", linewidth=1)
    axes[1].axhline(0.10, color="#222222", linestyle="--", linewidth=1)
    axes[0].set_title("Whitened reconstruction advantage")
    axes[1].set_title("Cosine advantage over random projection")
    axes[0].set_ylabel("Relative MSE reduction vs mean")
    axes[1].set_ylabel("Cosine gap")
    for axis in axes:
        axis.set_xlabel("Teacher layer")
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("TM-1 cross-fitted stitch gates")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"tm1_stitch_gates.{suffix}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cka_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = load_lock()
    panel = read_jsonl(args.panel)
    panel_ids = [str(row["item_id"]) for row in panel]
    student_ids, student = full_cache(args.cache_root / "student", key="h0")
    if student_ids != panel_ids:
        raise RuntimeError("TM-1 student cache order differs from the frozen panel")
    whitening = rmt_whitener(student.reshape(-1, student.shape[-1]))
    cka = json.loads(args.cka_summary.read_text(encoding="utf-8"))
    row_halves = torch.tensor(
        [deterministic_half(item_id, int(lock["tm1"]["split_seed"])) for item_id in panel_ids]
    )
    results = {}
    model_states: dict[str, Any] = {
        "kind": "paper2_tm0_crossfit_stitch_states_v1",
        "panel_sha256": sha256_file(args.panel),
        "whitening": {
            "mean": whitening["mean"],
            "transform": whitening["transform"],
        },
        "teachers": {},
    }
    prefit = {
        "kind": "paper2_tm0_tm1_stitch_prefit_v1",
        "panel_sha256": sha256_file(args.panel),
        "cka_summary_sha256": sha256_file(args.cka_summary),
        "split_seed": int(lock["tm1"]["split_seed"]),
        "view_contract": "last_and_mean_stacked_but_grouped_by_row_half",
        "ridge_selection": "train_only_exact_eigendecomposition_gcv",
        "ridge_multipliers": lock["tm1"]["ridge_multipliers"],
        "whitening": {
            key: value
            for key, value in whitening.items()
            if key not in {"mean", "transform"}
        },
        "random_control": "seeded_rademacher_projection_row_norm_matched",
        "teachers": {},
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    for teacher in ("teacher_7b", "teacher_14b"):
        curves = [
            cka["subsets"][subset]["teachers"][teacher]["arithmetic_mean_curve"]
            for subset in ("a", "b")
        ]
        selected = int(np.argmax(np.mean(np.asarray(curves), axis=0)))
        layers = int(cka["subsets"]["a"]["teachers"][teacher]["layers"])
        candidates = sorted(
            {max(0, min(layers - 1, selected + offset)) for offset in (-2, 0, 2)}
        )
        trajectory_boundaries = window_boundaries(
            selected, layers - 1, int(lock["tm2g"]["windows"])
        )
        fitted_layers = sorted(set(candidates + trajectory_boundaries))
        prefit["teachers"][teacher] = {
            "selected_layer_zero_based": selected,
            "selected_layer_one_based": selected + 1,
            "candidate_layers_zero_based": candidates,
            "candidate_layers_one_based": [value + 1 for value in candidates],
            "trajectory_boundaries_zero_based": trajectory_boundaries,
            "trajectory_boundaries_one_based": [
                value + 1 for value in trajectory_boundaries
            ],
            "fitted_layers_zero_based": fitted_layers,
            "fitted_layers_one_based": [value + 1 for value in fitted_layers],
            "selection_rule": "argmax_mean_of_disjoint_subset_mean_pool_curves",
        }
    atomic_json(args.output_dir / "tm1_stitch_prefit.json", prefit)
    for teacher_index, teacher in enumerate(("teacher_7b", "teacher_14b")):
        candidates = prefit["teachers"][teacher]["fitted_layers_zero_based"]
        teacher_ids, values = full_cache(
            args.cache_root / teacher, key="layers", layer_indices=candidates
        )
        if teacher_ids != panel_ids:
            raise RuntimeError(f"TM-1 {teacher} cache order differs from frozen panel")
        teacher_result = {"layers": {}}
        teacher_state = {
            "selected_layer_zero_based": prefit["teachers"][teacher][
                "selected_layer_zero_based"
            ],
            "layers": {},
        }
        for local_index, layer in enumerate(candidates):
            half_metrics = []
            half_states = []
            for heldout in (0, 1):
                metrics, state = fit_half(
                    values[:, local_index],
                    student,
                    row_halves,
                    heldout_half=heldout,
                    multipliers=[
                        float(value) for value in lock["tm1"]["ridge_multipliers"]
                    ],
                    whitening=whitening,
                    random_seed=(
                        int(lock["tm1"]["split_seed"])
                        + teacher_index * 100
                        + layer * 2
                        + heldout
                    ),
                )
                half_metrics.append(metrics)
                half_states.append(state)
            gate = all(
                row["primary_mse_gate_pass"] and row["primary_cosine_gate_pass"]
                for row in half_metrics
            )
            teacher_result["layers"][str(layer)] = {
                "layer_one_based": layer + 1,
                "halves": half_metrics,
                "gate_pass_both_halves": gate,
            }
            teacher_state["layers"][str(layer)] = half_states
            print(f"tm1_stitch teacher={teacher} layer={layer + 1} gate={gate}", flush=True)
        selected = str(prefit["teachers"][teacher]["selected_layer_zero_based"])
        teacher_result["selected_layer_gate_pass"] = teacher_result["layers"][
            selected
        ]["gate_pass_both_halves"]
        results[teacher] = teacher_result
        model_states["teachers"][teacher] = teacher_state
    summary = {
        "kind": "paper2_tm0_tm1_stitch_summary_v1",
        "teachers": results,
        "gate_key": (
            "G-TM1-PASS"
            if any(value["selected_layer_gate_pass"] for value in results.values())
            else "STITCH-DEAD"
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "injection_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    torch.save(model_states, args.output_dir / "tm1_stitch_states.pt")
    summary["state_sha256"] = sha256_file(args.output_dir / "tm1_stitch_states.pt")
    atomic_json(args.output_dir / "tm1_stitch_summary.json", summary)
    write_figure(summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
