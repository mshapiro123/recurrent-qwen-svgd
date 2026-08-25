"""Run the ratified TM-2g-J jet-geometric read without model stitches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.analyze_paper2_tm0_tm1_stitch import (
    deterministic_half,
    full_cache,
    rmt_whitener,
)
from analysis.analyze_paper2_tm0_tm2 import (
    bootstrap_mean_difference,
    score_map,
    strata_masks,
    target_entropy_audit,
)
from models.sidecar_v2 import fast_wht
from training.paper2_tm0 import atomic_json, load_lock, read_jsonl, sha256_file


TEACHERS = ("teacher_7b", "teacher_14b")
VIEWS = ("last_active_token", "active_token_mean")
WHT_BLOCK = 128
SKETCH_BLOCKS = 7
SKETCH_WIDTH = WHT_BLOCK * SKETCH_BLOCKS
PLANE_PROBES = 256
SKETCH_SEED = 202608257
PLANE_SEED = 202608258
NOISE_SEED = 202608259
EPSILON = 1e-12
PIVOT_NULL_QUANTILE = 0.95
PIVOT_PRECEDING_LAYERS = 2
PIVOT_FOLLOWING_LAYERS = 3
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 202608260
DISCRIMINATIVE_SEED = 202608261


def sparse_wht_sketch(
    values: torch.Tensor,
    *,
    seed: int,
    output_blocks: int = SKETCH_BLOCKS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Frozen signed block-hash JL map after orthonormal WHT-128 blocks."""

    width = values.shape[-1]
    if width % WHT_BLOCK:
        raise ValueError("TM-2g-J hidden width must be divisible by 128")
    input_blocks = width // WHT_BLOCK
    generator = torch.Generator().manual_seed(seed)
    signs = (
        torch.randint(0, 2, (input_blocks, WHT_BLOCK), generator=generator)
        .mul(2)
        .sub(1)
        .to(values.dtype)
    )
    order = torch.randperm(input_blocks, generator=generator)
    assignment = torch.empty(input_blocks, dtype=torch.long)
    assignment[order] = torch.arange(input_blocks).remainder(output_blocks)
    transformed = fast_wht(values.reshape(-1, WHT_BLOCK)).reshape(
        *values.shape[:-1], input_blocks, WHT_BLOCK
    )
    transformed = transformed * signs
    output = torch.zeros(
        *values.shape[:-1], output_blocks, WHT_BLOCK, dtype=values.dtype
    )
    counts = torch.bincount(assignment, minlength=output_blocks).clamp_min(1)
    for source in range(input_blocks):
        output[..., assignment[source], :] += transformed[..., source, :]
    output /= counts.sqrt().to(output.dtype).view(
        *((1,) * (output.ndim - 2)), output_blocks, 1
    )
    receipt = {
        "kind": "paper2_tm0_frozen_sparse_wht_jl_v1",
        "seed": seed,
        "input_width": width,
        "input_blocks": input_blocks,
        "output_blocks": output_blocks,
        "output_width": output_blocks * WHT_BLOCK,
        "block_size": WHT_BLOCK,
        "block_counts": counts.tolist(),
        "construction": "WHT128_then_seeded_rademacher_sign_then_balanced_block_hash",
    }
    return output.reshape(*values.shape[:-1], -1), receipt


def jet_invariants(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    epsilon: float = EPSILON,
) -> dict[str, torch.Tensor]:
    """Translation-invariant 2-jet Gram statistics."""

    speed2 = velocity.square().sum(dim=-1)
    accel2 = acceleration.square().sum(dim=-1)
    dot = (velocity * acceleration).sum(dim=-1)
    wedge2 = (speed2 * accel2 - dot.square()).clamp_min(0.0)
    wedge = wedge2.sqrt()
    trace = (speed2 + accel2).clamp_min(epsilon)
    discriminant = ((speed2 - accel2).square() + 4.0 * dot.square()).sqrt()
    eigen_max = ((trace + discriminant) * 0.5).clamp_min(epsilon)
    eigen_min = ((trace - discriminant) * 0.5).clamp_min(0.0)
    return {
        "speed": speed2.sqrt(),
        "acceleration_norm": accel2.sqrt(),
        "velocity_acceleration_dot": dot,
        "normalized_dot": dot / (speed2.sqrt() * accel2.sqrt()).clamp_min(epsilon),
        "wedge_norm": wedge,
        "curvature": wedge / (speed2.sqrt().pow(3) + epsilon),
        "gram_eigenvalue_ratio": eigen_min / eigen_max,
    }


def plane_probe_features(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Unbiased Gaussian JL coordinates for a unit simple bivector.

    For B = v ^ a and phi(B) = (v.p)(a.q) - (v.q)(a.p), independent
    standard-Gaussian p and q satisfy E[phi(B) phi(C)] = 2 <B, C>.
    Scaling by sqrt(2m) therefore makes the sketched dot product unbiased.
    """

    velocity_unit = F.normalize(velocity, dim=-1, eps=EPSILON)
    radial = (acceleration * velocity_unit).sum(dim=-1, keepdim=True)
    perpendicular = F.normalize(
        acceleration - radial * velocity_unit, dim=-1, eps=EPSILON
    )
    features = (
        (velocity_unit @ p.T) * (perpendicular @ q.T)
        - (velocity_unit @ q.T) * (perpendicular @ p.T)
    )
    return features / math.sqrt(2.0 * p.shape[0])


def registered_layer(cka: dict[str, Any], teacher: str) -> int:
    curves = [
        cka["subsets"][subset]["teachers"][teacher]["arithmetic_mean_curve"]
        for subset in ("a", "b")
    ]
    return int(np.argmax(np.mean(np.asarray(curves), axis=0)))


def profile_features(profiles: dict[str, torch.Tensor]) -> torch.Tensor:
    def safe_log(value: torch.Tensor) -> torch.Tensor:
        return value.clamp_min(EPSILON).log()

    values = [
        safe_log(profiles["speed"]),
        safe_log(profiles["acceleration_norm"]),
        profiles["normalized_dot"],
        safe_log(profiles["curvature"]),
        profiles["gram_eigenvalue_ratio"],
        profiles["plane_consistency"],
    ]
    return torch.cat(values, dim=1)


def fit_half_read(
    positive: torch.Tensor,
    positive_halves: torch.Tensor,
    negative: torch.Tensor,
    negative_halves: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    entropy = target_entropy_audit(len(positive), len(negative))
    if min(len(positive), len(negative)) < 4:
        return {"status": "UNDERPOWERED", "target_entropy_audit": entropy}
    rows = []
    for heldout in (0, 1):
        train_x = torch.cat(
            (positive[positive_halves.ne(heldout)], negative[negative_halves.ne(heldout)])
        ).numpy()
        train_y = np.concatenate(
            (
                np.ones(int(positive_halves.ne(heldout).sum())),
                np.zeros(int(negative_halves.ne(heldout).sum())),
            )
        )
        test_x = torch.cat(
            (positive[positive_halves.eq(heldout)], negative[negative_halves.eq(heldout)])
        ).numpy()
        test_y = np.concatenate(
            (
                np.ones(int(positive_halves.eq(heldout).sum())),
                np.zeros(int(negative_halves.eq(heldout).sum())),
            )
        )
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            rows.append({"heldout_half": heldout, "status": "UNDERPOWERED"})
            continue
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                solver="liblinear",
                random_state=seed + heldout,
            ),
        )
        estimator.fit(train_x, train_y)
        probability = estimator.predict_proba(test_x)[:, 1]
        rows.append(
            {
                "heldout_half": heldout,
                "status": "MEASURED",
                "train_rows": int(len(train_y)),
                "heldout_rows": int(len(test_y)),
                "roc_auc": float(roc_auc_score(test_y, probability)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(test_y, probability >= 0.5)
                ),
            }
        )
    return {
        "status": (
            "MEASURED"
            if all(row["status"] == "MEASURED" for row in rows)
            else "UNDERPOWERED"
        ),
        "target_entropy_audit": entropy,
        "halves": rows,
    }


def pivot_signatures(
    curvature: torch.Tensor,
    speed: torch.Tensor,
    noise_curvature: torch.Tensor,
) -> dict[str, torch.Tensor]:
    band = torch.quantile(noise_curvature, PIVOT_NULL_QUANTILE, dim=0)
    rows, layers = curvature.shape
    pivot = torch.zeros((rows, layers), dtype=torch.bool)
    for layer in range(1, layers - 1):
        if layer < PIVOT_PRECEDING_LAYERS:
            continue
        after = min(PIVOT_FOLLOWING_LAYERS, layers - layer - 1)
        if after < 1:
            continue
        local_peak = (
            curvature[:, layer] > curvature[:, layer - 1]
        ) & (curvature[:, layer] >= curvature[:, layer + 1])
        above_null = curvature[:, layer] > band[layer]
        preceding_low = (
            curvature[:, layer - PIVOT_PRECEDING_LAYERS : layer]
            <= band[layer - PIVOT_PRECEDING_LAYERS : layer]
        ).all(dim=1)
        following = speed[:, layer : layer + after + 1]
        converging = (following[:, 1:] <= following[:, :-1]).all(dim=1)
        pivot[:, layer] = local_peak & above_null & preceding_low & converging
    has_pivot = pivot.any(dim=1)
    first = torch.where(
        has_pivot,
        pivot.float().argmax(dim=1).float() / max(layers - 1, 1),
        torch.full((rows,), float("nan")),
    )
    return {
        "mask": pivot,
        "has_pivot": has_pivot.float(),
        "count": pivot.sum(dim=1).float(),
        "first_depth_quantile": first,
        "null_band": band,
    }


def generate_profiles(
    layers: torch.Tensor,
    *,
    selected_layer: int,
    teacher_index: int,
    view_index: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    selected = layers[:, selected_layer]
    reduced_selected, reduction_receipt = sparse_wht_sketch(
        selected.reshape(-1, selected.shape[-1]),
        seed=SKETCH_SEED + teacher_index,
    )
    whitener = rmt_whitener(reduced_selected)
    mean = whitener["mean"]
    transform = whitener["transform"]
    generator = torch.Generator().manual_seed(PLANE_SEED)
    p = torch.randn((PLANE_PROBES, SKETCH_WIDTH), generator=generator)
    q = torch.randn((PLANE_PROBES, SKETCH_WIDTH), generator=generator)
    noise_generator = torch.Generator().manual_seed(
        NOISE_SEED + teacher_index * 10 + view_index
    )

    def whiten(layer_values: torch.Tensor) -> torch.Tensor:
        reduced, _ = sparse_wht_sketch(
            layer_values,
            seed=SKETCH_SEED + teacher_index,
        )
        return (reduced - mean) @ transform

    previous2 = whiten(layers[:, selected_layer, view_index])
    previous1 = whiten(layers[:, selected_layer + 1, view_index])
    previous_velocity = previous1 - previous2
    previous_plane = None
    previous_state_plane = None
    previous_noise_velocity = None
    previous_noise_plane = None
    scalar_parts: dict[str, list[torch.Tensor]] = {}
    plane_parts = []
    state_plane_parts = []
    noise_scalar_parts: dict[str, list[torch.Tensor]] = {}
    noise_plane_parts = []
    for layer in range(selected_layer + 2, layers.shape[1]):
        current = whiten(layers[:, layer, view_index])
        velocity = current - previous1
        acceleration = velocity - previous_velocity
        invariants = jet_invariants(velocity, acceleration)
        for key, value in invariants.items():
            scalar_parts.setdefault(key, []).append(value)
        plane = plane_probe_features(velocity, acceleration, p, q)
        if previous_plane is not None:
            plane_parts.append((previous_plane * plane).sum(dim=-1))
        state_plane = plane_probe_features(previous1, velocity, p, q)
        if previous_state_plane is not None:
            state_plane_parts.append((previous_state_plane * state_plane).sum(dim=-1))

        random_direction = F.normalize(
            torch.randn(velocity.shape, generator=noise_generator),
            dim=-1,
            eps=EPSILON,
        )
        noise_velocity = random_direction * invariants["speed"].unsqueeze(1)
        if previous_noise_velocity is None:
            noise_acceleration = torch.zeros_like(noise_velocity)
        else:
            noise_acceleration = noise_velocity - previous_noise_velocity
        noise_invariants = jet_invariants(noise_velocity, noise_acceleration)
        for key, value in noise_invariants.items():
            noise_scalar_parts.setdefault(key, []).append(value)
        noise_plane = plane_probe_features(noise_velocity, noise_acceleration, p, q)
        if previous_noise_plane is not None:
            noise_plane_parts.append((previous_noise_plane * noise_plane).sum(dim=-1))

        previous2, previous1 = previous1, current
        previous_velocity = velocity
        previous_plane = plane
        previous_state_plane = state_plane
        previous_noise_velocity = noise_velocity
        previous_noise_plane = noise_plane

    profiles = {key: torch.stack(values, dim=1) for key, values in scalar_parts.items()}
    noise_profiles = {
        key: torch.stack(values, dim=1) for key, values in noise_scalar_parts.items()
    }
    profiles["plane_consistency"] = torch.stack(plane_parts, dim=1)
    profiles["state_velocity_plane_consistency"] = torch.stack(
        state_plane_parts, dim=1
    )
    noise_profiles["plane_consistency"] = torch.stack(noise_plane_parts, dim=1)
    pivot = pivot_signatures(
        profiles["curvature"], profiles["speed"], noise_profiles["curvature"]
    )
    noise_pivot = pivot_signatures(
        noise_profiles["curvature"],
        noise_profiles["speed"],
        noise_profiles["curvature"],
    )
    profiles["pivot"] = pivot["has_pivot"]
    profiles["pivot_count"] = pivot["count"]
    profiles["pivot_first_depth_quantile"] = pivot["first_depth_quantile"]
    noise_profiles["pivot"] = noise_pivot["has_pivot"]
    receipt = {
        "selected_layer_zero_based": selected_layer,
        "analyzed_layer_zero_based": list(
            range(selected_layer + 2, layers.shape[1])
        ),
        "reduction": reduction_receipt,
        "whitening": {
            key: value
            for key, value in whitener.items()
            if key not in {"mean", "transform"}
        },
        "plane_probe_seed": PLANE_SEED,
        "plane_probes": PLANE_PROBES,
        "noise_seed": NOISE_SEED + teacher_index * 10 + view_index,
        "epsilon": EPSILON,
        "pivot_null_quantile": PIVOT_NULL_QUANTILE,
        "pivot_preceding_layers": PIVOT_PRECEDING_LAYERS,
        "pivot_following_layers": PIVOT_FOLLOWING_LAYERS,
    }
    profiles["_noise"] = noise_profiles  # type: ignore[assignment]
    profiles["_pivot_null_band"] = pivot["null_band"]  # type: ignore[assignment]
    return profiles, receipt


def analyze_cells(
    profiles: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    battery_masks: dict[str, torch.Tensor],
    halves: torch.Tensor,
    *,
    teacher_index: int,
    view_index: int,
) -> dict[str, Any]:
    noise = profiles["_noise"]
    features = profile_features(profiles)
    result = {}
    for battery_index, (battery, battery_mask) in enumerate(battery_masks.items()):
        dnone = battery_mask & masks["D_none"]
        cells = {}
        for stratum_index, stratum in enumerate(("D_7>0.5", "D_14>0.5", "D_14>7")):
            positive = battery_mask & masks[stratum]
            if int(positive.sum()) < 2 or int(dnone.sum()) < 2:
                cells[stratum] = {
                    "status": "UNDERPOWERED",
                    "positive_rows": int(positive.sum()),
                    "dnone_rows": int(dnone.sum()),
                }
                continue
            seed = (
                BOOTSTRAP_SEED
                + teacher_index * 1000
                + view_index * 100
                + battery_index * 10
                + stratum_index
            )
            success_consistency = profiles["plane_consistency"][positive].mean(dim=1).numpy()
            dnone_consistency = profiles["plane_consistency"][dnone].mean(dim=1).numpy()
            noise_consistency = noise["plane_consistency"][positive].mean(dim=1).numpy()
            success_pivot = profiles["pivot"][positive].numpy()
            dnone_pivot = profiles["pivot"][dnone].numpy()
            noise_pivot = noise["pivot"][positive].numpy()
            consistency_vs_dnone = bootstrap_mean_difference(
                success_consistency, dnone_consistency, draws=BOOTSTRAP_DRAWS, seed=seed
            )
            consistency_vs_noise = bootstrap_mean_difference(
                success_consistency, noise_consistency, draws=BOOTSTRAP_DRAWS, seed=seed + 1
            )
            pivot_vs_dnone = bootstrap_mean_difference(
                success_pivot, dnone_pivot, draws=BOOTSTRAP_DRAWS, seed=seed + 2
            )
            pivot_vs_noise = bootstrap_mean_difference(
                success_pivot, noise_pivot, draws=BOOTSTRAP_DRAWS, seed=seed + 3
            )
            consistency_pass = (
                consistency_vs_dnone["ci95_low"] > 0.0
                and consistency_vs_noise["ci95_low"] > 0.0
            )
            pivot_pass = (
                pivot_vs_dnone["ci95_low"] > 0.0
                and pivot_vs_noise["ci95_low"] > 0.0
            )
            final_quarter = max(1, profiles["normalized_dot"].shape[1] // 4)
            cells[stratum] = {
                "status": "MEASURED",
                "positive_rows": int(positive.sum()),
                "dnone_rows": int(dnone.sum()),
                "plane_consistency": {
                    "success_mean": float(success_consistency.mean()),
                    "dnone_mean": float(dnone_consistency.mean()),
                    "smooth_noise_mean": float(noise_consistency.mean()),
                    "success_minus_dnone": consistency_vs_dnone,
                    "success_minus_smooth_noise": consistency_vs_noise,
                },
                "pivot_signature": {
                    "success_rate": float(success_pivot.mean()),
                    "dnone_rate": float(dnone_pivot.mean()),
                    "smooth_noise_rate": float(noise_pivot.mean()),
                    "success_minus_dnone": pivot_vs_dnone,
                    "success_minus_smooth_noise": pivot_vs_noise,
                },
                "late_normalized_velocity_acceleration_dot": {
                    "success_mean": float(
                        profiles["normalized_dot"][positive, -final_quarter:].mean()
                    ),
                    "dnone_mean": float(
                        profiles["normalized_dot"][dnone, -final_quarter:].mean()
                    ),
                },
                "claim_pass": consistency_pass or pivot_pass,
                "claim_channel": (
                    "both"
                    if consistency_pass and pivot_pass
                    else "plane_consistency"
                    if consistency_pass
                    else "pivot_signature"
                    if pivot_pass
                    else "none"
                ),
                "discriminative": fit_half_read(
                    features[positive],
                    halves[positive],
                    features[dnone],
                    halves[dnone],
                    seed=DISCRIMINATIVE_SEED + seed,
                ),
            }
        result[battery] = cells
    return result


def profile_means(
    profiles: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    battery_masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    output = {}
    for battery, battery_mask in battery_masks.items():
        output[battery] = {}
        for stratum, stratum_mask in masks.items():
            selected = battery_mask & stratum_mask
            if not bool(selected.any()):
                continue
            output[battery][stratum] = {
                "rows": int(selected.sum()),
                "speed_mean_by_layer": profiles["speed"][selected].mean(dim=0).tolist(),
                "curvature_mean_by_layer": profiles["curvature"][selected].mean(dim=0).tolist(),
                "gram_eigenvalue_ratio_mean_by_layer": profiles[
                    "gram_eigenvalue_ratio"
                ][selected].mean(dim=0).tolist(),
                "normalized_dot_mean_by_layer": profiles["normalized_dot"][selected]
                .mean(dim=0)
                .tolist(),
                "plane_consistency_mean_by_layer": profiles["plane_consistency"][selected]
                .mean(dim=0)
                .tolist(),
                "pivot_rate": float(profiles["pivot"][selected].mean()),
            }
    return output


def interpolate_profile(values: list[float], points: int = 32) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, np.asarray(values, dtype=np.float64))


def cross_scale_shapes(summary: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for view in VIEWS:
        left = summary["teachers"]["teacher_7b"]["views"][view]["profile_means"]
        right = summary["teachers"]["teacher_14b"]["views"][view]["profile_means"]
        cells = {}
        for battery in sorted(set(left) & set(right)):
            for stratum in sorted(set(left[battery]) & set(right[battery])):
                cell = {}
                for metric in (
                    "curvature_mean_by_layer",
                    "gram_eigenvalue_ratio_mean_by_layer",
                    "normalized_dot_mean_by_layer",
                ):
                    a = interpolate_profile(left[battery][stratum][metric])
                    b = interpolate_profile(right[battery][stratum][metric])
                    cell[metric] = {
                        "pearson": float(np.corrcoef(a, b)[0, 1]),
                        "cosine": float(
                            np.dot(a, b)
                            / max(np.linalg.norm(a) * np.linalg.norm(b), EPSILON)
                        ),
                    }
                cells[f"{battery}:{stratum}"] = cell
        output[view] = cells
    return output


def write_figures(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "D_7>0.5": "#176B87",
        "D_14>0.5": "#C54F3D",
        "D_none": "#777777",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, teacher in enumerate(TEACHERS):
        profiles = summary["teachers"][teacher]["views"]["active_token_mean"][
            "profile_means"
        ].get("gsm8k", {})
        for column, metric in enumerate(
            ("curvature_mean_by_layer", "normalized_dot_mean_by_layer")
        ):
            axis = axes[row, column]
            for stratum, color in colors.items():
                if stratum not in profiles:
                    continue
                axis.plot(profiles[stratum][metric], color=color, label=stratum)
            axis.axhline(0.0, color="#222222", linewidth=0.8, linestyle="--")
            axis.set_title(
                f"{teacher.replace('teacher_', '').upper()} - "
                + ("curvature" if column == 0 else "velocity-acceleration cosine")
            )
            axis.set_xlabel("Layer offset after j*+2")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("TM-2g-J: reasoning trajectories are read as per-layer 2-jets")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"tm2g_jet_profiles.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, teacher in zip(axes, TEACHERS):
        cells = summary["teachers"][teacher]["views"]["active_token_mean"][
            "cells"
        ].get("gsm8k", {})
        labels = []
        estimates = []
        lows = []
        highs = []
        colors_used = []
        for stratum in ("D_7>0.5", "D_14>0.5"):
            cell = cells.get(stratum, {})
            if cell.get("status") != "MEASURED":
                continue
            for channel, key in (
                ("plane", "plane_consistency"),
                ("pivot", "pivot_signature"),
            ):
                estimate = cell[key]["success_minus_dnone"]["estimate"]
                low = cell[key]["success_minus_dnone"]["ci95_low"]
                high = cell[key]["success_minus_dnone"]["ci95_high"]
                labels.append(f"{stratum.replace('D_', '')}\n{channel}")
                estimates.append(estimate)
                lows.append(estimate - low)
                highs.append(high - estimate)
                colors_used.append(colors[stratum])
        x = np.arange(len(labels))
        axis.bar(x, estimates, color=colors_used, alpha=0.85)
        axis.errorbar(x, estimates, yerr=[lows, highs], fmt="none", color="#222222", capsize=3)
        axis.axhline(0.0, color="#222222", linewidth=0.8, linestyle="--")
        axis.set_xticks(x, labels)
        axis.set_title(teacher.replace("teacher_", "Qwen2.5 ").upper())
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Success stratum minus D_none (95% bootstrap CI)")
    figure.suptitle("TM-2g-J decisive GSM8K contrasts")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"tm2g_jet_decisive_contrasts.{suffix}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cka_summary", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--teacher_7b_scores", type=Path, required=True)
    parser.add_argument("--teacher_14b_scores", type=Path, required=True)
    parser.add_argument("--failed_loop_archive_receipt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = load_lock()
    registered = lock["tm2g_jet_r4"]
    observed = {
        "wht_block": WHT_BLOCK,
        "output_blocks": SKETCH_BLOCKS,
        "sketch_seed": SKETCH_SEED,
        "plane_probes": PLANE_PROBES,
        "plane_probe_seed": PLANE_SEED,
        "smooth_noise_seed": NOISE_SEED,
        "epsilon": EPSILON,
        "pivot_null_quantile": PIVOT_NULL_QUANTILE,
        "pivot_preceding_low_layers": PIVOT_PRECEDING_LAYERS,
        "pivot_following_nonincreasing_speed_layers": PIVOT_FOLLOWING_LAYERS,
    }
    mismatches = {
        key: {"registered": registered[key], "observed": value}
        for key, value in observed.items()
        if registered[key] != value
    }
    if mismatches:
        raise RuntimeError(f"TM-2g-J machine-lock mismatch: {mismatches}")
    panel = read_jsonl(args.panel)
    item_ids = [str(row["item_id"]) for row in panel]
    batteries = [str(row["battery"]) for row in panel]
    battery_masks = {
        battery: torch.tensor([value == battery for value in batteries])
        for battery in sorted(set(batteries))
    }
    battery_masks["pooled"] = torch.ones(len(panel), dtype=torch.bool)
    masks = strata_masks(
        item_ids,
        score_map(args.base_scores),
        score_map(args.teacher_7b_scores),
        score_map(args.teacher_14b_scores),
    )
    cka = json.loads(args.cka_summary.read_text(encoding="utf-8"))
    halves = torch.tensor(
        [deterministic_half(item_id, 202608251) for item_id in item_ids]
    )
    amendment_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "STRATEGY_TM0_R4_JET_AMENDMENT_20260825.md"
    )
    pre_run = {
        "kind": "paper2_tm0_tm2g_jet_pre_run_v1",
        "written_before_any_jet_statistic": True,
        "authority": {
            "drive_id": "1GDZE-YnYU-RNHoBcMWBKcuWXjW3pxyaH",
            "path": str(amendment_path),
            "bytes": amendment_path.stat().st_size,
            "sha256": sha256_file(amendment_path),
        },
        "panel_sha256": sha256_file(args.panel),
        "machine_lock_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "training" / "paper2_tm0_lock.json"
        ),
        "primary_object": "velocity_wedge_acceleration",
        "secondary_object": "mean_relative_state_wedge_velocity",
        "finite_differences": "causal_v_l=z_l-z_l-1_a_l=v_l-v_l-1",
        "per_model_frame": "frozen_sparse_WHT_JL_then_RMT_whitening_no_cross_model_stitch",
        "sparse_reduction": {
            "seed_base": SKETCH_SEED,
            "block_size": WHT_BLOCK,
            "output_blocks": SKETCH_BLOCKS,
            "output_width": SKETCH_WIDTH,
        },
        "plane_probes": {
            "count": PLANE_PROBES,
            "seed": PLANE_SEED,
            "construction": "iid_gaussian_row_normalized_shared_across_teachers",
        },
        "epsilon": EPSILON,
        "smooth_noise_null": {
            "seed_base": NOISE_SEED,
            "construction": "independent_gaussian_random_walk_steps_row_layer_norm_matched",
        },
        "pivot_signature": {
            "null_quantile": PIVOT_NULL_QUANTILE,
            "preceding_low_layers": PIVOT_PRECEDING_LAYERS,
            "following_nonincreasing_speed_layers": PIVOT_FOLLOWING_LAYERS,
        },
        "pooling": {
            "primary": "active_token_mean",
            "secondary": "last_active_token",
            "never_independent": True,
        },
        "decision_rule": (
            "ROTATION-CONSISTENT iff either plane-consistency or pivot-signature "
            "success-minus-D_none and success-minus-smooth-noise bootstrap lower bounds "
            "are positive for D_7>0.5 and D_14>0.5 in GSM8K, active-token mean, both teachers"
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "tm2g_jet_pre_run_receipt.json", pre_run)

    summary: dict[str, Any] = {
        "kind": "paper2_tm0_tm2g_jet_summary_v1",
        "panel_sha256": sha256_file(args.panel),
        "authority_sha256": sha256_file(amendment_path),
        "stratum_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "teachers": {},
        "failed_loop_postdiction": json.loads(
            args.failed_loop_archive_receipt.read_text(encoding="utf-8")
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "injection_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    for teacher_index, teacher in enumerate(TEACHERS):
        cache_ids, layers = full_cache(args.cache_root / teacher, key="layers")
        if cache_ids != item_ids:
            raise RuntimeError(f"TM-2g-J cache order mismatch: {teacher}")
        selected = registered_layer(cka, teacher)
        teacher_result = {"selected_layer_zero_based": selected, "views": {}}
        for view_index, view in enumerate(VIEWS):
            profiles, receipt = generate_profiles(
                layers,
                selected_layer=selected,
                teacher_index=teacher_index,
                view_index=view_index,
            )
            teacher_result["views"][view] = {
                "frame_receipt": receipt,
                "cells": analyze_cells(
                    profiles,
                    masks,
                    battery_masks,
                    halves,
                    teacher_index=teacher_index,
                    view_index=view_index,
                ),
                "profile_means": profile_means(profiles, masks, battery_masks),
                "state_velocity_secondary": {
                    "mean_plane_consistency": float(
                        profiles["state_velocity_plane_consistency"].mean()
                    )
                },
            }
        summary["teachers"][teacher] = teacher_result
        del layers

    summary["cross_scale_scalar_profile_shapes"] = cross_scale_shapes(summary)
    required = []
    noise_only = []
    for teacher in TEACHERS:
        cells = summary["teachers"][teacher]["views"]["active_token_mean"]["cells"].get(
            "gsm8k", {}
        )
        for stratum in ("D_7>0.5", "D_14>0.5"):
            cell = cells.get(stratum, {})
            if cell.get("status") == "MEASURED":
                required.append(bool(cell.get("claim_pass")))
                noise_only.append(
                    cell["plane_consistency"]["success_minus_smooth_noise"]["ci95_low"] > 0
                    or cell["pivot_signature"]["success_minus_smooth_noise"]["ci95_low"] > 0
                )
    if required and all(required):
        key = "ROTATION-CONSISTENT"
    elif any(required):
        key = "STRATUM-SPLIT"
    elif noise_only and all(noise_only):
        key = "ROTATION-GENERIC"
    else:
        key = "ROTATION-ABSENT"
    summary["decision_key"] = key
    atomic_json(args.output_dir / "tm2g_jet_summary.json", summary)
    write_figures(summary, args.output_dir)
    print(json.dumps({"tm2g_jet": key}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
