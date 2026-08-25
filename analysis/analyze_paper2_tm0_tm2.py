"""Run the registered TM-2 and TM-2g desk analyses after a TM-1 pass."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from analysis.analyze_paper2_tm0_tm1_cka import cache_shards
from analysis.analyze_paper2_tm0_tm1_stitch import deterministic_half
from training.paper2_tm0 import (
    atomic_json,
    geometric_features,
    load_lock,
    random_orthoproject,
    rademacher_sketch,
    read_jsonl,
    sha256_file,
)


VIEWS = ("last_active_token", "active_token_mean")
TEACHERS = ("teacher_7b", "teacher_14b")


def target_entropy_audit(positive_rows: int, negative_rows: int) -> dict[str, float | int | bool]:
    total = positive_rows + negative_rows
    fraction = positive_rows / total if total else 0.0
    entropy = 0.0
    for probability in (fraction, 1.0 - fraction):
        if probability > 0.0:
            entropy -= probability * np.log2(probability)
    return {
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "positive_fraction": fraction,
        "binary_entropy_bits": float(entropy),
        "both_labels_present": positive_rows > 0 and negative_rows > 0,
    }


def full_cache(model_dir: Path, key: str) -> tuple[list[str], torch.Tensor]:
    item_ids: list[str] = []
    tensors: list[torch.Tensor] = []
    for shard in cache_shards(model_dir):
        item_ids.extend(str(value) for value in shard["item_ids"])
        tensors.append(shard[key].float())
    if len(item_ids) != len(set(item_ids)):
        raise RuntimeError(f"duplicate cache ids in {model_dir}")
    return item_ids, torch.cat(tensors)


def score_map(path: Path) -> dict[str, bool]:
    rows = read_jsonl(path)
    result = {str(row["item_id"]): bool(row["correct"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate score item ids: {path}")
    return result


def crossfit_stitched(
    values: torch.Tensor,
    item_ids: list[str],
    states: dict[str, Any],
    layer: int,
    split_seed: int,
) -> torch.Tensor:
    output = torch.empty(
        values.shape[0], values.shape[2], states["layers"][str(layer)][0]["student_mean"].numel()
    )
    halves = torch.tensor([deterministic_half(item_id, split_seed) for item_id in item_ids])
    for heldout in (0, 1):
        mask = halves.eq(heldout)
        fit = states["layers"][str(layer)][heldout]
        output[mask] = (
            values[mask, layer] - fit["teacher_mean"]
        ) @ fit["weight"] + fit["student_mean"]
    return output


def remove_common_mode(values: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    unit = F.normalize(values, dim=-1, eps=1e-12)
    direction = F.normalize(unit.mean(dim=0), dim=0, eps=1e-12)
    projected = values - (values @ direction).unsqueeze(1) * direction
    return projected, {
        "mean_unit_projection": float((unit @ direction).mean()),
        "unit_projection_energy_fraction": float((unit @ direction).square().mean()),
    }


def svd_receipt(values: torch.Tensor, ranks: list[int]) -> tuple[dict[str, Any], torch.Tensor]:
    if values.shape[0] < 2:
        return {"rows": int(values.shape[0]), "underpowered": True}, torch.empty(0, values.shape[1])
    centered = values - values.mean(dim=0)
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    energy = singular.square()
    total = float(energy.sum())
    receipt = {
        "rows": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "variance_explained": {
            str(rank): float(energy[: min(rank, energy.numel())].sum() / max(total, 1e-30))
            for rank in ranks
        },
        "effective_rank": float(
            torch.exp(-(energy / energy.sum().clamp_min(1e-30) * torch.log(
                (energy / energy.sum().clamp_min(1e-30)).clamp_min(1e-30)
            )).sum())
        ),
    }
    return receipt, right


def subspace_overlap(left: torch.Tensor, right: torch.Tensor, rank: int = 32) -> dict[str, float]:
    use = min(rank, left.shape[0], right.shape[0])
    if use == 0:
        return {"rank": 0, "mean_cosine": float("nan"), "minimum_cosine": float("nan")}
    cosines = torch.linalg.svdvals(left[:use] @ right[:use].T).clamp(0.0, 1.0)
    return {
        "rank": use,
        "mean_cosine": float(cosines.mean()),
        "minimum_cosine": float(cosines.min()),
        "mean_squared_cosine": float(cosines.square().mean()),
    }


def bootstrap_mean_difference(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = float(positive.mean() - negative.mean())
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = (
            rng.choice(positive, positive.size, replace=True).mean()
            - rng.choice(negative, negative.size, replace=True).mean()
        )
    return {
        "estimate": observed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def discriminative_read(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    folds: int,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    x = torch.cat((positive, negative)).numpy()
    y = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    entropy = target_entropy_audit(len(positive), len(negative))
    if min(len(positive), len(negative)) < folds:
        return {
            "status": "UNDERPOWERED",
            "target_entropy_audit": entropy,
        }
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probability = np.empty(len(y), dtype=np.float64)
    for train, test in splitter.split(x, y):
        model = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=seed)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    auc = float(roc_auc_score(y, probability))
    balanced = float(balanced_accuracy_score(y, probability >= 0.5))
    rng = np.random.default_rng(seed + 17)
    auc_draws = []
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    for _ in range(draws):
        sampled = np.concatenate((
            rng.choice(positive_indices, positive_indices.size, replace=True),
            rng.choice(negative_indices, negative_indices.size, replace=True),
        ))
        auc_draws.append(roc_auc_score(y[sampled], probability[sampled]))
    return {
        "status": "MEASURED",
        "target_entropy_audit": entropy,
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "roc_auc_ci95": [float(np.quantile(auc_draws, 0.025)), float(np.quantile(auc_draws, 0.975))],
    }


def two_half_discriminative_read(
    positive: torch.Tensor,
    positive_halves: torch.Tensor,
    negative: torch.Tensor,
    negative_halves: torch.Tensor,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    entropy = target_entropy_audit(len(positive), len(negative))
    if min(len(positive), len(negative)) < 4:
        return {
            "status": "UNDERPOWERED",
            "target_entropy_audit": entropy,
        }
    results = []
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
            results.append({"heldout_half": heldout, "status": "UNDERPOWERED"})
            continue
        model = LogisticRegression(
            C=1.0, max_iter=2000, solver="liblinear", random_state=seed + heldout
        )
        model.fit(train_x, train_y)
        probability = model.predict_proba(test_x)[:, 1]
        auc = float(roc_auc_score(test_y, probability))
        rng = np.random.default_rng(seed + 101 + heldout)
        positive_indices = np.flatnonzero(test_y == 1)
        negative_indices = np.flatnonzero(test_y == 0)
        auc_draws = []
        for _ in range(draws):
            sampled = np.concatenate(
                (
                    rng.choice(positive_indices, positive_indices.size, replace=True),
                    rng.choice(negative_indices, negative_indices.size, replace=True),
                )
            )
            auc_draws.append(roc_auc_score(test_y[sampled], probability[sampled]))
        results.append(
            {
                "heldout_half": heldout,
                "status": "MEASURED",
                "train_rows": int(len(train_y)),
                "heldout_rows": int(len(test_y)),
                "roc_auc": auc,
                "roc_auc_ci95": [
                    float(np.quantile(auc_draws, 0.025)),
                    float(np.quantile(auc_draws, 0.975)),
                ],
                "balanced_accuracy": float(
                    balanced_accuracy_score(test_y, probability >= 0.5)
                ),
            }
        )
    return {
        "status": "MEASURED" if all(row["status"] == "MEASURED" for row in results) else "UNDERPOWERED",
        "target_entropy_audit": entropy,
        "halves": results,
        "both_halves_above_chance": all(
            row.get("roc_auc_ci95", [0.0])[0] > 0.5 for row in results
        ),
    }


def strata_masks(
    item_ids: list[str], base: dict[str, bool], seven: dict[str, bool], fourteen: dict[str, bool]
) -> dict[str, torch.Tensor]:
    b = torch.tensor([base[item_id] for item_id in item_ids])
    s = torch.tensor([seven[item_id] for item_id in item_ids])
    f = torch.tensor([fourteen[item_id] for item_id in item_ids])
    return {
        "D_7>0.5": s & ~b,
        "D_14>0.5": f & ~b,
        "D_14>7": f & ~s,
        "D_all": b & s & f,
        "D_none": ~b & ~s & ~f,
    }


def classify_direction_cell(cell: dict[str, Any], threshold: float) -> str:
    success = ("D_7>0.5", "D_14>0.5", "D_14>7")
    powered = [name for name in success if not cell[name].get("under_minimum_rows", True)]
    if len(powered) < 2 or cell["D_none"].get("under_minimum_rows", True):
        return "UNDERPOWERED"
    low_rank = all(cell[name]["variance_explained"]["32"] >= threshold for name in powered)
    discriminative = all(
        cell["discriminative"][name].get("both_halves_above_chance", False)
        for name in powered
    )
    overlaps = cell["principal_angles"]
    consistent = True
    for left_index, left in enumerate(powered):
        for right in powered[left_index + 1 :]:
            pair = overlaps[f"{left}__vs__{right}"]["mean_squared_cosine"]
            dnone_left = overlaps[f"{left}__vs__D_none"]["mean_squared_cosine"]
            dnone_right = overlaps[f"{right}__vs__D_none"]["mean_squared_cosine"]
            consistent &= pair > max(dnone_left, dnone_right)
    if low_rank and consistent and discriminative:
        return "STRUCTURED"
    if low_rank and cell["D_none"]["variance_explained"]["32"] >= threshold:
        return "GENERIC"
    return "DIFFUSE"


def write_figures(
    summary: dict[str, Any], output_dir: Path, *, include_tm2g: bool = True
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "D_7>0.5": "#176B87",
        "D_14>0.5": "#C54F3D",
        "D_14>7": "#7C5AA6",
        "D_none": "#777777",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for row_index, teacher in enumerate(TEACHERS):
        for column_index, view in enumerate(VIEWS):
            axis = axes[row_index, column_index]
            cell = summary["teachers"][teacher]["views"][view]["batteries"].get(
                "gsm8k", {}
            )
            names = [name for name in colors if name in cell]
            axis.bar(
                range(len(names)),
                [cell[name].get("variance_explained", {}).get("32", 0.0) for name in names],
                color=[colors[name] for name in names],
            )
            axis.axhline(0.30, color="#222222", linestyle="--", linewidth=1)
            axis.set_xticks(range(len(names)), [name.replace("D_", "") for name in names], rotation=20)
            axis.set_title(f"{teacher.replace('teacher_', '')} · {view.replace('_', ' ')}")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Top-32 variance explained")
    axes[1, 0].set_ylabel("Top-32 variance explained")
    figure.suptitle("TM-2 GSM8K displacement concentration")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"tm2_displacement_concentration.{suffix}", dpi=180)
    plt.close(figure)

    if not include_tm2g:
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, teacher in zip(axes, TEACHERS):
        position = 0
        ticks = []
        labels = []
        for view in VIEWS:
            cells = summary["teachers"][teacher]["views"][view]["tm2g"][
                "plane_consistency"
            ].get("gsm8k", {})
            for stratum in ("D_7>0.5", "D_14>0.5", "D_14>7"):
                cell = cells.get(stratum, {})
                if cell.get("status") != "MEASURED":
                    continue
                estimate = cell["success_minus_dnone"]["estimate"]
                low = cell["success_minus_dnone"]["ci95_low"]
                high = cell["success_minus_dnone"]["ci95_high"]
                axis.errorbar(
                    position,
                    estimate,
                    yerr=[[estimate - low], [high - estimate]],
                    fmt="o",
                    color=colors[stratum],
                    capsize=3,
                )
                ticks.append(position)
                labels.append(f"{view.split('_')[0]}\n{stratum.replace('D_', '')}")
                position += 1
            position += 0.5
        axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
        axis.set_xticks(ticks, labels, rotation=25)
        axis.set_title(teacher.replace("teacher_", "Qwen2.5 ").upper())
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Plane consistency minus D_none (95% row bootstrap CI)")
    figure.suptitle("TM-2g success-specific rotational consistency")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"tm2g_plane_consistency.{suffix}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--stitch_prefit", type=Path, required=True)
    parser.add_argument("--stitch_states", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--teacher_7b_scores", type=Path, required=True)
    parser.add_argument("--teacher_14b_scores", type=Path, required=True)
    parser.add_argument("--w1_seed0", type=Path, required=True)
    parser.add_argument("--w1_seed1", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--tm2g_mode", choices=("legacy", "disabled"), default="legacy"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = load_lock()
    panel = read_jsonl(args.panel)
    item_ids = [str(row["item_id"]) for row in panel]
    batteries = [str(row["battery"]) for row in panel]
    battery_values = sorted(set(batteries))
    battery_masks = {name: torch.tensor([value == name for value in batteries]) for name in battery_values}
    battery_masks["pooled"] = torch.ones(len(panel), dtype=torch.bool)
    analysis_batteries = battery_values + ["pooled"]
    masks = strata_masks(
        item_ids,
        score_map(args.base_scores),
        score_map(args.teacher_7b_scores),
        score_map(args.teacher_14b_scores),
    )
    panel_halves = torch.tensor(
        [
            deterministic_half(item_id, int(lock["tm1"]["split_seed"]))
            for item_id in item_ids
        ]
    )
    prefit = json.loads(args.stitch_prefit.read_text(encoding="utf-8"))
    state_bundle = torch.load(args.stitch_states, map_location="cpu", weights_only=False)
    transform = state_bundle["whitening"]["transform"].float()
    broad_mean = state_bundle["whitening"]["mean"].float()
    pre_run = {
        "kind": "paper2_tm0_tm2g_pre_run_v1",
        "written_before_geometric_statistics": True,
        "panel_sha256": sha256_file(args.panel),
        "stitch_states_sha256": sha256_file(args.stitch_states),
        "windows": {
            teacher: prefit["teachers"][teacher]["trajectory_boundaries_zero_based"]
            for teacher in TEACHERS
        },
        "pooling_views": list(VIEWS),
        "q": {"rows": lock["tm2g"]["q_rows"], "seed": lock["tm2g"]["q_seed"], "construction": "seeded_gaussian_qr_orthoproject"},
        "bivector_sketch": {"features": lock["tm2g"]["bivector_features"], "seed": lock["tm2g"]["bivector_sketch_seed"], "construction": "seeded_rademacher"},
        "noise_null": {"seed": lock["tm2g"]["noise_seed"], "construction": "seeded_gaussian_projected_perpendicular_to_each_state"},
        "common_mode": "unit_global_mean_direction_per_teacher_and_pooling_then_raw_projection",
        "direction_key_rule": (
            "within-battery cell is STRUCTURED iff at least two success strata and D_none "
            "have >=40 rows, each powered success stratum has top32 variance >=0.30, "
            "each success-pair mean squared principal cosine exceeds both corresponding "
            "success-vs-D_none values, and both deterministic fit-half discriminator CIs "
            "exclude AUC 0.5; pooled-only structure is BATTERY-CONFOUNDED; mixed cells split"
        ),
        "rotation_key_rule": (
            "ROTATION-CONSISTENT iff the row-bootstrap 95% lower bounds for success-minus-"
            "D_none and success-minus-matched-noise are positive for D_7>0.5 and D_14>0.5, "
            "both pooling views and both teachers, within GSM8K; partial replication splits"
        ),
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    if args.tm2g_mode == "disabled":
        for key in ("q", "bivector_sketch", "noise_null", "rotation_key_rule"):
            pre_run.pop(key, None)
        pre_run["kind"] = "paper2_tm0_tm2_pre_run_v1"
        pre_run["tm2g"] = {
            "status": "SUPERSEDED_BY_TM2G_J_R4",
            "authority_drive_id": "1GDZE-YnYU-RNHoBcMWBKcuWXjW3pxyaH",
        }
        atomic_json(args.output_dir / "tm2_pre_run_receipt.json", pre_run)
    else:
        atomic_json(args.output_dir / "tm2g_pre_run_receipt.json", pre_run)

    summary: dict[str, Any] = {
        "kind": "paper2_tm0_tm2_tm2g_summary_v1",
        "panel_sha256": sha256_file(args.panel),
        "stratum_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "stratum_battery_counts": {
            name: dict(Counter(value for value, keep in zip(batteries, mask.tolist()) if keep))
            for name, mask in masks.items()
        },
        "teachers": {},
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "injection_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    teacher_subspaces: dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]] = {}
    for teacher_index, teacher in enumerate(TEACHERS):
        cache_ids, layers = full_cache(args.cache_root / teacher, "layers")
        if cache_ids != item_ids:
            raise RuntimeError(f"{teacher} cache order mismatch")
        teacher_states = state_bundle["teachers"][teacher]
        boundaries = prefit["teachers"][teacher]["trajectory_boundaries_zero_based"]
        stitched = {
            layer: crossfit_stitched(
                layers, item_ids, teacher_states, layer, int(lock["tm1"]["split_seed"])
            )
            for layer in boundaries
        }
        white = {layer: (value - broad_mean) @ transform for layer, value in stitched.items()}
        total_delta = white[boundaries[-1]] - white[boundaries[0]]
        teacher_result: dict[str, Any] = {"views": {}}
        teacher_subspaces[teacher] = {}
        for view_index, view in enumerate(VIEWS):
            start_norm = white[boundaries[0]][:, view_index].norm(dim=-1)
            final_norm = white[boundaries[-1]][:, view_index].norm(dim=-1)
            trajectory_norm = total_delta[:, view_index].norm(dim=-1)
            residual, common = remove_common_mode(total_delta[:, view_index])
            view_result: dict[str, Any] = {
                "common_mode": common,
                "trajectory_scale_telemetry": {
                    "start_state_norm_mean": float(start_norm.mean()),
                    "final_state_norm_mean": float(final_norm.mean()),
                    "total_displacement_norm_mean": float(trajectory_norm.mean()),
                    "total_displacement_norm_median": float(trajectory_norm.median()),
                    "mean_row_displacement_to_start_ratio": float(
                        (trajectory_norm / start_norm.clamp_min(1e-12)).mean()
                    ),
                },
                "batteries": {},
            }
            teacher_subspaces[teacher][view] = {}
            for battery in analysis_batteries:
                battery_result: dict[str, Any] = {}
                teacher_subspaces[teacher][view][battery] = {}
                raw_subspaces = {}
                for stratum, stratum_mask in masks.items():
                    selected = battery_masks[battery] & stratum_mask
                    cell, subspace = svd_receipt(residual[selected], [int(v) for v in lock["tm2"]["ranks"]])
                    raw_cell, raw_subspace = svd_receipt(
                        total_delta[selected, view_index],
                        [int(v) for v in lock["tm2"]["ranks"]],
                    )
                    cell["raw_before_common_projection"] = raw_cell
                    cell["under_minimum_rows"] = int(selected.sum()) < int(lock["tm2"]["minimum_stratum_rows"])
                    battery_result[stratum] = cell
                    teacher_subspaces[teacher][view][battery][stratum] = subspace
                    raw_subspaces[stratum] = raw_subspace
                overlaps = {}
                raw_overlaps = {}
                names = list(masks)
                for left_index, left in enumerate(names):
                    for right in names[left_index + 1 :]:
                        key = f"{left}__vs__{right}"
                        overlaps[key] = subspace_overlap(
                            teacher_subspaces[teacher][view][battery][left],
                            teacher_subspaces[teacher][view][battery][right],
                        )
                        raw_overlaps[key] = subspace_overlap(
                            raw_subspaces[left], raw_subspaces[right]
                        )
                battery_result["principal_angles"] = overlaps
                battery_result["raw_principal_angles"] = raw_overlaps
                dnone = battery_masks[battery] & masks["D_none"]
                discriminative = {}
                for stratum in ("D_7>0.5", "D_14>0.5", "D_14>7"):
                    positive = battery_masks[battery] & masks[stratum]
                    discriminative[stratum] = two_half_discriminative_read(
                        residual[positive],
                        panel_halves[positive],
                        residual[dnone],
                        panel_halves[dnone],
                        seed=int(lock["tm2g"]["discriminative_seed"]) + teacher_index * 100 + view_index * 10,
                        draws=int(lock["tm2"]["bootstrap_draws"]),
                    )
                battery_result["discriminative"] = discriminative
                view_result["batteries"][battery] = battery_result

            if args.tm2g_mode == "disabled":
                view_result["tm2g"] = {
                    "status": "SUPERSEDED_BY_TM2G_J_R4",
                    "authority_drive_id": "1GDZE-YnYU-RNHoBcMWBKcuWXjW3pxyaH",
                }
                teacher_result["views"][view] = view_result
                continue

            q = random_orthoproject(int(lock["tm2g"]["q_rows"]), total_delta.shape[-1], seed=int(lock["tm2g"]["q_seed"]))
            wedge_features = q.shape[0] * (q.shape[0] - 1) // 2
            sketch = rademacher_sketch(wedge_features, int(lock["tm2g"]["bivector_features"]), seed=int(lock["tm2g"]["bivector_sketch_seed"]))
            radial_parts = []
            plane_parts = []
            noise_parts = []
            generator = torch.Generator().manual_seed(int(lock["tm2g"]["noise_seed"]) + teacher_index * 10 + view_index)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                states = white[start][:, view_index]
                delta = white[end][:, view_index] - states
                radial, plane, _ = geometric_features(states, delta, q, sketch)
                random_delta = torch.randn(delta.shape, generator=generator)
                state_unit = F.normalize(states, dim=-1, eps=1e-12)
                random_delta -= (random_delta * state_unit).sum(dim=-1, keepdim=True) * state_unit
                _, noise_plane, _ = geometric_features(states, random_delta, q, sketch)
                radial_parts.append(radial)
                plane_parts.append(plane)
                noise_parts.append(noise_plane)
            radial_matrix = torch.stack(radial_parts, dim=1)
            plane_matrix = torch.stack(plane_parts, dim=1)
            noise_matrix = torch.stack(noise_parts, dim=1)
            consistency = (plane_matrix[:, :-1] * plane_matrix[:, 1:]).sum(dim=-1).mean(dim=1)
            noise_consistency = (noise_matrix[:, :-1] * noise_matrix[:, 1:]).sum(dim=-1).mean(dim=1)
            geometric_x = torch.cat((radial_matrix, consistency[:, None], plane_matrix.mean(dim=1)), dim=1)
            geometric_cells = {}
            for battery_index, battery in enumerate(analysis_batteries):
                cells = {}
                dnone_mask = battery_masks[battery] & masks["D_none"]
                for stratum_index, stratum in enumerate(("D_7>0.5", "D_14>0.5", "D_14>7")):
                    positive_mask = battery_masks[battery] & masks[stratum]
                    if positive_mask.sum() < 2 or dnone_mask.sum() < 2:
                        cells[stratum] = {"status": "UNDERPOWERED", "positive_rows": int(positive_mask.sum()), "dnone_rows": int(dnone_mask.sum())}
                        continue
                    success = consistency[positive_mask].numpy()
                    failed = consistency[dnone_mask].numpy()
                    noise = noise_consistency[positive_mask].numpy()
                    seed = int(lock["tm2"]["bootstrap_seed"]) + teacher_index * 1000 + view_index * 100 + battery_index * 10 + stratum_index
                    vs_failed = bootstrap_mean_difference(success, failed, draws=int(lock["tm2"]["bootstrap_draws"]), seed=seed)
                    vs_noise = bootstrap_mean_difference(success, noise, draws=int(lock["tm2"]["bootstrap_draws"]), seed=seed + 1)
                    cells[stratum] = {
                        "status": "MEASURED",
                        "positive_rows": int(positive_mask.sum()),
                        "dnone_rows": int(dnone_mask.sum()),
                        "success_mean": float(success.mean()),
                        "dnone_mean": float(failed.mean()),
                        "matched_noise_mean": float(noise.mean()),
                        "success_minus_dnone": vs_failed,
                        "success_minus_noise": vs_noise,
                        "claim_pass": vs_failed["ci95_low"] > 0.0 and vs_noise["ci95_low"] > 0.0,
                        "discriminative": discriminative_read(
                            geometric_x[positive_mask], geometric_x[dnone_mask],
                            folds=int(lock["tm2g"]["discriminative_folds"]), seed=int(lock["tm2g"]["discriminative_seed"]) + seed, draws=int(lock["tm2"]["bootstrap_draws"]),
                        ),
                    }
                geometric_cells[battery] = cells
            view_result["tm2g"] = {
                "radial_fraction": {
                    "mean_by_window": radial_matrix.mean(dim=0).tolist(),
                    "median_by_window": radial_matrix.median(dim=0).values.tolist(),
                },
                "plane_consistency": geometric_cells,
            }
            teacher_result["views"][view] = view_result
        summary["teachers"][teacher] = teacher_result

    bridge = {}
    for seed, path in ((0, args.w1_seed0), (1, args.w1_seed1)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        index = {str(item_id): row for row, item_id in enumerate(payload["item_ids"])}
        correction = payload["families"]["l0a"].float()
        seed_result = {}
        for teacher in TEACHERS:
            teacher_result = {}
            for view in VIEWS:
                cells = {}
                for battery in analysis_batteries:
                    for stratum in ("D_7>0.5", "D_14>0.5", "D_14>7"):
                        ids = [item_ids[row] for row, keep in enumerate((battery_masks[battery] & masks[stratum]).tolist()) if keep and item_ids[row] in index]
                        if len(ids) < 2:
                            continue
                        teacher_space = teacher_subspaces[teacher][view][battery][stratum]
                        rows = torch.stack([correction[index[item_id]] for item_id in ids])
                        rows, _ = remove_common_mode(rows)
                        _, correction_space = svd_receipt(rows, [32])
                        cells[f"{battery}:{stratum}"] = subspace_overlap(teacher_space, correction_space)
                teacher_result[view] = cells
            seed_result[teacher] = teacher_result
        bridge[str(seed)] = seed_result
    summary["bridge_read"] = bridge

    if args.tm2g_mode == "disabled":
        rotation_key = "SUPERSEDED_BY_TM2G_J_R4"
    else:
        gsm8k_required = []
        generic_above_noise = []
        for teacher in TEACHERS:
            for view in VIEWS:
                cells = summary["teachers"][teacher]["views"][view]["tm2g"]["plane_consistency"].get("gsm8k", {})
                for stratum in ("D_7>0.5", "D_14>0.5"):
                    cell = cells.get(stratum, {})
                    if cell.get("status") == "MEASURED":
                        gsm8k_required.append(bool(cell.get("claim_pass")))
                        generic_above_noise.append(cell["success_minus_noise"]["ci95_low"] > 0.0)
        if gsm8k_required and all(gsm8k_required):
            rotation_key = "ROTATION-CONSISTENT"
        elif any(gsm8k_required):
            rotation_key = "STRATUM-SPLIT"
        elif generic_above_noise and all(generic_above_noise):
            rotation_key = "ROTATION-GENERIC"
        else:
            rotation_key = "ROTATION-ABSENT"
    direction_cells = {}
    for teacher in TEACHERS:
        for view in VIEWS:
            direction_cells[f"{teacher}:{view}"] = {
                battery: classify_direction_cell(
                    summary["teachers"][teacher]["views"][view]["batteries"][battery],
                    float(lock["tm2"]["low_rank_prediction_fraction"]),
                )
                for battery in analysis_batteries
            }
    within = [
        value
        for cells in direction_cells.values()
        for battery, value in cells.items()
        if battery != "pooled" and value != "UNDERPOWERED"
    ]
    pooled = [cells["pooled"] for cells in direction_cells.values()]
    if within and all(value == "STRUCTURED" for value in within):
        direction_key = "DIRECTIONS-STRUCTURED"
    elif any(value == "STRUCTURED" for value in pooled) and not any(
        value == "STRUCTURED" for value in within
    ):
        direction_key = "BATTERY-CONFOUNDED"
    elif within and all(value == "GENERIC" for value in within):
        direction_key = "DIRECTIONS-GENERIC"
    elif len(set(within)) > 1:
        direction_key = "STRATUM-SPLIT"
    else:
        direction_key = "DIRECTIONS-DIFFUSE"
    summary["direction_cell_classification"] = direction_cells
    summary["decision_keys"] = {"tm2": direction_key, "tm2g": rotation_key}
    atomic_json(args.output_dir / "tm2_tm2g_summary.json", summary)
    write_figures(summary, args.output_dir, include_tm2g=args.tm2g_mode == "legacy")
    print(json.dumps(summary["decision_keys"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
