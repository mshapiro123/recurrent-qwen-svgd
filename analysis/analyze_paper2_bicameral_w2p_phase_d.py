"""Run the W2-prime D1-D3 conditional-mixer desk battery."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from training.paper2_stage2b_autopsy import spherical_kmeans

from training.paper2_bicameral_w2p import (
    BOOTSTRAP_DRAWS,
    CONDITIONAL_COSINE_GATE,
    HEMISPHERIC_RISK_REDUCTION_GATE,
    RIDGE_MULTIPLIER_GRID,
    bootstrap_mean_interval,
    conditional_row_cosine,
    deterministic_derangement,
    fit_block_map,
    normalized_row_risk,
    paired_relative_risk_reduction,
    select_crossfitted_map,
    validate_deployment_features,
)


KIND = "paper2_bicameral_w2p_phase_d_v1"
AUTHORITY_SHA256 = "f89b45ef100fa46536dd93a3ef936aa8c9cfa1fc624b401b4bfc0d2b50bc2aa4"
RULINGS_SHA256 = "34352161cb69612bfc996658fab0f2d24eed381cc3895eda99a7c5a3d2e835fd"
FS2_PRIME_FEATURE_BLOCKS = (
    "interface_mean_m18",
    "interface_difference_d18",
    "mean_history_concat_m8_m12_m16_dm8to12_dm12to16_dm16to18",
    "difference_history_concat_d8_d12_d16_dd8to12_dd12to16_dd16to18",
)
FS2_PRIME_CONTROL = (
    "base_history_concat_h8_h12_h16_h18_dh8to12_dh12to16_dh16to18"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_target(
    target_artifact: Mapping[str, Any], item_ids: Sequence[str], family: str
) -> torch.Tensor:
    index = {str(item_id): position for position, item_id in enumerate(target_artifact["item_ids"])}
    missing = [item_id for item_id in item_ids if item_id not in index]
    if missing:
        raise RuntimeError(f"W2-prime target alignment missing {len(missing)} rows")
    values = target_artifact["families"][family]
    aligned = values[torch.tensor([index[item_id] for item_id in item_ids], dtype=torch.long)].float()
    return F.normalize(aligned, dim=-1, eps=1e-12)


def summarize_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batteries: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    rows = conditional_row_cosine(prediction, target)
    per_battery = {}
    for battery in sorted(set(batteries)):
        mask = torch.tensor([value == battery for value in batteries])
        per_battery[battery] = {
            "rows": int(mask.sum()),
            **bootstrap_mean_interval(rows[mask], seed=seed + len(per_battery) + 1),
        }
    return {
        "pooled": bootstrap_mean_interval(rows, seed=seed),
        "positive_fraction": float((rows > 0).float().mean()),
        "per_battery": per_battery,
    }


def _project_off(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return values - (values @ basis.T) @ basis


def nuisance_deflated_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    folds: torch.Tensor,
    cluster_labels: torch.Tensor,
    *,
    rank: int = 8,
) -> dict[str, float]:
    values = []
    for fold in sorted(set(int(value) for value in folds.tolist())):
        for cluster in sorted(set(int(value) for value in cluster_labels.tolist())):
            evaluate = (folds == fold) & (cluster_labels == cluster)
            fit = (folds != fold) & (cluster_labels == cluster)
            other = cluster_labels != cluster
            nuisance_rows = torch.cat([target[fit], target[other]], dim=0).double()
            _u, _s, vh = torch.linalg.svd(nuisance_rows, full_matrices=False)
            nuisance = vh[: min(rank, vh.shape[0])]
            fit_projected = _project_off(target[fit].double(), nuisance)
            cluster_direction = F.normalize(
                fit_projected.mean(dim=0), dim=0, eps=1e-12
            ).unsqueeze(0)
            combined = torch.cat([nuisance, cluster_direction], dim=0)
            combined = torch.linalg.qr(combined.T, mode="reduced").Q.T
            target_eval = _project_off(target[evaluate].double(), combined)
            prediction_eval = _project_off(prediction[evaluate].double(), combined)
            values.append(
                F.cosine_similarity(
                    prediction_eval.float(), target_eval.float(), dim=-1, eps=1e-12
                )
            )
    rows = torch.cat(values)
    return {
        "mean": float(rows.mean()),
        "positive_fraction": float((rows > 0).float().mean()),
        "rank": int(rank),
        "basis_estimation": "R-S0-A opposite-fold within-cluster rank-8 nuisance plus cluster direction",
        "cluster_count": int(cluster_labels.unique().numel()),
    }


def serialize_model(model: Any) -> dict[str, Any]:
    return {
        "x_means": [value.cpu() for value in model.x_means],
        "projections": [value.cpu() for value in model.projections],
        "y_mean": model.y_mean.cpu(),
        "theta": model.theta.cpu(),
        "ridge_effective": list(model.ridge_effective),
    }


def build_registered_feature_sets(
    cache: Mapping[str, Any],
    *,
    branch_b_permutation: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the prospectively frozen prompt-only FS-1 and FS-2-prime blocks."""

    required_sites = (8, 12, 16, 18)
    if tuple(sorted(int(value) for value in cache["sites"])) != required_sites:
        raise RuntimeError("W2-prime D4 site inventory changed")
    base: dict[int, torch.Tensor] = {}
    mean: dict[int, torch.Tensor] = {}
    difference: dict[int, torch.Tensor] = {}
    for site in required_sites:
        payload = cache["sites"][site]
        h_a = payload["branch_a"].float()
        h_b = payload["branch_b"].float()
        if branch_b_permutation is not None:
            h_b = h_b[branch_b_permutation]
        base[site] = payload["base"].float()
        mean[site] = (h_a + h_b) / 2
        difference[site] = (h_a - h_b) / 2

    mean_history = torch.cat(
        [
            mean[8],
            mean[12],
            mean[16],
            mean[12] - mean[8],
            mean[16] - mean[12],
            mean[18] - mean[16],
        ],
        dim=-1,
    )
    difference_history = torch.cat(
        [
            difference[8],
            difference[12],
            difference[16],
            difference[12] - difference[8],
            difference[16] - difference[12],
            difference[18] - difference[16],
        ],
        dim=-1,
    )
    base_history = torch.cat(
        [
            base[8],
            base[12],
            base[16],
            base[18],
            base[12] - base[8],
            base[16] - base[12],
            base[18] - base[16],
        ],
        dim=-1,
    )
    return {
        "fs1_md": [mean[18], difference[18]],
        "fs2_prime": [mean[18], difference[18], mean_history, difference_history],
        "fs0_h": [base[18]],
        "fs0_prime": [base_history],
        "fs_m": [mean[18]],
        "fs_ab": [mean[18] + difference[18], mean[18] - difference[18]],
    }


def block_energy_shares(model: Any, blocks: Sequence[torch.Tensor]) -> list[float]:
    contributions = []
    offset = 0
    for block, mean, projection in zip(blocks, model.x_means, model.projections):
        width = projection.shape[1]
        scores = (block.float() - mean) @ projection
        contribution = scores @ model.theta[offset : offset + width]
        contributions.append(float(contribution.square().sum(dim=-1).mean()))
        offset += width
    denominator = max(sum(contributions), 1e-12)
    return [value / denominator for value in contributions]


def fit_feature_set(
    blocks: Sequence[torch.Tensor],
    target: torch.Tensor,
    batteries: Sequence[str],
    *,
    seed: int,
    cluster_labels: torch.Tensor,
    rank_options: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    selected = select_crossfitted_map(
        blocks,
        target,
        batteries,
        seed=seed,
        rank_options=rank_options,
    )
    prediction = selected.pop("prediction")
    folds = selected.pop("folds")
    row_risk = normalized_row_risk(prediction, target)
    return {
        "prediction": prediction,
        "folds": folds,
        "row_risk": row_risk,
        "fit": selected,
        "risk": bootstrap_mean_interval(row_risk, seed=seed + 100),
        "conditional_cosine": summarize_cosine(
            prediction, target, batteries, seed=seed + 200
        ),
        "rank8_nuisance_deflated_cosine": nuisance_deflated_cosine(
            prediction, target, folds, cluster_labels
        ),
    }


def final_model(
    blocks: Sequence[torch.Tensor], target: torch.Tensor, selected: Mapping[str, Any]
) -> Any:
    return fit_block_map(
        blocks,
        target,
        ranks=selected["ranks"],
        ridge_multipliers=selected["ridge_multipliers"],
    )


def site_incremental_table(
    sites: Mapping[int, Mapping[str, torch.Tensor]],
    target: torch.Tensor,
    batteries: Sequence[str],
    *,
    ranks: Sequence[int],
    cluster_labels: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    table = {}
    for site_index, site in enumerate(sorted(int(value) for value in sites)):
        payload = sites[site]
        h_a = payload["branch_a"].float()
        h_b = payload["branch_b"].float()
        directions = {}
        for direction_index, (name, receiver, sender) in enumerate(
            (
                ("a_to_b", h_b, h_a),
                ("b_to_a", h_a, h_b),
            )
        ):
            pair = fit_feature_set(
                [receiver, sender],
                target,
                batteries,
                seed=seed + site_index * 20 + direction_index,
                cluster_labels=cluster_labels,
                rank_options=((int(ranks[0]),), (int(ranks[1]),)),
            )
            receiver_only = fit_feature_set(
                [receiver],
                target,
                batteries,
                seed=seed + 100 + site_index * 20 + direction_index,
                cluster_labels=cluster_labels,
                rank_options=((int(sum(ranks)),),),
            )
            incremental = paired_relative_risk_reduction(
                receiver_only["row_risk"],
                pair["row_risk"],
                seed=seed + 200 + site_index * 20 + direction_index,
            )
            directions[name] = {
                "incremental_relative_risk_reduction": incremental,
                "pair_selected": pair["fit"]["selected"],
                "receiver_selected": receiver_only["fit"]["selected"],
                "pair_conditional_cosine": pair["conditional_cosine"],
                "receiver_conditional_cosine": receiver_only["conditional_cosine"],
            }
        table[str(site)] = directions
    return table


def composition_desk_item(root: Path, seed: int) -> dict[str, Any]:
    rows = {}
    for arm in ("l1", "l2"):
        path = root / f"seed_{seed}_phase_b_{arm}.jsonl"
        values = read_jsonl(path)
        rows[arm] = {str(row["item_id"]): row for row in values}
    if rows["l1"].keys() != rows["l2"].keys():
        raise RuntimeError("W2-prime A-3 source populations changed")
    grouped: dict[str, list[float]] = defaultdict(list)
    for item_id in rows["l1"]:
        first = rows["l1"][item_id]
        second = rows["l2"][item_id]
        grouped[str(first["battery"])].append(
            float(first["margin_delta"]) - float(second["margin_delta"])
        )
    return {
        battery: {"rows": len(values), **bootstrap_mean_interval(torch.tensor(values), seed=20260900 + seed + index)}
        for index, (battery, values) in enumerate(sorted(grouped.items()))
    }


def run_seed(
    args: argparse.Namespace,
    seed: int,
    *,
    secondary_family: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_path = args.cache_root / f"seed_{seed}_w2p_d4_cache.pt"
    target_path = args.target_root / f"seed_{seed}_full_student_targets.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    targets = torch.load(target_path, map_location="cpu", weights_only=False)
    validate_deployment_features(cache)
    if cache.get("kind") != "paper2_bicameral_w2p_d4_cache_v1":
        raise RuntimeError("wrong W2-prime D4 cache kind")
    item_ids = [str(value) for value in cache["item_ids"]]
    batteries = [str(value) for value in cache["batteries"]]
    feature_blocks = build_registered_feature_sets(cache)
    seed_result: dict[str, Any] = {
        "seed": seed,
        "rows": len(item_ids),
        "inputs": {
            "cache_sha256": sha256_file(cache_path),
            "target_sha256": sha256_file(target_path),
            "input_features": cache["feature_list"],
        },
        "fs2": {
            "status": "BLOCKED_SOURCE_CONFLICT",
            "reason": "banked W3 loop states were extracted from forced-target prompt-plus-gold sequences",
        },
        "fs2_prime": {
            "status": "REGISTERED_PRE_OUTCOME",
            "gate_eligible": True,
            "feature_blocks": list(FS2_PRIME_FEATURE_BLOCKS),
            "matched_single_stream_control": FS2_PRIME_CONTROL,
        },
        "targets": {},
        "a3_composition": composition_desk_item(args.phase_b_private, seed),
    }
    models = {}
    for target_index, family in enumerate(("l0a", secondary_family)):
        target = align_target(targets, item_ids, family)
        cluster_labels, cluster_silhouette = spherical_kmeans(
            target,
            clusters=2,
            seed=20260819 + seed,
            restarts=8,
            iterations=50,
        )
        permutation = deterministic_derangement(len(item_ids), tag=f"seed{seed}:{family}")
        shuffled_blocks = build_registered_feature_sets(
            cache, branch_b_permutation=permutation
        )
        feature_results = {}
        target_models = {}
        for feature_index, (feature_name, control_name) in enumerate(
            (("fs1_md", "fs0_h"), ("fs2_prime", "fs0_prime"))
        ):
            fitted = fit_feature_set(
                feature_blocks[feature_name],
                target,
                batteries,
                seed=20260825 + seed * 1000 + target_index * 100 + feature_index,
                cluster_labels=cluster_labels,
            )
            ranks = fitted["fit"]["selected"]["ranks"]
            total_rank = int(sum(ranks))
            control = fit_feature_set(
                feature_blocks[control_name],
                target,
                batteries,
                seed=20261025 + seed * 1000 + target_index * 100 + feature_index,
                cluster_labels=cluster_labels,
                rank_options=((total_rank,),),
            )
            additional_controls = {}
            if feature_name == "fs1_md":
                additional_controls = {
                    "fs_m": fit_feature_set(
                        feature_blocks["fs_m"],
                        target,
                        batteries,
                        seed=20261225 + seed * 1000 + target_index * 100,
                        cluster_labels=cluster_labels,
                        rank_options=((total_rank,),),
                    ),
                    "fs_ab": fit_feature_set(
                        feature_blocks["fs_ab"],
                        target,
                        batteries,
                        seed=20261425 + seed * 1000 + target_index * 100,
                        cluster_labels=cluster_labels,
                        rank_options=((ranks[0],), (ranks[1],)),
                    ),
                }
            risk_reduction = paired_relative_risk_reduction(
                control["row_risk"],
                fitted["row_risk"],
                seed=20261625 + seed * 1000 + target_index * 100 + feature_index,
            )
            selected_model = final_model(
                feature_blocks[feature_name], target, fitted["fit"]["selected"]
            )
            frozen_branch_prediction = selected_model.predict(
                shuffled_blocks[feature_name]
            )
            frozen_permuted_prediction = selected_model.predict(
                [block[permutation] for block in feature_blocks[feature_name]]
            )
            clean_fitted = {
                key: item
                for key, item in fitted.items()
                if key not in {"prediction", "folds", "row_risk"}
            }
            clean_control = {
                key: item
                for key, item in control.items()
                if key not in {"prediction", "folds", "row_risk"}
            }
            clean_additional_controls = {
                name: {
                    key: item
                    for key, item in value.items()
                    if key not in {"prediction", "folds", "row_risk"}
                }
                for name, value in additional_controls.items()
            }
            conditional = float(fitted["conditional_cosine"]["pooled"]["mean"])
            feature_results[feature_name] = {
                "matched_total_rank": total_rank,
                "fit": clean_fitted,
                "matched_single_stream_control": {
                    "name": control_name,
                    "fit": clean_control,
                },
                "additional_d2_controls": clean_additional_controls,
                "mean_over_base_relative_risk_reduction": (
                    paired_relative_risk_reduction(
                        control["row_risk"],
                        additional_controls["fs_m"]["row_risk"],
                        seed=20261725 + seed * 100 + target_index,
                    )
                    if "fs_m" in additional_controls
                    else None
                ),
                "block_energy_shares": block_energy_shares(
                    selected_model, feature_blocks[feature_name]
                ),
                "hemispheric_relative_risk_reduction": risk_reduction,
                "g_d1_pass": conditional >= CONDITIONAL_COSINE_GATE,
                "g_d2_pass": float(risk_reduction["relative_risk_reduction"])
                >= HEMISPHERIC_RISK_REDUCTION_GATE,
                "d3": {
                    "permutation_sha256": hashlib.sha256(
                        permutation.numpy().tobytes()
                    ).hexdigest(),
                    "frozen_map_branch_pair_shuffle_risk": bootstrap_mean_interval(
                        normalized_row_risk(frozen_branch_prediction, target),
                        seed=20261825 + seed * 100 + feature_index,
                    ),
                    "frozen_map_permuted_input_risk": bootstrap_mean_interval(
                        normalized_row_risk(frozen_permuted_prediction, target),
                        seed=20261925 + seed * 100 + feature_index,
                    ),
                },
            }
            target_models[feature_name] = {
                "kind": "paper2_bicameral_w2p_selected_map_v1",
                "seed": seed,
                "target_family": family,
                "target_role": "sole_gate_family" if family == "l0a" else "diagnostic_only",
                "feature_set": feature_name,
                "item_ids": item_ids,
                "batteries": batteries,
                "hyperparameters": fitted["fit"]["selected"],
                "model": serialize_model(selected_model),
                "input_provenance": "student_prompt_only",
            }

        fs1_ranks = feature_results["fs1_md"]["fit"]["fit"]["selected"]["ranks"]
        seed_result["targets"][family] = {
            "primary": family == "l0a",
            "gate_eligible": family == "l0a",
            "construction": "loss_gradient" if family == "l0a" else "teacher_forced_state_delta_h_gold_minus_h_free",
            "rs0a_clusters": {
                "method": "spherical_kmeans_k2_seed_20260819_plus_seed",
                "silhouette": float(cluster_silhouette),
                "counts": [
                    int((cluster_labels == value).sum())
                    for value in cluster_labels.unique(sorted=True)
                ],
                "assignment_sha256": hashlib.sha256(
                    cluster_labels.numpy().tobytes()
                ).hexdigest(),
            },
            "feature_sets": feature_results,
            "d4_site_screen": site_incremental_table(
                cache["sites"],
                target,
                batteries,
                ranks=fs1_ranks,
                cluster_labels=cluster_labels,
                seed=20262025 + seed * 1000 + target_index * 400,
            ),
        }
        models[family] = target_models
    return seed_result, models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("training/paper2_bicameral_w2p_lock.json"),
    )
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phase_b_private", type=Path, required=True)
    parser.add_argument("--prefit_receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock["rulings"]["sha256"] != RULINGS_SHA256:
        raise RuntimeError("W2-prime rulings hash changed")
    secondary_binding = lock["phase_d"]["secondary_target_binding"]
    if secondary_binding["status"] != "RESOLVED_DIAGNOSTIC_ONLY":
        raise RuntimeError(
            "W2-prime diagnostic target binding does not match R-W2P-1"
        )
    secondary_family = str(secondary_binding["resolved_family"])
    if secondary_family != "l0d" or secondary_binding["gate_eligible"]:
        raise RuntimeError("W2-prime R-W2P-1 requires diagnostic-only L0d")
    if lock["fs2_prime"]["feature_blocks"] != list(FS2_PRIME_FEATURE_BLOCKS):
        raise RuntimeError("W2-prime FS-2-prime feature inventory changed")
    if lock["fs2_prime"]["matched_single_stream_control"] != FS2_PRIME_CONTROL:
        raise RuntimeError("W2-prime FS-2-prime matched control changed")
    prefit = {
        "kind": "paper2_bicameral_w2p_prefit_receipt_v1",
        "status": "frozen_before_any_fit",
        "authority_sha256": AUTHORITY_SHA256,
        "rulings_sha256": RULINGS_SHA256,
        "lock_sha256": sha256_file(args.lock),
        "sole_gate_target": "l0a_loss_gradient",
        "diagnostic_target": "l0d_teacher_forced_state_delta_h_gold_minus_h_free",
        "fs2_status": "BLOCKED_SOURCE_CONFLICT",
        "fs2_prime_feature_blocks": list(FS2_PRIME_FEATURE_BLOCKS),
        "fs2_prime_matched_single_stream_control": FS2_PRIME_CONTROL,
        "selection_precedence": "fs1_md_if_double_pass_else_fs2_prime_if_double_pass",
        "selection_rule": "nested_blockwise_inner_cv_then_joint_refit",
        "standing_law": "SL-3",
        "secondary_read": "R-S0-A opposite-fold within-cluster rank-8 nuisance plus cluster direction",
        "rs0a_cluster_rule": "target-family spherical_kmeans k=2 seed 20260819+model_seed restarts=8 iterations=50",
        "d2_controls": ["matched_single_stream", "mean_only", "raw_branch_pair"],
        "outer_folds": 4,
        "inner_folds": 3,
        "rank_grid": [2, 4, 8, 16, 32],
        "ridge_multiplier_grid": list(RIDGE_MULTIPLIER_GRID),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.prefit_receipt, prefit)
    print(
        json.dumps(
            {
                "prefit_receipt": str(args.prefit_receipt),
                "prefit_sha256": sha256_file(args.prefit_receipt),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    seeds = {}
    for seed in (0, 1):
        result, models = run_seed(args, seed, secondary_family=secondary_family)
        seeds[str(seed)] = result
        for family, feature_models in models.items():
            for feature_name, payload in feature_models.items():
                torch.save(
                    payload,
                    args.model_dir / f"seed_{seed}_{family}_{feature_name}.pt",
                )
    feature_gates = {}
    seed_split_key = None
    for feature_name in ("fs1_md", "fs2_prime"):
        d1 = [
            bool(
                seeds[str(seed)]["targets"]["l0a"]["feature_sets"][feature_name][
                    "g_d1_pass"
                ]
            )
            for seed in (0, 1)
        ]
        d2 = [
            bool(
                seeds[str(seed)]["targets"]["l0a"]["feature_sets"][feature_name][
                    "g_d2_pass"
                ]
            )
            for seed in (0, 1)
        ]
        feature_gates[feature_name] = {
            "g_d1_by_seed": d1,
            "g_d2_by_seed": d2,
            "double_pass": all(d1) and all(d2),
        }
        if d1[0] != d1[1]:
            seed_split_key = "SEED-SPLIT-G-D1"
        elif d2[0] != d2[1]:
            seed_split_key = "SEED-SPLIT-G-D2"
    selected_feature_set = None
    if seed_split_key is not None:
        key = seed_split_key
    elif feature_gates["fs1_md"]["double_pass"]:
        key = "DESK-DOUBLE-PASS"
        selected_feature_set = "fs1_md"
    elif feature_gates["fs2_prime"]["double_pass"]:
        key = "DESK-DOUBLE-PASS"
        selected_feature_set = "fs2_prime"
    elif not any(all(value["g_d1_by_seed"]) for value in feature_gates.values()):
        key = "MAP-MIRAGE"
    else:
        key = "HEMISPHERES-UNINFORMATIVE"
    payload = {
        "kind": KIND,
        "status": "complete_cpu_only",
        "authority_sha256": AUTHORITY_SHA256,
        "rulings_sha256": RULINGS_SHA256,
        "prefit_receipt_sha256": sha256_file(args.prefit_receipt),
        "rank_grid": [2, 4, 8, 16, 32],
        "ridge_multiplier_grid": list(RIDGE_MULTIPLIER_GRID),
        "ridge_scaling": "multiplier_times_mean_diagonal_xtx",
        "selection_rule": "nested_blockwise_inner_cv_then_joint_refit",
        "outer_folds": 4,
        "inner_folds": 3,
        "sole_gate_target_family": "l0a",
        "diagnostic_target_family": secondary_family,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "fs2_status": "BLOCKED_SOURCE_CONFLICT",
        "fs2_prime_status": "REGISTERED_PRE_OUTCOME_AND_EVALUATED",
        "feature_gates": feature_gates,
        "selected_feature_set": selected_feature_set,
        "desk_key": key,
        "phase_g_authorized_by_desk_result": key == "DESK-DOUBLE-PASS",
        "seeds": seeds,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"desk_key": key, "output": str(args.output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
