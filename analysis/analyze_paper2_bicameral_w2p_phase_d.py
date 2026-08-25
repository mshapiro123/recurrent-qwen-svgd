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


def nuisance_deflated_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    folds: torch.Tensor,
    *,
    rank: int = 8,
) -> dict[str, float]:
    values = []
    for fold in sorted(set(int(value) for value in folds.tolist())):
        evaluate = folds == fold
        train = ~evaluate
        mean = target[train].double().mean(dim=0, keepdim=True)
        centered = target[train].double() - mean
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[: min(rank, vh.shape[0])]
        target_eval = target[evaluate].double() - mean
        prediction_eval = prediction[evaluate].double() - mean
        target_eval = target_eval - (target_eval @ basis.T) @ basis
        prediction_eval = prediction_eval - (prediction_eval @ basis.T) @ basis
        values.append(F.cosine_similarity(prediction_eval.float(), target_eval.float(), dim=-1, eps=1e-12))
    rows = torch.cat(values)
    return {
        "mean": float(rows.mean()),
        "positive_fraction": float((rows > 0).float().mean()),
        "rank": int(rank),
        "basis_estimation": "opposite_fold_target_pca",
    }


def serialize_model(model: Any) -> dict[str, Any]:
    return {
        "x_means": [value.cpu() for value in model.x_means],
        "projections": [value.cpu() for value in model.projections],
        "y_mean": model.y_mean.cpu(),
        "theta": model.theta.cpu(),
        "ridge_effective": list(model.ridge_effective),
    }


def fit_feature_set(
    blocks: Sequence[torch.Tensor],
    target: torch.Tensor,
    batteries: Sequence[str],
    *,
    seed: int,
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
            prediction, target, folds
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
                rank_options=((int(ranks[0]),), (int(ranks[1]),)),
            )
            receiver_only = fit_feature_set(
                [receiver],
                target,
                batteries,
                seed=seed + 100 + site_index * 20 + direction_index,
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
    site = cache["sites"][int(cache["interface_site"])]
    h = site["base"].float()
    h_a = site["branch_a"].float()
    h_b = site["branch_b"].float()
    m = (h_a + h_b) / 2
    d = (h_a - h_b) / 2
    feature_blocks = {"fs0_h": [h], "fs_m": [m], "fs1_md": [m, d], "fs_ab": [h_a, h_b]}
    seed_result: dict[str, Any] = {
        "seed": seed,
        "rows": len(item_ids),
        "inputs": {
            "cache_sha256": sha256_file(cache_path),
            "target_sha256": sha256_file(target_path),
            "input_features": cache["feature_list"],
        },
        "fs2": {
            "status": "blocked_by_leak_boundary",
            "reason": "banked W3 loop states were extracted from forced-target prompt-plus-gold sequences",
        },
        "targets": {},
        "a3_composition": composition_desk_item(args.phase_b_private, seed),
    }
    models = {}
    for target_index, family in enumerate(("l0a", secondary_family)):
        target = align_target(targets, item_ids, family)
        md = fit_feature_set(feature_blocks["fs1_md"], target, batteries, seed=20260825 + seed * 100 + target_index)
        ranks = md["fit"]["selected"]["ranks"]
        total_rank = int(sum(ranks))
        controls = {
            "fs0_h": fit_feature_set([h], target, batteries, seed=20261025 + seed * 100 + target_index, rank_options=((total_rank,),)),
            "fs_m": fit_feature_set([m], target, batteries, seed=20261225 + seed * 100 + target_index, rank_options=((total_rank,),)),
            "fs_ab": fit_feature_set([h_a, h_b], target, batteries, seed=20261425 + seed * 100 + target_index, rank_options=((ranks[0],), (ranks[1],))),
        }
        fits = {"fs1_md": md, **controls}
        risk_reduction = paired_relative_risk_reduction(
            controls["fs0_h"]["row_risk"],
            md["row_risk"],
            seed=20261625 + seed * 100 + target_index,
        )
        permutation = deterministic_derangement(len(item_ids), tag=f"seed{seed}:{family}")
        selected_model = final_model(
            feature_blocks["fs1_md"], target, md["fit"]["selected"]
        )
        shuffled_m = (h_a + h_b[permutation]) / 2
        shuffled_d = (h_a - h_b[permutation]) / 2
        frozen_branch_prediction = selected_model.predict([shuffled_m, shuffled_d])
        frozen_permuted_prediction = selected_model.predict([m[permutation], d[permutation]])
        d3 = {
            "permutation_sha256": hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
            "frozen_map_branch_pair_shuffle_risk": bootstrap_mean_interval(
                normalized_row_risk(frozen_branch_prediction, target), seed=20261825 + seed
            ),
            "frozen_map_permuted_input_risk": bootstrap_mean_interval(
                normalized_row_risk(frozen_permuted_prediction, target), seed=20261925 + seed
            ),
        }
        clean = {}
        for name, value in fits.items():
            clean[name] = {
                key: item
                for key, item in value.items()
                if key not in {"prediction", "folds", "row_risk"}
            }
        conditional = float(md["conditional_cosine"]["pooled"]["mean"])
        seed_result["targets"][family] = {
            "primary": family == "l0a",
            "matched_total_rank": total_rank,
            "fits": clean,
            "hemispheric_relative_risk_reduction": risk_reduction,
            "g_d1_pass": conditional >= CONDITIONAL_COSINE_GATE,
            "g_d2_pass": float(risk_reduction["relative_risk_reduction"]) >= HEMISPHERIC_RISK_REDUCTION_GATE,
            "d3": d3,
            "d4_site_screen": site_incremental_table(
                cache["sites"],
                target,
                batteries,
                ranks=ranks,
                seed=20262025 + seed * 1000 + target_index * 400,
            ),
        }
        models[family] = {
            "kind": "paper2_bicameral_w2p_selected_map_v1",
            "seed": seed,
            "target_family": family,
            "feature_set": "fs1_md",
            "item_ids": item_ids,
            "batteries": batteries,
            "hyperparameters": md["fit"]["selected"],
            "model": serialize_model(selected_model),
            "input_provenance": "student_prompt_only",
        }
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    secondary_binding = lock["phase_d"]["secondary_target_binding"]
    if secondary_binding["status"] != "RESOLVED":
        raise RuntimeError(
            "W2-prime secondary target is blocked by the L0c/L0d authority conflict; "
            "strategy must bind the target before D1-D3"
        )
    secondary_family = str(secondary_binding["resolved_family"])
    if secondary_family not in {"l0c", "l0d"}:
        raise RuntimeError("W2-prime resolved secondary target must be l0c or l0d")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    seeds = {}
    for seed in (0, 1):
        result, models = run_seed(args, seed, secondary_family=secondary_family)
        seeds[str(seed)] = result
        for family, payload in models.items():
            torch.save(payload, args.model_dir / f"seed_{seed}_{family}_fs1_md.pt")
    primary_d1 = [bool(seeds[str(seed)]["targets"]["l0a"]["g_d1_pass"]) for seed in (0, 1)]
    primary_d2 = [bool(seeds[str(seed)]["targets"]["l0a"]["g_d2_pass"]) for seed in (0, 1)]
    secondary_pass = [
        bool(seeds[str(seed)]["targets"][secondary_family]["g_d1_pass"])
        and bool(seeds[str(seed)]["targets"][secondary_family]["g_d2_pass"])
        for seed in (0, 1)
    ]
    if all(primary_d1) and all(primary_d2):
        key = "DESK-DOUBLE-PASS"
    elif all(secondary_pass) and not (all(primary_d1) and all(primary_d2)):
        key = "TARGET-FAMILY-SPLIT"
    elif not all(primary_d1):
        key = "MAP-MIRAGE" if primary_d1[0] == primary_d1[1] else "SEED-SPLIT-G-D1"
    else:
        key = "HEMISPHERES-UNINFORMATIVE" if primary_d2[0] == primary_d2[1] else "SEED-SPLIT-G-D2"
    payload = {
        "kind": KIND,
        "status": "complete_cpu_only",
        "authority_sha256": AUTHORITY_SHA256,
        "rank_grid": [2, 4, 8, 16, 32],
        "ridge_multiplier_grid": list(RIDGE_MULTIPLIER_GRID),
        "ridge_scaling": "multiplier_times_mean_diagonal_xtx",
        "selection_rule": "nested_blockwise_inner_cv_then_joint_refit",
        "outer_folds": 4,
        "inner_folds": 3,
        "secondary_target_family": secondary_family,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "fs2_status": "blocked_by_leak_boundary_pending_strategy_clarification",
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
