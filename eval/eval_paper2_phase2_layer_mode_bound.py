"""Cached CPU layer-mode bound using the corrected randomized-fit protocol."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from eval.eval_paper2_phase2_canonicalizer_arbitration import (
    BOOTSTRAP_SEED,
    FIT_SEEDS,
    PRIMARY_ALPHA,
    _average_metrics,
    _load_legacy,
    _row_metrics,
    _save_torch_atomic,
    _transform,
    paired_bootstrap_ci,
)
from eval.eval_paper2_phase2_exp0a import (
    CALIBRATION_FRACTION,
    LATENT_DIM,
    N_SLOTS,
    SPLIT_SEED,
    _canonical_statistics,
    _fit_probe,
    _fit_projector,
    _load_samples,
    _pool,
    _rms_unit,
    load_stage0a_arrays,
)
from training.paper2_phase2_stage0a import sha256_file
from training.paper2_phase2_stage0ab import build_anchor_targets, document_split


FIT_PROTOCOL_VERSION = "layer_mode_bound_randomized_rrr_r2"
SPREAD_GATE = 0.0025
SWAP_CI_LOWER_BOUND = 0.005
REGISTERED_ROWS = 166_708
REGISTERED_WIDTH = 15_360
REGISTERED_RANK = 256
REGISTERED_ADDENDUM_DRIVE_ID = "1LL8djMUG9o17iS4ctMTRCQ_T4jn7mXy1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resource_plan(*, rows: int, width: int, rank: int) -> dict[str, Any]:
    """Record why the r2 randomized design path is mandatory."""

    return {
        "method": "randomized_low_rank_svd_streamed_design",
        "rows": int(rows),
        "width": int(width),
        "rank": int(rank),
        "fit_seeds": list(FIT_SEEDS),
        "dense_covariance_materialized": False,
        "dense_covariance_bytes_fp64": int(width) * int(width) * 8,
        "dense_covariance_formation_mac_proxy": int(rows) * int(width) * int(width),
        "dense_eigendecomposition_flop_proxy": int(width) ** 3,
        "design_matrix_bytes_fp32": int(rows) * int(width) * 4,
        "expected_relative_cost_vs_5120_width_fit": float(width) / 5120.0,
        "strategy_revision": "layer_mode_bound_r2_lapack_route_superseded",
    }


def build_concat_design(
    states: torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Per-layer RMS-normalize, then concatenate without a dense covariance."""

    if states.ndim != 3 or states.shape[1] != 3:
        raise ValueError("layer-mode bound requires [rows, 3, width] teacher states")
    if indices is None:
        indices = torch.arange(states.shape[0], device=states.device)
    indices = indices.long()
    design = torch.empty(
        (indices.numel(), states.shape[1] * states.shape[2]),
        dtype=torch.float32,
        device=states.device,
    )
    for start in range(0, indices.numel(), int(chunk_size)):
        stop = min(indices.numel(), start + int(chunk_size))
        selected = states.index_select(0, indices[start:stop])
        design[start:stop] = _rms_unit(selected.float()).reshape(
            stop - start, -1
        )
        if start == 0 or stop == indices.numel() or (start // int(chunk_size)) % 32 == 0:
            print(f"layer_mode_concat_progress rows={stop}/{indices.numel()}", flush=True)
    return design


def layer_mode_decision(
    *,
    agreement_ci: tuple[float, float],
    stratum_deltas: dict[str, float],
    concat_agreements: Sequence[float],
    learned_agreements: Sequence[float] = (),
) -> dict[str, Any]:
    if len(concat_agreements) != len(FIT_SEEDS):
        raise ValueError("layer-mode decision requires exactly three concat fits")
    concat_spread = max(map(float, concat_agreements)) - min(map(float, concat_agreements))
    learned_spread = (
        max(map(float, learned_agreements)) - min(map(float, learned_agreements))
        if learned_agreements
        else 0.0
    )
    spread = max(concat_spread, learned_spread)
    if spread >= SPREAD_GATE:
        return {
            "primary": "learned_mixture_rrr",
            "reason": "concat_fit_spread_exceeds_0p25pp_noise_gate",
            "swap_branch_expired": True,
            "authorize_tucker_layer_rank_2": False,
            "max_within_concat_agreement_range": concat_spread,
            "max_within_learned_agreement_range": learned_spread,
            "max_within_arm_agreement_range": spread,
        }
    sign_consistent = bool(stratum_deltas) and all(
        float(delta) > 0.0 for delta in stratum_deltas.values()
    )
    if float(agreement_ci[0]) >= SWAP_CI_LOWER_BOUND and sign_consistent:
        return {
            "primary": "concat_rrr",
            "reason": "paired_ci_lower_at_least_0p5pp_and_both_strata_positive",
            "swap_branch_expired": False,
            "authorize_tucker_layer_rank_2": True,
            "max_within_concat_agreement_range": concat_spread,
            "max_within_learned_agreement_range": learned_spread,
            "max_within_arm_agreement_range": spread,
        }
    return {
        "primary": "learned_mixture_rrr",
        "reason": "concat_did_not_clear_locked_0p5pp_ci_and_stratum_rule",
        "swap_branch_expired": False,
        "authorize_tucker_layer_rank_2": False,
        "max_within_concat_agreement_range": concat_spread,
        "max_within_learned_agreement_range": learned_spread,
        "max_within_arm_agreement_range": spread,
    }


def _validate_artifact(path: Path, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != sha256:
        raise RuntimeError(f"artifact hash mismatch: {path}")


def _calibration_metrics(
    *,
    fit: dict[str, Any],
    design: torch.Tensor,
    calibration: torch.Tensor,
    arrays: dict[str, Any],
    lm_head: torch.Tensor,
) -> dict[str, torch.Tensor]:
    z = _transform(
        design[calibration],
        teacher_mean=fit["teacher_mean"].float(),
        projector=fit["projector_weight"].float(),
        canonical_mean=fit["canonical_mean"].float(),
        basis=fit["whiten_basis"].float(),
        eigenvalues=fit["whiten_eigenvalues"].float(),
        alpha=PRIMARY_ALPHA,
    )
    return _row_metrics(
        z=z,
        decoder=fit["decoder_weight"].float(),
        decoder_bias=fit["decoder_bias"].float(),
        horizons=arrays["horizons"][calibration],
        topk_ids=arrays["topk_ids"][calibration],
        topk_log_probs=arrays["topk_log_probs"][calibration],
        lm_head=lm_head,
    )


def _metric_summary(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "teacher_top1_agreement": float(metrics["teacher_top1"].float().mean()),
        "future_kl": float(metrics["future_kl"].float().mean()),
    }


def _rank_spectrum(projector: torch.Tensor) -> dict[str, Any]:
    singular = torch.linalg.svdvals(projector.float())
    energy = singular.square()
    fraction = energy / energy.sum().clamp_min(1e-12)
    cumulative = fraction.cumsum(dim=0)
    effective_rank = torch.exp(
        -(fraction * fraction.clamp_min(1e-30).log()).sum()
    )
    thresholds = {}
    for threshold in (0.5, 0.9, 0.95, 0.99):
        thresholds[str(threshold)] = int(
            torch.searchsorted(cumulative, torch.tensor(threshold)).item() + 1
        )
    return {
        "singular_value_count": int(singular.numel()),
        "numerical_rank_rtol_1e_6": int(
            (singular > singular.max() * 1e-6).sum()
        ),
        "entropy_effective_rank": float(effective_rank),
        "energy_rank": thresholds,
        "top_32_singular_values": singular[:32].tolist(),
    }


def _load_learned_fits(
    arbitration_summary: dict[str, Any], arbitration_private: Path
) -> list[dict[str, Any]]:
    receipts = arbitration_summary["seed_controlled_refits"]["learned_mixture_rrr"]
    by_seed = {int(row["seed"]): row for row in receipts}
    fits = []
    for seed in FIT_SEEDS:
        receipt = by_seed[int(seed)]["artifact"]
        path = arbitration_private / f"learned_mixture_rrr_seed_{seed}.pt"
        _validate_artifact(path, receipt["sha256"])
        fit = torch.load(path, map_location="cpu", weights_only=False)
        if fit.get("arm") != "learned_mixture_rrr" or int(fit.get("seed", -1)) != seed:
            raise RuntimeError(f"learned-mixture artifact metadata mismatch: {path}")
        fits.append(fit)
    return fits


def _fit_concat_seed(
    *,
    seed: int,
    calibration_design: torch.Tensor,
    holdout_design: torch.Tensor,
    calibration_targets: torch.Tensor,
    calibration_hidden: torch.Tensor,
    calibration_horizons: torch.Tensor,
    calibration_topk_ids: torch.Tensor,
    calibration_topk_log_probs: torch.Tensor,
    holdout_targets: dict[str, Any],
    lm_head: torch.Tensor,
    cache_path: Path,
) -> dict[str, Any]:
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached.get("protocol_version") != FIT_PROTOCOL_VERSION
            or cached.get("arm") != "concat_rrr"
            or int(cached.get("seed", -1)) != int(seed)
        ):
            raise RuntimeError(f"resume cache metadata mismatch: {cache_path}")
        print(f"layer_mode_resume arm=concat_rrr seed={seed} path={cache_path}", flush=True)
        return cached
    print(f"layer_mode_fit_start arm=concat_rrr seed={seed}", flush=True)
    mean, projector, fit_receipt = _fit_projector(
        calibration_design,
        calibration_targets,
        method="predictive_rrr",
        seed=int(seed),
    )
    canonical_mean, basis, effective, condition = _canonical_statistics(
        calibration_design, mean, projector
    )
    z_calibration = _transform(
        calibration_design,
        teacher_mean=mean,
        projector=projector,
        canonical_mean=canonical_mean,
        basis=basis,
        eigenvalues=effective,
        alpha=PRIMARY_ALPHA,
    )
    z_holdout = _transform(
        holdout_design,
        teacher_mean=mean,
        projector=projector,
        canonical_mean=canonical_mean,
        basis=basis,
        eigenvalues=effective,
        alpha=PRIMARY_ALPHA,
    )
    decoder, decoder_bias = _fit_probe(
        z_calibration, calibration_hidden, calibration_horizons
    )
    calibration_metrics = _row_metrics(
        z=z_calibration,
        decoder=decoder,
        decoder_bias=decoder_bias,
        horizons=calibration_horizons,
        topk_ids=calibration_topk_ids,
        topk_log_probs=calibration_topk_log_probs,
        lm_head=lm_head,
    )
    metrics = _row_metrics(
        z=z_holdout,
        decoder=decoder,
        decoder_bias=decoder_bias,
        horizons=holdout_targets["horizons"],
        topk_ids=holdout_targets["topk_ids"],
        topk_log_probs=holdout_targets["topk_log_probs"],
        lm_head=lm_head,
    )
    payload = {
        "kind": "paper2_phase2_layer_mode_concat_rrr_fit",
        "protocol_version": FIT_PROTOCOL_VERSION,
        "arm": "concat_rrr",
        "seed": int(seed),
        "alpha": PRIMARY_ALPHA,
        "input_mode": "three_per_layer_rms_normalized_states_concatenated",
        "teacher_mean": mean,
        "projector_weight": projector,
        "canonical_mean": canonical_mean,
        "whiten_basis": basis,
        "whiten_eigenvalues": effective,
        "decoder_weight": decoder,
        "decoder_bias": decoder_bias,
        "fit": fit_receipt,
        "screening_condition": condition,
        "metrics": metrics,
        "calibration_metrics": calibration_metrics,
        "training_started": False,
        "optimizer_steps": 0,
    }
    _save_torch_atomic(cache_path, payload)
    print(
        f"layer_mode_fit_complete arm=concat_rrr seed={seed} "
        f"agreement={float(metrics['teacher_top1'].mean()):.6f}",
        flush=True,
    )
    return payload


def run_layer_mode_bound(
    *,
    stage0a_private: Path,
    exp0a_summary_path: Path,
    arbitration_summary_path: Path,
    arbitration_private: Path,
    output_private: Path,
    output_summary: Path,
) -> dict[str, Any]:
    started = time.time()
    exp0a = json.loads(exp0a_summary_path.read_text(encoding="utf-8"))
    arbitration = json.loads(arbitration_summary_path.read_text(encoding="utf-8"))
    if arbitration.get("canonicalizer_decision", {}).get("primary") != "learned_mixture_rrr":
        raise RuntimeError("layer-mode bound requires the banked learned-mixture decision")
    arrays = load_stage0a_arrays(stage0a_private)
    samples = arrays["samples"]
    documents = [str(row["document_id"]) for row in samples]
    calibration = document_split(
        documents, calibration_fraction=CALIBRATION_FRACTION, seed=SPLIT_SEED
    )
    arrays["horizons"] = torch.tensor([int(row["horizon"]) for row in samples])
    anchor_indices = torch.tensor([int(row["anchor_index"]) for row in samples])
    anchor_targets, _slot_mask = build_anchor_targets(
        topk_ids=arrays["topk_ids"],
        topk_log_probs=arrays["topk_log_probs"],
        middle_states=arrays["states"][:, 1],
        horizons=arrays["horizons"],
        anchor_indices=anchor_indices,
        anchor_count=int(anchor_indices.max()) + 1,
        latent_dim=LATENT_DIM,
        n_slots=N_SLOTS,
        seed=SPLIT_SEED,
    )
    sample_targets = anchor_targets.index_select(0, anchor_indices).reshape(len(samples), -1)
    target_receipt = exp0a["holdout_targets"]
    targets_path = Path(target_receipt["path"])
    _validate_artifact(targets_path, target_receipt["sha256"])
    holdout_targets = torch.load(targets_path, map_location="cpu", weights_only=False)
    holdout_indices = holdout_targets["sample_indices"].long()
    holdout_strata = [str(samples[index]["stratum"]) for index in holdout_indices.tolist()]
    model_summary = json.loads(
        (stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    lm_head_path = Path(model_summary["lm_head"]["path"])
    _validate_artifact(lm_head_path, model_summary["lm_head"]["sha256"])
    lm_head = torch.load(lm_head_path, map_location="cpu", weights_only=False)["weight_bfloat16"]

    learned_fits = _load_learned_fits(arbitration, arbitration_private)
    mixture_artifact, _mixture_endpoint = _load_legacy(
        exp0a_summary=exp0a, method="tucker_predictive"
    )
    learned_design = _pool(arrays["states"], mixture_artifact["layer_weights"].float())
    learned_calibration = [
        _calibration_metrics(
            fit=fit,
            design=learned_design,
            calibration=calibration,
            arrays=arrays,
            lm_head=lm_head,
        )
        for fit in learned_fits
    ]
    del learned_design
    gc.collect()

    calibration_indices = torch.where(calibration)[0]
    holdout_indices_from_split = torch.where(~calibration)[0]
    if not torch.equal(holdout_indices_from_split, holdout_indices):
        raise RuntimeError("layer-mode holdout order differs from the frozen 0A target order")
    calibration_design = build_concat_design(
        arrays["states"], indices=calibration_indices
    )
    holdout_design = build_concat_design(
        arrays["states"], indices=holdout_indices_from_split
    )
    concat_fits = []
    concat_calibration = []
    output_private.mkdir(parents=True, exist_ok=True)
    for seed in FIT_SEEDS:
        cache_path = output_private / f"concat_rrr_seed_{seed}.pt"
        fit = _fit_concat_seed(
            seed=seed,
            calibration_design=calibration_design,
            holdout_design=holdout_design,
            calibration_targets=sample_targets[calibration],
            calibration_hidden=arrays["final_hidden"][calibration],
            calibration_horizons=arrays["horizons"][calibration],
            calibration_topk_ids=arrays["topk_ids"][calibration],
            calibration_topk_log_probs=arrays["topk_log_probs"][calibration],
            holdout_targets=holdout_targets,
            lm_head=lm_head,
            cache_path=cache_path,
        )
        concat_fits.append(fit)
        concat_calibration.append(fit["calibration_metrics"])

    learned_average = _average_metrics(learned_fits)
    concat_average = _average_metrics(concat_fits)
    agreement_delta = concat_average["teacher_top1"] - learned_average["teacher_top1"]
    kl_delta = concat_average["future_kl"] - learned_average["future_kl"]
    agreement_ci = paired_bootstrap_ci(agreement_delta, seed=BOOTSTRAP_SEED + 400)
    kl_ci = paired_bootstrap_ci(kl_delta, seed=BOOTSTRAP_SEED + 401)
    stratum_deltas = {}
    by_stratum = {}
    for stratum in sorted(set(holdout_strata)):
        mask = torch.tensor([value == stratum for value in holdout_strata], dtype=torch.bool)
        stratum_deltas[stratum] = float(agreement_delta[mask].mean())
        by_stratum[stratum] = {
            "rows": int(mask.sum()),
            "delta_teacher_top1_agreement": stratum_deltas[stratum],
            "delta_future_kl": float(kl_delta[mask].mean()),
        }
    concat_agreements = [float(fit["metrics"]["teacher_top1"].mean()) for fit in concat_fits]
    learned_agreements = [float(fit["metrics"]["teacher_top1"].mean()) for fit in learned_fits]
    decision = layer_mode_decision(
        agreement_ci=(float(agreement_ci["ci95_low"]), float(agreement_ci["ci95_high"])),
        stratum_deltas=stratum_deltas,
        concat_agreements=concat_agreements,
        learned_agreements=learned_agreements,
    )
    selected_fits = concat_fits if decision["primary"] == "concat_rrr" else learned_fits
    selected_seed_fit = selected_fits[0]
    selected_path = (
        output_private / f"concat_rrr_seed_{FIT_SEEDS[0]}.pt"
        if decision["primary"] == "concat_rrr"
        else arbitration_private / f"learned_mixture_rrr_seed_{FIT_SEEDS[0]}.pt"
    )
    calibration_by_arm = {
        "learned_mixture_rrr": [_metric_summary(value) for value in learned_calibration],
        "concat_rrr": [_metric_summary(value) for value in concat_calibration],
    }
    holdout_by_arm = {
        "learned_mixture_rrr": [_metric_summary(fit["metrics"]) for fit in learned_fits],
        "concat_rrr": [_metric_summary(fit["metrics"]) for fit in concat_fits],
    }
    gaps = {}
    for arm in calibration_by_arm:
        cal_agreement = sum(row["teacher_top1_agreement"] for row in calibration_by_arm[arm]) / 3
        hold_agreement = sum(row["teacher_top1_agreement"] for row in holdout_by_arm[arm]) / 3
        cal_kl = sum(row["future_kl"] for row in calibration_by_arm[arm]) / 3
        hold_kl = sum(row["future_kl"] for row in holdout_by_arm[arm]) / 3
        gaps[arm] = {
            "calibration_minus_holdout_teacher_top1_agreement": cal_agreement - hold_agreement,
            "calibration_minus_holdout_future_kl": cal_kl - hold_kl,
        }
    result = {
        "kind": "paper2_phase2_layer_mode_bound",
        "status": "complete_development_only_no_training",
        "governing_addendum": {
            "revision": "r2",
            "drive_id": REGISTERED_ADDENDUM_DRIVE_ID,
            "r1_dense_lapack_route_superseded": True,
        },
        "resource_plan": resource_plan(
            rows=int(calibration.sum()), width=REGISTERED_WIDTH, rank=REGISTERED_RANK
        ),
        "locked_constants": {
            "fit_seeds": list(FIT_SEEDS),
            "spread_gate": SPREAD_GATE,
            "swap_ci_lower_bound": SWAP_CI_LOWER_BOUND,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": PRIMARY_ALPHA,
            "rank": REGISTERED_RANK,
        },
        "fit_variability": {
            "concat_teacher_top1_agreement": concat_agreements,
            "learned_teacher_top1_agreement": learned_agreements,
            "concat_within_arm_range": max(concat_agreements) - min(concat_agreements),
            "learned_within_arm_range": max(learned_agreements) - min(learned_agreements),
            "max_within_arm_range": max(
                max(concat_agreements) - min(concat_agreements),
                max(learned_agreements) - min(learned_agreements),
            ),
            "spread_gate_pass": decision["max_within_arm_agreement_range"] < SPREAD_GATE,
        },
        "paired_holdout": {
            "delta_definition": "concat_rrr_minus_learned_mixture_rrr",
            "teacher_top1_agreement": agreement_ci,
            "future_kl": kl_ci,
            "by_stratum": by_stratum,
        },
        "calibration_metrics_by_seed": calibration_by_arm,
        "holdout_metrics_by_seed": holdout_by_arm,
        "calibration_minus_holdout_gap": gaps,
        "canonicalizer_decision": decision,
        "selected_artifact": {
            "arm": decision["primary"],
            "selection_seed": FIT_SEEDS[0],
            "selection_seed_rule": "first preregistered fit seed fixed before results",
            "path": str(selected_path),
            "sha256": sha256_file(selected_path),
            "bytes": selected_path.stat().st_size,
        },
        "selected_projector_rank_spectrum": _rank_spectrum(
            selected_seed_fit["projector_weight"]
        ),
        "source_hashes": {
            "exp0a_summary": sha256_file(exp0a_summary_path),
            "arbitration_summary": sha256_file(arbitration_summary_path),
        },
        "elapsed_seconds": time.time() - started,
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "do_not_claim": [
            "concat_rrr is a deterministic exact-LAPACK fit",
            "common seeds cancel sketch noise across differently shaped inputs",
            "concat_rrr universally upper-bounds every possible Tucker training procedure",
            "development-only probe agreement is downstream verified acceptance",
            "the layer-mode bound selects whitening alpha",
        ],
    }
    write_json(output_summary, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--exp0a_summary", dest="exp0a_summary_path", type=Path, required=True)
    parser.add_argument(
        "--arbitration_summary", dest="arbitration_summary_path", type=Path, required=True
    )
    parser.add_argument("--arbitration_private", type=Path, required=True)
    parser.add_argument("--output_private", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = run_layer_mode_bound(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
