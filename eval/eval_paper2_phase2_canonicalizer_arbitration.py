"""CPU-only canonicalizer arbitration from cached Stage 0A/Experiment 0A artifacts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

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
    _transform,
    load_stage0a_arrays,
)
from training.paper2_phase2_stage0a import sha256_file
from training.paper2_phase2_stage0ab import (
    WHITEN_EPS_ABS,
    WHITEN_TAU,
    build_anchor_targets,
    document_split,
)


BOOTSTRAP_SEED = 20260804
BOOTSTRAP_REPLICATES = 10_000
PRIMARY_ALPHA = 0.5
FIT_SEEDS = (20260814, 20260824, 20260834)
PREVIOUS_EDGE = 0.0068
FIT_PROTOCOL_VERSION = "seed_controlled_mixture_rrr_v2"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def paired_bootstrap_ci(
    delta: torch.Tensor,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 64,
) -> dict[str, float | int]:
    """Percentile paired bootstrap over rows with bounded working memory."""

    values = delta.detach().double().cpu().flatten()
    if values.numel() < 2:
        raise ValueError("paired bootstrap requires at least two rows")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("paired bootstrap values must be finite")
    generator = torch.Generator().manual_seed(int(seed))
    means = torch.empty(int(replicates), dtype=torch.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        indices = torch.randint(
            values.numel(),
            (stop - start, values.numel()),
            generator=generator,
        )
        means[start:stop] = values[indices].mean(dim=1)
    return {
        "rows": int(values.numel()),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(values.mean()),
        "ci95_low": float(torch.quantile(means, 0.025)),
        "ci95_high": float(torch.quantile(means, 0.975)),
    }


def arbitration_decision(
    *,
    agreement_ci: tuple[float, float],
    stratum_deltas: dict[str, float],
    fit_noise_comparable_to_previous_edge: bool = False,
) -> dict[str, Any]:
    """Apply the strategy-locked agreement/sign/parsimony rule."""

    low, high = map(float, agreement_ci)
    positive_ci = low > 0.0
    positive_strata = bool(stratum_deltas) and all(value > 0.0 for value in stratum_deltas.values())
    if fit_noise_comparable_to_previous_edge:
        return {
            "primary": "uniform_mixture_rrr",
            "fallback": "learned_mixture_rrr_optional_pilot_ablation",
            "reason": "multi_seed_fit_noise_is_comparable_to_prior_0p68pp_edge_parsimony_wins",
        }
    if positive_ci and positive_strata:
        return {
            "primary": "learned_mixture_rrr",
            "fallback": "uniform_mixture_rrr",
            "reason": "agreement_ci_excludes_zero_positive_and_sign_is_consistent",
        }
    return {
        "primary": "uniform_mixture_rrr",
        "fallback": "learned_mixture_rrr_optional_pilot_ablation",
        "reason": "paired_edge_not_stable_under_locked_rule_parsimony_wins",
    }


def eigenvalue_floor_support(
    effective: torch.Tensor, *, tau: float = WHITEN_TAU
) -> dict[str, Any]:
    """Audit floor mass from legacy effective eigenvalues.

    Legacy 0A artifacts did not persist raw eigenvalues. Equality to the clamp
    image is recoverable; the strict raw-below-floor fraction is not.
    """

    values = effective.detach().double().cpu().flatten()
    floor = float(values.max()) * float(tau)
    at_floor = torch.isclose(
        values, torch.tensor(floor, dtype=values.dtype), rtol=1e-5, atol=max(1e-12, floor * 1e-7)
    )
    return {
        "rank": int(values.numel()),
        "effective_floor": floor,
        "at_floor_count": int(at_floor.sum()),
        "floored_fraction": float(at_floor.double().mean()),
        "raw_fraction_recoverable": False,
        "legacy_receipt_scope": "effective clamp-image mass; exact raw fraction recomputed in refit",
    }


def _validate_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != receipt["sha256"]:
        raise RuntimeError(f"artifact hash mismatch: {path}")


def _row_metrics(
    *,
    z: torch.Tensor,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    horizons: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_log_probs: torch.Tensor,
    lm_head: torch.Tensor,
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    rows = torch.arange(z.shape[0], device=z.device)
    selected = z[rows, horizons.long() - 1]
    agreement: list[torch.Tensor] = []
    kl: list[torch.Tensor] = []
    for start in range(0, z.shape[0], batch_size):
        stop = min(z.shape[0], start + batch_size)
        predicted_hidden = selected[start:stop] @ decoder + decoder_bias
        ids = topk_ids[start:stop].long()
        token_weights = lm_head.index_select(0, ids.reshape(-1)).view(
            stop - start, ids.shape[1], -1
        )
        predicted_logits = torch.einsum(
            "bd,bkd->bk", predicted_hidden.to(token_weights.dtype), token_weights
        ).float()
        teacher_log = torch.log_softmax(topk_log_probs[start:stop].float(), dim=-1)
        predicted_log = torch.log_softmax(predicted_logits, dim=-1)
        kl.append((teacher_log.exp() * (teacher_log - predicted_log)).sum(dim=-1).cpu())
        predicted_ids = ids.gather(1, predicted_logits.argmax(dim=1, keepdim=True)).squeeze(1)
        agreement.append(predicted_ids.eq(ids[:, 0]).float().cpu())
    return {"teacher_top1": torch.cat(agreement), "future_kl": torch.cat(kl)}


def _metrics_from_legacy_artifact(
    *,
    artifact: dict[str, Any],
    endpoint: dict[str, Any],
    targets: dict[str, Any],
    lm_head: torch.Tensor,
) -> dict[str, torch.Tensor]:
    screening = artifact["screening"]
    raw = endpoint["raw_centered"].float()
    basis = endpoint["whiten_basis"].float()
    eigenvalues = endpoint["whiten_eigenvalues"].float()
    z = (raw @ basis) * eigenvalues.pow(-0.5 * PRIMARY_ALPHA)
    decoder = screening["decoders"][str(PRIMARY_ALPHA)]
    return _row_metrics(
        z=z,
        decoder=decoder["weight"].float(),
        decoder_bias=decoder["bias"].float(),
        horizons=targets["horizons"],
        topk_ids=targets["topk_ids"],
        topk_log_probs=targets["topk_log_probs"],
        lm_head=lm_head,
    )


def _paired_report(
    *,
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    strata: Sequence[str],
    seed_offset: int,
) -> dict[str, Any]:
    agreement_delta = candidate["teacher_top1"] - baseline["teacher_top1"]
    kl_delta = candidate["future_kl"] - baseline["future_kl"]
    strata_tensor: dict[str, Any] = {}
    agreement_by_stratum: dict[str, float] = {}
    for stratum in sorted(set(strata)):
        mask = torch.tensor([value == stratum for value in strata], dtype=torch.bool)
        agreement_by_stratum[stratum] = float(agreement_delta[mask].mean())
        strata_tensor[stratum] = {
            "rows": int(mask.sum()),
            "delta_teacher_top1_agreement": agreement_by_stratum[stratum],
            "delta_future_kl": float(kl_delta[mask].mean()),
        }
    agreement_bootstrap = paired_bootstrap_ci(
        agreement_delta, seed=BOOTSTRAP_SEED + seed_offset
    )
    kl_bootstrap = paired_bootstrap_ci(kl_delta, seed=BOOTSTRAP_SEED + seed_offset + 1)
    decision = arbitration_decision(
        agreement_ci=(
            float(agreement_bootstrap["ci95_low"]),
            float(agreement_bootstrap["ci95_high"]),
        ),
        stratum_deltas=agreement_by_stratum,
    )
    return {
        "delta_definition": "candidate_minus_uniform_predictive_rrr; lower future_kl is better",
        "teacher_top1_agreement": agreement_bootstrap,
        "future_kl": kl_bootstrap,
        "by_stratum": strata_tensor,
        "sign_consistent_positive_agreement": all(value > 0 for value in agreement_by_stratum.values()),
        "locked_rule_decision": decision,
    }


def _load_legacy(
    *, exp0a_summary: dict[str, Any], method: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = exp0a_summary["methods"][method]
    artifact_path = Path(receipt["artifact"]["path"])
    endpoint_path = Path(receipt["holdout_endpoints"]["path"])
    _validate_receipt(artifact_path, receipt["artifact"])
    _validate_receipt(endpoint_path, receipt["holdout_endpoints"])
    return (
        torch.load(artifact_path, map_location="cpu", weights_only=False),
        torch.load(endpoint_path, map_location="cpu", weights_only=False),
    )


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _fit_arm_seed(
    *,
    arm: str,
    seed: int,
    pooled_states: torch.Tensor,
    calibration: torch.Tensor,
    sample_targets: torch.Tensor,
    arrays: dict[str, Any],
    targets: dict[str, Any],
    lm_head: torch.Tensor,
    cache_path: Path,
) -> dict[str, Any]:
    """Fit one RRR arm/seed, scoring and caching enough state to resume."""

    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached.get("protocol_version") != FIT_PROTOCOL_VERSION
            or cached.get("arm") != arm
            or int(cached.get("seed", -1)) != int(seed)
        ):
            raise RuntimeError(f"resume cache metadata mismatch: {cache_path}")
        print(f"arbitration_resume arm={arm} seed={seed} path={cache_path}", flush=True)
        return cached

    print(f"arbitration_fit_start arm={arm} seed={seed}", flush=True)
    mean, projector, fit_receipt = _fit_projector(
        pooled_states[calibration],
        sample_targets[calibration],
        method="predictive_rrr",
        seed=int(seed),
    )
    canonical_mean, basis, effective, condition = _canonical_statistics(
        pooled_states[calibration], mean, projector
    )
    z_calibration = _transform(
        pooled_states[calibration],
        teacher_mean=mean,
        projector=projector,
        canonical_mean=canonical_mean,
        basis=basis,
        eigenvalues=effective,
        alpha=PRIMARY_ALPHA,
    )
    z_holdout = _transform(
        pooled_states[~calibration],
        teacher_mean=mean,
        projector=projector,
        canonical_mean=canonical_mean,
        basis=basis,
        eigenvalues=effective,
        alpha=PRIMARY_ALPHA,
    )
    decoder, decoder_bias = _fit_probe(
        z_calibration, arrays["final_hidden"][calibration], arrays["horizons"][calibration]
    )
    metrics = _row_metrics(
        z=z_holdout,
        decoder=decoder,
        decoder_bias=decoder_bias,
        horizons=targets["horizons"],
        topk_ids=targets["topk_ids"],
        topk_log_probs=targets["topk_log_probs"],
        lm_head=lm_head,
    )
    payload = {
        "kind": "paper2_phase2_seed_controlled_mixture_rrr_fit",
        "protocol_version": FIT_PROTOCOL_VERSION,
        "arm": arm,
        "seed": int(seed),
        "alpha": PRIMARY_ALPHA,
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
        "training_started": False,
        "optimizer_steps": 0,
    }
    _save_torch_atomic(cache_path, payload)
    print(
        f"arbitration_fit_complete arm={arm} seed={seed} "
        f"agreement={float(metrics['teacher_top1'].mean()):.6f}",
        flush=True,
    )
    return payload


def _average_metrics(fits: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not fits:
        raise ValueError("at least one fit is required")
    return {
        metric: torch.stack([fit["metrics"][metric].float() for fit in fits]).mean(dim=0)
        for metric in ("teacher_top1", "future_kl")
    }


def run_arbitration(
    *,
    stage0a_private: Path,
    exp0a_summary_path: Path,
    output_private: Path,
    output_summary: Path,
) -> dict[str, Any]:
    started = time.time()
    exp0a = json.loads(exp0a_summary_path.read_text(encoding="utf-8"))
    if exp0a.get("status") != "complete_development_only":
        raise RuntimeError("canonicalizer arbitration requires the completed 0A receipt")
    target_receipt = exp0a["holdout_targets"]
    targets_path = Path(target_receipt["path"])
    _validate_receipt(targets_path, target_receipt)
    targets = torch.load(targets_path, map_location="cpu", weights_only=False)

    samples = _load_samples(stage0a_private / "sample_manifest.jsonl")
    holdout_indices = targets["sample_indices"].long()
    holdout_strata = [str(samples[index]["stratum"]) for index in holdout_indices.tolist()]
    if len(holdout_strata) != int(exp0a["split"]["holdout_samples"]):
        raise RuntimeError("holdout rows do not match the document-disjoint 0A split")

    model_summary = json.loads(
        (stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    lm_head_path = Path(model_summary["lm_head"]["path"])
    _validate_receipt(lm_head_path, model_summary["lm_head"])
    lm_head = torch.load(lm_head_path, map_location="cpu", weights_only=False)["weight_bfloat16"]

    uniform_artifact, uniform_endpoint = _load_legacy(
        exp0a_summary=exp0a, method="predictive_rrr"
    )
    mixture_artifact, mixture_endpoint = _load_legacy(
        exp0a_summary=exp0a, method="tucker_predictive"
    )
    uniform_metrics = _metrics_from_legacy_artifact(
        artifact=uniform_artifact, endpoint=uniform_endpoint, targets=targets, lm_head=lm_head
    )
    legacy_mixture_metrics = _metrics_from_legacy_artifact(
        artifact=mixture_artifact, endpoint=mixture_endpoint, targets=targets, lm_head=lm_head
    )
    legacy_pair = _paired_report(
        candidate=legacy_mixture_metrics,
        baseline=uniform_metrics,
        strata=holdout_strata,
        seed_offset=0,
    )

    # The legacy arm named tucker_predictive is predictive RRR after a learned
    # layer mixture. Refit both mixtures with the identical three-seed set so
    # mixture is the only between-arm variable and fit noise is visible.
    arrays = load_stage0a_arrays(stage0a_private)
    documents = [str(row["document_id"]) for row in arrays["samples"]]
    calibration = document_split(
        documents, calibration_fraction=CALIBRATION_FRACTION, seed=SPLIT_SEED
    )
    anchor_indices = torch.tensor([int(row["anchor_index"]) for row in arrays["samples"]])
    horizons = torch.tensor([int(row["horizon"]) for row in arrays["samples"]])
    arrays["horizons"] = horizons
    anchor_targets, _slot_mask = build_anchor_targets(
        topk_ids=arrays["topk_ids"],
        topk_log_probs=arrays["topk_log_probs"],
        middle_states=arrays["states"][:, 1],
        horizons=horizons,
        anchor_indices=anchor_indices,
        anchor_count=int(anchor_indices.max()) + 1,
        latent_dim=LATENT_DIM,
        n_slots=N_SLOTS,
        seed=SPLIT_SEED,
    )
    sample_targets = anchor_targets.index_select(0, anchor_indices).reshape(len(samples), -1)
    learned_weights = mixture_artifact["layer_weights"].float()
    output_private.mkdir(parents=True, exist_ok=True)
    arm_weights = {
        "uniform_mixture_rrr": torch.full((3,), 1 / 3),
        "learned_mixture_rrr": learned_weights,
    }
    fits: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arm_weights}
    fit_receipts: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arm_weights}
    for arm, weights in arm_weights.items():
        pooled_states = _pool(arrays["states"], weights)
        for seed in FIT_SEEDS:
            cache_path = output_private / f"{arm}_seed_{seed}.pt"
            fit = _fit_arm_seed(
                arm=arm,
                seed=seed,
                pooled_states=pooled_states,
                calibration=calibration,
                sample_targets=sample_targets,
                arrays=arrays,
                targets=targets,
                lm_head=lm_head,
                cache_path=cache_path,
            )
            fits[arm].append(fit)
            fit_receipts[arm].append(
                {
                    "seed": seed,
                    "fit": fit["fit"],
                    "screening_condition": fit["screening_condition"],
                    "teacher_top1_agreement": float(fit["metrics"]["teacher_top1"].mean()),
                    "future_kl": float(fit["metrics"]["future_kl"].mean()),
                    "artifact": {
                        "path": str(cache_path),
                        "sha256": sha256_file(cache_path),
                        "bytes": cache_path.stat().st_size,
                    },
                }
            )
        del pooled_states

    per_seed_pairs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(FIT_SEEDS):
        per_seed_pairs.append(
            {
                "seed": seed,
                "comparison": _paired_report(
                    candidate=fits["learned_mixture_rrr"][seed_index]["metrics"],
                    baseline=fits["uniform_mixture_rrr"][seed_index]["metrics"],
                    strata=holdout_strata,
                    seed_offset=100 + seed_index * 10,
                ),
            }
        )

    averaged = {arm: _average_metrics(arm_fits) for arm, arm_fits in fits.items()}
    seed_averaged_pair = _paired_report(
        candidate=averaged["learned_mixture_rrr"],
        baseline=averaged["uniform_mixture_rrr"],
        strata=holdout_strata,
        seed_offset=200,
    )
    agreement_by_arm = {
        arm: [float(fit["metrics"]["teacher_top1"].mean()) for fit in arm_fits]
        for arm, arm_fits in fits.items()
    }
    agreement_ranges = {
        arm: max(values) - min(values) for arm, values in agreement_by_arm.items()
    }
    delta_by_seed = [
        agreement_by_arm["learned_mixture_rrr"][index]
        - agreement_by_arm["uniform_mixture_rrr"][index]
        for index in range(len(FIT_SEEDS))
    ]
    max_arm_range = max(agreement_ranges.values())
    fit_noise_comparable = max_arm_range >= PREVIOUS_EDGE
    pooled_stratum_deltas = {
        stratum: float(values["delta_teacher_top1_agreement"])
        for stratum, values in seed_averaged_pair["by_stratum"].items()
    }
    agreement_ci = seed_averaged_pair["teacher_top1_agreement"]
    canonicalizer_decision = arbitration_decision(
        agreement_ci=(float(agreement_ci["ci95_low"]), float(agreement_ci["ci95_high"])),
        stratum_deltas=pooled_stratum_deltas,
        fit_noise_comparable_to_previous_edge=fit_noise_comparable,
    )

    floor_support: dict[str, Any] = {
        "predictive_rrr_legacy": eigenvalue_floor_support(
            uniform_artifact["screening"]["whiten_eigenvalues"]
        ),
        "learned_mixture_rrr_legacy": eigenvalue_floor_support(
            mixture_artifact["screening"]["whiten_eigenvalues"]
        ),
    }
    for arm, arm_fits in fits.items():
        floor_support[arm] = [
            {
                "seed": int(fit["seed"]),
                **eigenvalue_floor_support(fit["whiten_eigenvalues"]),
                "condition_receipt": fit["screening_condition"],
                "raw_fraction_recoverable": True,
                "floored_fraction": float(fit["screening_condition"]["floored_fraction"]),
            }
            for fit in arm_fits
        ]

    result = {
        "kind": "paper2_phase2_canonicalizer_arbitration",
        "status": "complete_development_only",
        "source_exp0a_summary_sha256": sha256_file(exp0a_summary_path),
        "semantic_correction": {
            "legacy_label": "tucker_predictive",
            "actual_estimator": "predictive_rrr_after_learned_layer_mixture",
            "separate_tucker_factorization_present": False,
            "legacy_seed_confound": True,
            "future_label": "learned_mixture_rrr",
            "consequence": "paired same-seed mixture comparison governs canonicalizer arbitration",
        },
        "locked_analysis_constants": {
            "primary_alpha": PRIMARY_ALPHA,
            "fit_seeds": list(FIT_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "prior_edge_teacher_top1": PREVIOUS_EDGE,
            "fit_noise_comparable_operationalization": "maximum within-arm three-seed agreement range >= 0.0068",
            "document_disjoint_holdout": True,
            "eps_abs_future_fits": WHITEN_EPS_ABS,
            "tau": WHITEN_TAU,
        },
        "decomposition_choice": {
            "method": "common-seed randomized low-rank SVD with three-seed spread",
            "exact_lapack_used": False,
            "reason": "exact decomposition of the 166708-by-5120 fit matrix is not practical for the authorized cached CPU job",
        },
        "legacy_mixture_rrr_vs_uniform_rrr": legacy_pair,
        "seed_controlled_pairs": per_seed_pairs,
        "seed_averaged_pair": seed_averaged_pair,
        "fit_variability": {
            "teacher_top1_agreement_by_arm": agreement_by_arm,
            "within_arm_ranges": agreement_ranges,
            "delta_learned_minus_uniform_by_seed": delta_by_seed,
            "delta_range": max(delta_by_seed) - min(delta_by_seed),
            "max_within_arm_range": max_arm_range,
            "comparable_to_prior_0p68pp_edge": fit_noise_comparable,
        },
        "canonicalizer_decision": canonicalizer_decision,
        "floored_fraction_audit": floor_support,
        "learned_layer_mixture": learned_weights.tolist(),
        "seed_controlled_refits": fit_receipts,
        "elapsed_seconds": time.time() - started,
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "do_not_claim": [
            "the legacy tucker_predictive arm used a distinct Tucker factorization",
            "the legacy 0.68-point edge was a mixture effect before seed-controlled arbitration",
            "the canonicalizer arbitration selects whitening alpha",
            "development-only probe agreement is downstream acceptance",
            "legacy effective-floor mass alone recovers strict raw-below-floor counts",
        ],
    }
    write_json(output_summary, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--exp0a_summary", type=Path, required=True)
    parser.add_argument("--output_private", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_arbitration(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
