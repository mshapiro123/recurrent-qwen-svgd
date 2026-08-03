"""DEV-only Experiment 0A canonicalizer and partial-whitening screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eval.cache_paper2_phase2_stage0a import _load_flat_shard
from training.paper2_phase2_stage0a import sha256_file
from training.paper2_phase2_stage0ab import (
    WHITEN_ALPHAS,
    WHITEN_EPS_ABS,
    WHITEN_TAU,
    build_anchor_targets,
    document_split,
    effective_eigenvalues,
    finite_quantiles,
)


N_SLOTS = 8
LATENT_DIM = 128
INTERNAL_RANK = 256
CALIBRATION_FRACTION = 0.8
SPLIT_SEED = 20260804
METHODS = ("pca", "predictive_rrr", "tucker_predictive")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rms_unit(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)


def _load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    samples.sort(key=lambda row: int(row["sample_index"]))
    if [int(row["sample_index"]) for row in samples] != list(range(len(samples))):
        raise RuntimeError("Experiment 0A sample manifest is not contiguous")
    return samples


def load_stage0a_arrays(private_dir: Path) -> dict[str, Any]:
    manifest_path = private_dir / "sample_manifest.jsonl"
    model_summary_path = private_dir / "model_cache/teacher_14b/summary.json"
    if not manifest_path.is_file() or not model_summary_path.is_file():
        raise FileNotFoundError("Experiment 0A requires the complete Stage 0A private cache")
    samples = _load_samples(manifest_path)
    summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    count = len(samples)
    width = 5120
    states = torch.empty((count, 3, width), dtype=torch.bfloat16)
    final_hidden = torch.empty((count, width), dtype=torch.bfloat16)
    topk_ids = torch.empty((count, 128), dtype=torch.int32)
    topk_log_probs = torch.empty((count, 128), dtype=torch.bfloat16)
    seen = torch.zeros(count, dtype=torch.bool)
    for shard_number, receipt in enumerate(summary["shards"], start=1):
        path = Path(receipt["path"])
        if sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"Experiment 0A Stage 0A shard hash mismatch: {path}")
        flat = _load_flat_shard(path)
        indices = flat["sample_indices"].long()
        states[indices] = flat["teacher_states_bfloat16"]
        final_hidden[indices] = flat["final_hidden_bfloat16"]
        topk_ids[indices] = flat["topk_ids"]
        topk_log_probs[indices] = flat["topk_log_probs"]
        seen[indices] = True
        if shard_number == 1 or shard_number % 16 == 0 or shard_number == len(summary["shards"]):
            print(
                f"exp0a_cache_load shard={shard_number}/{len(summary['shards'])} "
                f"samples={int(seen.sum())}",
                flush=True,
            )
    if not bool(seen.all()):
        raise RuntimeError("Experiment 0A teacher-state cache is incomplete")
    return {
        "samples": samples,
        "states": states,
        "final_hidden": final_hidden,
        "topk_ids": topk_ids,
        "topk_log_probs": topk_log_probs,
        "model_summary": summary,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _pool(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    normalized = _rms_unit(states.float())
    weights = weights.float() / weights.float().sum()
    return (normalized * weights.view(1, -1, 1)).sum(dim=1)


def _fit_tucker_weights(
    states: torch.Tensor, targets: torch.Tensor, *, seed: int, sample_count: int = 8192
) -> torch.Tensor:
    generator = torch.Generator(device=states.device).manual_seed(seed)
    indices = torch.randperm(states.shape[0], generator=generator, device=states.device)[:sample_count]
    target_projection = torch.randn(
        targets.shape[1], 64, generator=generator, device=states.device
    ) / math.sqrt(targets.shape[1])
    compressed_target = targets.index_select(0, indices).float() @ target_projection
    compressed_target -= compressed_target.mean(dim=0)
    scores = []
    for layer in range(states.shape[1]):
        values = _rms_unit(states.index_select(0, indices)[:, layer].float())
        state_projection = torch.randn(
            values.shape[1], 64, generator=generator, device=states.device
        ) / math.sqrt(values.shape[1])
        compressed = values @ state_projection
        compressed -= compressed.mean(dim=0)
        cross = compressed.T @ compressed_target / max(1, compressed.shape[0] - 1)
        scores.append(cross.square().sum().sqrt())
    stacked = torch.stack(scores)
    scale = stacked.std().clamp_min(1e-6)
    return torch.softmax((stacked - stacked.mean()) / scale, dim=0)


def _fit_projector(
    x: torch.Tensor, y: torch.Tensor, *, method: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mean = x.mean(dim=0)
    centered = x - mean
    rank = min(INTERNAL_RANK, centered.shape[0] - 1, centered.shape[1])
    torch.manual_seed(seed)
    _u, singular, basis = torch.pca_lowrank(
        centered, q=rank, center=False, niter=2
    )
    if method == "pca":
        generator = torch.Generator(device=x.device).manual_seed(seed + 1)
        expansion = torch.randn(
            y.shape[1], rank, generator=generator, device=x.device
        )
        expansion = torch.linalg.qr(expansion, mode="reduced").Q.T
        projector = basis[:, :rank] @ expansion
    elif method in {"predictive_rrr", "tucker_predictive"}:
        scores = centered @ basis[:, :rank]
        gram = scores.T @ scores
        ridge = max(float(torch.diagonal(gram).mean()) * 1e-5, 1e-6)
        coefficient = torch.linalg.solve(
            gram + ridge * torch.eye(rank, device=x.device), scores.T @ y
        )
        projector = basis[:, :rank] @ coefficient
    else:
        raise ValueError(f"unknown Experiment 0A method: {method}")
    return mean, projector, {
        "internal_rank": rank,
        "pca_singular_values": finite_quantiles(singular.detach().cpu().tolist()),
    }


def _canonical_statistics(
    x: torch.Tensor, teacher_mean: torch.Tensor, projector: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw = ((x - teacher_mean) @ projector).view(-1, N_SLOTS, LATENT_DIM)
    canonical_mean = raw.mean(dim=0)
    centered = raw - canonical_mean
    flat = centered.reshape(-1, LATENT_DIM)
    covariance = torch.zeros(
        (LATENT_DIM, LATENT_DIM), dtype=torch.float64, device=flat.device
    )
    for start in range(0, flat.shape[0], 8192):
        chunk = flat[start : start + 8192].double()
        covariance.add_(chunk.T @ chunk)
    covariance.div_(max(1, flat.shape[0] - 1))
    eigenvalues, basis = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    raw_eigenvalues = eigenvalues[order].clamp_min(0).float()
    basis = basis[:, order].float()
    effective = effective_eigenvalues(
        raw_eigenvalues, tau=WHITEN_TAU, eps_abs=WHITEN_EPS_ABS
    )
    positive = raw_eigenvalues[raw_eigenvalues > 0]
    floor = max(float(raw_eigenvalues.max()) * WHITEN_TAU, WHITEN_EPS_ABS)
    floored = raw_eigenvalues < floor
    condition = {
        "raw": (
            float(raw_eigenvalues.max() / positive.min()) if positive.numel() else None
        ),
        "zero_eigenvalue_count": int(raw_eigenvalues.eq(0).sum()),
        "effective": float(effective.max() / effective.min()),
        "floor": float(effective.min()),
        "floored_count": int(floored.sum()),
        "floored_fraction": float(floored.float().mean()),
        "rank": int(raw_eigenvalues.numel()),
        "fit_accumulation_dtype": "float64",
        "fit_health_relative_floor_exceeds_absolute": bool(
            WHITEN_TAU * float(raw_eigenvalues.max()) > WHITEN_EPS_ABS
        ),
    }
    return canonical_mean, basis, effective, condition


def _transform(
    x: torch.Tensor, *, teacher_mean: torch.Tensor, projector: torch.Tensor,
    canonical_mean: torch.Tensor, basis: torch.Tensor, eigenvalues: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    raw = ((x - teacher_mean) @ projector).view(-1, N_SLOTS, LATENT_DIM)
    centered = raw - canonical_mean
    scale = eigenvalues.pow(-0.5 * float(alpha))
    return (centered @ basis) * scale


def _fit_probe(
    z: torch.Tensor, hidden: torch.Tensor, horizons: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(z.shape[0], device=z.device)
    selected = z[rows, horizons.long() - 1]
    hidden_mean = hidden.float().mean(dim=0)
    centered_hidden = hidden.float() - hidden_mean
    gram = selected.T @ selected
    ridge = max(float(torch.diagonal(gram).mean()) * 1e-4, 1e-6)
    cross = selected.T @ centered_hidden
    decoder = torch.linalg.solve(
        gram + ridge * torch.eye(LATENT_DIM, device=z.device), cross
    )
    return decoder, hidden_mean


def _probe_metrics(
    *, z: torch.Tensor, decoder: torch.Tensor, decoder_bias: torch.Tensor,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor, topk_log_probs: torch.Tensor, horizons: torch.Tensor,
    observed_token_ids: torch.Tensor, lm_head: torch.Tensor, batch_size: int = 64
) -> dict[str, float | int]:
    mse_values = []
    cosine_values = []
    kl_values = []
    top1_matches = 0
    observed_correct = 0
    observed_covered = 0
    total = z.shape[0]
    rows = torch.arange(total, device=z.device)
    selected = z[rows, horizons.long() - 1]
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        predicted_hidden = selected[start:stop] @ decoder + decoder_bias
        target_hidden = hidden[start:stop].float()
        mse_values.extend(
            ((predicted_hidden - target_hidden).square().mean(dim=1) /
             target_hidden.square().mean(dim=1).clamp_min(1e-12)).detach().cpu().tolist()
        )
        cosine_values.extend(
            F.cosine_similarity(predicted_hidden, target_hidden, dim=1).detach().cpu().tolist()
        )
        ids = topk_ids[start:stop].long()
        token_weights = lm_head.index_select(0, ids.reshape(-1)).view(
            stop - start, ids.shape[1], -1
        )
        predicted_logits = torch.einsum(
            "bd,bkd->bk", predicted_hidden.to(token_weights.dtype), token_weights
        ).float()
        predicted_log = torch.log_softmax(predicted_logits, dim=-1)
        target_log = torch.log_softmax(topk_log_probs[start:stop].float(), dim=-1)
        target_prob = target_log.exp()
        kl_values.extend(
            torch.sum(target_prob * (target_log - predicted_log), dim=-1).detach().cpu().tolist()
        )
        predicted_ids = ids.gather(1, predicted_logits.argmax(dim=1, keepdim=True)).squeeze(1)
        teacher_ids = ids[:, 0]
        top1_matches += int(predicted_ids.eq(teacher_ids).sum())
        observed = observed_token_ids[start:stop]
        covered = ids.eq(observed.unsqueeze(1)).any(dim=1)
        observed_covered += int(covered.sum())
        observed_correct += int((predicted_ids.eq(observed) & covered).sum())
    return {
        "samples": total,
        "normalized_hidden_mse_mean": float(torch.tensor(mse_values).mean()),
        "hidden_cosine_mean": float(torch.tensor(cosine_values).mean()),
        "future_topk_kl_mean": float(torch.tensor(kl_values).mean()),
        "future_topk_kl_median": float(torch.tensor(kl_values).median()),
        "teacher_top1_agreement": top1_matches / total,
        "observed_token_topk_coverage": observed_covered / total,
        "observed_token_accuracy": observed_correct / total,
    }


def _gradient_audit(z: torch.Tensor, anchor_indices: torch.Tensor, horizons: torch.Tensor) -> dict[str, Any]:
    _unique, local_anchor_indices = torch.unique(
        anchor_indices, sorted=True, return_inverse=True
    )
    anchor_count = int(_unique.numel())
    first = torch.empty((anchor_count, N_SLOTS, LATENT_DIM), device=z.device)
    last = torch.empty_like(first)
    first[local_anchor_indices[horizons.eq(1)]] = z[horizons.eq(1)]
    last[local_anchor_indices[horizons.eq(4)]] = z[horizons.eq(4)]
    gradient = (last - first).clamp(-1.0, 1.0)
    coordinate_rms = gradient.square().mean(dim=(0, 1)).sqrt()
    return {
        "coordinate_rms": finite_quantiles(coordinate_rms.detach().cpu().tolist()),
        "coefficient_of_variation": float(
            coordinate_rms.std() / coordinate_rms.mean().clamp_min(1e-12)
        ),
        "max_to_median": float(
            coordinate_rms.max() / coordinate_rms.median().clamp_min(1e-12)
        ),
    }


def _save_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def run_exp0a(
    *, stage0a_private: Path, stage0a_summary: Path, repaired_summary: Path,
    output_private: Path, output_summary: Path, device: str
) -> dict[str, Any]:
    started = time.time()
    source = json.loads(stage0a_summary.read_text(encoding="utf-8"))
    repair = json.loads(repaired_summary.read_text(encoding="utf-8"))
    if source.get("status") != "complete_development_only" or repair.get("status") != "complete_development_only":
        raise RuntimeError("Experiment 0A requires complete Stage 0A and repair receipts")
    if source.get("training_started") or source.get("frozen_evaluation_partitions_touched"):
        raise RuntimeError("Experiment 0A source violates the DEV-only contract")
    if repair.get("source_summary_sha256") != sha256_file(stage0a_summary):
        raise RuntimeError("Experiment 0A repair receipt points to a different Stage 0A summary")
    arrays = load_stage0a_arrays(stage0a_private)
    samples = arrays["samples"]
    sample_count = len(samples)
    if sample_count != 200_000:
        raise RuntimeError(f"Experiment 0A expected 200,000 samples, observed {sample_count}")
    metadata = {
        "anchor_indices": torch.tensor([int(row["anchor_index"]) for row in samples]),
        "horizons": torch.tensor([int(row["horizon"]) for row in samples]),
        "observed": torch.tensor([int(row["observed_next_token_id"]) for row in samples]),
        "strata": [str(row["stratum"]) for row in samples],
        "documents": [str(row["document_id"]) for row in samples],
    }
    calibration = document_split(
        metadata["documents"], calibration_fraction=CALIBRATION_FRACTION, seed=SPLIT_SEED
    )
    split_receipt = {
        "seed": SPLIT_SEED,
        "calibration_samples": int(calibration.sum()),
        "holdout_samples": int((~calibration).sum()),
        "calibration_documents": len(
            {metadata["documents"][index] for index in torch.where(calibration)[0].tolist()}
        ),
        "holdout_documents": len(
            {metadata["documents"][index] for index in torch.where(~calibration)[0].tolist()}
        ),
        "document_disjoint": True,
        "fit_count_resolution": (
            "document-disjoint screening fit uses the calibration subset; final candidate refit uses all "
            "200000 DEV-C boundary samples and is never scored as held-out evidence"
        ),
    }
    target_device = torch.device(device)
    states = arrays["states"].to(target_device)
    final_hidden = arrays["final_hidden"].to(target_device)
    topk_ids = arrays["topk_ids"].to(target_device)
    topk_log_probs = arrays["topk_log_probs"].to(target_device)
    anchor_indices = metadata["anchor_indices"].to(target_device)
    horizons = metadata["horizons"].to(target_device)
    observed = metadata["observed"].to(target_device)
    calibration_device = calibration.to(target_device)
    anchor_targets, slot_mask = build_anchor_targets(
        topk_ids=topk_ids,
        topk_log_probs=topk_log_probs,
        middle_states=states[:, 1],
        horizons=horizons,
        anchor_indices=anchor_indices,
        anchor_count=int(anchor_indices.max()) + 1,
        latent_dim=LATENT_DIM,
        n_slots=N_SLOTS,
        seed=SPLIT_SEED,
    )
    sample_targets = anchor_targets.index_select(0, anchor_indices).reshape(sample_count, -1)
    uniform_weights = torch.full((3,), 1 / 3, device=target_device)
    uniform_x = _pool(states, uniform_weights)
    tucker_weights = _fit_tucker_weights(
        states[calibration_device], sample_targets[calibration_device], seed=SPLIT_SEED
    )
    lm_head_path = Path(arrays["model_summary"]["lm_head"]["path"])
    if sha256_file(lm_head_path) != arrays["model_summary"]["lm_head"]["sha256"]:
        raise RuntimeError("Experiment 0A 14B LM-head hash mismatch")
    lm_head_payload = torch.load(lm_head_path, map_location="cpu", weights_only=False)
    lm_head = lm_head_payload["weight_bfloat16"].to(target_device)

    method_summaries: dict[str, Any] = {}
    output_private.mkdir(parents=True, exist_ok=True)
    shared_endpoint = _save_artifact(
        output_private / "holdout_targets.pt",
        {
            "kind": "paper2_phase2_exp0a_holdout_targets",
            "sample_indices": torch.where(~calibration)[0],
            "anchor_indices": metadata["anchor_indices"][~calibration],
            "horizons": metadata["horizons"][~calibration],
            "final_hidden_bfloat16": arrays["final_hidden"][~calibration],
            "topk_ids": arrays["topk_ids"][~calibration],
            "topk_log_probs": arrays["topk_log_probs"][~calibration],
            "observed_token_ids": metadata["observed"][~calibration],
        },
    )
    for method_index, method in enumerate(METHODS):
        method_path = output_private / f"{method}.pt"
        method_summary_path = output_private / f"{method}_summary.json"
        if method_path.is_file() and method_summary_path.is_file():
            saved = json.loads(method_summary_path.read_text(encoding="utf-8"))
            if saved.get("artifact", {}).get("sha256") == sha256_file(method_path):
                method_summaries[method] = saved
                print(f"exp0a_resume_method={method}", flush=True)
                continue
        layer_weights = tucker_weights if method == "tucker_predictive" else uniform_weights
        x = _pool(states, layer_weights) if method == "tucker_predictive" else uniform_x
        fit_method = "predictive_rrr" if method == "tucker_predictive" else method
        screening_mean, screening_projector, fit_receipt = _fit_projector(
            x[calibration_device], sample_targets[calibration_device],
            method=fit_method, seed=SPLIT_SEED + method_index * 10
        )
        canonical_mean, basis, eigenvalues, condition = _canonical_statistics(
            x[calibration_device], screening_mean, screening_projector
        )
        alpha_summaries: dict[str, Any] = {}
        decoders: dict[str, torch.Tensor] = {}
        endpoint_raw = ((x[~calibration_device] - screening_mean) @ screening_projector).view(
            -1, N_SLOTS, LATENT_DIM
        ) - canonical_mean
        for alpha in WHITEN_ALPHAS:
            key = str(alpha)
            z_calibration = _transform(
                x[calibration_device], teacher_mean=screening_mean,
                projector=screening_projector, canonical_mean=canonical_mean,
                basis=basis, eigenvalues=eigenvalues, alpha=alpha
            )
            z_holdout = _transform(
                x[~calibration_device], teacher_mean=screening_mean,
                projector=screening_projector, canonical_mean=canonical_mean,
                basis=basis, eigenvalues=eigenvalues, alpha=alpha
            )
            decoder, decoder_bias = _fit_probe(
                z_calibration, final_hidden[calibration_device], horizons[calibration_device]
            )
            decoders[key] = {
                "weight": decoder.detach().cpu(),
                "bias": decoder_bias.detach().cpu(),
            }
            metrics = _probe_metrics(
                z=z_holdout, decoder=decoder, decoder_bias=decoder_bias,
                hidden=final_hidden[~calibration_device],
                topk_ids=topk_ids[~calibration_device],
                topk_log_probs=topk_log_probs[~calibration_device],
                horizons=horizons[~calibration_device],
                observed_token_ids=observed[~calibration_device], lm_head=lm_head
            )
            canonical_eigenvalues = eigenvalues.pow(1.0 - alpha)
            metrics.update(
                {
                    "alpha": alpha,
                    "canonical_condition_number": float(
                        canonical_eigenvalues.max() / canonical_eigenvalues.min()
                    ),
                    "gradient_audit": _gradient_audit(
                        z_calibration, anchor_indices[calibration_device], horizons[calibration_device]
                    ),
                }
            )
            alpha_summaries[key] = metrics
        final_mean, final_projector, final_fit_receipt = _fit_projector(
            x, sample_targets, method=fit_method, seed=SPLIT_SEED + method_index * 10
        )
        final_canonical_mean, final_basis, final_eigenvalues, final_condition = _canonical_statistics(
            x, final_mean, final_projector
        )
        artifact_payload = {
            "kind": "paper2_phase2_exp0a_canonicalizer",
            "method": method,
            "n_slots": N_SLOTS,
            "latent_dim": LATENT_DIM,
            "layer_weights": layer_weights.detach().cpu(),
            "screening": {
                "projector_weight": screening_projector.detach().cpu(),
                "teacher_mean": screening_mean.detach().cpu(),
                "canonical_mean": canonical_mean.detach().cpu(),
                "whiten_basis": basis.detach().cpu(),
                "whiten_eigenvalues": eigenvalues.detach().cpu(),
                "decoders": decoders,
            },
            "final_refit_200k": {
                "projector_weight": final_projector.detach().cpu(),
                "teacher_mean": final_mean.detach().cpu(),
                "canonical_mean": final_canonical_mean.detach().cpu(),
                "whiten_basis": final_basis.detach().cpu(),
                "whiten_eigenvalues": final_eigenvalues.detach().cpu(),
            },
            "single_effective_eigenvalue_rule": {
                "tau": WHITEN_TAU,
                "eps_abs": WHITEN_EPS_ABS,
                "second_forward_epsilon": False,
            },
            "shared_basis_across_alphas": True,
            "persistent_state_renormalization": False,
            "training_started": False,
            "optimizer_steps": 0,
        }
        artifact = _save_artifact(method_path, artifact_payload)
        endpoint = _save_artifact(
            output_private / f"{method}_holdout_raw_endpoints.pt",
            {
                "kind": "paper2_phase2_exp0a_holdout_raw_endpoints",
                "method": method,
                "raw_centered": endpoint_raw.detach().to(torch.bfloat16).cpu(),
                "whiten_basis": basis.detach().cpu(),
                "whiten_eigenvalues": eigenvalues.detach().cpu(),
            },
        )
        method_summary = {
            "method": method,
            "layer_weights": layer_weights.detach().cpu().tolist(),
            "screening_fit": fit_receipt,
            "screening_condition": condition,
            "alpha": alpha_summaries,
            "final_refit_samples": sample_count,
            "final_fit": final_fit_receipt,
            "final_condition": final_condition,
            "artifact": artifact,
            "holdout_endpoints": endpoint,
            "projection_multiply_adds_per_sample": int(5120 * N_SLOTS * LATENT_DIM),
        "probe_parameter_count": int(LATENT_DIM * 5120 + 5120),
        }
        write_json(method_summary_path, method_summary)
        method_summaries[method] = method_summary
        print(f"exp0a_method_complete={method}", flush=True)

    pca_kl = method_summaries["pca"]["alpha"]["0.5"]["future_topk_kl_mean"]
    rrr_kl = method_summaries["predictive_rrr"]["alpha"]["0.5"]["future_topk_kl_mean"]
    linear_underfit_trigger = rrr_kl >= 0.99 * pca_kl
    summary = {
        "kind": "paper2_phase2_exp0a_canonicalizer_screening",
        "status": "complete_development_only",
        "source_stage0a_summary_sha256": sha256_file(stage0a_summary),
        "source_repair_summary_sha256": sha256_file(repaired_summary),
        "source_manifest_sha256": arrays["manifest_sha256"],
        "split": split_receipt,
        "geometry": {
            "n_slots": N_SLOTS,
            "available_future_slots": 4,
            "unavailable_span_slots": 4,
            "latent_dim": LATENT_DIM,
            "internal_rank": INTERNAL_RANK,
            "alphas": list(WHITEN_ALPHAS),
            "shared_frozen_pca_basis": True,
            "single_eigenvalue_floor": {"tau": WHITEN_TAU, "eps_abs": WHITEN_EPS_ABS},
            "persistent_state_renormalization": False,
        },
        "methods": method_summaries,
        "conditional_nonlinear_ablations": {
            "triggered": linear_underfit_trigger,
            "attention_pooling": "build-ready_trigger_held",
            "deterministic_autoencoder": "build-ready_trigger_held",
            "rule": (
                "run only if deterministic linear models show underfitting; they are not part of "
                "the default v1 path without that evidence"
            ),
        },
        "screening_survivors": [
            method
            for method, result in method_summaries.items()
            if all(math.isfinite(result["alpha"][str(alpha)]["future_topk_kl_mean"]) for alpha in WHITEN_ALPHAS)
        ],
        "holdout_targets": shared_endpoint,
        "elapsed_seconds": time.time() - started,
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "do_not_claim": [
            "Experiment 0A selects the final alpha before matched module pilots",
            "development-only probe fidelity is downstream task quality",
            "top-k conditional KL is full-vocabulary KL",
            "a final 200k refit has a document-disjoint performance estimate",
        ],
    }
    write_json(output_summary, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--repaired_summary", type=Path, required=True)
    parser.add_argument("--output_private", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_exp0a(
        stage0a_private=args.stage0a_private,
        stage0a_summary=args.stage0a_summary,
        repaired_summary=args.repaired_summary,
        output_private=args.output_private,
        output_summary=args.output_summary,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
