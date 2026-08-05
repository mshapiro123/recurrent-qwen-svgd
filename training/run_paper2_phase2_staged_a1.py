"""Run the locked Phase-2 staged re-pilot A1 state-construction stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules
from training.paper2_phase2_matched_alpha import (
    build_adamw_groups,
    distribution_overlap,
    document_partition,
    expected_accepted_length,
    masked_sparse_kl,
    normalize_sparse_with_tail,
    quality_noninferior,
    reconstruct_sparse_residual_seed,
    wilson_lower,
)
from training.paper2_phase2_staged_repilot import (
    PROTOCOL_LOCK_COMMIT,
    a1_gate,
    realized_gradient_shares,
    relative_improvement,
    solve_static_weights,
    trust_tripwire,
)
from training.run_paper2_phase2_matched_alpha import (
    _assert_zero_loop_identity,
    _batch,
    _decoder_for_alpha,
    _local_source,
    _position_buckets,
    _tensor_digest,
    build_pilot_cache,
    sha256_file,
    sha256_lf_file,
    write_json,
)


PROTOCOL_RELATIVE = Path("training/paper2_phase2_staged_repilot_preregistration.json")
LOSS_MASK_CONTRACT = "masked_sparse_candidates_plus_tail_v3_finite_residual_seed"
CALIBRATION_KIND = "paper2_phase2_staged_a1_calibration_v1"
AMENDMENT_LOCK_COMMIT = "01ae5f28c0b52f4635dc15298bd07269bd853055"
AUDIT_RECEIPT_RELATIVE = Path(
    "outputs/stage5/stage5_paper2_phase2_a1_matched_estimator_audit_20260805/summary.json"
)


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _assert_lock(root: Path) -> dict[str, Any]:
    registration = json.loads((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    if registration.get("status") != "locked_before_training":
        raise RuntimeError("staged re-pilot registration is not locked")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", AMENDMENT_LOCK_COMMIT, "HEAD"], cwd=root
    ).returncode:
        raise RuntimeError(f"launcher does not descend from amendment lock {AMENDMENT_LOCK_COMMIT}")
    amendment = registration.get("resume_amendment_20260805", {})
    if amendment.get("status") != "locked_before_resumed_training":
        raise RuntimeError("A1 resume amendment is not locked")
    audit_path = root / AUDIT_RECEIPT_RELATIVE
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing matched-estimator audit receipt: {audit_path}")
    if sha256_lf_file(audit_path) != amendment["audit_receipt_lf_sha256"]:
        raise RuntimeError("matched-estimator audit receipt differs from the amendment lock")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("decision") != "resume_saved_step_200" or not audit.get(
        "resume_authorized"
    ):
        raise RuntimeError("matched-estimator audit did not authorize checkpoint resume")
    return registration


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_a1_trainable(module: Phase2StudentModules) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for parameter in module.flow.parameters():
        parameter.requires_grad_(True)


def _frozen_hash(module: Phase2StudentModules) -> str:
    return _tensor_digest(
        {
            name: value
            for name, value in module.named_parameters()
            if not name.startswith("flow.")
        }
    )


def _a1_losses(
    *,
    module: Phase2StudentModules,
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    huber_delta: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hidden4 = batch["hidden"].float()
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    candidate_embeddings = embedding(batch["candidate_ids"].clamp_min(0)).float()
    raw_base_candidate_logits = torch.einsum("bhd,bhcd->bhc", hidden4, candidate_embeddings)
    residual_seed = reconstruct_sparse_residual_seed(
        batch["base_candidates"], raw_base_candidate_logits, batch["candidate_mask"]
    ).detach()
    # A1 executes only the state-construction path. The deployed bridge, control,
    # and draft gates stay closed; the bridge is called separately below only to
    # define the explicitly counterfactual preservation loss.
    scratch0 = module.initializer(hidden, attention)
    flow_output = module.flow(
        scratch0,
        hidden.float().mean(dim=1),
        steps=1,
        target_state=batch["target_scratch"].float(),
        apply_trust_penalty=False,
    )
    target_log = normalize_sparse_with_tail(
        batch["teacher_candidates"], batch["teacher_tail"], batch["candidate_mask"]
    ).detach()
    base_log = normalize_sparse_with_tail(
        batch["base_candidates"], batch["base_tail"], batch["candidate_mask"]
    ).detach()

    start = flow_output.states[0]
    target = batch["target_scratch"].float().detach()
    desired = start.detach() + 0.5 * (target - start.detach())
    selected = torch.zeros(start.shape[:2], dtype=torch.bool, device=start.device)
    selected[:, :4] = True
    coordinates = selected.unsqueeze(-1).expand_as(flow_output.state)
    flow_huber = F.huber_loss(
        flow_output.state[coordinates], desired[coordinates], delta=float(huber_delta)
    )
    flow_cosine = 1.0 - F.cosine_similarity(
        flow_output.state[:, :4].reshape(start.shape[0], -1),
        desired[:, :4].reshape(start.shape[0], -1),
        dim=1,
    ).mean()
    flow = flow_huber + 0.1 * flow_cosine
    flow_mse = F.mse_loss(flow_output.state[coordinates], desired[coordinates])

    predicted_hidden = torch.einsum(
        "bhd,df->bhf", flow_output.state[:, :4].float(), decoder
    ) + decoder_bias
    topk_embeddings = teacher_embedding(batch["teacher_topk_ids"].long()).float()
    probe_logits = torch.einsum("bhf,bhkf->bhk", predicted_hidden, topk_embeddings)
    probe_log = torch.log_softmax(probe_logits, dim=-1)
    probe_target = torch.log_softmax(
        batch["teacher_topk_log_probs"].float(), dim=-1
    ).detach()
    functional = (probe_target.exp() * (probe_target - probe_log)).sum(dim=-1).mean()

    # Training-only counterfactual: actual A1 execution remains zero-loop and
    # never consumes this bridge output.
    counterfactual_bridge = module.bridge(
        h0=hidden,
        previous=hidden,
        scratch=flow_output.state,
        loop_index=0,
        active=True,
    )
    bridge_delta = torch.einsum(
        "bhd,bhcd->bhc",
        counterfactual_bridge.hidden[:, 1:].float() - hidden4,
        candidate_embeddings,
    )
    counterfactual_log = normalize_sparse_with_tail(
        residual_seed + bridge_delta, batch["base_tail"], batch["candidate_mask"]
    )
    preserve_mask = base_log.argmax(dim=-1).eq(target_log.argmax(dim=-1))
    preserve_rows = masked_sparse_kl(base_log, counterfactual_log, batch["candidate_mask"])
    counterfactual_preserve = (
        preserve_rows[preserve_mask].mean()
        if bool(preserve_mask.any())
        else preserve_rows.mean() * 0
    )

    update = flow_output.updates[0][:, :4].float()
    prior = flow_output.states[0][:, :4].float()
    endpoint = target[:, :4]
    update_rms = update.square().mean(dim=(1, 2)).sqrt()
    state_rms = prior.square().mean(dim=(1, 2)).sqrt()
    endpoint_rms = endpoint.square().mean(dim=(1, 2)).sqrt()
    state_ratio = update_rms / state_rms.clamp_min(1e-6)
    endpoint_ratio = update_rms / torch.maximum(state_rms, endpoint_rms).add(1e-6)
    target_increment = desired[:, :4] - start.detach()[:, :4]
    losses = {
        "flow": flow,
        "functional_probe_kl": functional,
        "counterfactual_preserve_kl": counterfactual_preserve,
    }
    metrics = {
        "flow_mse": flow_mse,
        "probe_kl_rows": (probe_target.exp() * (probe_target - probe_log)).sum(dim=-1).mean(-1),
        "base_log": base_log,
        "target_log": target_log,
        "state_ratio": state_ratio,
        "endpoint_ratio": endpoint_ratio,
        "target_increment": target_increment,
        "flow_state": flow_output.state[:, :4],
        "probe_log": probe_log,
        "probe_target": probe_target,
        "counterfactual_bridge_gate": counterfactual_bridge.gate.detach(),
        "executed_bridge_gate": update.new_zeros(()),
        "executed_draft_gate": update.new_zeros(()),
    }
    return losses, metrics


def _loss_gradients(
    losses: dict[str, torch.Tensor], parameters: list[nn.Parameter]
) -> dict[str, list[torch.Tensor]]:
    output: dict[str, list[torch.Tensor]] = {}
    for index, (name, loss) in enumerate(losses.items()):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index + 1 < len(losses),
            allow_unused=True,
        )
        output[name] = [
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)
        ]
    return output


def _gradient_norms(gradients: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    return {
        name: math.sqrt(
            sum(float(value.detach().double().square().sum()) for value in values)
        )
        for name, values in gradients.items()
    }


def _gradient_gram(gradients: dict[str, list[torch.Tensor]]) -> dict[str, dict[str, float]]:
    names = list(gradients)
    return {
        left: {
            right: sum(
                float(a.detach().double().mul(b.detach().double()).sum())
                for a, b in zip(gradients[left], gradients[right])
            )
            for right in names
        }
        for left in names
    }


def _weighted_norm(gram: dict[str, dict[str, float]], weights: dict[str, float]) -> float:
    squared = sum(
        weights[left] * weights[right] * gram[left][right]
        for left in weights
        for right in weights
    )
    return math.sqrt(max(0.0, squared))


def calibrate_a1(
    *,
    module: Phase2StudentModules,
    cache: dict[str, Any],
    train_indices: torch.Tensor,
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    seed: int,
    registration: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    module.train()
    initial_hash = _tensor_digest(
        {name: value for name, value in module.named_parameters() if value.requires_grad}
    )
    generator = torch.Generator().manual_seed(seed + 34001)
    batches = [
        train_indices.index_select(
            0,
            torch.randint(
                train_indices.numel(),
                (int(registration["batch_size"]),),
                generator=generator,
            ),
        )
        for _ in range(int(registration["calibration"]["batches"]))
    ]
    increments = []
    with torch.no_grad():
        for indices in batches:
            batch = _batch(cache, indices, alpha=0.5, device=device)
            _losses, metrics = _a1_losses(
                module=module,
                batch=batch,
                embedding=embedding,
                teacher_embedding=teacher_embedding,
                decoder=decoder,
                decoder_bias=decoder_bias,
                huber_delta=1.0,
            )
            increments.append(metrics["target_increment"].abs().reshape(-1).cpu())
    huber_delta = float(torch.quantile(torch.cat(increments), 0.75))
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise RuntimeError("calibration produced an invalid Huber delta")

    parameters = [value for value in module.flow.parameters() if value.requires_grad]
    start = int(registration["calibration"]["measurement_first_batch"]) - 1
    stop = int(registration["calibration"]["measurement_last_batch"])
    norm_rows = []
    grams = []
    for batch_number, indices in enumerate(batches[start:stop], start=start + 1):
        batch = _batch(cache, indices, alpha=0.5, device=device)
        losses, _metrics = _a1_losses(
            module=module,
            batch=batch,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            huber_delta=huber_delta,
        )
        gradients = _loss_gradients(losses, parameters)
        norm_rows.append(_gradient_norms(gradients))
        grams.append(_gradient_gram(gradients))
        if batch_number == start + 1 or batch_number % 10 == 0 or batch_number == stop:
            print(f"staged_a1_calibration seed={seed} batch={batch_number}/{stop}", flush=True)
    names = list(norm_rows[0])
    mean_norms = {
        name: sum(row[name] for row in norm_rows) / len(norm_rows) for name in names
    }
    target = registration["a1"]["target_gradient_shares"]
    weights = solve_static_weights(
        mean_norms,
        target,
        anchor="flow",
        minimum_norm=float(registration["calibration"]["minimum_gradient_norm"]),
    )
    weighted_norms = torch.tensor(
        [_weighted_norm(gram, weights) for gram in grams], dtype=torch.float64
    )
    clip_ceiling = float(
        torch.quantile(weighted_norms, 0.99)
        * float(registration["clip"]["ceiling_multiplier_over_calibration_p99"])
    )
    final_hash = _tensor_digest(
        {name: value for name, value in module.named_parameters() if value.requires_grad}
    )
    if final_hash != initial_hash:
        raise RuntimeError("gradient-only calibration mutated A1 parameters")
    return {
        "kind": CALIBRATION_KIND,
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "loss_mask_contract": LOSS_MASK_CONTRACT,
        "seed": seed,
        "optimizer_updates": 0,
        "sampled_batches": len(batches),
        "measurement_batches": len(norm_rows),
        "huber_delta": huber_delta,
        "mean_gradient_norms": mean_norms,
        "target_gradient_shares": target,
        "static_loss_weights": weights,
        "calibration_realized_shares": realized_gradient_shares(mean_norms, weights),
        "weighted_total_gradient_norm_p99": float(torch.quantile(weighted_norms, 0.99)),
        "clip_ceiling": clip_ceiling,
        "parameter_hash_before": initial_hash,
        "parameter_hash_after": final_hash,
    }


@torch.no_grad()
def evaluate_a1(
    *,
    module: Phase2StudentModules,
    cache: dict[str, Any],
    eval_indices: torch.Tensor,
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    huber_delta: float,
    device: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    module.eval()
    accumulated = {
        "flow_loss": 0.0,
        "flow_mse": 0.0,
        "functional_probe_kl": 0.0,
        "counterfactual_preserve_kl": 0.0,
        "base_accepted_length": 0.0,
        "endpoint_ratio_mean": 0.0,
        "endpoint_ratio_max": 0.0,
        "state_ratio_mean": 0.0,
        "endpoint_error_mean": 0.0,
        "probe_top1": 0.0,
    }
    rows = 0
    baseline_correct = 0
    for offset in range(0, eval_indices.numel(), batch_size):
        selected = eval_indices[offset : offset + batch_size]
        batch = _batch(cache, selected, alpha=0.5, device=device)
        losses, metrics = _a1_losses(
            module=module,
            batch=batch,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            huber_delta=huber_delta,
        )
        count = int(selected.numel())
        overlap = distribution_overlap(metrics["target_log"], metrics["base_log"])
        base_accepted = expected_accepted_length(overlap)
        accumulated["flow_loss"] += float(losses["flow"]) * count
        accumulated["flow_mse"] += float(metrics["flow_mse"]) * count
        accumulated["functional_probe_kl"] += float(losses["functional_probe_kl"]) * count
        accumulated["counterfactual_preserve_kl"] += float(
            losses["counterfactual_preserve_kl"]
        ) * count
        accumulated["base_accepted_length"] += float(base_accepted.sum())
        accumulated["endpoint_ratio_mean"] += float(metrics["endpoint_ratio"].sum())
        accumulated["state_ratio_mean"] += float(metrics["state_ratio"].sum())
        endpoint_error = (
            metrics["flow_state"] - batch["target_scratch"][:, :4].float()
        ).square().mean(dim=(1, 2)).sqrt()
        accumulated["endpoint_error_mean"] += float(endpoint_error.sum())
        accumulated["probe_top1"] += float(
            metrics["probe_log"]
            .argmax(dim=-1)
            .eq(metrics["probe_target"].argmax(dim=-1))
            .float()
            .mean(dim=-1)
            .sum()
        )
        baseline_correct += int(
            metrics["base_log"].argmax(dim=-1).eq(metrics["target_log"].argmax(dim=-1)).sum()
        )
        accumulated["endpoint_ratio_max"] = max(
            accumulated["endpoint_ratio_max"], float(metrics["endpoint_ratio"].max())
        )
        rows += count
    result = {name: value / rows for name, value in accumulated.items()}
    result["endpoint_ratio_max"] = accumulated["endpoint_ratio_max"]
    retention = baseline_correct / max(1, baseline_correct)
    retention_wilson = (
        wilson_lower(baseline_correct, baseline_correct) if baseline_correct else 0.0
    )
    result.update(
        {
            "rows": rows,
            "baseline_correct": baseline_correct,
            "retained_correct": baseline_correct,
            "executed_bridge_gate": 0.0,
            "executed_draft_gate": 0.0,
            "accepted_length_delta": 0.0,
            "quality_retention": retention,
            "quality_wilson_95_lower": retention_wilson,
            "quality_noninferior": quality_noninferior(baseline_correct, baseline_correct),
        }
    )
    return result


def _share_audit(
    *,
    module: Phase2StudentModules,
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    huber_delta: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    module.train()
    losses, _metrics = _a1_losses(
        module=module,
        batch=batch,
        embedding=embedding,
        teacher_embedding=teacher_embedding,
        decoder=decoder,
        decoder_bias=decoder_bias,
        huber_delta=huber_delta,
    )
    parameters = [value for value in module.flow.parameters() if value.requires_grad]
    norms = _gradient_norms(_loss_gradients(losses, parameters))
    return {"gradient_norms": norms, "shares": realized_gradient_shares(norms, weights)}


def _calibration_measurement_batches(
    train_indices: torch.Tensor, *, seed: int, registration: dict[str, Any]
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 34001)
    batches = [
        train_indices.index_select(
            0,
            torch.randint(
                train_indices.numel(),
                (int(registration["batch_size"]),),
                generator=generator,
            ),
        )
        for _ in range(int(registration["calibration"]["batches"]))
    ]
    start = int(registration["calibration"]["measurement_first_batch"]) - 1
    stop = int(registration["calibration"]["measurement_last_batch"])
    return batches[start:stop]


def _fixed_dev_share_batches(
    eval_indices: torch.Tensor, *, registration: dict[str, Any]
) -> list[torch.Tensor]:
    amendment = registration["resume_amendment_20260805"]
    count = int(amendment["hard_share_estimator"]["measurement_batches"])
    generator = torch.Generator().manual_seed(20260805)
    return [
        eval_indices.index_select(
            0,
            torch.randint(
                eval_indices.numel(),
                (int(registration["batch_size"]),),
                generator=generator,
            ),
        )
        for _ in range(count)
    ]


def _matched_share_audit(
    *,
    module: Phase2StudentModules,
    batches: list[torch.Tensor],
    cache: dict[str, Any],
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    huber_delta: float,
    weights: dict[str, float],
    device: str,
    seed: int,
    step: int,
    flow_minimum: float,
    probe_maximum: float,
    population: str,
) -> dict[str, Any]:
    module.train()
    parameter_hash_before = _tensor_digest(dict(module.named_parameters()))
    parameters = [value for value in module.flow.parameters() if value.requires_grad]
    norm_rows = []
    for batch_number, indices in enumerate(batches, start=1):
        batch = _batch(cache, indices, alpha=0.5, device=device)
        losses, _metrics = _a1_losses(
            module=module,
            batch=batch,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            huber_delta=huber_delta,
        )
        norm_rows.append(_gradient_norms(_loss_gradients(losses, parameters)))
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(batches):
            print(
                f"staged_a1_matched_share seed={seed} step={step} "
                f"batch={batch_number}/{len(batches)}",
                flush=True,
            )
    names = list(norm_rows[0])
    mean_norms = {
        name: sum(row[name] for row in norm_rows) / len(norm_rows) for name in names
    }
    shares = realized_gradient_shares(mean_norms, weights)
    batch_shares = [realized_gradient_shares(row, weights) for row in norm_rows]
    contract = {
        "flow_at_least_0p50": float(shares["flow"]) >= flow_minimum,
        "probe_at_most_0p25": float(shares["functional_probe_kl"]) <= probe_maximum,
    }
    contract["joint"] = all(contract.values())
    parameter_hash_after = _tensor_digest(dict(module.named_parameters()))
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("matched share audit mutated A1 parameters")
    return {
        "kind": "paper2_phase2_staged_a1_matched_share_v1",
        "step": step,
        "seed": seed,
        "population": population,
        "batches": len(norm_rows),
        "batch_size": int(batches[0].numel()),
        "mean_gradient_norms": mean_norms,
        "shares": shares,
        "contract": contract,
        "batch_contract_fraction": {
            "flow_at_least_0p50": sum(
                float(row["flow"]) >= flow_minimum for row in batch_shares
            )
            / len(batch_shares),
            "probe_at_most_0p25": sum(
                float(row["functional_probe_kl"]) <= probe_maximum for row in batch_shares
            )
            / len(batch_shares),
            "joint": sum(
                float(row["flow"]) >= flow_minimum
                and float(row["functional_probe_kl"]) <= probe_maximum
                for row in batch_shares
            )
            / len(batch_shares),
        },
        "counterfactual_preserve_share_role": "descriptive",
        "optimizer_updates": 0,
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    torch.cuda.set_rng_state_all(payload["cuda"])


def run_seed(
    *,
    cache: dict[str, Any],
    embedding_weight: torch.Tensor,
    teacher_embedding: nn.Embedding,
    train_indices: torch.Tensor,
    eval_indices: torch.Tensor,
    registration: dict[str, Any],
    seed: int,
    output_dir: Path,
    private_dir: Path,
    source_hashes: dict[str, str],
    audit_summary: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    arm = f"alpha_0p5_seed_{seed}"
    arm_private = private_dir / arm
    arm_private.mkdir(parents=True, exist_ok=True)
    source_checkpoint_path = arm_private / "a1_resume.pt"
    checkpoint_path = arm_private / "a1_resume_amended.pt"
    calibration_path = arm_private / "a1_calibration.json"
    _seed_everything(seed)
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(device)
    module = Phase2StudentModules(
        tied_embedding=embedding,
        hidden_size=896,
        rms_cap=float(registration["constants"]["state_rms_cap"]),
    ).to(device=device, dtype=torch.float32)
    _set_a1_trainable(module)
    initial_trainable_hash = _tensor_digest(
        {name: value for name, value in module.named_parameters() if value.requires_grad}
    )
    initial_frozen_hash = _frozen_hash(module)
    decoder, decoder_bias = _decoder_for_alpha(cache, alpha=0.5, device=device)

    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("kind") != CALIBRATION_KIND or int(calibration["seed"]) != seed:
            raise RuntimeError("A1 calibration receipt does not match the arm")
        if calibration.get("protocol_lock_commit") != PROTOCOL_LOCK_COMMIT:
            raise RuntimeError("A1 calibration protocol lock mismatch")
        if calibration.get("loss_mask_contract") != LOSS_MASK_CONTRACT:
            raise RuntimeError("A1 calibration loss contract mismatch")
        if calibration.get("parameter_hash_before") != initial_trainable_hash:
            raise RuntimeError("A1 calibration initialization hash mismatch")
    else:
        calibration = calibrate_a1(
            module=module,
            cache=cache,
            train_indices=train_indices,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            seed=seed,
            registration=registration,
            device=device,
        )
        write_json(calibration_path, calibration)
    weights = {name: float(value) for name, value in calibration["static_loss_weights"].items()}
    huber_delta = float(calibration["huber_delta"])
    clip_ceiling = float(calibration["clip_ceiling"])
    optimizer = torch.optim.AdamW(build_adamw_groups(module.flow, weight_decay=0.01), lr=3e-4)
    generator = torch.Generator().manual_seed(seed + 9173)
    history: list[dict[str, Any]] = []
    trust_history: list[bool] = []
    clip_events: list[bool] = []
    warnings: list[dict[str, Any]] = []
    matched_share_audits: list[dict[str, Any]] = []
    dev_share_audits: list[dict[str, Any]] = []
    step = 0
    target_steps = int(registration["resume_amendment_20260805"]["stop_for_strategy_review_at_step"])
    extension_used = False
    abort_reason = None
    resume_source = checkpoint_path if checkpoint_path.is_file() else source_checkpoint_path

    if not resume_source.is_file():
        raise FileNotFoundError("A1 resume source checkpoint is missing")

    if resume_source.is_file():
        saved = torch.load(resume_source, map_location="cpu", weights_only=False)
        if saved["initial_trainable_hash"] != initial_trainable_hash:
            raise RuntimeError("A1 resume initialization mismatch")
        if resume_source == source_checkpoint_path:
            expected_source_sha = registration["resume_amendment_20260805"][
                "resume_source_checkpoints"
            ][f"seed_{seed}_sha256"]
            observed_source_sha = sha256_file(source_checkpoint_path)
            if observed_source_sha != expected_source_sha:
                raise RuntimeError("pre-amendment step-200 checkpoint SHA mismatch")
            if int(saved["step"]) != 200:
                raise RuntimeError("amended resume must start from the audited step-200 checkpoint")
        elif saved.get("amendment_lock_commit") != AMENDMENT_LOCK_COMMIT:
            raise RuntimeError("amended checkpoint was written under a different lock")
        with torch.no_grad():
            current = dict(module.named_parameters())
            for name, value in saved["flow_state"].items():
                current[name].copy_(value.to(device=device, dtype=current[name].dtype))
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["batch_generator_state"])
        _restore_rng(saved["rng"])
        step = int(saved["step"])
        target_steps = int(
            registration["resume_amendment_20260805"]["stop_for_strategy_review_at_step"]
        )
        extension_used = False
        history = list(saved["history"])
        trust_history = list(saved["trust_history"])
        clip_events = list(saved["clip_events"])
        warnings = list(saved["warnings"])
        if resume_source == source_checkpoint_path:
            warnings = [
                {**warning, "lineage": "superseded_pre_amendment_contract"}
                for warning in warnings
            ]
        matched_share_audits = list(saved.get("matched_share_audits", []))
        dev_share_audits = list(saved.get("dev_share_audits", []))
        print(f"staged_a1_resume arm={arm} step={step} target={target_steps}", flush=True)

    audit_arm = next(row for row in audit_summary["arms"] if int(row["seed"]) == seed)
    audited_step_200 = audit_arm["checkpoints"]["step_200"]["training_matched_primary"]
    audited_step_200_dev = audit_arm["checkpoints"]["step_200"][
        "dev_population_shift_descriptive"
    ]
    if not matched_share_audits:
        matched_share_audits.append(
            {
                "kind": "paper2_phase2_staged_a1_matched_share_external_audit_v1",
                "step": 200,
                "seed": seed,
                "population": audit_summary["primary_population"],
                "batches": int(audited_step_200["batches"]),
                "batch_size": int(audited_step_200["batch_size"]),
                "mean_gradient_norms": audited_step_200["mean_gradient_norms"],
                "shares": audited_step_200["aggregate_shares"],
                "contract": audited_step_200["aggregate_contract"],
                "batch_contract_fraction": audited_step_200["batch_contract_fraction"],
                "optimizer_updates": 0,
                "audit_receipt_lf_sha256": registration["resume_amendment_20260805"][
                    "audit_receipt_lf_sha256"
                ],
            }
        )
    if not dev_share_audits:
        dev_share_audits.append(
            {
                "kind": "paper2_phase2_staged_a1_dev_share_external_audit_v1",
                "step": 200,
                "seed": seed,
                "population": "fixed_51_batch_dev_population_shift_descriptive",
                "batches": int(audited_step_200_dev["batches"]),
                "batch_size": int(audited_step_200_dev["batch_size"]),
                "mean_gradient_norms": audited_step_200_dev["mean_gradient_norms"],
                "shares": audited_step_200_dev["aggregate_shares"],
                "contract_descriptive": audited_step_200_dev["aggregate_contract"],
                "batch_contract_fraction_descriptive": audited_step_200_dev[
                    "batch_contract_fraction"
                ],
                "optimizer_updates": 0,
                "verdict_role": "none",
                "audit_receipt_lf_sha256": registration["resume_amendment_20260805"][
                    "audit_receipt_lf_sha256"
                ],
            }
        )

    def save() -> None:
        payload = {
            "kind": "paper2_phase2_staged_a1_resume_amended_v2",
            "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
            "amendment_lock_commit": AMENDMENT_LOCK_COMMIT,
            "loss_mask_contract": LOSS_MASK_CONTRACT,
            "seed": seed,
            "step": step,
            "target_steps": target_steps,
            "extension_used": extension_used,
            "initial_trainable_hash": initial_trainable_hash,
            "initial_frozen_hash": initial_frozen_hash,
            "flow_state": {
                name: value.detach().cpu()
                for name, value in module.named_parameters()
                if name.startswith("flow.")
            },
            "optimizer": optimizer.state_dict(),
            "history": history,
            "trust_history": trust_history,
            "clip_events": clip_events,
            "warnings": warnings,
            "matched_share_audits": matched_share_audits,
            "dev_share_audits": dev_share_audits,
            "source_step_200_checkpoint": {
                "path": str(source_checkpoint_path),
                "sha256": registration["resume_amendment_20260805"][
                    "resume_source_checkpoints"
                ][f"seed_{seed}_sha256"],
            },
            "batch_generator_state": generator.get_state(),
            "rng": _rng_state(),
        }
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)

    matched_batches = _calibration_measurement_batches(
        train_indices, seed=seed, registration=registration
    )
    dev_share_batches = _fixed_dev_share_batches(eval_indices, registration=registration)

    identity_batch = _batch(cache, eval_indices[:16], alpha=0.5, device=device)
    zero_loop_identity = _assert_zero_loop_identity(module=module, batch=identity_batch)
    if not history:
        evaluation = evaluate_a1(
            module=module,
            cache=cache,
            eval_indices=eval_indices,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            huber_delta=huber_delta,
            device=device,
        )
        history.append({"step": 0, **evaluation})
        save()

    while step < target_steps and abort_reason is None:
        step += 1
        selected = train_indices.index_select(
            0,
            torch.randint(
                train_indices.numel(),
                (int(registration["batch_size"]),),
                generator=generator,
            ),
        )
        batch = _batch(cache, selected, alpha=0.5, device=device)
        losses, metrics = _a1_losses(
            module=module,
            batch=batch,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            huber_delta=huber_delta,
        )
        total = sum(weights[name] * losses[name] for name in weights)
        if not bool(torch.isfinite(total)):
            abort_reason = "nonfinite_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if any(
            value.grad is not None
            for name, value in module.named_parameters()
            if not name.startswith("flow.")
        ):
            abort_reason = "frozen_parameter_received_gradient"
            break
        active = [value for value in module.flow.parameters() if value.grad is not None]
        raw_norm = torch.nn.utils.clip_grad_norm_(active, clip_ceiling)
        clip_events.append(float(raw_norm) > clip_ceiling)
        learning_rate = 3e-4 * min(1.0, step / 100.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        step_exceeds = float(metrics["endpoint_ratio"].mean()) > float(
            registration["trust"]["endpoint_ratio_tripwire"]
        )
        trust_history.append(step_exceeds)
        if step > int(registration["trust"]["starts_after_optimizer_step"]) and trust_tripwire(
            trust_history,
            window=int(registration["trust"]["rolling_window"]),
            maximum_exceeding=int(registration["trust"]["maximum_exceeding_steps"]),
        ):
            abort_reason = "endpoint_ratio_catastrophe"
            break

        if step % int(registration["evaluation_every"]) == 0:
            evaluation = evaluate_a1(
                module=module,
                cache=cache,
                eval_indices=eval_indices,
                embedding=embedding,
                teacher_embedding=teacher_embedding,
                decoder=decoder,
                decoder_bias=decoder_bias,
                huber_delta=huber_delta,
                device=device,
            )
            hard_share_steps = set(
                registration["resume_amendment_20260805"]["hard_share_estimator"][
                    "optimizer_steps"
                ]
            )
            matched_share = None
            if step in hard_share_steps:
                matched_share = _matched_share_audit(
                    module=module,
                    batches=matched_batches,
                    cache=cache,
                    embedding=embedding,
                    teacher_embedding=teacher_embedding,
                    decoder=decoder,
                    decoder_bias=decoder_bias,
                    huber_delta=huber_delta,
                    weights=weights,
                    device=device,
                    seed=seed,
                    step=step,
                    flow_minimum=float(
                        registration["resume_amendment_20260805"]["hard_share_contract"][
                            "flow_minimum"
                        ]
                    ),
                    probe_maximum=float(
                        registration["resume_amendment_20260805"]["hard_share_contract"][
                            "functional_probe_kl_maximum"
                        ]
                    ),
                    population="training_partition_exact_calibration_batches_50_through_100",
                )
                matched_share_audits.append(matched_share)
                dev_share = _matched_share_audit(
                    module=module,
                    batches=dev_share_batches,
                    cache=cache,
                    embedding=embedding,
                    teacher_embedding=teacher_embedding,
                    decoder=decoder,
                    decoder_bias=decoder_bias,
                    huber_delta=huber_delta,
                    weights=weights,
                    device=device,
                    seed=seed,
                    step=step,
                    flow_minimum=float(
                        registration["resume_amendment_20260805"]["hard_share_contract"][
                            "flow_minimum"
                        ]
                    ),
                    probe_maximum=float(
                        registration["resume_amendment_20260805"]["hard_share_contract"][
                            "functional_probe_kl_maximum"
                        ]
                    ),
                    population="fixed_51_batch_dev_population_shift_descriptive",
                )
                dev_share["verdict_role"] = "none"
                dev_share["contract_descriptive"] = dev_share.pop("contract")
                dev_share_audits.append(dev_share)
                if not matched_share["contract"]["flow_at_least_0p50"]:
                    abort_reason = "amended_flow_share_contract_miss"
                elif not matched_share["contract"]["probe_at_most_0p25"]:
                    abort_reason = "amended_probe_share_contract_miss_recalibration_review"

            preserve_alarm_multiple = float(
                registration["resume_amendment_20260805"]["preserve_loss_alarm"][
                    "threshold_multiple_over_step_zero_dev"
                ]
            )
            preserve_alarm_threshold = preserve_alarm_multiple * float(
                history[0]["counterfactual_preserve_kl"]
            )
            preserve_alarm = float(evaluation["counterfactual_preserve_kl"]) > float(
                preserve_alarm_threshold
            )
            if preserve_alarm:
                warnings.append(
                    {
                        "step": step,
                        "kind": "counterfactual_preserve_loss_value_alarm_log_only",
                        "observed": float(evaluation["counterfactual_preserve_kl"]),
                        "threshold": preserve_alarm_threshold,
                    }
                )
            quality_tripwire_passed = (
                float(evaluation["quality_retention"])
                >= float(registration["quality"]["point_retention_floor"])
                and float(evaluation["quality_wilson_95_lower"])
                >= float(registration["quality"]["wilson_95_lower_floor"])
                and bool(evaluation["quality_noninferior"])
            )
            previous_quality_failed = bool(
                history and history[-1].get("quality_tripwire_passed") is False
            )
            if not quality_tripwire_passed:
                warnings.append({"step": step, "kind": "quality_tripwire_single_failure"})
                if previous_quality_failed:
                    abort_reason = "quality_tripwire_two_consecutive_failures"
            evaluation.update(
                {
                    "step": step,
                    "learning_rate": learning_rate,
                    "total_training_loss": float(total.detach()),
                    "matched_realized_gradient_shares": (
                        matched_share["shares"] if matched_share is not None else None
                    ),
                    "matched_gradient_share_contract": (
                        matched_share["contract"] if matched_share is not None else None
                    ),
                    "counterfactual_preserve_share_role": "descriptive",
                    "counterfactual_preserve_loss_alarm_log_only": preserve_alarm,
                    "counterfactual_preserve_loss_alarm_threshold": preserve_alarm_threshold,
                    "quality_tripwire_passed": quality_tripwire_passed,
                    "clip_active_fraction": sum(clip_events) / len(clip_events),
                    "flow_loss_relative_improvement_from_previous_eval": relative_improvement(
                        float(history[-1]["flow_loss"]), float(evaluation["flow_loss"])
                    ),
                }
            )
            history.append(evaluation)
            print(
                f"staged_a1_eval arm={arm} step={step} "
                f"flow_mse={evaluation['flow_mse']:.6f} "
                f"probe_kl={evaluation['functional_probe_kl']:.6f} "
                f"clip_fraction={evaluation['clip_active_fraction']:.4f}",
                flush=True,
            )
            save()

    save()
    initial = history[0]
    final = history[-1]
    gate_passed = a1_gate(
        initial_probe_kl=float(initial["functional_probe_kl"]),
        final_probe_kl=float(final["functional_probe_kl"]),
        initial_flow_mse=float(initial["flow_mse"]),
        final_flow_mse=float(final["flow_mse"]),
        minimum_probe_improvement=float(
            registration["a1"]["functional_probe_kl_minimum_improvement_nats"]
        ),
    )
    if abort_reason:
        verdict = "blocked"
    elif not gate_passed:
        verdict = "a1_gate_candidate_negative"
    else:
        verdict = "a1_gate_candidate_pass"
    final_frozen_hash = _frozen_hash(module)
    if final_frozen_hash != initial_frozen_hash:
        raise RuntimeError("A1 mutated a stage-frozen parameter")
    source_checkpoint_sha = sha256_file(source_checkpoint_path)
    expected_source_checkpoint_sha = registration["resume_amendment_20260805"][
        "resume_source_checkpoints"
    ][f"seed_{seed}_sha256"]
    if source_checkpoint_sha != expected_source_checkpoint_sha:
        raise RuntimeError("original step-200 checkpoint changed during amended training")
    result = {
        "kind": "paper2_phase2_staged_a1_resume_amended_arm_v2",
        "status": "complete" if abort_reason is None else "blocked_with_receipts",
        "verdict": verdict,
        "abort_reason": abort_reason,
        "seed": seed,
        "alpha": 0.5,
        "step": step,
        "target_steps": target_steps,
        "extension_used": extension_used,
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "amendment_lock_commit": AMENDMENT_LOCK_COMMIT,
        "launcher_commit": _git_head(Path(__file__).resolve().parents[1]),
        "source_hashes": source_hashes,
        "calibration": calibration,
        "history": history,
        "a1_gate_passed": gate_passed,
        "probe_kl_improvement": float(initial["functional_probe_kl"])
        - float(final["functional_probe_kl"]),
        "flow_mse_improvement": float(initial["flow_mse"]) - float(final["flow_mse"]),
        "zero_loop_identity": zero_loop_identity,
        "execution_gates_forced_closed": True,
        "counterfactual_preserve_is_training_only": True,
        "frozen_hash_before": initial_frozen_hash,
        "frozen_hash_after": final_frozen_hash,
        "clip_active_fraction": sum(clip_events) / max(1, len(clip_events)),
        "trust_exceeding_steps": sum(trust_history),
        "trust_observed_steps": len(trust_history),
        "warnings": warnings,
        "matched_share_audits": matched_share_audits,
        "dev_share_audits": dev_share_audits,
        "resume_lineage": {
            "steps_0_through_200_contract": "superseded_symmetric_band",
            "conformance_reestablished_by": str(AUDIT_RECEIPT_RELATIVE),
            "source_checkpoint": {
                "path": str(source_checkpoint_path),
                "sha256": source_checkpoint_sha,
            },
            "source_checkpoint_preserved": True,
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
    }
    write_json(output_dir / f"{arm}.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    registration = _assert_lock(root)
    audit_summary = json.loads((root / AUDIT_RECEIPT_RELATIVE).read_text(encoding="utf-8"))
    constants_path = root / "training/paper2_phase2_dc2_constants.json"
    if sha256_lf_file(constants_path) != registration["constants"]["lf_sha256"]:
        raise RuntimeError("V1d constants file does not match the staged registration")
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30
    ):
        raise RuntimeError("staged A1 requires an A100-class GPU with at least 35 GiB VRAM")
    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.cache,
    )
    if cache["source"]["canonicalizer_sha256"] != registration["canonicalizer"]["sha256"]:
        raise RuntimeError("canonicalizer does not match the staged registration")
    stage0a_summary = json.loads(args.stage0a_summary.read_text(encoding="utf-8"))
    for key in ("data_sha256", "sample_manifest_sha256", "position_key_sha256"):
        observed = (
            stage0a_summary["config"][key]
            if key == "data_sha256"
            else stage0a_summary["manifest"][key]
        )
        if observed != registration["frozen_data"][key]:
            raise RuntimeError(f"Stage 0A {key} differs from the staged registration")
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0]
    train_indices = torch.where(~eval_mask)[0]
    student_summary = json.loads(
        (args.stage0a_private / "model_cache/student_0p5b/summary.json").read_text(
            encoding="utf-8"
        )
    )
    teacher_summary = json.loads(
        (args.stage0a_private / "model_cache/teacher_14b/summary.json").read_text(
            encoding="utf-8"
        )
    )
    student_head = _local_source(student_summary["lm_head"]["path"], args.stage0a_private)
    teacher_head = _local_source(teacher_summary["lm_head"]["path"], args.stage0a_private)
    if sha256_file(student_head) != registration["frozen_data"]["student_lm_head_sha256"]:
        raise RuntimeError("student LM head differs from the staged registration")
    if sha256_file(teacher_head) != teacher_summary["lm_head"]["sha256"]:
        raise RuntimeError("teacher LM head hash mismatch")
    embedding_weight = torch.load(student_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    teacher_weight = torch.load(teacher_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    teacher_embedding = nn.Embedding.from_pretrained(teacher_weight.float(), freeze=True).to(
        args.device
    )
    source_hashes = {
        **cache["source"],
        "student_lm_head_sha256": sha256_file(student_head),
        "teacher_lm_head_sha256": sha256_file(teacher_head),
        "constants_lf_sha256": registration["constants"]["lf_sha256"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        run_seed(
            cache=cache,
            embedding_weight=embedding_weight,
            teacher_embedding=teacher_embedding,
            train_indices=train_indices,
            eval_indices=eval_indices,
            registration=registration,
            seed=int(seed),
            output_dir=args.output_dir,
            private_dir=args.private_dir,
            source_hashes=source_hashes,
            audit_summary=audit_summary,
            device=args.device,
        )
        for seed in registration["seeds"]
    ]
    summary = {
        "kind": "paper2_phase2_staged_a1_resume_amended_v2",
        "status": "complete_with_strategy_gate_required"
        if all(arm["status"] == "complete" for arm in arms)
        else "blocked_with_receipts",
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "amendment_lock_commit": AMENDMENT_LOCK_COMMIT,
        "launcher_commit": _git_head(root),
        "train_anchors": int(train_indices.numel()),
        "evaluation_anchors": int(eval_indices.numel()),
        "document_isolated": True,
        "arms": arms,
        "a2_launched": False,
        "strategy_review_required_before_a2": True,
        "automatic_extension_disabled": True,
        "prior_run_classification": "protocol_bug_not_registered_attempt",
        "frozen_confirmatory_partitions_touched": [],
        "do_not_claim": [
            "A1 state construction establishes useful state use",
            "alpha 0.5 was selected",
            "DEV results are E1 confirmation evidence",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdicts": [arm["verdict"] for arm in result["arms"]],
                "a2_launched": result["a2_launched"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
