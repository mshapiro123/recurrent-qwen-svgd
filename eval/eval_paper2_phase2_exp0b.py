"""DEV-only Experiment 0B interpolation and serial-flow geometry screening."""

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

from eval.eval_paper2_phase2_exp0a import LATENT_DIM, METHODS, N_SLOTS, _probe_metrics
from training.paper2_phase2_stage0a import sha256_file
from training.paper2_phase2_stage0ab import (
    WHITEN_ALPHAS,
    SharedResidualFlowPilot,
    affine_interpolate,
    finite_quantiles,
)


TAU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FLOW_BETAS = (0.5, 0.6, 1.0)
FLOW_STEPS = 600
FLOW_BATCH_SIZE = 128
FLOW_SEED = 20260805


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _anchor_split(anchor_ids: torch.Tensor, *, seed: int) -> torch.Tensor:
    values = []
    for anchor in anchor_ids.tolist():
        digest = hashlib.sha256(f"{seed}:{int(anchor)}".encode("ascii")).digest()
        values.append(int.from_bytes(digest[:8], "big") / float(2**64 - 1) < 0.8)
    mask = torch.tensor(values, dtype=torch.bool, device=anchor_ids.device)
    if not bool(mask.any()) or not bool((~mask).any()):
        cutoff = max(1, min(anchor_ids.numel() - 1, round(anchor_ids.numel() * 0.8)))
        mask[:] = False
        mask[:cutoff] = True
    return mask


def _paired_endpoints(
    raw: torch.Tensor, anchor_indices: torch.Tensor, horizons: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique = torch.unique(anchor_indices, sorted=True)
    lookup = {int(value): index for index, value in enumerate(unique.tolist())}
    local = torch.tensor(
        [lookup[int(value)] for value in anchor_indices.tolist()], dtype=torch.long
    )
    start = torch.empty((unique.numel(), N_SLOTS, LATENT_DIM), dtype=raw.dtype)
    stop = torch.empty_like(start)
    start[local[horizons.eq(1)]] = raw[horizons.eq(1)]
    stop[local[horizons.eq(4)]] = raw[horizons.eq(4)]
    return unique, local, start, stop


def _target_for_horizon_four(
    targets: dict[str, torch.Tensor], unique_anchors: torch.Tensor
) -> dict[str, torch.Tensor]:
    horizons = targets["horizons"].long()
    anchor_indices = targets["anchor_indices"].long()
    mask = horizons.eq(4)
    lookup = {int(value): index for index, value in enumerate(anchor_indices[mask].tolist())}
    indices = torch.tensor([lookup[int(value)] for value in unique_anchors.tolist()])
    horizon_four_rows = torch.where(mask)[0].index_select(0, indices)
    return {
        key: value.index_select(0, horizon_four_rows)
        for key, value in targets.items()
        if isinstance(value, torch.Tensor) and value.shape[0] == horizons.shape[0]
    }


def _topk_kl_rows(
    *, state: torch.Tensor, decoder: torch.Tensor, decoder_bias: torch.Tensor,
    target_ids: torch.Tensor,
    target_log_probs: torch.Tensor, lm_head: torch.Tensor, batch_size: int = 64
) -> torch.Tensor:
    values = []
    selected = state[:, 3]
    for start in range(0, state.shape[0], batch_size):
        stop = min(state.shape[0], start + batch_size)
        hidden = selected[start:stop] @ decoder + decoder_bias
        ids = target_ids[start:stop].long()
        weights = lm_head.index_select(0, ids.reshape(-1)).view(
            stop - start, ids.shape[1], -1
        )
        logits = torch.einsum("bd,bkd->bk", hidden.to(weights.dtype), weights).float()
        predicted = torch.log_softmax(logits, dim=-1)
        target = torch.log_softmax(target_log_probs[start:stop].float(), dim=-1)
        values.append(torch.sum(target.exp() * (target - predicted), dim=-1).detach().cpu())
    return torch.cat(values)


def _path_metrics(
    *, start: torch.Tensor, stop: torch.Tensor, decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    target: dict[str, torch.Tensor], lm_head: torch.Tensor
) -> dict[str, Any]:
    kl_rows = []
    per_tau: dict[str, Any] = {}
    predicted_hidden = []
    for tau in TAU_GRID:
        state = affine_interpolate(start, stop, tau)
        rows = _topk_kl_rows(
            state=state, decoder=decoder, decoder_bias=decoder_bias,
            target_ids=target["topk_ids"],
            target_log_probs=target["topk_log_probs"], lm_head=lm_head
        )
        kl_rows.append(rows)
        predicted = state[:, 3] @ decoder + decoder_bias
        predicted_hidden.append(predicted)
        per_tau[str(tau)] = {
            "future_topk_kl": finite_quantiles(rows.tolist()),
            "hidden_cosine_to_target_mean": float(
                F.cosine_similarity(predicted, target["final_hidden_bfloat16"].float(), dim=1).mean()
            ),
            "state_rms": float(state.square().mean().sqrt()),
        }
    stack = torch.stack(kl_rows, dim=1)
    differences = stack[:, 1:] - stack[:, :-1]
    monotonic = differences.le(1e-6).all(dim=1)
    hidden_stack = torch.stack(predicted_hidden, dim=1)
    second = hidden_stack[:, 2:] - 2 * hidden_stack[:, 1:-1] + hidden_stack[:, :-2]
    return {
        "tau": per_tau,
        "monotonic_improvement_fraction": float(monotonic.float().mean()),
        "mean_kl_change_start_to_stop": float((stack[:, -1] - stack[:, 0]).mean()),
        "probe_path_second_difference_rms": float(second.square().mean().sqrt()),
        "off_manifold_norm_contraction_midpoint": float(
            affine_interpolate(start, stop, 0.5).norm(dim=-1).mean()
            / torch.stack([start.norm(dim=-1), stop.norm(dim=-1)]).mean().clamp_min(1e-12)
        ),
    }


def _serial_targets(state: torch.Tensor, stop: torch.Tensor, beta: float) -> torch.Tensor:
    return (1.0 - float(beta)) * state.detach() + float(beta) * stop.detach()


def _train_flow(
    *, start: torch.Tensor, stop: torch.Tensor, train_mask: torch.Tensor,
    decoder: torch.Tensor, decoder_bias: torch.Tensor,
    target: dict[str, torch.Tensor], lm_head: torch.Tensor,
    output_path: Path, device: str
) -> dict[str, Any]:
    torch.manual_seed(FLOW_SEED)
    module = SharedResidualFlowPilot(
        latent_dim=LATENT_DIM, context_dim=LATENT_DIM, max_steps=4
    ).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=3e-4, weight_decay=0.01)
    train_indices = torch.where(train_mask)[0]
    losses = []
    for step in range(1, FLOW_STEPS + 1):
        sampled = train_indices[
            torch.randint(0, train_indices.numel(), (FLOW_BATCH_SIZE,), device=train_indices.device)
        ]
        current = start.index_select(0, sampled)
        target_stop = stop.index_select(0, sampled)
        context = current.mean(dim=1)
        loss = torch.zeros((), device=device)
        for loop_index, beta in enumerate(FLOW_BETAS):
            desired = _serial_targets(current, target_stop, beta)
            predicted = module.step(current, context, loop_index)
            huber = F.huber_loss(predicted, desired)
            cosine = 1.0 - F.cosine_similarity(
                predicted.reshape(predicted.shape[0], -1),
                desired.reshape(desired.shape[0], -1),
                dim=1,
            ).mean()
            loss = loss + huber + 0.1 * cosine
            current = predicted
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 100 == 0:
            print(f"exp0b_flow_progress step={step}/{FLOW_STEPS} loss={losses[-1]:.6f}", flush=True)

    validation = ~train_mask
    with torch.no_grad():
        current = start[validation]
        context = current.mean(dim=1)
        for loop_index in range(len(FLOW_BETAS)):
            current = module.step(current, context, loop_index)
        before_mse = F.mse_loss(start[validation], stop[validation])
        after_mse = F.mse_loss(current, stop[validation])
        validation_target = {
            key: value[validation] for key, value in target.items()
        }
        before_probe = _probe_metrics(
            z=start[validation], decoder=decoder, decoder_bias=decoder_bias,
            hidden=validation_target["final_hidden_bfloat16"],
            topk_ids=validation_target["topk_ids"],
            topk_log_probs=validation_target["topk_log_probs"],
            horizons=torch.full((int(validation.sum()),), 4, device=device),
            observed_token_ids=validation_target["observed_token_ids"], lm_head=lm_head
        )
        after_probe = _probe_metrics(
            z=current, decoder=decoder, decoder_bias=decoder_bias,
            hidden=validation_target["final_hidden_bfloat16"],
            topk_ids=validation_target["topk_ids"],
            topk_log_probs=validation_target["topk_log_probs"],
            horizons=torch.full((int(validation.sum()),), 4, device=device),
            observed_token_ids=validation_target["observed_token_ids"], lm_head=lm_head
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "paper2_phase2_exp0b_serial_flow_pilot",
            "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
            "seed": FLOW_SEED,
            "steps": FLOW_STEPS,
        },
        output_path,
    )
    return {
        "optimizer_steps": FLOW_STEPS,
        "train_anchors": int(train_mask.sum()),
        "validation_anchors": int(validation.sum()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_100_mean": sum(losses[-100:]) / min(100, len(losses)),
        "validation_mse_before": float(before_mse),
        "validation_mse_after": float(after_mse),
        "validation_mse_ratio": float(after_mse / before_mse.clamp_min(1e-12)),
        "probe_before": before_probe,
        "probe_after": after_probe,
        "checkpoint": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }


def run_exp0b(
    *, exp0a_summary: Path, exp0a_private: Path, stage0a_private: Path,
    output_private: Path, output_summary: Path, device: str
) -> dict[str, Any]:
    started = time.time()
    source = json.loads(exp0a_summary.read_text(encoding="utf-8"))
    if source.get("status") != "complete_development_only":
        raise RuntimeError("Experiment 0B requires a complete Experiment 0A receipt")
    target_path = Path(source["holdout_targets"]["path"])
    if sha256_file(target_path) != source["holdout_targets"]["sha256"]:
        raise RuntimeError("Experiment 0B holdout-target hash mismatch")
    targets_cpu = torch.load(target_path, map_location="cpu", weights_only=False)
    model_summary = json.loads(
        (stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    head_path = Path(model_summary["lm_head"]["path"])
    if sha256_file(head_path) != model_summary["lm_head"]["sha256"]:
        raise RuntimeError("Experiment 0B 14B LM-head hash mismatch")
    head_payload = torch.load(head_path, map_location="cpu", weights_only=False)
    lm_head = head_payload["weight_bfloat16"].to(device)
    targets = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in targets_cpu.items()
    }
    summaries: dict[str, Any] = {}
    total_optimizer_steps = 0
    output_private.mkdir(parents=True, exist_ok=True)
    for method in source["screening_survivors"]:
        artifact_path = Path(source["methods"][method]["artifact"]["path"])
        endpoint_path = Path(source["methods"][method]["holdout_endpoints"]["path"])
        if sha256_file(artifact_path) != source["methods"][method]["artifact"]["sha256"]:
            raise RuntimeError(f"Experiment 0B canonicalizer hash mismatch: {method}")
        if sha256_file(endpoint_path) != source["methods"][method]["holdout_endpoints"]["sha256"]:
            raise RuntimeError(f"Experiment 0B endpoint hash mismatch: {method}")
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
        endpoint_payload = torch.load(
            endpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        raw = endpoint_payload["raw_centered"].float()
        unique, _local, start_raw, stop_raw = _paired_endpoints(
            raw, targets_cpu["anchor_indices"].long(), targets_cpu["horizons"].long()
        )
        target_cpu = _target_for_horizon_four(targets_cpu, unique)
        target = {key: value.to(device) for key, value in target_cpu.items()}
        method_result: dict[str, Any] = {}
        for alpha in WHITEN_ALPHAS:
            key = str(alpha)
            basis = artifact["screening"]["whiten_basis"].float()
            eigenvalues = artifact["screening"]["whiten_eigenvalues"].float()
            scale = eigenvalues.pow(-0.5 * alpha)
            start_state = ((start_raw @ basis) * scale).to(device)
            stop_state = ((stop_raw @ basis) * scale).to(device)
            decoder_payload = artifact["screening"]["decoders"][key]
            decoder = decoder_payload["weight"].float().to(device)
            decoder_bias = decoder_payload["bias"].float().to(device)
            path = _path_metrics(
                start=start_state, stop=stop_state, decoder=decoder,
                decoder_bias=decoder_bias,
                target=target, lm_head=lm_head
            )
            train_mask = _anchor_split(unique.to(device), seed=FLOW_SEED)
            flow = _train_flow(
                start=start_state, stop=stop_state, train_mask=train_mask,
                decoder=decoder, decoder_bias=decoder_bias,
                target=target, lm_head=lm_head,
                output_path=output_private / f"{method}_alpha_{key}_flow.pt",
                device=device,
            )
            total_optimizer_steps += int(flow["optimizer_steps"])
            method_result[key] = {"path": path, "serial_flow_trainability": flow}
            print(f"exp0b_arm_complete method={method} alpha={alpha}", flush=True)
        summaries[method] = method_result

    screening_survivors = []
    for method, alpha_results in summaries.items():
        for alpha, result in alpha_results.items():
            path = result["path"]
            flow = result["serial_flow_trainability"]
            if (
                math.isfinite(path["mean_kl_change_start_to_stop"])
                and math.isfinite(path["probe_path_second_difference_rms"])
                and math.isfinite(flow["validation_mse_ratio"])
            ):
                screening_survivors.append({"method": method, "alpha": float(alpha)})
    summary = {
        "kind": "paper2_phase2_exp0b_flow_path_screening",
        "status": "complete_development_only",
        "source_exp0a_summary_sha256": sha256_file(exp0a_summary),
        "tau_grid": list(TAU_GRID),
        "flow_betas": list(FLOW_BETAS),
        "geometry_contract": {
            "affine_interpolation": True,
            "target_or_persistent_state_renormalization": False,
            "rms_normalization_scope": "module inputs and innovations only",
            "loop_cap": 4,
        },
        "methods": summaries,
        "screening_survivors": screening_survivors,
        "alpha_selected": False,
        "selection_deferred_until_matched_built_module_pilots": True,
        "flow_pilot_training_started": True,
        "optimizer_steps": total_optimizer_steps,
        "backbone_training_started": False,
        "backbone_parameters_mutated": False,
        "frozen_evaluation_partitions_touched": [],
        "elapsed_seconds": time.time() - started,
        "do_not_claim": [
            "Experiment 0B selects the final alpha",
            "linear probe-path smoothness proves a trained recurrent module will follow the path",
            "the disposable DEV flow pilot is an E1 result",
            "development-only interpolation metrics generalize to frozen evaluation data",
        ],
    }
    write_json(output_summary, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp0a_summary", type=Path, required=True)
    parser.add_argument("--exp0a_private", type=Path, required=True)
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--output_private", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_exp0b(
        exp0a_summary=args.exp0a_summary,
        exp0a_private=args.exp0a_private,
        stage0a_private=args.stage0a_private,
        output_private=args.output_private,
        output_summary=args.output_summary,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
