"""Read-only audit of the terminal Phase-2 matched-alpha pilot checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules
from training.run_paper2_phase2_matched_alpha import (
    BETAS,
    LOSS_MASK_CONTRACT,
    _batch,
    _decoder_for_alpha,
    _load_trainable_state,
    _local_source,
    _losses,
    _parameter_groups,
    _seed_everything,
    _tensor_digest,
    build_pilot_cache,
    evaluate,
    sha256_file,
    write_json,
)
from training.paper2_phase2_matched_alpha import PROTOCOL_LOCK_COMMIT, document_partition


STRATEGY_DRIVE_ID = "1ByHrPod0MVjVzgrMQcBZub_PM3Mgp-gL"
STRATEGY_SHA256_ABBREVIATED = "b4d8adfd...091012"
PILOT_LAUNCHER_COMMIT = "72082031b50e054a2677e281e7eb96cad8113432"
LOSS_WEIGHTS = {
    "final_ce": 1.0,
    "preserve_kl": 0.1,
    "flow": 1.0,
    "functional_probe_kl": 0.5,
    "cumulative_kl": 1.0,
    "trust": 0.01,
}
CLIP_CEILINGS = {"refiner": 1.0, "bridge": 0.5, "heads": 1.0}


def quantile_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    flat = values.detach().float().cpu().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if not finite.numel():
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean()),
        "p50": float(torch.quantile(finite, 0.50)),
        "p90": float(torch.quantile(finite, 0.90)),
        "p95": float(torch.quantile(finite, 0.95)),
        "p99": float(torch.quantile(finite, 0.99)),
        "max": float(finite.max()),
    }


def effective_gradient_attribution(
    vectors: dict[str, torch.Tensor], *, weights: dict[str, float], ceiling: float
) -> dict[str, Any]:
    names = list(vectors)
    weighted = {name: vectors[name].float() * float(weights[name]) for name in names}
    total = sum(weighted.values(), torch.zeros_like(next(iter(weighted.values()))))
    total_norm = torch.linalg.vector_norm(total)
    clip_scale = min(1.0, float(ceiling) / max(float(total_norm), 1e-30))
    post = {name: value * clip_scale for name, value in weighted.items()}
    norm_sum = sum(float(torch.linalg.vector_norm(value)) for value in post.values())
    post_total = total * clip_scale
    denominator = float(torch.dot(post_total, post_total))
    return {
        "preclip_total_norm": float(total_norm),
        "clip_ceiling": float(ceiling),
        "clip_scale": clip_scale,
        "clipped": clip_scale < 1.0,
        "losses": {
            name: {
                "nominal_weight": float(weights[name]),
                "weighted_preclip_norm": float(torch.linalg.vector_norm(weighted[name])),
                "postclip_norm_share": (
                    float(torch.linalg.vector_norm(post[name])) / norm_sum if norm_sum else 0.0
                ),
                "signed_update_alignment_share": (
                    float(torch.dot(post[name], post_total)) / denominator if denominator else 0.0
                ),
            }
            for name in names
        },
    }


def gradient_attribution(
    *,
    module: Phase2StudentModules,
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
) -> dict[str, Any]:
    module.train()
    module.zero_grad(set_to_none=True)
    losses, _ = _losses(
        module=module,
        batch=batch,
        embedding=embedding,
        teacher_embedding=teacher_embedding,
        decoder=decoder,
        decoder_bias=decoder_bias,
    )
    groups = _parameter_groups(module)
    parameters = [parameter for values in groups.values() for parameter in values]
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    raw_norms: dict[str, dict[str, float]] = {}
    for loss_name, loss in losses.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        cursor = 0
        vectors[loss_name] = {}
        raw_norms[loss_name] = {}
        for group_name, group_parameters in groups.items():
            pieces = []
            for parameter in group_parameters:
                gradient = gradients[cursor]
                cursor += 1
                pieces.append(
                    torch.zeros_like(parameter).reshape(-1)
                    if gradient is None
                    else gradient.detach().reshape(-1)
                )
            vector = torch.cat(pieces).float().cpu()
            vectors[loss_name][group_name] = vector
            raw_norms[loss_name][group_name] = float(torch.linalg.vector_norm(vector))
    module.zero_grad(set_to_none=True)
    module.eval()
    return {
        "raw_loss_module_gradient_norms": raw_norms,
        "postclip_effective_attribution": {
            group_name: effective_gradient_attribution(
                {loss_name: vectors[loss_name][group_name] for loss_name in losses},
                weights=LOSS_WEIGHTS,
                ceiling=CLIP_CEILINGS[group_name],
            )
            for group_name in groups
        },
    }


def demanded_permitted_audit(rows: dict[str, torch.Tensor]) -> dict[str, Any]:
    start = rows["flow_start"].float()
    target = rows["target_scratch"].float()
    state = rows["flow_state"].float()
    start_rms = start.square().mean(dim=(1, 2)).sqrt()
    target_rms = target.square().mean(dim=(1, 2)).sqrt()
    permitted = 0.5 * torch.maximum(start_rms, target_rms)
    by_beta = {}
    for beta in (float(BETAS[0]), 1.0):
        demanded = beta * (target - start)
        demanded_rms = demanded.square().mean(dim=(1, 2)).sqrt()
        demand_ratio = demanded_rms / permitted.clamp_min(1e-12)
        by_beta[str(beta)] = {
            "demanded_over_permitted": quantile_summary(demand_ratio),
            "fraction_demand_exceeds_permission": float((demand_ratio > 1.0).float().mean()),
        }
    beta = float(BETAS[0])
    desired = start + beta * (target - start)
    residual = state - desired
    absolute = residual.abs()
    return {
        "loop": 1,
        "beta": beta,
        "trust_ceiling": 0.5,
        "demanded_over_permitted": by_beta[str(beta)]["demanded_over_permitted"],
        "fraction_demand_exceeds_permission": by_beta[str(beta)][
            "fraction_demand_exceeds_permission"
        ],
        "demanded_over_permitted_by_beta": by_beta,
        "huber_delta": 1.0,
        "huber_linear_regime_fraction": float((absolute > 1.0).float().mean()),
        "huber_quadratic_regime_fraction": float((absolute <= 1.0).float().mean()),
        "huber_absolute_residual": quantile_summary(absolute),
        "effective_coordinate_gradient_magnitude": quantile_summary(absolute.clamp_max(1.0)),
        "realized_endpoint_ratio": quantile_summary(rows["endpoint_ratio"]),
    }


def correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    a = left.detach().float().reshape(-1)
    b = right.detach().float().reshape(-1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.numel() < 2 or not bool((a != a[0]).any()) or not bool((b != b[0]).any()):
        return None
    a = a - a.mean()
    b = b - b.mean()
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def _group_rows(
    *, rows: dict[str, torch.Tensor], labels: list[str], group_name: str
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[str(label)].append(index)
    output = {}
    delta = rows["acceptance_delta"].float()
    base_correct = rows["base_correct_by_horizon"].bool()
    bridge_correct = rows["bridge_correct_by_horizon"].bool()
    quality_loss = (base_correct & ~bridge_correct).any(dim=1)
    gate = rows["draft_gate"].float().mean(dim=1)
    for label, indexes in sorted(grouped.items()):
        selected = torch.tensor(indexes, dtype=torch.long)
        local_delta = delta.index_select(0, selected)
        output[label] = {
            "rows": int(selected.numel()),
            "mean_acceptance_delta": float(local_delta.mean()),
            "acceptance_improved_fraction": float((local_delta > 0).float().mean()),
            "acceptance_worsened_fraction": float((local_delta < 0).float().mean()),
            "quality_loss_row_fraction": float(quality_loss.index_select(0, selected).float().mean()),
            "mean_draft_gate": float(gate.index_select(0, selected).mean()),
        }
    return {"group": group_name, "values": output}


def row_tradeoff_audit(
    *, rows: dict[str, torch.Tensor], cache: dict[str, Any], eval_indices: torch.Tensor
) -> dict[str, Any]:
    delta = rows["acceptance_delta"].float()
    base_correct = rows["base_correct_by_horizon"].bool()
    bridge_correct = rows["bridge_correct_by_horizon"].bool()
    quality_loss = (base_correct & ~bridge_correct).any(dim=1)
    gate = rows["draft_gate"].float().mean(dim=1)
    positions = cache["positions"].index_select(0, eval_indices)
    position_labels = []
    for position in positions.tolist():
        if position == 0:
            position_labels.append("position_0")
        elif position <= 3:
            position_labels.append("position_1_3")
        elif position <= 31:
            position_labels.append("position_4_31")
        elif position <= 127:
            position_labels.append("position_32_127")
        else:
            position_labels.append("position_128_plus")
    strata = [cache["strata"][int(index)] for index in eval_indices.tolist()]
    disagreement = (~base_correct).sum(dim=1)
    disagreement_labels = [f"teacher_disagreements_{int(value)}" for value in disagreement]
    return {
        "rows": int(delta.numel()),
        "mean_acceptance_delta": float(delta.mean()),
        "acceptance_improved_fraction": float((delta > 0).float().mean()),
        "acceptance_worsened_fraction": float((delta < 0).float().mean()),
        "quality_loss_rows": int(quality_loss.sum()),
        "quality_loss_row_fraction": float(quality_loss.float().mean()),
        "draft_gate_vs_acceptance_delta_correlation": correlation(gate, delta),
        "draft_gate_vs_quality_loss_correlation": correlation(gate, quality_loss.float()),
        "by_stratum": _group_rows(rows=rows, labels=strata, group_name="stratum"),
        "by_position": _group_rows(rows=rows, labels=position_labels, group_name="position"),
        "by_teacher_disagreement_count": _group_rows(
            rows=rows, labels=disagreement_labels, group_name="teacher_disagreement_count"
        ),
    }


def historical_row_comparison(
    *, arm: dict[str, Any], exact_rows: dict[str, torch.Tensor]
) -> dict[str, Any]:
    passing = [row for row in arm.get("history", []) if row.get("quality_noninferior")]
    if not passing:
        return {"available": False, "reason": "no quality-passing scheduled evaluation"}
    reference_step = int(passing[-1]["step"])
    reference_path = Path(arm["checkpoint"]["path"]).parent / f"rows_step_{reference_step:04d}.pt"
    if not reference_path.is_file():
        return {
            "available": False,
            "reason": "scheduled row artifact missing",
            "reference_step": reference_step,
            "reference_path": str(reference_path),
        }
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    reference_delta = (
        reference["accepted_length"].float() - reference["base_accepted_length"].float()
    )
    exact_delta = exact_rows["acceptance_delta"].float()
    if reference_delta.shape != exact_delta.shape:
        raise RuntimeError("historical and exact row artifacts do not share the fixed DEV ordering")
    reference_quality = reference["quality_correct"].float()
    exact_quality = exact_rows["quality_correct"].float()
    return {
        "available": True,
        "reference_step": reference_step,
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "abort_step": int(arm["step"]),
        "rows": int(exact_delta.numel()),
        "acceptance_delta_change_mean": float((exact_delta - reference_delta).mean()),
        "acceptance_rows_improved_since_reference": float(
            (exact_delta > reference_delta).float().mean()
        ),
        "acceptance_rows_worsened_since_reference": float(
            (exact_delta < reference_delta).float().mean()
        ),
        "quality_change_mean": float((exact_quality - reference_quality).mean()),
        "quality_rows_worsened_since_reference": float(
            (exact_quality < reference_quality).float().mean()
        ),
        "quality_rows_improved_since_reference": float(
            (exact_quality > reference_quality).float().mean()
        ),
    }


def trust_history_audit(arm: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    history = list(checkpoint.get("trust_history", []))
    scheduled = []
    for row in arm.get("history", []):
        losses = row.get("losses", {})
        weighted_total = sum(float(losses.get(name, 0.0)) * weight for name, weight in LOSS_WEIGHTS.items())
        trust_rent = float(losses.get("trust", 0.0)) * LOSS_WEIGHTS["trust"]
        scheduled.append(
            {
                "step": int(row["step"]),
                "trust_rent": trust_rent,
                "weighted_total_loss": weighted_total,
                "trust_rent_share": trust_rent / weighted_total if weighted_total else 0.0,
            }
        )
    return {
        "trust_active_boolean_steps": int(sum(bool(value) for value in history)),
        "trust_observed_steps": len(history),
        "rolling_stop_rule_reproduced": (
            len(history) >= 100 and sum(bool(value) for value in history[-100:]) > 50
        ),
        "scheduled_evaluation_proxy": scheduled,
        "per_training_step_trust_magnitude_recoverable": False,
        "recoverability_note": (
            "The pilot checkpoint preserves per-step active booleans but not trust-rent magnitudes "
            "or the original training batches. Scheduled evaluation shares are a proxy, not a "
            "reconstruction of the training trajectory."
        ),
    }


def _load_heads(
    *, stage0a_private: Path, protocol: dict[str, Any], device: str
) -> tuple[torch.Tensor, nn.Embedding, dict[str, str]]:
    student_summary = json.loads(
        (stage0a_private / "model_cache/student_0p5b/summary.json").read_text(encoding="utf-8")
    )
    teacher_summary = json.loads(
        (stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    student_path = _local_source(student_summary["lm_head"]["path"], stage0a_private)
    teacher_path = _local_source(teacher_summary["lm_head"]["path"], stage0a_private)
    if sha256_file(student_path) != protocol["frozen_data"]["student_lm_head_sha256"]:
        raise RuntimeError("student LM-head hash differs from the locked pilot")
    if sha256_file(teacher_path) != teacher_summary["lm_head"]["sha256"]:
        raise RuntimeError("teacher LM-head hash differs from the Stage 0A receipt")
    student_weight = torch.load(student_path, map_location="cpu", weights_only=False)["weight_bfloat16"]
    teacher_weight = torch.load(teacher_path, map_location="cpu", weights_only=False)["weight_bfloat16"]
    teacher_embedding = nn.Embedding.from_pretrained(teacher_weight.float(), freeze=True).to(device)
    return student_weight, teacher_embedding, {
        "student_lm_head_sha256": sha256_file(student_path),
        "teacher_lm_head_sha256": sha256_file(teacher_path),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("exact-stop matched-alpha audit requires CUDA inference")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_summary.read_text(encoding="utf-8"))
    if pilot.get("launcher_commit") != PILOT_LAUNCHER_COMMIT:
        raise RuntimeError("pilot summary is not the identifying finite-support run")
    if pilot.get("protocol_lock_commit") != PROTOCOL_LOCK_COMMIT:
        raise RuntimeError("pilot and protocol lock commits differ")
    if pilot.get("frozen_evaluation_partitions_touched"):
        raise RuntimeError("pilot unexpectedly touched a frozen evaluation partition")
    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.cache,
    )
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0]
    student_weight, teacher_embedding, head_hashes = _load_heads(
        stage0a_private=args.stage0a_private, protocol=protocol, device=args.device
    )
    output_arms = []
    args.private_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in pilot["arms"]:
        if arm.get("loss_mask_contract") != LOSS_MASK_CONTRACT:
            raise RuntimeError("arm loss-mask contract differs from identifying run")
        checkpoint_path = Path(arm["checkpoint"]["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing exact-stop checkpoint: {checkpoint_path}")
        if sha256_file(checkpoint_path) != arm["checkpoint"]["sha256"]:
            raise RuntimeError(f"exact-stop checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["step"]) != int(arm["step"]):
            raise RuntimeError("checkpoint does not represent the recorded abort step")
        alpha = float(arm["alpha"])
        seed = int(arm["seed"])
        _seed_everything(seed)
        embedding = nn.Embedding.from_pretrained(student_weight.float(), freeze=True).to(args.device)
        module = Phase2StudentModules(
            tied_embedding=embedding,
            hidden_size=896,
            rms_cap=float(args.rms_cap),
        ).to(device=args.device, dtype=torch.float32)
        initial = {
            name: value for name, value in module.named_parameters() if value.requires_grad
        }
        if _tensor_digest(initial) != checkpoint["initial_hash"]:
            raise RuntimeError("module initialization does not reproduce the checkpoint lineage")
        _load_trainable_state(module, checkpoint["trainable_state"])
        loaded_trainable = {
            name: value for name, value in module.named_parameters() if value.requires_grad
        }
        loaded_trainable_hash = _tensor_digest(loaded_trainable)
        decoder, decoder_bias = _decoder_for_alpha(cache, alpha=alpha, device=args.device)
        exact_summary, rows = evaluate(
            module=module,
            cache=cache,
            indices=eval_indices,
            alpha=alpha,
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            device=args.device,
            batch_size=args.batch_size,
        )
        atlas = gradient_attribution(
            module=module,
            batch=_batch(cache, eval_indices[:16], alpha=alpha, device=args.device),
            embedding=embedding,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
        )
        post_audit_trainable_hash = _tensor_digest(
            {name: value for name, value in module.named_parameters() if value.requires_grad}
        )
        if post_audit_trainable_hash != loaded_trainable_hash:
            raise RuntimeError("read-only audit mutated trainable module parameters")
        arm_name = f"alpha_{str(alpha).replace('.', 'p')}_seed_{seed}"
        private_rows = args.private_dir / f"{arm_name}_exact_step_{int(arm['step']):04d}_rows.pt"
        torch.save(rows, private_rows)
        result = {
            "arm": arm_name,
            "alpha": alpha,
            "seed": seed,
            "abort_reason": arm.get("abort_reason"),
            "abort_step": int(arm["step"]),
            "previous_last_evaluated_step": int(arm["final"]["step"]),
            "exact_abort_evaluation": exact_summary,
            "change_since_previous_evaluation": {
                "retention": exact_summary["retention"] - float(arm["final"]["retention"]),
                "acceptance_delta": (
                    exact_summary["acceptance_delta"] - float(arm["final"]["acceptance_delta"])
                ),
                "flow_validation_loss": (
                    exact_summary["flow_validation_loss"] - float(arm["final"]["flow_validation_loss"])
                ),
            },
            "demanded_permitted_and_huber": demanded_permitted_audit(rows),
            "row_tradeoff": row_tradeoff_audit(
                rows=rows, cache=cache, eval_indices=eval_indices
            ),
            "last_quality_passing_to_abort_rows": historical_row_comparison(
                arm=arm, exact_rows=rows
            ),
            "gradient_attribution": atlas,
            "parameter_immutability": {
                "loaded_sha256": loaded_trainable_hash,
                "post_audit_sha256": post_audit_trainable_hash,
                "unchanged": True,
            },
            "training_trust_recoverability": trust_history_audit(arm, checkpoint),
            "clip_fractions_from_training": arm.get("clip_events", {}),
            "private_rows": {
                "path": str(private_rows),
                "sha256": sha256_file(private_rows),
            },
        }
        write_json(args.output_dir / f"{arm_name}.json", result)
        output_arms.append(result)
        del module, embedding, decoder, decoder_bias, rows
        torch.cuda.empty_cache()

    demanded = defaultdict(list)
    huber = defaultdict(list)
    for arm in output_arms:
        key = str(arm["alpha"])
        demanded[key].append(
            arm["demanded_permitted_and_huber"]["fraction_demand_exceeds_permission"]
        )
        huber[key].append(
            arm["demanded_permitted_and_huber"]["huber_linear_regime_fraction"]
        )
    summary = {
        "kind": "paper2_phase2_matched_alpha_read_only_audit",
        "status": "complete_read_only",
        "strategy_handoff": {
            "drive_id": STRATEGY_DRIVE_ID,
            "sha256_abbreviated_as_reported_in_chat": STRATEGY_SHA256_ABBREVIATED,
            "full_sha256_locked_here": False,
            "revision": "r3",
        },
        "pilot_summary_sha256": sha256_file(args.pilot_summary),
        "pilot_launcher_commit": PILOT_LAUNCHER_COMMIT,
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "evaluation_anchors": int(eval_indices.numel()),
        "document_isolated": True,
        "frozen_evaluation_partitions_touched": [],
        "model_optimizer_constructed": False,
        "model_parameter_updates": 0,
        "head_hashes": head_hashes,
        "guardrail_taxonomy": {
            "tripwires": ["nonfinite", "frozen_lineage_mutation", "identity_failure"],
            "pending_empirical_catastrophe_threshold": ["quality_collapse"],
            "endpoint_qualification_not_catastrophe": ["retention_0p997", "wilson_lower_0p990"],
            "shapers": ["trust_rent", "per_module_gradient_clipping"],
            "standing_rule": "unmeasured shapers observe and log with only a generous catastrophe stop",
        },
        "loss_scale_contract": {
            "source_identical_across_arms": all(
                arm.get("launcher_commit") == PILOT_LAUNCHER_COMMIT for arm in pilot["arms"]
            ),
            "huber_delta": 1.0,
            "cosine_weight": 0.1,
            "clip_ceilings": CLIP_CEILINGS,
            "loss_weights": LOSS_WEIGHTS,
            "effective_shape_measured_by_arm": True,
        },
        "pooled_chain_diagnostics": {
            alpha: {
                "mean_fraction_demand_exceeds_permission": sum(values) / len(values),
                "mean_huber_linear_regime_fraction": sum(huber[alpha]) / len(huber[alpha]),
            }
            for alpha, values in demanded.items()
        },
        "arms": output_arms,
        "audit_limitations": [
            "Per-training-step trust-rent magnitudes were not stored and cannot be reconstructed.",
            "Post-clip attribution is recomputed on the fixed 16-anchor audit batch at each exact checkpoint.",
            "All results are DEV-only and do not touch frozen E1 evaluation partitions.",
            "Endpoint non-inferiority remains a qualification metric, not an empirically grounded catastrophe threshold.",
        ],
        "do_not_claim": [
            "The old alpha arms were geometry-only comparisons.",
            "Scheduled evaluation trust rent reconstructs training-step rent.",
            "An endpoint qualification threshold is a catastrophe tripwire.",
            "The audit selects alpha or authorizes E1.",
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
    parser.add_argument("--pilot_summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--rms_cap", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": len(result["arms"]),
                "evaluation_anchors": result["evaluation_anchors"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
