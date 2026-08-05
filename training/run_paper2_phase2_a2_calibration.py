"""Measure the zero-update A2 gradient contract on both banked A1 lineages."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules
from training.paper2_phase2_matched_alpha import (
    document_partition,
    masked_sparse_kl,
    normalize_sparse_with_tail,
    reconstruct_sparse_residual_seed,
)
from training.paper2_phase2_staged_repilot import (
    realized_gradient_shares,
    solve_static_weights,
)
from training.run_paper2_phase2_matched_alpha import (
    _batch,
    _local_source,
    _tensor_digest,
    build_pilot_cache,
    sha256_file,
    sha256_lf_file,
    write_json,
)
from training.run_paper2_phase2_staged_a1 import _frozen_hash, _seed_everything


PROTOCOL_RELATIVE = Path("training/paper2_phase2_staged_repilot_preregistration.json")
A1_SUMMARY_KIND = "paper2_phase2_staged_a1_resume_amended_v2"
A1_CHECKPOINT_KIND = "paper2_phase2_staged_a1_resume_amended_v2"
CALIBRATION_KIND = "paper2_phase2_a2_zero_update_calibration_v1"
STRATEGY_DRIVE_ID = "1CCIZqKgIvaveFit8IEOzcXfEcf-4YYWZ"
LOSSES = ("final_ce", "cumulative_kl", "local_ce", "preserve_kl")
PRIMARY_LOSSES = ("cumulative_kl", "local_ce")


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _assert_authorization(root: Path) -> dict[str, Any]:
    registration = json.loads((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    authorization = registration.get("a2_calibration_authorization_20260805", {})
    if authorization.get("status") != "authorized_zero_update_prelock":
        raise RuntimeError("A2 zero-update calibration is not authorized")
    if authorization.get("strategy_drive_id") != STRATEGY_DRIVE_ID:
        raise RuntimeError("A2 calibration strategy authority differs from the registered resource")
    if authorization.get("optimizer_updates") != 0:
        raise RuntimeError("A2 calibration authorization must forbid optimizer updates")
    if authorization.get("a2_training_authorized") is not False:
        raise RuntimeError("A2 training must remain closed during calibration")
    return registration


def _set_a2_trainable(module: Phase2StudentModules) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for name, parameter in module.named_parameters():
        if (
            name.startswith("bridge.")
            or name.startswith("control.")
            or name.startswith("draft.down.")
            or name.startswith("draft.up.")
            or name.startswith("draft.write_gate.")
        ):
            parameter.requires_grad_(True)


def _a2_losses(
    *,
    module: Phase2StudentModules,
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
) -> dict[str, torch.Tensor]:
    hidden4 = batch["hidden"].float()
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    candidate_embeddings = embedding(batch["candidate_ids"].clamp_min(0)).float()
    raw_base_candidate_logits = torch.einsum(
        "bhd,bhcd->bhc", hidden4, candidate_embeddings
    )
    residual_seed = reconstruct_sparse_residual_seed(
        batch["base_candidates"], raw_base_candidate_logits, batch["candidate_mask"]
    ).detach()
    output = module(
        hidden=hidden,
        previous_logits=residual_seed,
        steps=1,
        attention_mask=attention,
        position_bucket=batch["position_bucket"],
        target_scratch=batch["target_scratch"].float(),
        apply_trust_penalty=False,
        candidate_ids=batch["candidate_ids"],
    )
    draft_log = normalize_sparse_with_tail(
        output.logits, batch["base_tail"], batch["candidate_mask"]
    )
    target_log = normalize_sparse_with_tail(
        batch["teacher_candidates"], batch["teacher_tail"], batch["candidate_mask"]
    ).detach()
    base_log = normalize_sparse_with_tail(
        batch["base_candidates"], batch["base_tail"], batch["candidate_mask"]
    ).detach()
    bridge_delta = torch.einsum(
        "bhd,bhcd->bhc", output.hidden[:, 1:].float() - hidden4, candidate_embeddings
    )
    bridge_log = normalize_sparse_with_tail(
        residual_seed + bridge_delta, batch["base_tail"], batch["candidate_mask"]
    )
    teacher_argmax = target_log.argmax(dim=-1)
    final_ce = F.nll_loss(
        bridge_log.reshape(-1, bridge_log.shape[-1]), teacher_argmax.reshape(-1)
    )
    cumulative_kl = masked_sparse_kl(
        target_log, draft_log, batch["candidate_mask"]
    ).mean()
    # The four cached horizon positions are valid by construction.
    local_ce = F.nll_loss(
        draft_log.reshape(-1, draft_log.shape[-1]), teacher_argmax.reshape(-1)
    )
    preserve = base_log.argmax(dim=-1).eq(teacher_argmax)
    preserve_rows = masked_sparse_kl(
        base_log, bridge_log, batch["candidate_mask"]
    )
    preserve_kl = (
        preserve_rows[preserve].mean()
        if bool(preserve.any())
        else preserve_rows.mean() * 0
    )
    return {
        "final_ce": final_ce,
        "cumulative_kl": cumulative_kl,
        "local_ce": local_ce,
        "preserve_kl": preserve_kl,
    }


def _active_named_parameters(module: Phase2StudentModules) -> list[tuple[str, nn.Parameter]]:
    return [(name, value) for name, value in module.named_parameters() if value.requires_grad]


def _parameter_group(name: str) -> str:
    if name.startswith("bridge."):
        return "bridge"
    if name.startswith("control."):
        return "control"
    if name.startswith("draft."):
        return "draft"
    raise RuntimeError(f"unexpected A2 trainable parameter: {name}")


def _loss_gradients(
    losses: dict[str, torch.Tensor], named: list[tuple[str, nn.Parameter]]
) -> dict[str, list[torch.Tensor]]:
    parameters = [value for _name, value in named]
    output: dict[str, list[torch.Tensor]] = {}
    for index, name in enumerate(LOSSES):
        gradients = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=index + 1 < len(LOSSES),
            allow_unused=True,
        )
        output[name] = [
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)
        ]
    return output


def _dot(left: list[torch.Tensor], right: list[torch.Tensor], selected: list[int]) -> float:
    return sum(
        float(left[index].double().mul(right[index].double()).sum()) for index in selected
    )


def _norm(values: list[torch.Tensor], selected: list[int]) -> float:
    return math.sqrt(max(0.0, _dot(values, values, selected)))


def _cosine(
    left: list[torch.Tensor], right: list[torch.Tensor], selected: list[int]
) -> float | None:
    left_norm = _norm(left, selected)
    right_norm = _norm(right, selected)
    if left_norm <= 1e-30 or right_norm <= 1e-30:
        return None
    return _dot(left, right, selected) / (left_norm * right_norm)


def _distribution(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "minimum": float(tensor.min()),
        "p01": float(torch.quantile(tensor, 0.01)),
        "p05": float(torch.quantile(tensor, 0.05)),
        "p25": float(torch.quantile(tensor, 0.25)),
        "median": float(torch.quantile(tensor, 0.50)),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0,
        "p75": float(torch.quantile(tensor, 0.75)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "maximum": float(tensor.max()),
    }


def _weighted_total_norm(
    gradients: dict[str, list[torch.Tensor]],
    weights: dict[str, float],
    selected: list[int],
) -> float:
    squared = 0.0
    for left in LOSSES:
        for right in LOSSES:
            squared += (
                weights[left]
                * weights[right]
                * _dot(gradients[left], gradients[right], selected)
            )
    return math.sqrt(max(0.0, squared))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def calibrate_seed(
    *,
    seed: int,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    a1_arm: dict[str, Any],
    cache: dict[str, Any],
    train_indices: torch.Tensor,
    embedding_weight: torch.Tensor,
    registration: dict[str, Any],
    private_dir: Path,
    device: str,
) -> dict[str, Any]:
    observed_checkpoint_sha = sha256_file(checkpoint_path)
    if observed_checkpoint_sha != expected_checkpoint_sha256:
        raise RuntimeError(f"seed {seed} A1 checkpoint SHA mismatch")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("kind") != A1_CHECKPOINT_KIND or int(saved.get("seed", -1)) != seed:
        raise RuntimeError(f"seed {seed} A1 checkpoint identity mismatch")
    if int(saved.get("step", -1)) != 1000:
        raise RuntimeError(f"seed {seed} A1 checkpoint is not the banked step-1000 endpoint")

    _seed_everything(seed)
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(device)
    module = Phase2StudentModules(
        tied_embedding=embedding,
        hidden_size=896,
        rms_cap=float(registration["constants"]["state_rms_cap"]),
    ).to(device=device, dtype=torch.float32)
    if _frozen_hash(module) != saved["initial_frozen_hash"]:
        raise RuntimeError(f"seed {seed} non-flow A1 initialization did not reconstruct")
    current_parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in saved["flow_state"].items():
            if not name.startswith("flow.") or name not in current_parameters:
                raise RuntimeError(f"seed {seed} checkpoint contains an unexpected flow key: {name}")
            current_parameters[name].copy_(value.to(current_parameters[name]))
    _set_a2_trainable(module)
    module.train()

    named = _active_named_parameters(module)
    all_indices = list(range(len(named)))
    group_indices = {
        group: [index for index, (name, _value) in enumerate(named) if _parameter_group(name) == group]
        for group in ("bridge", "control", "draft")
    }
    if any(not indices for indices in group_indices.values()):
        raise RuntimeError("A2 parameter grouping produced an empty trainable module")
    if any(name.startswith(("flow.", "initializer.", "draft.tied_embedding.")) for name, _ in named):
        raise RuntimeError("A2 calibration exposed a frozen parameter to gradients")

    full_hash_before = _tensor_digest(dict(module.named_parameters()))
    flow_hash_before = _tensor_digest(module.flow.state_dict())
    trainable_hash_before = _tensor_digest(dict(named))
    generator = torch.Generator().manual_seed(seed + 34001)
    sampled = [
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
    first = int(registration["calibration"]["measurement_first_batch"])
    last = int(registration["calibration"]["measurement_last_batch"])
    batch_rows: list[dict[str, Any]] = []
    for batch_number in range(first, last + 1):
        batch = _batch(cache, sampled[batch_number - 1], alpha=0.5, device=device)
        losses = _a2_losses(module=module, batch=batch, embedding=embedding)
        if any(not bool(torch.isfinite(value)) for value in losses.values()):
            raise RuntimeError(f"seed {seed} batch {batch_number} produced non-finite A2 loss")
        gradients = _loss_gradients(losses, named)
        raw = {loss: _norm(gradients[loss], all_indices) for loss in LOSSES}
        by_group = {
            group: {loss: _norm(gradients[loss], indices) for loss in LOSSES}
            for group, indices in group_indices.items()
        }
        global_cosines: dict[str, float | None] = {}
        group_cosines: dict[str, dict[str, float | None]] = {
            group: {} for group in group_indices
        }
        for left_index, left in enumerate(LOSSES):
            for right in LOSSES[left_index + 1 :]:
                key = f"{left}__{right}"
                global_cosines[key] = _cosine(
                    gradients[left], gradients[right], all_indices
                )
                for group, indices in group_indices.items():
                    group_cosines[group][key] = _cosine(
                        gradients[left], gradients[right], indices
                    )
        batch_rows.append(
            {
                "batch": batch_number,
                "loss_values": {name: float(value.detach()) for name, value in losses.items()},
                "raw_gradient_norms": raw,
                "raw_gradient_norms_by_parameter_group": by_group,
                "global_conflict_cosines": global_cosines,
                "conflict_cosines_by_parameter_group": group_cosines,
            }
        )
        if batch_number == first or batch_number % 10 == 0 or batch_number == last:
            print(f"a2_calibration seed={seed} batch={batch_number}/{last}", flush=True)

    mean_norms = {loss: _mean([row["raw_gradient_norms"][loss] for row in batch_rows]) for loss in LOSSES}
    legacy_targets = {name: float(value) for name, value in registration["a2"]["target_gradient_shares"].items()}
    legacy_weights = solve_static_weights(
        mean_norms,
        legacy_targets,
        anchor="final_ce",
        minimum_norm=float(registration["calibration"]["minimum_gradient_norm"]),
    )
    for row in batch_rows:
        weighted = {
            loss: legacy_weights[loss] * row["raw_gradient_norms"][loss] for loss in LOSSES
        }
        denominator = sum(weighted.values())
        row["legacy_weighted_gradient_norms"] = weighted
        row["legacy_independent_gradient_shares"] = {
            loss: weighted[loss] / denominator for loss in LOSSES
        }

    # Recompute combined norms with a second deterministic pass to avoid retaining
    # full gradient tensors in the private receipt.
    combined_norms: list[float] = []
    for batch_number in range(first, last + 1):
        batch = _batch(cache, sampled[batch_number - 1], alpha=0.5, device=device)
        gradients = _loss_gradients(
            _a2_losses(module=module, batch=batch, embedding=embedding), named
        )
        combined_norms.append(
            _weighted_total_norm(gradients, legacy_weights, all_indices)
        )
    weighted_p99 = float(torch.quantile(torch.tensor(combined_norms, dtype=torch.float64), 0.99))
    clip_ceiling = weighted_p99 * float(
        registration["clip"]["ceiling_multiplier_over_calibration_p99"]
    )

    full_hash_after = _tensor_digest(dict(module.named_parameters()))
    flow_hash_after = _tensor_digest(module.flow.state_dict())
    trainable_hash_after = _tensor_digest(dict(named))
    if (full_hash_after, flow_hash_after, trainable_hash_after) != (
        full_hash_before,
        flow_hash_before,
        trainable_hash_before,
    ):
        raise RuntimeError(f"seed {seed} zero-update calibration mutated model parameters")

    raw_distributions = {
        loss: _distribution([row["raw_gradient_norms"][loss] for row in batch_rows])
        for loss in LOSSES
    }
    weighted_distributions = {
        loss: _distribution(
            [row["legacy_weighted_gradient_norms"][loss] for row in batch_rows]
        )
        for loss in LOSSES
    }
    raw_group_distributions = {
        group: {
            loss: _distribution(
                [
                    row["raw_gradient_norms_by_parameter_group"][group][loss]
                    for row in batch_rows
                ]
            )
            for loss in LOSSES
        }
        for group in group_indices
    }
    conflict_distributions: dict[str, dict[str, float]] = {}
    for key in batch_rows[0]["global_conflict_cosines"]:
        values = [
            row["global_conflict_cosines"][key]
            for row in batch_rows
            if row["global_conflict_cosines"][key] is not None
        ]
        if values:
            conflict_distributions[key] = _distribution(values)
    group_conflict_distributions: dict[str, dict[str, dict[str, float]]] = {}
    for group in group_indices:
        group_conflict_distributions[group] = {}
        for key in batch_rows[0]["conflict_cosines_by_parameter_group"][group]:
            values = [
                row["conflict_cosines_by_parameter_group"][group][key]
                for row in batch_rows
                if row["conflict_cosines_by_parameter_group"][group][key] is not None
            ]
            if values:
                group_conflict_distributions[group][key] = _distribution(values)
    primary_key = "cumulative_kl__local_ce"
    primary_values = [
        row["global_conflict_cosines"][primary_key]
        for row in batch_rows
        if row["global_conflict_cosines"][primary_key] is not None
    ]
    raw_spread = max(mean_norms.values()) / min(mean_norms.values())
    a1_norms = {
        name: float(value)
        for name, value in a1_arm["calibration"]["mean_gradient_norms"].items()
    }
    a1_spread = max(a1_norms.values()) / min(a1_norms.values())
    private_dir.mkdir(parents=True, exist_ok=True)
    private_path = private_dir / f"seed_{seed}_batch_rows.json"
    write_json(private_path, {"kind": f"{CALIBRATION_KIND}_batch_rows", "seed": seed, "rows": batch_rows})

    return {
        "seed": seed,
        "status": "complete_zero_update",
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": observed_checkpoint_sha,
            "step": int(saved["step"]),
        },
        "optimizer_updates": 0,
        "measurement_batches": len(batch_rows),
        "batch_size": int(registration["batch_size"]),
        "loss_contract": {
            "active": list(LOSSES),
            "primary_acceptance_facing": list(PRIMARY_LOSSES),
            "local_ce": "horizon-wise NLL of cumulative draft logits against cached teacher top-1",
        },
        "trainable_parameter_counts": {
            group: sum(named[index][1].numel() for index in indices)
            for group, indices in group_indices.items()
        },
        "raw_mean_gradient_norms": mean_norms,
        "raw_gradient_norm_distributions": raw_distributions,
        "raw_gradient_norm_distributions_by_parameter_group": raw_group_distributions,
        "raw_norm_spread": raw_spread,
        "a1_calibration_raw_norm_spread": a1_spread,
        "raw_norm_spread_exceeds_a1": raw_spread > a1_spread,
        "legacy_initialization": {
            "role": "initialization_target_only_pending_a2_amendment",
            "target_shares": legacy_targets,
            "static_weights": legacy_weights,
            "mean_realized_independent_shares": realized_gradient_shares(
                mean_norms, legacy_weights
            ),
            "weighted_gradient_norm_distributions": weighted_distributions,
            "directional_contract_compatible": (
                legacy_targets["cumulative_kl"] + legacy_targets["local_ce"] >= 0.50
                and legacy_targets["final_ce"] <= 0.25
                and legacy_targets["preserve_kl"] <= 0.25
            ),
        },
        "conflict_cosine_distributions": conflict_distributions,
        "conflict_cosine_distributions_by_parameter_group": group_conflict_distributions,
        "primary_primary_conflict": {
            "pair": primary_key,
            "distribution": _distribution(primary_values),
            "fraction_below_minus_0p5": sum(value < -0.5 for value in primary_values)
            / len(primary_values),
            "mean_below_minus_0p5": _mean(primary_values) < -0.5,
        },
        "clip_observation": {
            "weighted_total_norm_distribution": _distribution(combined_norms),
            "p99": weighted_p99,
            "candidate_catastrophe_tripwire_p99_times_10": clip_ceiling,
            "role": "calibration_input_for_a2_amendment_not_active_shaper",
        },
        "mutation_assertions": {
            "full_parameter_hash_before": full_hash_before,
            "full_parameter_hash_after": full_hash_after,
            "flow_hash_before": flow_hash_before,
            "flow_hash_after": flow_hash_after,
            "a2_trainable_hash_before": trainable_hash_before,
            "a2_trainable_hash_after": trainable_hash_after,
            "all_unchanged": True,
        },
        "private_batch_receipt": {
            "path": str(private_path),
            "sha256": sha256_file(private_path),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    registration = _assert_authorization(root)
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise RuntimeError("A2 calibration requires an A100-class GPU with at least 35 GiB VRAM")
    a1_summary = json.loads(args.a1_summary.read_text(encoding="utf-8"))
    if a1_summary.get("kind") != A1_SUMMARY_KIND:
        raise RuntimeError("A1 source summary kind mismatch")
    if [arm.get("verdict") for arm in a1_summary["arms"]] != [
        "a1_gate_candidate_pass",
        "a1_gate_candidate_pass",
    ]:
        raise RuntimeError("both A1 source arms must be machine candidate passes")
    if registration["a1_strategy_bank_20260805"]["status"] != "banked_a1_pass":
        raise RuntimeError("strategy has not banked the replicated A1 pass")
    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.cache,
    )
    if cache["source"]["canonicalizer_sha256"] != registration["canonicalizer"]["sha256"]:
        raise RuntimeError("canonicalizer does not match the staged registration")
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    train_indices = torch.where(~eval_mask)[0]
    student_summary = json.loads(
        (args.stage0a_private / "model_cache/student_0p5b/summary.json").read_text(
            encoding="utf-8"
        )
    )
    student_head = _local_source(student_summary["lm_head"]["path"], args.stage0a_private)
    if sha256_file(student_head) != registration["frozen_data"]["student_lm_head_sha256"]:
        raise RuntimeError("student LM head differs from the staged registration")
    embedding_weight = torch.load(student_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    arms_by_seed = {int(arm["seed"]): arm for arm in a1_summary["arms"]}
    expected = registration["a2_calibration_authorization_20260805"][
        "a1_checkpoint_sha256_by_seed"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        calibrate_seed(
            seed=seed,
            checkpoint_path=(args.a1_checkpoint_seed_0 if seed == 0 else args.a1_checkpoint_seed_1),
            expected_checkpoint_sha256=expected[str(seed)],
            a1_arm=arms_by_seed[seed],
            cache=cache,
            train_indices=train_indices,
            embedding_weight=embedding_weight,
            registration=registration,
            private_dir=args.private_dir,
            device=args.device,
        )
        for seed in (0, 1)
    ]
    summary = {
        "kind": CALIBRATION_KIND,
        "status": "complete_with_a2_amendment_required",
        "launcher_commit": _git_head(root),
        "strategy_authority": {
            "drive_id": STRATEGY_DRIVE_ID,
            "decision": "BANK_A1_AND_AUTHORIZE_A2_CALIBRATION",
        },
        "a1_banked": True,
        "a1_summary": {"path": str(args.a1_summary), "sha256": sha256_file(args.a1_summary)},
        "optimizer_updates": 0,
        "a2_training_launched": False,
        "draft_head_only_control_launched": False,
        "frozen_confirmatory_partitions_touched": [],
        "train_anchors": int(train_indices.numel()),
        "arms": arms,
        "strategy_review_required_before_a2_training": True,
        "next_action": "fold calibration receipts into a committed A2 amendment lock",
        "v1d_banked": registration["v1d"],
        "do_not_claim": [
            "A2 state use was trained or evaluated",
            "the legacy 35/35/10/20 initialization is the locked A2 contract",
            "DEV calibration is E1 confirmation evidence",
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
    parser.add_argument("--a1_summary", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_0", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_1", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "optimizer_updates": result["optimizer_updates"],
                "a2_training_launched": result["a2_training_launched"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
