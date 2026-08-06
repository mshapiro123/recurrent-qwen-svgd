"""Read-only audit of the A2 step-237 gradient-tripwire event."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.paper2_phase2_matched_alpha import build_adamw_groups, document_partition
from training.run_paper2_phase2_a2 import (
    LOSSES,
    _active_named,
    _batch,
    _decoder_for_alpha,
    _forward,
    _load_module,
    _load_trainable_state,
    _tensor_digest,
    evaluate,
)
from training.run_paper2_phase2_a2_calibration import _distribution
from training.run_paper2_phase2_matched_alpha import (
    _local_source,
    build_pilot_cache,
    sha256_file,
    write_json,
)


KIND = "paper2_phase2_a2_tripwire_audit_v1"
STOP_STEP = 237
STOP_ATTEMPT = 238
WINDOW_START = 228
WINDOW_END = 248
BATCH_SIZE = 128


def batch_hash(indices: torch.Tensor) -> str:
    return hashlib.sha256(indices.cpu().contiguous().numpy().tobytes()).hexdigest()


def reconstruct_scheduled_batches(
    train_indices: torch.Tensor,
    *,
    seed: int,
    first_attempt: int,
    last_attempt: int,
    batch_size: int = BATCH_SIZE,
) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    result: dict[int, torch.Tensor] = {}
    for attempt in range(1, last_attempt + 1):
        selected = train_indices.index_select(
            0,
            torch.randint(train_indices.numel(), (batch_size,), generator=generator),
        )
        if attempt >= first_attempt:
            result[attempt] = selected
    return result


def matched_reference_batches(
    train_indices: torch.Tensor, *, seed: int, batch_size: int = BATCH_SIZE
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed) + 34001)
    batches = [
        train_indices.index_select(
            0,
            torch.randint(train_indices.numel(), (batch_size,), generator=generator),
        )
        for _ in range(100)
    ]
    return batches[49:100]


def _named_gradient_norm(
    gradients: tuple[torch.Tensor | None, ...],
    named: list[tuple[str, nn.Parameter]],
    prefix: str | None = None,
) -> float:
    total = 0.0
    for gradient, (name, _parameter) in zip(gradients, named):
        if gradient is not None and (prefix is None or name.startswith(prefix)):
            total += float(gradient.detach().double().square().sum())
    return math.sqrt(total)


def _gradient_dot(
    left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...]
) -> float:
    total = 0.0
    for lhs, rhs in zip(left, right):
        if lhs is not None and rhs is not None:
            total += float((lhs.detach().double() * rhs.detach().double()).sum())
    return total


def total_gradient_profile(
    *,
    module: nn.Module,
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
    weights: dict[str, float],
) -> dict[str, Any]:
    named = _active_named(module)
    parameters = [parameter for _name, parameter in named]
    losses, _metrics = _forward(module=module, batch=batch, embedding=embedding, arm="full_a2")
    total = sum(float(weights[name]) * losses[name] for name in LOSSES)
    gradients = torch.autograd.grad(total, parameters, allow_unused=True)
    result = {
        "total_loss": float(total.detach()),
        "total_gradient_norm": _named_gradient_norm(gradients, named),
        "module_gradient_norms": {
            group: _named_gradient_norm(gradients, named, f"{group}.")
            for group in ("bridge", "control", "draft")
        },
        "finite": bool(torch.isfinite(total))
        and all(
            gradient is None or bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
    }
    module.zero_grad(set_to_none=True)
    return result


def detailed_gradient_profile(
    *,
    module: nn.Module,
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
    weights: dict[str, float],
) -> dict[str, Any]:
    named = _active_named(module)
    parameters = [parameter for _name, parameter in named]
    losses, _metrics = _forward(module=module, batch=batch, embedding=embedding, arm="full_a2")
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for index, name in enumerate(LOSSES):
        gradients[name] = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=index + 1 < len(LOSSES),
            allow_unused=True,
        )
    raw_norms = {
        name: _named_gradient_norm(gradients[name], named) for name in LOSSES
    }
    weighted_norms = {
        name: abs(float(weights[name])) * raw_norms[name] for name in LOSSES
    }
    pairwise_cosines: dict[str, float] = {}
    for left_index, left in enumerate(LOSSES):
        for right in LOSSES[left_index + 1 :]:
            denominator = raw_norms[left] * raw_norms[right]
            pairwise_cosines[f"{left}__{right}"] = (
                _gradient_dot(gradients[left], gradients[right]) / denominator
                if denominator > 0
                else 0.0
            )
    by_group = {}
    for group in ("bridge", "control", "draft"):
        by_group[group] = {
            name: {
                "raw_norm": _named_gradient_norm(gradients[name], named, f"{group}."),
                "weighted_norm": abs(float(weights[name]))
                * _named_gradient_norm(gradients[name], named, f"{group}."),
            }
            for name in LOSSES
        }
    module.zero_grad(set_to_none=True)
    return {
        "loss_values": {name: float(losses[name].detach()) for name in LOSSES},
        "raw_gradient_norms": raw_norms,
        "weighted_gradient_norms": weighted_norms,
        "independent_weighted_shares": {
            name: weighted_norms[name] / max(1e-30, sum(weighted_norms.values()))
            for name in LOSSES
        },
        "pairwise_loss_gradient_cosines": pairwise_cosines,
        "loss_gradient_norms_by_module": by_group,
    }


def batch_composition(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    base_top = batch["base_candidates"].argmax(-1)
    teacher_top = batch["teacher_candidates"].argmax(-1)
    buckets = batch["position_bucket"].detach().cpu()
    unique, counts = torch.unique(buckets, return_counts=True)
    return {
        "position_bucket_counts": {
            str(int(key)): int(value) for key, value in zip(unique, counts)
        },
        "base_teacher_top1_agreement": float(base_top.eq(teacher_top).float().mean()),
        "hidden_rms": float(batch["hidden"].float().square().mean().sqrt()),
        "target_rms": float(batch["target_scratch"].float().square().mean().sqrt()),
        "mean_candidates": float(batch["candidate_mask"].float().sum(-1).mean()),
    }


@torch.no_grad()
def population_composition(
    cache: dict[str, Any], indices: torch.Tensor, *, device: str, chunk_size: int = 1024
) -> dict[str, Any]:
    bucket_counts: dict[str, int] = {}
    agreement = 0
    agreement_total = 0
    hidden_sq = 0.0
    hidden_count = 0
    target_sq = 0.0
    target_count = 0
    candidate_count = 0.0
    candidate_rows = 0
    for start in range(0, indices.numel(), chunk_size):
        batch = _batch(
            cache,
            indices[start : start + chunk_size],
            alpha=0.5,
            device=device,
        )
        base_top = batch["base_candidates"].argmax(-1)
        teacher_top = batch["teacher_candidates"].argmax(-1)
        agreement += int(base_top.eq(teacher_top).sum())
        agreement_total += base_top.numel()
        hidden_sq += float(batch["hidden"].float().square().sum())
        hidden_count += batch["hidden"].numel()
        target_sq += float(batch["target_scratch"].float().square().sum())
        target_count += batch["target_scratch"].numel()
        candidate_count += float(batch["candidate_mask"].float().sum())
        candidate_rows += batch["candidate_mask"].shape[0] * batch["candidate_mask"].shape[1]
        unique, counts = torch.unique(batch["position_bucket"].detach().cpu(), return_counts=True)
        for key, value in zip(unique, counts):
            label = str(int(key))
            bucket_counts[label] = bucket_counts.get(label, 0) + int(value)
    return {
        "anchors": int(indices.numel()),
        "position_bucket_counts": bucket_counts,
        "base_teacher_top1_agreement": agreement / max(1, agreement_total),
        "hidden_rms": math.sqrt(hidden_sq / max(1, hidden_count)),
        "target_rms": math.sqrt(target_sq / max(1, target_count)),
        "mean_candidates": candidate_count / max(1, candidate_rows),
    }


def simulated_update(
    *,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    saved_optimizer: dict[str, Any],
    batch: dict[str, torch.Tensor],
    embedding: nn.Embedding,
    weights: dict[str, float],
    cache: dict[str, Any],
    eval_indices: torch.Tensor,
    teacher_embedding: nn.Embedding,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    before_state = {
        name: value.detach().cpu().clone()
        for name, value in module.named_parameters()
        if value.requires_grad
    }
    before_hash = _tensor_digest(dict(_active_named(module)))
    before_eval, _ = evaluate(
        module=module,
        cache=cache,
        indices=eval_indices,
        embedding=embedding,
        teacher_embedding=teacher_embedding,
        decoder=decoder,
        decoder_bias=decoder_bias,
        arm="full_a2",
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    losses, _metrics = _forward(module=module, batch=batch, embedding=embedding, arm="full_a2")
    total = sum(float(weights[name]) * losses[name] for name in LOSSES)
    total.backward()
    optimizer.step()
    named_after = dict(_active_named(module))
    total_parameter_sq = 0.0
    total_delta_sq = 0.0
    group_values: dict[str, dict[str, float]] = {}
    for group in ("bridge", "control", "draft"):
        parameter_sq = 0.0
        delta_sq = 0.0
        for name, after in named_after.items():
            if name.startswith(f"{group}."):
                before = before_state[name].to(after)
                parameter_sq += float(before.double().square().sum())
                delta_sq += float((after.detach().double() - before.double()).square().sum())
        group_values[group] = {
            "parameter_norm": math.sqrt(parameter_sq),
            "update_norm": math.sqrt(delta_sq),
            "relative_update_norm": math.sqrt(delta_sq) / max(1e-30, math.sqrt(parameter_sq)),
        }
        total_parameter_sq += parameter_sq
        total_delta_sq += delta_sq
    finite_after = all(bool(torch.isfinite(value).all()) for value in named_after.values())
    after_eval, _ = evaluate(
        module=module,
        cache=cache,
        indices=eval_indices,
        embedding=embedding,
        teacher_embedding=teacher_embedding,
        decoder=decoder,
        decoder_bias=decoder_bias,
        arm="full_a2",
        device=device,
    )
    _load_trainable_state(module, before_state)
    optimizer.load_state_dict(saved_optimizer)
    optimizer.zero_grad(set_to_none=True)
    restored_hash = _tensor_digest(dict(_active_named(module)))
    if restored_hash != before_hash:
        raise RuntimeError("counterfactual update did not restore the trainable state")
    return {
        "optimizer_update_persisted": False,
        "finite_parameters_after_simulated_update": finite_after,
        "parameter_norm": math.sqrt(total_parameter_sq),
        "update_norm": math.sqrt(total_delta_sq),
        "relative_update_norm": math.sqrt(total_delta_sq)
        / max(1e-30, math.sqrt(total_parameter_sq)),
        "by_module": group_values,
        "dev_before": before_eval,
        "dev_after": after_eval,
        "dev_delta": {
            "mean_accepted_length": after_eval["mean_accepted_length"]
            - before_eval["mean_accepted_length"],
            "retention": after_eval["retention"] - before_eval["retention"],
            "oracle_headroom_relative": after_eval["quality_safe_oracle_headroom_relative"]
            - before_eval["quality_safe_oracle_headroom_relative"],
        },
        "trainable_hash_before": before_hash,
        "trainable_hash_after_restore": restored_hash,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    registration = json.loads(args.protocol.read_text(encoding="utf-8"))
    lock = registration["a2_lock_amendment_20260805"]
    public = json.loads(args.a2_summary.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.cache,
    )
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0]
    train_indices = torch.where(~eval_mask)[0]
    training_composition = population_composition(
        cache, train_indices, device=args.device
    )
    scheduled = reconstruct_scheduled_batches(
        train_indices,
        seed=int(lock["common_training_row_seed"]),
        first_attempt=WINDOW_START,
        last_attempt=WINDOW_END,
    )
    stop_hash = batch_hash(scheduled[STOP_ATTEMPT])
    public_arms = {
        (int(arm["seed"]), arm["arm"]): arm for arm in public["arms"]
    }
    expected_stop_hashes = {
        arm["preupdate_attempts"][-1]["row_hash"]
        for arm in public["arms"]
        if arm["arm"] == "full_a2"
    }
    if expected_stop_hashes != {stop_hash}:
        raise RuntimeError("reconstructed stopping batch does not match the landed receipt")
    student_summary = json.loads(
        (args.stage0a_private / "model_cache/student_0p5b/summary.json").read_text(encoding="utf-8")
    )
    teacher_summary = json.loads(
        (args.stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    student_head = _local_source(student_summary["lm_head"]["path"], args.stage0a_private)
    teacher_head = _local_source(teacher_summary["lm_head"]["path"], args.stage0a_private)
    embedding_weight = torch.load(student_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    teacher_weight = torch.load(teacher_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    teacher_embedding = nn.Embedding.from_pretrained(teacher_weight.float(), freeze=True).to(args.device)
    decoder, decoder_bias = _decoder_for_alpha(cache, alpha=0.5, device=args.device)
    calibration_by_seed = {int(row["seed"]): row for row in calibration["arms"]}
    seed_results = []
    source_hashes_before: dict[str, str] = {}
    for seed in (0, 1):
        print(f"a2_tripwire_audit_seed_start seed={seed}", flush=True)
        public_arm = public_arms[(seed, "full_a2")]
        resume_path = args.resume_checkpoint_seed_0 if seed == 0 else args.resume_checkpoint_seed_1
        a1_path = args.a1_checkpoint_seed_0 if seed == 0 else args.a1_checkpoint_seed_1
        expected_resume_sha = public_arm["checkpoint"]["sha256"]
        expected_a1_sha = lock["a1_checkpoint_sha256_by_seed"][str(seed)]
        if sha256_file(resume_path) != expected_resume_sha:
            raise RuntimeError(f"seed {seed} resume checkpoint SHA mismatch")
        source_hashes_before[str(resume_path)] = expected_resume_sha
        if sha256_file(a1_path) != expected_a1_sha:
            raise RuntimeError(f"seed {seed} A1 checkpoint SHA mismatch")
        source_hashes_before[str(a1_path)] = expected_a1_sha
        embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(args.device)
        module, _source = _load_module(
            seed=seed,
            checkpoint_path=a1_path,
            expected_sha=expected_a1_sha,
            embedding=embedding,
            rms_cap=float(registration["constants"]["state_rms_cap"]),
            device=args.device,
            arm="full_a2",
        )
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(saved["step"]) != STOP_STEP or saved["abort_reason"] != "catastrophe_gradient_norm_tripwire":
            raise RuntimeError(f"seed {seed} is not the registered step-237 tripwire state")
        _load_trainable_state(module, saved["trainable_state"])
        optimizer = torch.optim.AdamW(build_adamw_groups(module, weight_decay=0.01), lr=3e-4)
        optimizer.load_state_dict(saved["optimizer"])
        weights = {
            name: float(value)
            for name, value in lock["full_a2_static_weights_by_seed"][str(seed)].items()
        }
        threshold = float(lock["catastrophe_gradient_norm_by_seed"][str(seed)])
        window_rows = []
        for attempt, indices in scheduled.items():
            profile = total_gradient_profile(
                module=module,
                batch=_batch(cache, indices, alpha=0.5, device=args.device),
                embedding=embedding,
                weights=weights,
            )
            window_rows.append(
                {
                    "attempt": attempt,
                    "row_hash": batch_hash(indices),
                    "is_stopping_batch": attempt == STOP_ATTEMPT,
                    "exceeds_registered_threshold": profile["total_gradient_norm"] > threshold,
                    **profile,
                }
            )
            if attempt in (WINDOW_START, STOP_ATTEMPT, WINDOW_END):
                print(
                    f"a2_tripwire_window seed={seed} attempt={attempt} "
                    f"norm={profile['total_gradient_norm']:.9g}",
                    flush=True,
                )
        reference_rows = []
        for number, indices in enumerate(
            matched_reference_batches(train_indices, seed=seed), start=50
        ):
            profile = total_gradient_profile(
                module=module,
                batch=_batch(cache, indices, alpha=0.5, device=args.device),
                embedding=embedding,
                weights=weights,
            )
            reference_rows.append({"batch": number, **profile})
            if number in (50, 75, 100):
                print(
                    f"a2_tripwire_reference seed={seed} batch={number}/100 "
                    f"norm={profile['total_gradient_norm']:.9g}",
                    flush=True,
                )
        stop_batch = _batch(cache, scheduled[STOP_ATTEMPT], alpha=0.5, device=args.device)
        stop_profile = next(row for row in window_rows if row["is_stopping_batch"])
        reference_norms = [row["total_gradient_norm"] for row in reference_rows]
        window_norms = [row["total_gradient_norm"] for row in window_rows]
        counterfactual = simulated_update(
            module=module,
            optimizer=optimizer,
            saved_optimizer=saved["optimizer"],
            batch=stop_batch,
            embedding=embedding,
            weights=weights,
            cache=cache,
            eval_indices=eval_indices,
            teacher_embedding=teacher_embedding,
            decoder=decoder,
            decoder_bias=decoder_bias,
            device=args.device,
        )
        print(
            f"a2_tripwire_counterfactual seed={seed} "
            f"relative_update={counterfactual['relative_update_norm']:.9g} "
            f"retention_delta={counterfactual['dev_delta']['retention']:.9g}",
            flush=True,
        )
        seed_results.append(
            {
                "seed": seed,
                "checkpoint": {"path": str(resume_path), "sha256": expected_resume_sha},
                "step": STOP_STEP,
                "registered_threshold": threshold,
                "calibration_p99": calibration_by_seed[seed]["clip_observation"]["p99"],
                "stopping_batch": {
                    **stop_profile,
                    "threshold_ratio": stop_profile["total_gradient_norm"] / threshold,
                    "reference_percentile_fraction_leq": sum(
                        value <= stop_profile["total_gradient_norm"] for value in reference_norms
                    )
                    / len(reference_norms),
                    "window_percentile_fraction_leq": sum(
                        value <= stop_profile["total_gradient_norm"] for value in window_norms
                    )
                    / len(window_norms),
                    "composition": batch_composition(stop_batch),
                    "decomposition": detailed_gradient_profile(
                        module=module,
                        batch=stop_batch,
                        embedding=embedding,
                        weights=weights,
                    ),
                },
                "schedule_window": {
                    "attempts": [WINDOW_START, WINDOW_END],
                    "distribution": _distribution(window_norms),
                    "threshold_exceedance_count": sum(value > threshold for value in window_norms),
                    "rows": window_rows,
                },
                "matched_reference": {
                    "batches": len(reference_rows),
                    "distribution": _distribution(reference_norms),
                    "threshold_exceedance_count": sum(value > threshold for value in reference_norms),
                    "rows": reference_rows,
                },
                "simulated_single_update": counterfactual,
            }
        )
        del module, optimizer, embedding
        torch.cuda.empty_cache()
    for path, expected in source_hashes_before.items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"source checkpoint mutated during audit: {path}")
    result = {
        "kind": KIND,
        "status": "complete_descriptive",
        "optimizer_updates_persisted": 0,
        "training_authorized": False,
        "stopping_attempt": STOP_ATTEMPT,
        "stopping_row_hash": stop_hash,
        "train_anchors": int(train_indices.numel()),
        "evaluation_anchors": int(eval_indices.numel()),
        "training_population_composition": training_composition,
        "seeds": seed_results,
        "source_checkpoint_hashes_unchanged": True,
        "interpretation_boundary": (
            "descriptive audit only; continuation requires a separate strategy amendment"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--a2_summary", type=Path, required=True)
    parser.add_argument("--calibration_summary", type=Path, required=True)
    parser.add_argument("--stage0a_summary", type=Path, required=True)
    parser.add_argument("--stage0a_private", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_0", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_1", type=Path, required=True)
    parser.add_argument("--resume_checkpoint_seed_0", type=Path, required=True)
    parser.add_argument("--resume_checkpoint_seed_1", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
