"""Run the locked Phase-2 Option B dose-then-data four-arm matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from eval.eval_paper2_phase2_a2_localization import population_hash_receipt

from training.paper2_phase2_a2 import (
    classify_inflight_quality,
    classify_relative_explosion,
    repeated_marginal_bounds,
)
from training.paper2_phase2_matched_alpha import build_adamw_groups, document_partition
from training.run_paper2_phase2_a2 import (
    LOSSES,
    PRIMARY,
    _active_named,
    _forward,
    _grad_norm,
    _load_module,
    _tensor_digest,
    directional_audit,
    evaluate,
)
from training.run_paper2_phase2_matched_alpha import (
    _decoder_for_alpha,
    _load_trainable_state,
    build_pilot_cache,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "training/paper2_phase2_option_b_preregistration.json"
RULE_INVENTORY = ROOT / "training/paper2_phase2_option_b_rule_inventory.json"
RUN_KIND = "paper2_phase2_option_b_matrix_v1"
ARM_KIND = "paper2_phase2_option_b_arm_v1"
EXPECTED_STATUS = "locked_post_endpoint_reserialization_erratum_training_authorized"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_training_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if registration.get("status") != EXPECTED_STATUS:
        raise RuntimeError("Option B post-generation amendment is not locked")
    for field in ("pre_splice_authorized", "post_splice_authorized", "training_authorized"):
        if registration.get(field) is not True:
            raise RuntimeError(f"Option B authorization is absent: {field}")
    amendment = registration["post_generation_hash_amendment"]
    amendment_path = ROOT / amendment["amendment_document"]
    if amendment_path.stat().st_size != int(amendment["amendment_document_bytes"]):
        raise RuntimeError("Option B hash-amendment byte lock differs")
    if sha256_file(amendment_path) != amendment["amendment_document_sha256"]:
        raise RuntimeError("Option B hash-amendment SHA differs")
    endpoint_erratum = registration["governing_documents"][
        "endpoint_reserialization_erratum"
    ]
    endpoint_erratum_path = ROOT / endpoint_erratum["path"]
    if endpoint_erratum_path.stat().st_size != int(endpoint_erratum["bytes"]):
        raise RuntimeError("Option B endpoint erratum byte lock differs")
    if sha256_file(endpoint_erratum_path) != endpoint_erratum["sha256"]:
        raise RuntimeError("Option B endpoint erratum SHA differs")
    if registration["endpoint_reserialization_erratum"].get(
        "locked_before_option_b_optimizer_updates"
    ) is not True:
        raise RuntimeError("Option B endpoint erratum was not locked before training")
    if int(amendment["recorded_splice_step"]) != int(
        registration["fixed_constants"]["target_splice_step"]
    ):
        raise RuntimeError("Option B recorded splice differs from the fixed target")
    inventory = json.loads(RULE_INVENTORY.read_text(encoding="utf-8"))
    if inventory.get("kind") != "paper2_phase2_option_b_complete_rule_inventory":
        raise RuntimeError("Option B rule inventory is invalid")
    if len(inventory.get("rules", [])) != 18:
        raise RuntimeError("Option B rule inventory is incomplete")
    return registration, inventory


def learning_rate_at_step(step: int, constants: dict[str, Any]) -> float:
    base = float(constants["learning_rate"])
    warmup = int(constants["warmup_steps"])
    stable = int(constants["stable_learning_rate_through_step"])
    total = int(constants["steps"])
    floor = float(constants["minimum_learning_rate_fraction"])
    if step <= 0:
        return 0.0
    if step <= warmup:
        return base * step / warmup
    if step <= stable:
        return base
    progress = min(1.0, (step - stable) / max(1, total - stable))
    return base * (1.0 - progress * (1.0 - floor))


def _pad_last(value: torch.Tensor, width: int, fill: float | int | bool) -> torch.Tensor:
    if value.shape[-1] == width:
        return value
    grown = torch.full((*value.shape[:-1], width), fill, dtype=value.dtype)
    grown[..., : value.shape[-1]] = value
    return grown


def merge_caches(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Concatenate the two frozen populations without changing either row order."""
    for key in ("whiten_basis", "whiten_eigenvalues", "decoder_weight_alpha_0p5", "decoder_bias"):
        if not torch.equal(old[key], new[key]):
            raise RuntimeError(f"Option B canonicalizer buffer differs across populations: {key}")
    width = max(int(old["candidate_ids"].shape[-1]), int(new["candidate_ids"].shape[-1]))
    padded = {
        "candidate_ids": (-1, old["candidate_ids"], new["candidate_ids"]),
        "candidate_mask": (False, old["candidate_mask"], new["candidate_mask"]),
        "base_log_probs": (float("-inf"), old["base_log_probs"], new["base_log_probs"]),
        "teacher_log_probs": (float("-inf"), old["teacher_log_probs"], new["teacher_log_probs"]),
    }
    merged: dict[str, Any] = {
        "kind": "paper2_phase2_option_b_merged_cache_v1",
        "documents": [*old["documents"], *new["documents"]],
        "strata": [*old["strata"], *new["strata"]],
        "source": {"old": old["source"], "new": new["source"]},
    }
    for key, (fill, left, right) in padded.items():
        merged[key] = torch.cat(
            [_pad_last(left, width, fill), _pad_last(right, width, fill)], dim=0
        )
    for key in (
        "positions",
        "student_hidden",
        "target_centered_raw",
        "base_tail",
        "teacher_tail",
        "teacher_topk_ids",
        "teacher_topk_log_probs",
    ):
        merged[key] = torch.cat([old[key], new[key]], dim=0)
    for key in ("whiten_basis", "whiten_eigenvalues", "decoder_weight_alpha_0p5", "decoder_bias"):
        merged[key] = old[key]
    return merged


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def fixed_old_subset(cache: dict[str, Any], train_indices: torch.Tensor) -> torch.Tensor:
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0].tolist()
    target_by_stratum: dict[str, int] = {}
    for index in eval_indices:
        stratum = str(cache["strata"][index])
        target_by_stratum[stratum] = target_by_stratum.get(stratum, 0) + 1
    selected: list[int] = []
    for stratum, target in sorted(target_by_stratum.items()):
        candidates = [
            int(index)
            for index in train_indices.tolist()
            if str(cache["strata"][int(index)]) == stratum
        ]
        candidates.sort(
            key=lambda index: hashlib.sha256(
                (
                    f"20260806:old_train_diagnostic:"
                    f"{cache['documents'][index]}:{index}"
                ).encode("utf-8")
            ).hexdigest()
        )
        selected.extend(candidates[:target])
    selected.sort()
    records = [
        {
            "anchor_index": index,
            "document_id": cache["documents"][index],
            "stratum": cache["strata"][index],
            "prediction_position": int(cache["positions"][index]),
        }
        for index in selected
    ]
    expected = "0f5d114c3dcf6c856956ba9a618f7957f0c3d18c317415c3a1eb23420cd609c5"
    if _canonical_sha256(records) != expected:
        raise RuntimeError("Option B fixed old-train subset differs from the lock")
    return torch.tensor(selected, dtype=torch.long)


def build_option_b_cache(
    *,
    old_summary: Path,
    old_private: Path,
    new_summary: Path,
    new_private: Path,
    canonicalizer: Path,
    old_cache_path: Path,
    new_cache_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    old = build_pilot_cache(
        stage0a_summary_path=old_summary,
        stage0a_private=old_private,
        canonicalizer_path=canonicalizer,
        output_path=old_cache_path,
        expected_samples=200_000,
    )
    new = build_pilot_cache(
        stage0a_summary_path=new_summary,
        stage0a_private=new_private,
        canonicalizer_path=canonicalizer,
        output_path=new_cache_path,
        expected_samples=560_000,
    )
    return old, new, merge_caches(old, new)


def _endpoint_module(
    *,
    seed: int,
    arm: str,
    a1_checkpoint: Path,
    a1_sha: str,
    endpoint: Path,
    endpoint_sha: str,
    endpoint_state_digest: str,
    embedding: nn.Embedding,
    rms_cap: float,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    module, source = _load_module(
        seed=seed,
        checkpoint_path=a1_checkpoint,
        expected_sha=a1_sha,
        embedding=embedding,
        rms_cap=rms_cap,
        device=device,
        arm=arm,
    )
    if sha256_file(endpoint) != endpoint_sha:
        raise RuntimeError(f"Option B endpoint SHA mismatch for seed={seed} arm={arm}")
    saved = torch.load(endpoint, map_location="cpu", weights_only=False)
    if (
        saved.get("kind") != "paper2_phase2_a2_arm_v1"
        or int(saved.get("seed", -1)) != seed
        or saved.get("arm") != arm
        or int(saved.get("step", -1)) != 2000
    ):
        raise RuntimeError(f"Option B endpoint metadata mismatch for seed={seed} arm={arm}")
    observed_state_digest = _tensor_digest(saved["trainable_state"])
    if observed_state_digest != endpoint_state_digest:
        raise RuntimeError(
            f"Option B endpoint state digest mismatch for seed={seed} arm={arm}"
        )
    _load_trainable_state(module, saved["trainable_state"])
    return module, {"a1": source, "a2_endpoint": saved}


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


def _checkpoint_write(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = sha256_file(path)
    reread = torch.load(path, map_location="cpu", weights_only=False)
    if int(reread["step"]) != int(payload["step"]):
        raise RuntimeError("Option B durable checkpoint round-trip failed")
    return digest


def run_arm(
    *,
    seed: int,
    arm: str,
    merged_cache: dict[str, Any],
    old_train_indices: torch.Tensor,
    expanded_train_indices: torch.Tensor,
    old_eval_indices: torch.Tensor,
    fixed_old_indices: torch.Tensor,
    fixed_new_indices: torch.Tensor,
    rms_cap: float,
    embedding_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    a1_checkpoint: Path,
    a1_sha: str,
    endpoint: Path,
    endpoint_sha: str,
    endpoint_state_digest: str,
    registration: dict[str, Any],
    inventory: dict[str, Any],
    output_dir: Path,
    private_dir: Path,
    device: str,
) -> dict[str, Any]:
    name = f"seed_{seed}_{arm}"
    constants = registration["fixed_constants"]
    splice = int(registration["post_generation_hash_amendment"]["recorded_splice_step"])
    target_steps = int(constants["steps"])
    arm_private = private_dir / name
    resume_path = arm_private / "resume.pt"
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(device)
    teacher_embedding = nn.Embedding.from_pretrained(teacher_weight.float(), freeze=True).to(device)
    module, source = _endpoint_module(
        seed=seed,
        arm=arm,
        a1_checkpoint=a1_checkpoint,
        a1_sha=a1_sha,
        endpoint=endpoint,
        endpoint_sha=endpoint_sha,
        endpoint_state_digest=endpoint_state_digest,
        embedding=embedding,
        rms_cap=rms_cap,
        device=device,
    )
    trainable = dict(_active_named(module))
    frozen_before = _tensor_digest(
        {key: value for key, value in module.named_parameters() if not value.requires_grad}
    )
    endpoint_payload = source["a2_endpoint"]
    weights = {key: float(value) for key, value in endpoint_payload["static_loss_weights"].items()}
    active_losses = LOSSES if arm == "full_a2" else PRIMARY
    optimizer = torch.optim.AdamW(build_adamw_groups(module, weight_decay=0.01), lr=0.0)
    schedule_generator = torch.Generator().manual_seed(20260805 + seed)
    audit_generator = torch.Generator().manual_seed(20270805 + seed)
    audit_batches = [
        torch.randint(len(merged_cache["documents"]), (128,), generator=audit_generator)
        for _ in range(int(constants["directional_audit_batches"]))
    ]
    decoder, decoder_bias = _decoder_for_alpha(merged_cache, alpha=0.5, device=device)
    step = 0
    history: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    schedule_hashes: list[str] = []
    observed_anchors: set[int] = set()
    gradient_norms: list[float] = []
    gradient_events: list[dict[str, Any]] = []
    relative_exceedances = 0
    quality_misses = 0
    previous_marginal: list[str] = []
    abort_reason: str | None = None
    splice_receipt: dict[str, Any] | None = None

    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("kind") != ARM_KIND or saved.get("name") != name:
            raise RuntimeError(f"Option B resume identity mismatch for {name}")
        _load_trainable_state(module, saved["trainable_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        step = int(saved["step"])
        history = list(saved["history"])
        audits = list(saved["directional_audits"])
        schedule_hashes = list(saved["schedule_hashes"])
        observed_anchors = set(int(value) for value in saved["observed_anchors"])
        gradient_norms = list(saved["gradient_norms"])
        gradient_events = list(saved["gradient_events"])
        relative_exceedances = int(saved["relative_exceedances"])
        quality_misses = int(saved["quality_misses"])
        previous_marginal = list(saved["previous_marginal"])
        abort_reason = saved.get("abort_reason")
        splice_receipt = saved.get("splice_receipt")
        schedule_generator.set_state(saved["schedule_generator_state"])
        audit_generator.set_state(saved["audit_generator_state"])
        _restore_rng(saved["rng"])
        print(f"option_b_resume arm={name} step={step}", flush=True)

    def save(archive: bool = False) -> dict[str, Any]:
        payload = {
            "kind": ARM_KIND,
            "name": name,
            "seed": seed,
            "arm": arm,
            "step": step,
            "target_steps": target_steps,
            "source_endpoint_sha256": endpoint_sha,
            "trainable_state": {
                key: value.detach().cpu() for key, value in module.named_parameters() if value.requires_grad
            },
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "directional_audits": audits,
            "schedule_hashes": schedule_hashes,
            "observed_anchors": sorted(observed_anchors),
            "gradient_norms": gradient_norms,
            "gradient_events": gradient_events,
            "relative_exceedances": relative_exceedances,
            "quality_misses": quality_misses,
            "previous_marginal": previous_marginal,
            "abort_reason": abort_reason,
            "splice_receipt": splice_receipt,
            "schedule_generator_state": schedule_generator.get_state(),
            "audit_generator_state": audit_generator.get_state(),
            "rng": _rng_state(),
        }
        digest = _checkpoint_write(resume_path, payload)
        if archive:
            archive_path = arm_private / f"checkpoint_step_{step:05d}.pt"
            shutil.copy2(resume_path, archive_path)
            archive_digest = sha256_file(archive_path)
            if archive_digest != digest:
                raise RuntimeError("Option B archive and resume checkpoint differ")
        return {"path": str(resume_path), "sha256": digest}

    def run_evaluations() -> dict[str, Any]:
        evaluations: dict[str, Any] = {}
        for label, indices in (
            ("fixed_evaluation", old_eval_indices),
            ("fixed_old_train", fixed_old_indices),
            ("fixed_new_train", fixed_new_indices),
        ):
            metrics, rows = evaluate(
                module=module,
                cache=merged_cache,
                indices=indices,
                embedding=embedding,
                teacher_embedding=teacher_embedding,
                decoder=decoder,
                decoder_bias=decoder_bias,
                arm=arm,
                device=device,
            )
            evaluations[label] = metrics
            torch.save(rows, arm_private / f"rows_{label}_step_{step:05d}.pt")
        return evaluations

    if not history:
        evaluations = run_evaluations()
        history.append({"step": 0, "learning_rate": 0.0, "evaluations": evaluations})
        if arm == "draft_only_control" and not evaluations["fixed_evaluation"][
            "control_executed_path_bit_exact"
        ]:
            raise RuntimeError("Option B draft-only control failed step-zero identity")
        save(archive=True)

    while step < target_steps and abort_reason is None:
        population = old_train_indices if step < splice else expanded_train_indices
        selected = population.index_select(
            0,
            torch.randint(
                population.numel(),
                (int(constants["batch_size_anchors"]),),
                generator=schedule_generator,
            ),
        )
        selected_hash = hashlib.sha256(selected.numpy().tobytes()).hexdigest()
        batch_cache = merged_cache
        batch = __import__(
            "training.run_paper2_phase2_matched_alpha", fromlist=["_batch"]
        )._batch(batch_cache, selected, alpha=0.5, device=device)
        losses, _metrics = _forward(module=module, batch=batch, embedding=embedding, arm=arm)
        total = sum(weights[key] * losses[key] for key in active_losses)
        if not bool(torch.isfinite(total)):
            abort_reason = "non_finite_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradients = [value.grad for value in trainable.values()]
        if any(value is not None and not bool(torch.isfinite(value).all()) for value in gradients):
            abort_reason = "non_finite_gradient"
            break
        total_norm = _grad_norm(gradients, list(trainable.values()))
        explosion = classify_relative_explosion(
            prior_norms=gradient_norms,
            current_norm=total_norm,
            previous_consecutive_exceedances=relative_exceedances,
            window=100,
            multiplier=10.0,
            consecutive_to_stop=3,
        )
        relative_exceedances = int(explosion["consecutive_exceedances"])
        gradient_events.append({"attempt": step + 1, "norm": total_norm, **explosion})
        if explosion["stop"]:
            abort_reason = "relative_gradient_explosion"
            break
        gradient_norms.append(total_norm)
        next_step = step + 1
        lr = learning_rate_at_step(next_step, constants)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step = next_step
        schedule_hashes.append(selected_hash)
        observed_anchors.update(int(value) for value in selected.tolist())

        if step == splice:
            before = {
                "model": _tensor_digest(dict(module.named_parameters())),
                "optimizer_groups": len(optimizer.param_groups),
                "rule_inventory_sha256": sha256_file(RULE_INVENTORY),
            }
            splice_receipt = {
                "step": step,
                "population_before": int(old_train_indices.numel()),
                "population_after": len(merged_cache["documents"]),
                "state_before_population_switch": before,
                "only_population_version_changed": True,
            }

        if arm == "full_a2" and step % int(constants["directional_audit_cadence_steps"]) == 0:
            audit = {"step": step, **directional_audit(
                module=module,
                cache=merged_cache,
                batches=audit_batches,
                embedding=embedding,
                weights=weights,
                device=device,
            )}
            audits.append(audit)
            if audit["classification"] == "gross":
                abort_reason = "directional_gross_miss"
            elif audit["classification"] == "marginal":
                repeated = repeated_marginal_bounds(previous_marginal, audit["marginal_bounds"])
                if repeated:
                    audit["repeated_bounds"] = repeated
                    abort_reason = "directional_repeated_marginal_miss"
                previous_marginal = list(audit["marginal_bounds"])
            else:
                previous_marginal = []

        if step % int(constants["evaluation_cadence_steps"]) == 0 or abort_reason:
            evaluations = run_evaluations()
            fixed = evaluations["fixed_evaluation"]
            quality = classify_inflight_quality(
                step_zero_retention=float(history[0]["evaluations"]["fixed_evaluation"]["retention"]),
                retention=float(fixed["retention"]),
                wilson_lower=float(fixed["retention_wilson_95_lower"]),
                previous_point_failures=quality_misses,
                point_drop=0.003,
                point_failures_to_stop=2,
                wilson_floor=0.99,
            )
            quality_misses = int(quality["consecutive_point_misses"])
            history.append(
                {
                    "step": step,
                    "learning_rate": lr,
                    "train_total_loss": float(total.detach()),
                    "evaluations": evaluations,
                    "quality_tripwire": quality,
                }
            )
            if arm == "draft_only_control" and not fixed["control_executed_path_bit_exact"]:
                abort_reason = "control_identity"
            elif quality["stop"]:
                abort_reason = str(quality["stop_reason"])
            checkpoint = save(archive=True)
            print(
                f"option_b_eval arm={name} step={step} "
                f"eal={fixed['mean_accepted_length']:.6f} "
                f"retention={fixed['retention']:.6f} abort={abort_reason}",
                flush=True,
            )

    frozen_after = _tensor_digest(
        {key: value for key, value in module.named_parameters() if not value.requires_grad}
    )
    if frozen_after != frozen_before:
        raise RuntimeError(f"Option B frozen-lineage mutation for {name}")
    checkpoint = save(archive=step % int(constants["checkpoint_cadence_steps"]) != 0)
    result = {
        "kind": ARM_KIND,
        "status": "aborted" if abort_reason else "complete",
        "abort_reason": abort_reason,
        "seed": seed,
        "arm": arm,
        "step": step,
        "target_steps": target_steps,
        "source_endpoint": {"path": str(endpoint), "sha256": endpoint_sha},
        "fresh_optimizer_state_at_step_zero": True,
        "active_training_losses": list(active_losses),
        "static_loss_weights": weights,
        "history": history,
        "directional_audits": audits,
        "schedule": {
            "seed": 20260805 + seed,
            "updates": len(schedule_hashes),
            "sha256": hashlib.sha256("\n".join(schedule_hashes).encode("ascii")).hexdigest(),
            "distinct_anchors_observed": len(observed_anchors),
            "per_update_sha256": schedule_hashes,
        },
        "splice_receipt": splice_receipt,
        "gradient_telemetry": {"norms": gradient_norms, "events": gradient_events},
        "frozen_parameter_hash_before": frozen_before,
        "frozen_parameter_hash_after": frozen_after,
        "checkpoint": checkpoint,
        "rule_inventory": inventory,
    }
    write_json(output_dir / f"{name}.json", result)
    return result


def _history_by_step(arm: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["step"]): row for row in arm["history"]}


def _eal(row: dict[str, Any], surface: str = "fixed_evaluation") -> float:
    return float(row["evaluations"][surface]["mean_accepted_length"])


def _segment_slope(history: dict[int, dict[str, Any]], start: int, end: int) -> float | None:
    if start not in history or end not in history or end == start:
        return None
    return (_eal(history[end]) - _eal(history[start])) / ((end - start) / 1000.0)


def _ols_slope(history: dict[int, dict[str, Any]], start: int, end: int) -> float | None:
    points = [(step / 1000.0, _eal(row)) for step, row in history.items() if start <= step <= end]
    if len(points) < 2:
        return None
    x = torch.tensor([row[0] for row in points], dtype=torch.float64)
    y = torch.tensor([row[1] for row in points], dtype=torch.float64)
    centered = x - x.mean()
    return float((centered * (y - y.mean())).sum() / centered.square().sum().clamp_min(1e-12))


def pair_curve_summary(full: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    full_history = _history_by_step(full)
    control_history = _history_by_step(control)
    common_steps = sorted(set(full_history) & set(control_history))
    curve = []
    for step in common_steps:
        full_eal = _eal(full_history[step])
        control_eal = _eal(control_history[step])
        curve.append(
            {
                "step": step,
                "full_eal": full_eal,
                "control_eal": control_eal,
                "writeback_increment": full_eal - control_eal,
                "relative_full_gain": (full_eal - control_eal) / max(control_eal, 1e-12),
                "full_train_eval_gap_old": _eal(full_history[step], "fixed_old_train") - full_eal,
                "control_train_eval_gap_old": _eal(control_history[step], "fixed_old_train")
                - control_eal,
            }
        )
    endpoint = curve[-1] if curve else None
    full_dose = _segment_slope(full_history, 2000, 4000)
    full_fresh = _segment_slope(full_history, 4000, 6000)
    control_dose = _segment_slope(control_history, 2000, 4000)
    control_fresh = _segment_slope(control_history, 4000, 6000)
    second_half = _ols_slope(full_history, 10_000, 20_000)
    return {
        "seed": int(full["seed"]),
        "status": "complete" if full["status"] == control["status"] == "complete" else "blocked",
        "curve": curve,
        "endpoint": endpoint,
        "segment_slopes_eal_per_1000_updates": {
            "full_dose_pre_splice_2000_4000": full_dose,
            "full_fresh_post_splice_4000_6000": full_fresh,
            "control_dose_pre_splice_2000_4000": control_dose,
            "control_fresh_post_splice_4000_6000": control_fresh,
        },
        "full_second_half_exposure_slope_eal_per_1000_updates": second_half,
        "one_percent_relative_endpoint_gain": bool(
            endpoint is not None and endpoint["relative_full_gain"] >= 0.01
        ),
        "positive_second_half_exposure_slope": bool(second_half is not None and second_half > 0),
        "e1_support_trigger": bool(
            endpoint is not None
            and (
                endpoint["relative_full_gain"] >= 0.01
                or (second_half is not None and second_half > 0)
            )
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    registration, inventory = load_training_lock()
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise RuntimeError("Option B requires an A100-class GPU with at least 35 GiB VRAM")
    public = json.loads(args.new_public_summary.read_text(encoding="utf-8"))
    amendment = registration["post_generation_hash_amendment"]
    if sha256_file(args.new_public_summary) != amendment["teacher_cache_summary_sha256"]:
        raise RuntimeError("Option B public teacher-cache summary SHA differs")
    checks = {
        "selected_anchor_count": int(amendment["new_training_anchor_count"]),
        "horizon_sample_count": int(amendment["new_horizon_sample_count"]),
        "sample_manifest_sha256": amendment["new_training_manifest_sha256"],
        "position_key_sha256": amendment["new_position_key_sha256"],
        "lattice_summary_sha256": amendment["new_lattice_summary_sha256"],
    }
    for key, expected in checks.items():
        if public[key] != expected:
            raise RuntimeError(f"Option B landed population mismatch: {key}")
    for key, expected in amendment["model_cache_ledger_hashes"].items():
        if public["model_cache_ledger_hashes"].get(key) != expected:
            raise RuntimeError(f"Option B model-cache ledger mismatch: {key}")
    if public["new_data_sha256"] != amendment["new_data_sha256"]:
        raise RuntimeError("Option B new-data hash differs")
    if public["new_document_id_sha256"] != amendment["new_document_partition_sha256"]:
        raise RuntimeError("Option B new-document hash differs")
    if public["excluded_document_id_sha256"] != amendment["excluded_document_partition_sha256"]:
        raise RuntimeError("Option B exclusion-document hash differs")
    if public["full_logit_audit_sample_keys_sha256"] != amendment[
        "full_logit_audit_sample_keys_sha256"
    ]:
        raise RuntimeError("Option B full-logit audit-set hash differs")
    if int(public["full_logit_audit_samples"]) != int(amendment["full_logit_audit_samples"]):
        raise RuntimeError("Option B full-logit audit count differs")
    if int(public["teacher_14b_state_samples"]) != int(
        amendment["teacher_14b_state_sample_count"]
    ):
        raise RuntimeError("Option B teacher-state coverage count differs")
    if public["fixed_new_train_subset"]["sha256"] != amendment[
        "fixed_new_train_subset_sha256"
    ]:
        raise RuntimeError("Option B fixed-new subset receipt differs")
    if public["per_anchor_label_tier_admission"]["sha256"] != amendment[
        "anchor_admission_ledger_sha256"
    ]:
        raise RuntimeError("Option B admission-ledger receipt differs")
    if public["exclusion_lineage_closure_sha256"] != amendment[
        "exclusion_lineage_closure_sha256"
    ]:
        raise RuntimeError("Option B exclusion-lineage closure differs")
    if public.get("zero_overlap_with_excluded_documents") is not True:
        raise RuntimeError("Option B new documents overlap the excluded population")
    old, new, merged = build_option_b_cache(
        old_summary=args.old_summary,
        old_private=args.old_private,
        new_summary=args.new_full_summary,
        new_private=args.new_private,
        canonicalizer=args.canonicalizer,
        old_cache_path=args.old_cache,
        new_cache_path=args.new_cache,
    )
    old_eval_mask = document_partition(old["documents"], evaluation_fraction=0.2, seed=20260804)
    old_eval = torch.where(old_eval_mask)[0]
    old_train = torch.where(~old_eval_mask)[0]
    if old_train.numel() != 41_969 or old_eval.numel() != 8_031:
        raise RuntimeError("Option B existing population counts differ from the lock")
    old_population = population_hash_receipt(
        {
            "documents": old["documents"],
            "strata": old["strata"],
            "positions": [int(value) for value in old["positions"].tolist()],
        },
        expected_train_anchors=41_969,
        expected_evaluation_anchors=8_031,
    )
    for key in (
        "existing_training_manifest_sha256",
        "existing_document_partition_sha256",
        "evaluation_exclusion_sha256",
        "fixed_old_train_subset_sha256",
    ):
        if old_population[key] != amendment[key]:
            raise RuntimeError(f"Option B existing-population hash differs: {key}")
    fixed_old = fixed_old_subset(old, old_train)
    fixed_new_payload = json.loads(args.fixed_new_subset.read_text(encoding="utf-8"))
    fixed_new_local = torch.tensor(fixed_new_payload["anchor_indices"], dtype=torch.long)
    if sha256_file(args.fixed_new_subset) != amendment["fixed_new_train_subset_sha256"]:
        raise RuntimeError("Option B fixed new subset SHA differs")
    fixed_new = fixed_new_local + len(old["documents"])
    expanded_train = torch.cat(
        [
            old_train,
            torch.arange(
                len(old["documents"]),
                len(old["documents"]) + len(new["documents"]),
                dtype=torch.long,
            ),
        ]
    )
    old_student = json.loads(
        (args.old_private / "model_cache/student_0p5b/summary.json").read_text(encoding="utf-8")
    )
    old_teacher = json.loads(
        (args.old_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    from training.run_paper2_phase2_matched_alpha import _local_source

    student_head = _local_source(old_student["lm_head"]["path"], args.old_private)
    teacher_head = _local_source(old_teacher["lm_head"]["path"], args.old_private)
    embedding_weight = torch.load(student_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    teacher_weight = torch.load(teacher_head, map_location="cpu", weights_only=False)[
        "weight_bfloat16"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    source_registration = json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    rms_cap = float(source_registration["constants"]["state_rms_cap"])
    arms = []
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            arms.append(
                run_arm(
                    seed=seed,
                    arm=arm,
                    merged_cache=merged,
                    old_train_indices=old_train,
                    expanded_train_indices=expanded_train,
                    old_eval_indices=old_eval,
                    fixed_old_indices=fixed_old,
                    fixed_new_indices=fixed_new,
                    rms_cap=rms_cap,
                    embedding_weight=embedding_weight,
                    teacher_weight=teacher_weight,
                    a1_checkpoint=args.a1_checkpoints[seed],
                    a1_sha=args.a1_shas[seed],
                    endpoint=args.endpoints[(seed, arm)],
                    endpoint_sha=registration["source_checkpoints"][
                        f"seed_{seed}_{'full_a2' if arm == 'full_a2' else 'draft_only_control'}"
                    ],
                    endpoint_state_digest=registration[
                        "source_checkpoint_semantic_digests"
                    ][f"seed_{seed}_{arm}"],
                    registration=registration,
                    inventory=inventory,
                    output_dir=args.output_dir,
                    private_dir=args.private_dir,
                    device=args.device,
                )
            )
    pairs = []
    for seed in (0, 1):
        pair = [row for row in arms if row["seed"] == seed]
        common = min(pair[0]["schedule"]["updates"], pair[1]["schedule"]["updates"])
        if (
            pair[0]["schedule"]["per_update_sha256"][:common]
            != pair[1]["schedule"]["per_update_sha256"][:common]
        ):
            raise RuntimeError(f"Option B matched schedule differs for seed {seed}")
        full = next(row for row in pair if row["arm"] == "full_a2")
        control = next(row for row in pair if row["arm"] == "draft_only_control")
        pairs.append(pair_curve_summary(full, control))
    summary = {
        "kind": RUN_KIND,
        "status": "complete" if all(row["status"] == "complete" for row in arms) else "blocked",
        "launcher_commit": _git_head(),
        "hash_amendment": amendment,
        "population": {
            "existing_train_anchors": int(old_train.numel()),
            "existing_evaluation_anchors": int(old_eval.numel()),
            "new_train_anchors": len(new["documents"]),
            "expanded_train_anchors": int(old_train.numel()) + len(new["documents"]),
        },
        "arms": arms,
        "pairs": pairs,
        "scripted_reading": {
            "e1_support_in_both_seeds": all(row["e1_support_trigger"] for row in pairs),
            "e1_support_in_any_seed": any(row["e1_support_trigger"] for row in pairs),
            "interpretation": (
                "curve_supports_E1_recipe_transfer"
                if all(row["e1_support_trigger"] for row in pairs)
                else "mixed_seed_curve_requires_strategy_review"
                if any(row["e1_support_trigger"] for row in pairs)
                else "plateau_below_locked_support_reading"
            ),
        },
        "rule_inventory": inventory,
        "frozen_confirmatory_partitions_touched": [],
        "do_not_claim": [
            "DEV accepted length is serving throughput",
            "this single splice is a general unique-data scaling law",
            "a flat curve proves architectural impossibility",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_public_summary", type=Path, required=True)
    parser.add_argument("--new_full_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--fixed_new_subset", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--old_cache", type=Path, required=True)
    parser.add_argument("--new_cache", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_0", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_1", type=Path, required=True)
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            parser.add_argument(f"--endpoint_seed_{seed}_{arm}", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.a1_checkpoints = {
        0: args.a1_checkpoint_seed_0,
        1: args.a1_checkpoint_seed_1,
    }
    a1_lock = json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )["a2_lock_amendment_20260805"]
    args.a1_shas = {seed: a1_lock["a1_checkpoint_sha256_by_seed"][str(seed)] for seed in (0, 1)}
    args.endpoints = {
        (seed, arm): getattr(args, f"endpoint_seed_{seed}_{arm}")
        for seed in (0, 1)
        for arm in ("full_a2", "draft_only_control")
    }
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"status": result["status"]}, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
