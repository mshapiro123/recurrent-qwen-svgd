"""Run the locked Phase-2 A2 full-versus-draft-only matrix."""

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
from training.paper2_phase2_a2 import (
    classify_inflight_quality,
    classify_relative_explosion,
    classify_directional_shares,
    final_window_slope,
    paired_verdict,
    relative_oracle_headroom,
    retention_slope_latest,
    repeated_marginal_bounds,
    should_extend,
    validate_guardrail_inventory,
)
from training.paper2_phase2_matched_alpha import (
    build_adamw_groups,
    document_partition,
    expected_accepted_length,
    masked_sparse_kl,
    normalize_sparse_with_tail,
    quality_noninferior,
    reconstruct_sparse_residual_seed,
    wilson_lower,
)
from training.run_paper2_phase2_a2_calibration import _distribution
from training.run_paper2_phase2_matched_alpha import (
    _batch,
    _decoder_for_alpha,
    _load_trainable_state,
    _local_source,
    _tensor_digest,
    build_pilot_cache,
    sha256_file,
    write_json,
)
from training.run_paper2_phase2_staged_a1 import _frozen_hash, _seed_everything


PROTOCOL_RELATIVE = Path("training/paper2_phase2_staged_repilot_preregistration.json")
LOCK_STATUS = "locked_before_a2_training"
RUN_KIND = "paper2_phase2_a2_matrix_v1"
ARM_KIND = "paper2_phase2_a2_arm_v1"
LOSSES = ("final_ce", "cumulative_kl", "local_ce", "preserve_kl")
PRIMARY = ("cumulative_kl", "local_ce")
NON_PRIMARY = ("final_ce", "preserve_kl")
PREUPDATE_ABORT_REASONS = frozenset(
    {
        "nonfinite_loss",
        "nonfinite_gradient",
        "catastrophe_gradient_norm_tripwire",
        "relative_gradient_explosion_tripwire",
    }
)
STEP237_LOCK_KEY = "a2_step237_tripwire_amendment_20260806"
STEP237_LOCK_STATUS = "locked_before_a2_step237_continuation"
RUNTIME_RULE_NAMES = {
    "nonfinite_loss",
    "nonfinite_gradient",
    "relative_gradient_explosion",
    "obsolete_static_gradient_ceiling",
    "source_checkpoint_identity",
    "batch_generator_identity",
    "frozen_parameter_identity",
    "frozen_confirmatory_partition_contact",
    "control_executed_path_identity",
    "quality_wilson_floor",
    "init_relative_retention",
    "negative_retention_slope",
    "directional_gross_miss",
    "directional_repeated_marginal_miss",
    "endpoint_quality_qualification",
    "endpoint_oracle_headroom",
    "paired_full_superiority",
    "single_extension_decision",
}


def reconcile_executed_schedule(
    *,
    step: int,
    batch_hashes: list[str],
    abort_reason: str | None,
    preupdate_attempts: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Repair the legacy schedule definition without clearing a safety stop.

    A schedule entry means that an optimizer update consumed the batch. Older
    A2 code appended before the pre-update tripwires, so an aborted attempt can
    leave exactly one excess hash. No other mismatch is repairable.
    """
    attempts = list(preupdate_attempts or [])
    if len(batch_hashes) == step:
        return list(batch_hashes), attempts
    if len(batch_hashes) == step + 1 and abort_reason in PREUPDATE_ABORT_REASONS:
        rejected_hash = batch_hashes[-1]
        attempts.append(
            {
                "row_hash": rejected_hash,
                "abort_reason": abort_reason,
                "optimizer_update_applied": False,
                "recovered_from_legacy_schedule": True,
            }
        )
        return list(batch_hashes[:-1]), attempts
    raise RuntimeError(
        "unrepairable A2 training schedule state: "
        f"step={step} executed_batch_hashes={len(batch_hashes)} "
        f"abort_reason={abort_reason!r}"
    )


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _assert_lock(
    root: Path, *, require_resume: bool = False, require_step237_resume: bool = False
) -> dict[str, Any]:
    registration = json.loads((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    lock = registration.get("a2_lock_amendment_20260805", {})
    if lock.get("status") != LOCK_STATUS:
        raise RuntimeError("A2 training lock is absent")
    document = root / lock["document"]
    if not document.is_file():
        raise RuntimeError("A2 lock document is absent")
    strategy = lock["strategy_resolution"]
    strategy_path = root / "docs/STRATEGY_TO_CODING_AGENT_A2_LOCK_RESOLUTION_20260805.md"
    if strategy_path.stat().st_size != int(strategy["bytes"]):
        raise RuntimeError("strategy resolution byte count differs from the lock")
    if sha256_file(strategy_path) != strategy["sha256"]:
        raise RuntimeError("strategy resolution SHA differs from the lock")
    if require_resume:
        resume = registration.get("a2_step200_resume_amendment_20260805", {})
        if resume.get("status") != "locked_before_a2_resumed_training":
            raise RuntimeError("A2 step-200 resume lock is absent")
        resume_document = root / resume["document"]
        if not resume_document.is_file():
            raise RuntimeError("A2 step-200 resume amendment is absent")
        resume_strategy = resume["strategy_resolution"]
        resume_strategy_path = root / resume_strategy["document"]
        if resume_strategy_path.stat().st_size != int(resume_strategy["bytes"]):
            raise RuntimeError("A2 resume strategy byte count differs from the lock")
        if sha256_file(resume_strategy_path) != resume_strategy["sha256"]:
            raise RuntimeError("A2 resume strategy SHA differs from the lock")
    if require_step237_resume:
        step237 = registration.get(STEP237_LOCK_KEY, {})
        if step237.get("status") != STEP237_LOCK_STATUS:
            raise RuntimeError("A2 step-237 continuation lock is absent")
        for row, label in (
            (step237, "step-237 amendment"),
            (step237["strategy_resolution"], "step-237 strategy resolution"),
            (step237["guardrail_doctrine"], "guardrail doctrine"),
        ):
            path = root / row["document"]
            expected_bytes = int(row.get("document_bytes", row.get("bytes", -1)))
            expected_sha = row.get("document_sha256", row.get("sha256"))
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise RuntimeError(f"{label} byte lock differs")
            if sha256_file(path) != expected_sha:
                raise RuntimeError(f"{label} SHA lock differs")
        validation = validate_guardrail_inventory(step237["rule_inventory"])
        if not validation["valid"]:
            raise RuntimeError(f"A2 guardrail inventory is invalid: {validation['errors']}")
        inventory_names = {row["name"] for row in step237["rule_inventory"]}
        if inventory_names != RUNTIME_RULE_NAMES:
            raise RuntimeError(
                "A2 runner rules differ from the locked inventory: "
                f"runtime_only={sorted(RUNTIME_RULE_NAMES - inventory_names)} "
                f"lock_only={sorted(inventory_names - RUNTIME_RULE_NAMES)}"
            )
        sweep_path = (
            root
            / "outputs/stage5/stage5_paper2_phase2_a2_guardrail_sweep_20260806/summary.json"
        )
        if not sweep_path.is_file():
            raise RuntimeError("A2 guardrail grounding sweep receipt is absent")
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        if sweep.get("status") != "complete_lock_valid" or sweep.get("lock_key") != STEP237_LOCK_KEY:
            raise RuntimeError("A2 guardrail grounding sweep receipt is invalid")
    return registration


def _set_trainable(module: Phase2StudentModules, *, arm: str) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    prefixes = (
        ("bridge.", "control.", "draft.down.", "draft.up.", "draft.write_gate.")
        if arm == "full_a2"
        else ("control.", "draft.down.", "draft.up.", "draft.write_gate.")
    )
    for name, parameter in module.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)


def _forward(
    *, module: Phase2StudentModules, batch: dict[str, torch.Tensor],
    embedding: nn.Embedding, arm: str,
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

    if arm == "full_a2":
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
        scratch = output.scratch
        draft_logits = output.logits
        draft_gate = output.draft.write_gates
        bridge_delta = torch.einsum(
            "bhd,bhcd->bhc", output.hidden[:, 1:].float() - hidden4, candidate_embeddings
        )
        bridge_sparse = residual_seed + bridge_delta
        bridge_gate = output.bridge.gate.expand(hidden4.shape[0])
    elif arm == "draft_only_control":
        scratch = module.initializer(hidden, attention)
        zeros = hidden.new_zeros((hidden.shape[0],))
        control = module.control(
            scratch=scratch,
            previous=None,
            innovation_norm=zeros,
            student_entropy=zeros,
            top2_margin=zeros,
            position_bucket=batch["position_bucket"],
        )
        draft = module.draft(
            previous_logits=residual_seed,
            scratch=scratch,
            control_state=control,
            candidate_ids=batch["candidate_ids"],
        )
        draft_logits = draft.logits
        draft_gate = draft.write_gates
        bridge_sparse = residual_seed
        bridge_gate = zeros
    else:
        raise ValueError(f"unknown A2 arm: {arm}")

    draft_log = normalize_sparse_with_tail(
        draft_logits, batch["base_tail"], batch["candidate_mask"]
    )
    target_log = normalize_sparse_with_tail(
        batch["teacher_candidates"], batch["teacher_tail"], batch["candidate_mask"]
    ).detach()
    base_log = normalize_sparse_with_tail(
        batch["base_candidates"], batch["base_tail"], batch["candidate_mask"]
    ).detach()
    bridge_log = (
        base_log
        if arm == "draft_only_control"
        else normalize_sparse_with_tail(
            bridge_sparse, batch["base_tail"], batch["candidate_mask"]
        )
    )
    teacher_argmax = target_log.argmax(dim=-1)
    final_ce = F.nll_loss(
        bridge_log.reshape(-1, bridge_log.shape[-1]), teacher_argmax.reshape(-1)
    )
    cumulative_kl = masked_sparse_kl(target_log, draft_log, batch["candidate_mask"]).mean()
    local_ce = F.nll_loss(
        draft_log.reshape(-1, draft_log.shape[-1]), teacher_argmax.reshape(-1)
    )
    preserve = base_log.argmax(dim=-1).eq(teacher_argmax)
    preserve_rows = masked_sparse_kl(base_log, bridge_log, batch["candidate_mask"])
    preserve_kl = preserve_rows[preserve].mean() if bool(preserve.any()) else preserve_rows.mean() * 0
    return (
        {
            "final_ce": final_ce,
            "cumulative_kl": cumulative_kl,
            "local_ce": local_ce,
            "preserve_kl": preserve_kl,
        },
        {
            "draft_log": draft_log,
            "bridge_log": bridge_log,
            "target_log": target_log,
            "base_log": base_log,
            "draft_gate": draft_gate,
            "bridge_gate": bridge_gate,
            "scratch": scratch,
        },
    )


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-12))


@torch.no_grad()
def evaluate(
    *, module: Phase2StudentModules, cache: dict[str, Any], indices: torch.Tensor,
    embedding: nn.Embedding, teacher_embedding: nn.Embedding,
    decoder: torch.Tensor, decoder_bias: torch.Tensor, arm: str,
    device: str, batch_size: int = 128,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    module.eval()
    accum: dict[str, list[torch.Tensor]] = {}
    loss_accum: dict[str, list[float]] = {}
    control_identity = True
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start : start + batch_size]
        batch = _batch(cache, selected, alpha=0.5, device=device)
        losses, metrics = _forward(module=module, batch=batch, embedding=embedding, arm=arm)
        for name, value in losses.items():
            loss_accum.setdefault(name, []).append(float(value))
        if arm == "draft_only_control":
            control_identity = control_identity and torch.equal(metrics["bridge_log"], metrics["base_log"])
        predicted_hidden = torch.einsum(
            "bhd,df->bhf", metrics["scratch"][:, :4].float(), decoder
        ) + decoder_bias
        topk_embeddings = teacher_embedding(batch["teacher_topk_ids"].long()).float()
        probe_logits = torch.einsum("bhf,bhkf->bhk", predicted_hidden, topk_embeddings)
        probe_log = torch.log_softmax(probe_logits, dim=-1)
        probe_target = torch.log_softmax(batch["teacher_topk_log_probs"].float(), dim=-1)
        for name, value in {
            **metrics,
            "probe_log": probe_log,
            "probe_target": probe_target,
        }.items():
            accum.setdefault(name, []).append(value.detach().cpu())
    values = {name: torch.cat(parts) for name, parts in accum.items()}
    overlap = (values["target_log"].exp().minimum(values["draft_log"].exp())).sum(-1)
    base_overlap = (values["target_log"].exp().minimum(values["base_log"].exp())).sum(-1)
    accepted = expected_accepted_length(overlap)
    base_accepted = expected_accepted_length(base_overlap)
    target_top = values["target_log"].argmax(-1)
    base_top = values["base_log"].argmax(-1)
    bridge_top = values["bridge_log"].argmax(-1)
    base_correct = base_top.eq(target_top)
    bridge_correct = bridge_top.eq(target_top)
    retained = bridge_correct & base_correct
    baseline_count = int(base_correct.sum())
    retained_count = int(retained.sum())
    probe_kl = (
        values["probe_target"].exp()
        * (values["probe_target"] - values["probe_log"])
    ).sum(-1).mean(-1)
    probe_top1 = values["probe_target"].argmax(-1).eq(
        values["probe_log"].argmax(-1)
    ).float().mean(-1)
    delta = accepted - base_accepted
    quality_loss = (base_correct & ~bridge_correct).any(dim=1)
    oracle_selected = delta > 0
    safe_selected = oracle_selected & ~quality_loss
    safe_delta = torch.where(safe_selected, delta, torch.zeros_like(delta))
    safe_headroom = relative_oracle_headroom(
        quality_safe_oracle_delta=float(safe_delta.mean()),
        zero_loop_mean=float(base_accepted.mean()),
    )
    initial_gate = float(torch.sigmoid(torch.tensor(-3.5)))
    gate_open = values["draft_gate"].mean(-1) > initial_gate
    improved = delta > 0
    monotonicity = {}
    for name, mask in (("gate_open", gate_open), ("gate_closed", ~gate_open)):
        selected = improved & mask
        monotonicity[name] = {
            "accepted_improvement_rows": int(selected.sum()),
            "final_quality_worsened_fraction": (
                float(quality_loss[selected].float().mean()) if bool(selected.any()) else None
            ),
        }
    summary = {
        "mean_accepted_length": float(accepted.mean()),
        "zero_loop_mean_accepted_length": float(base_accepted.mean()),
        "acceptance_delta": float(delta.mean()),
        "mean_overlap_by_horizon": overlap.mean(0).tolist(),
        "baseline_correct": baseline_count,
        "retained_correct": retained_count,
        "retention": retained_count / max(1, baseline_count),
        "retention_wilson_95_lower": wilson_lower(retained_count, baseline_count),
        "quality_noninferior": quality_noninferior(retained_count, baseline_count),
        "quality_safe_oracle_selected_fraction": float(safe_selected.float().mean()),
        "quality_safe_oracle_acceptance_delta": float(safe_delta.mean()),
        "quality_safe_oracle_headroom_relative": safe_headroom,
        "mean_draft_gate": float(values["draft_gate"].mean()),
        "mean_bridge_gate": float(values["bridge_gate"].mean()),
        "mean_probe_kl": float(probe_kl.mean()),
        "probe_top1": float(probe_top1.mean()),
        "correlations": {
            "probe_kl_vs_accepted_length": _correlation(probe_kl, accepted),
            "probe_top1_vs_accepted_length": _correlation(probe_top1, accepted),
            "probe_kl_vs_probe_top1": _correlation(probe_kl, probe_top1),
        },
        "trained_monotonicity": monotonicity,
        "losses": {name: sum(values_) / len(values_) for name, values_ in loss_accum.items()},
        "control_executed_path_bit_exact": control_identity if arm == "draft_only_control" else None,
    }
    rows = {
        "accepted_length": accepted,
        "base_accepted_length": base_accepted,
        "acceptance_delta": delta,
        "base_correct_by_horizon": base_correct,
        "bridge_correct_by_horizon": bridge_correct,
        "quality_loss": quality_loss,
        "probe_kl": probe_kl,
        "probe_top1": probe_top1,
        "draft_gate": values["draft_gate"],
    }
    module.train()
    return summary, rows


def _active_named(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, value) for name, value in module.named_parameters() if value.requires_grad]


def _grad_norm(gradients: list[torch.Tensor | None], parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for gradient, parameter in zip(gradients, parameters):
        if gradient is not None:
            total += float(gradient.detach().double().square().sum())
        elif parameter.requires_grad:
            total += 0.0
    return math.sqrt(total)


def directional_audit(
    *, module: Phase2StudentModules, cache: dict[str, Any], batches: list[torch.Tensor],
    embedding: nn.Embedding, weights: dict[str, float], device: str,
) -> dict[str, Any]:
    named = _active_named(module)
    parameters = [value for _name, value in named]
    rows = []
    for batch_number, selected in enumerate(batches, start=50):
        losses, _ = _forward(
            module=module,
            batch=_batch(cache, selected, alpha=0.5, device=device),
            embedding=embedding,
            arm="full_a2",
        )
        raw = {}
        for index, name in enumerate(LOSSES):
            gradients = torch.autograd.grad(
                losses[name], parameters, retain_graph=index + 1 < len(LOSSES), allow_unused=True
            )
            raw[name] = _grad_norm(list(gradients), parameters)
        weighted = {name: abs(weights[name]) * raw[name] for name in LOSSES}
        denominator = sum(weighted.values())
        shares = {name: weighted[name] / denominator for name in LOSSES}
        rows.append({"batch": batch_number, "raw_norms": raw, "weighted_norms": weighted, "shares": shares})
    mean_raw = {name: sum(row["raw_norms"][name] for row in rows) / len(rows) for name in LOSSES}
    mean_weighted = {name: abs(weights[name]) * mean_raw[name] for name in LOSSES}
    denominator = sum(mean_weighted.values())
    shares = {name: mean_weighted[name] / denominator for name in LOSSES}
    classification = classify_directional_shares(shares)
    return {
        "mean_raw_gradient_norms": mean_raw,
        "mean_weighted_gradient_norms": mean_weighted,
        "independent_gradient_shares": shares,
        **classification,
        "per_batch_share_distributions": {
            name: _distribution([row["shares"][name] for row in rows]) for name in LOSSES
        },
        "matched_batches": len(rows),
        "batch_size": int(batches[0].numel()),
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


def _load_module(
    *, seed: int, checkpoint_path: Path, expected_sha: str,
    embedding: nn.Embedding, rms_cap: float, device: str, arm: str,
) -> tuple[Phase2StudentModules, dict[str, Any]]:
    if sha256_file(checkpoint_path) != expected_sha:
        raise RuntimeError(f"seed {seed} A1 checkpoint SHA mismatch")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("seed") != seed or int(saved.get("step", -1)) != 1000:
        raise RuntimeError(f"seed {seed} A1 checkpoint identity mismatch")
    _seed_everything(seed)
    module = Phase2StudentModules(
        tied_embedding=embedding, hidden_size=896, rms_cap=float(rms_cap)
    ).to(device=device, dtype=torch.float32)
    if _frozen_hash(module) != saved["initial_frozen_hash"]:
        raise RuntimeError(f"seed {seed} A1 initialization did not reconstruct")
    current = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in saved["flow_state"].items():
            if not name.startswith("flow.") or name not in current:
                raise RuntimeError(f"unexpected A1 flow key: {name}")
            current[name].copy_(value.to(current[name]))
    _set_trainable(module, arm=arm)
    frozen = {name: value for name, value in module.named_parameters() if not value.requires_grad}
    return module, {"initial_frozen_hash": _tensor_digest(frozen), "a1": saved}


def run_arm(
    *, seed: int, arm: str, target_steps: int, cache: dict[str, Any],
    train_indices: torch.Tensor, eval_indices: torch.Tensor, embedding_weight: torch.Tensor,
    teacher_embedding: nn.Embedding, checkpoint_path: Path, expected_checkpoint_sha: str,
    registration: dict[str, Any], output_dir: Path, private_dir: Path, device: str,
    resume_from_step200: bool = False, resume_from_step237: bool = False,
) -> dict[str, Any]:
    if resume_from_step200 and resume_from_step237:
        raise ValueError("A2 resume modes are mutually exclusive")
    name = f"seed_{seed}_{arm}"
    arm_private = private_dir / name
    arm_private.mkdir(parents=True, exist_ok=True)
    resume_path = arm_private / "resume.pt"
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True).to(device)
    module, source = _load_module(
        seed=seed, checkpoint_path=checkpoint_path, expected_sha=expected_checkpoint_sha,
        embedding=embedding, rms_cap=float(registration["constants"]["state_rms_cap"]),
        device=device, arm=arm,
    )
    trainable = dict(_active_named(module))
    initial_trainable_hash = _tensor_digest(trainable)
    optimizer = torch.optim.AdamW(build_adamw_groups(module, weight_decay=0.01), lr=3e-4)
    lock = registration["a2_lock_amendment_20260805"]
    all_weights = {key: float(value) for key, value in lock["full_a2_static_weights_by_seed"][str(seed)].items()}
    active_losses = LOSSES if arm == "full_a2" else PRIMARY
    weights = {name_: all_weights[name_] for name_ in active_losses}
    decoder, decoder_bias = _decoder_for_alpha(cache, alpha=0.5, device=device)
    batch_generator = torch.Generator().manual_seed(int(lock["common_training_row_seed"]))
    audit_generator = torch.Generator().manual_seed(seed + 34001)
    audit_all = [
        train_indices.index_select(
            0, torch.randint(train_indices.numel(), (128,), generator=audit_generator)
        )
        for _ in range(100)
    ]
    audit_batches = audit_all[49:100]
    history: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    step = 0
    abort_reason = None
    consecutive_quality_failures = 0
    previous_marginal: list[str] = []
    batch_hashes: list[str] = []
    preupdate_attempts: list[dict[str, Any]] = []
    resume_lineage: dict[str, Any] | None = None
    gradient_norm_history: list[float] = []
    gradient_tripwire_events: list[dict[str, Any]] = []
    consecutive_relative_exceedances = 0
    static_threshold_exceedances = 0
    next_batch_assertion: dict[str, Any] | None = None
    if resume_path.is_file():
        observed_resume_sha = sha256_file(resume_path)
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("kind") != ARM_KIND or saved.get("arm") != arm or int(saved.get("seed", -1)) != seed:
            raise RuntimeError(f"resume identity mismatch for {name}")
        if saved.get("initial_trainable_hash") != initial_trainable_hash:
            raise RuntimeError(f"resume initialization mismatch for {name}")
        target_steps = max(target_steps, int(saved.get("target_steps", target_steps)))
        _load_trainable_state(module, saved["trainable_state"])
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved["step"])
        history = list(saved["history"])
        audits = list(saved["directional_audits"])
        abort_reason = saved.get("abort_reason")
        consecutive_quality_failures = int(saved["consecutive_quality_failures"])
        previous_marginal = list(saved["previous_marginal"])
        saved_batch_hashes = list(saved["batch_hashes"])
        preupdate_attempts = list(saved.get("preupdate_attempts", []))
        gradient_norm_history = [
            float(value) for value in saved.get("gradient_norm_history", [])
        ]
        gradient_tripwire_events = list(saved.get("gradient_tripwire_events", []))
        consecutive_relative_exceedances = int(
            saved.get("consecutive_relative_exceedances", 0)
        )
        static_threshold_exceedances = int(saved.get("static_threshold_exceedances", 0))
        next_batch_assertion = saved.get("next_batch_assertion")
        batch_hashes, preupdate_attempts = reconcile_executed_schedule(
            step=step,
            batch_hashes=saved_batch_hashes,
            abort_reason=abort_reason,
            preupdate_attempts=preupdate_attempts,
        )
        legacy_schedule_repaired = len(batch_hashes) != len(saved_batch_hashes)
        if legacy_schedule_repaired or resume_from_step237:
            batch_generator.manual_seed(int(lock["common_training_row_seed"]))
            for _ in range(step):
                torch.randint(train_indices.numel(), (128,), generator=batch_generator)
        else:
            batch_generator.set_state(saved["batch_generator_state"])
        _restore_rng(saved["rng"])
        if resume_from_step200:
            resume_lock = registration["a2_step200_resume_amendment_20260805"]
            expected_source_sha = resume_lock["source_resume_sha256_by_arm"][name]
            prior_lineage = saved.get("resume_lineage")
            if prior_lineage is None:
                if observed_resume_sha != expected_source_sha:
                    raise RuntimeError(f"registered step-200 source SHA mismatch for {name}")
                if step != int(resume_lock["resume_step"]):
                    raise RuntimeError(f"registered A2 resume step mismatch for {name}")
                allowed = resume_lock["resumable_abort"]
                if arm == allowed["arm"]:
                    if abort_reason != allowed["reason"]:
                        raise RuntimeError(f"unregistered A2 abort reason for {name}")
                    abort_reason = None
                elif abort_reason is not None:
                    raise RuntimeError(f"control resume unexpectedly aborted for {name}")
                consecutive_quality_failures = 0
                resume_lineage = {
                    "amendment_status": resume_lock["status"],
                    "source_sha256": expected_source_sha,
                    "source_step": step,
                    "historical_abort_cleared": arm == allowed["arm"],
                }
            else:
                if prior_lineage.get("source_sha256") != expected_source_sha:
                    raise RuntimeError(f"A2 resume lineage mismatch for {name}")
                resume_lineage = dict(prior_lineage)
            if legacy_schedule_repaired:
                resume_lineage["legacy_preupdate_schedule_repaired"] = True
        if resume_from_step237:
            step237_lock = registration[STEP237_LOCK_KEY]
            expected_source_sha = step237_lock["source_resume_sha256_by_arm"][name]
            step237_lineage = (
                resume_lineage.get("step237_tripwire_amendment")
                if resume_lineage is not None
                else None
            )
            if step237_lineage is None:
                if observed_resume_sha != expected_source_sha:
                    raise RuntimeError(f"registered step-237 source SHA mismatch for {name}")
                if step != int(step237_lock["resume_step"]):
                    raise RuntimeError(f"registered A2 step-237 resume mismatch for {name}")
                resumable = step237_lock["resumable_abort"]
                if name in set(resumable["arms"]):
                    if abort_reason != resumable["reason"]:
                        raise RuntimeError(f"unregistered step-237 abort reason for {name}")
                    abort_reason = None
                elif abort_reason is not None:
                    raise RuntimeError(f"control step-237 resume unexpectedly aborted for {name}")
                if resume_lineage is None:
                    resume_lineage = {}
                resume_lineage["step237_tripwire_amendment"] = {
                    "status": step237_lock["status"],
                    "source_sha256": expected_source_sha,
                    "source_step": step,
                    "historical_static_abort_cleared": name in set(resumable["arms"]),
                }
                gradient_norm_history = []
                gradient_tripwire_events = []
                consecutive_relative_exceedances = 0
                static_threshold_exceedances = 0
            elif step237_lineage.get("source_sha256") != expected_source_sha:
                raise RuntimeError(f"step-237 continuation lineage mismatch for {name}")
            preview_state = batch_generator.get_state()
            preview = train_indices.index_select(
                0, torch.randint(train_indices.numel(), (128,), generator=batch_generator)
            )
            preview_hash = hashlib.sha256(
                preview.cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            batch_generator.set_state(preview_state)
            if step == int(step237_lock["resume_step"]):
                expected_hash = step237_lock["generator_reconstruction"]["next_row_hash"]
                if preview_hash != expected_hash:
                    raise RuntimeError(
                        f"attempt-238 row hash mismatch for {name}: {preview_hash}"
                    )
                next_batch_assertion = {
                    "completed_draws": step,
                    "next_attempt": step + 1,
                    "expected_row_hash": expected_hash,
                    "observed_row_hash": preview_hash,
                    "passed": True,
                    "generator_reconstructed_from_common_seed": True,
                    "preview_state_restored_before_training": True,
                }
            elif next_batch_assertion is None or not next_batch_assertion.get("passed"):
                raise RuntimeError(f"missing persisted attempt-238 assertion for {name}")
        print(f"a2_resume arm={name} step={step}", flush=True)

    frozen_before = _tensor_digest(
        {name_: value for name_, value in module.named_parameters() if not value.requires_grad}
    )
    if frozen_before != source["initial_frozen_hash"]:
        raise RuntimeError(f"frozen parameters changed before {name} training")

    def save() -> None:
        payload = {
            "kind": ARM_KIND,
            "seed": seed,
            "arm": arm,
            "step": step,
            "target_steps": target_steps,
            "initial_trainable_hash": initial_trainable_hash,
            "trainable_state": {
                key: value.detach().cpu() for key, value in module.named_parameters() if value.requires_grad
            },
            "optimizer": optimizer.state_dict(),
            "history": history,
            "directional_audits": audits,
            "abort_reason": abort_reason,
            "consecutive_quality_failures": consecutive_quality_failures,
            "previous_marginal": previous_marginal,
            "batch_hashes": batch_hashes,
            "preupdate_attempts": preupdate_attempts,
            "batch_generator_state": batch_generator.get_state(),
            "gradient_norm_history": gradient_norm_history,
            "gradient_tripwire_events": gradient_tripwire_events,
            "consecutive_relative_exceedances": consecutive_relative_exceedances,
            "static_threshold_exceedances": static_threshold_exceedances,
            "next_batch_assertion": next_batch_assertion,
            "rng": _rng_state(),
            "resume_lineage": resume_lineage,
        }
        temporary = resume_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(resume_path)

    if step == 0 and not history:
        evaluation, rows = evaluate(
            module=module, cache=cache, indices=eval_indices, embedding=embedding,
            teacher_embedding=teacher_embedding, decoder=decoder, decoder_bias=decoder_bias,
            arm=arm, device=device,
        )
        history.append({"step": 0, **evaluation})
        torch.save(rows, arm_private / "rows_step_0000.pt")
        if arm == "draft_only_control" and not evaluation["control_executed_path_bit_exact"]:
            raise RuntimeError("draft-only control changed the executed hidden path at step zero")
        save()

    while step < target_steps and abort_reason is None:
        selected = train_indices.index_select(
            0, torch.randint(train_indices.numel(), (128,), generator=batch_generator)
        )
        selected_hash = hashlib.sha256(
            selected.cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        if resume_from_step237 and step == 237:
            if next_batch_assertion is None:
                raise RuntimeError(f"attempt-238 preview assertion missing for {name}")
            if selected_hash != next_batch_assertion["expected_row_hash"]:
                raise RuntimeError(f"actual attempt-238 row hash mismatch for {name}")
            next_batch_assertion["actual_selected_row_hash"] = selected_hash
            next_batch_assertion["actual_selection_passed"] = True
        batch = _batch(cache, selected, alpha=0.5, device=device)
        losses, _ = _forward(module=module, batch=batch, embedding=embedding, arm=arm)
        total = sum(weights[name_] * losses[name_] for name_ in active_losses)
        if not bool(torch.isfinite(total)):
            abort_reason = "nonfinite_loss"
            preupdate_attempts.append(
                {
                    "row_hash": selected_hash,
                    "abort_reason": abort_reason,
                    "optimizer_update_applied": False,
                }
            )
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradients = [value.grad for value in trainable.values()]
        if any(gradient is not None and not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            abort_reason = "nonfinite_gradient"
            preupdate_attempts.append(
                {
                    "row_hash": selected_hash,
                    "abort_reason": abort_reason,
                    "optimizer_update_applied": False,
                }
            )
            break
        total_norm = _grad_norm(gradients, list(trainable.values()))
        static_threshold = float(lock["catastrophe_gradient_norm_by_seed"][str(seed)])
        static_exceeds = total_norm > static_threshold
        if static_exceeds:
            static_threshold_exceedances += 1
        if resume_from_step237:
            relative = registration[STEP237_LOCK_KEY]["relative_explosion"]
            event = classify_relative_explosion(
                prior_norms=gradient_norm_history,
                current_norm=total_norm,
                previous_consecutive_exceedances=consecutive_relative_exceedances,
                window=int(relative["trailing_window_steps"]),
                multiplier=float(relative["multiplier"]),
                consecutive_to_stop=int(relative["consecutive_exceedances_to_stop"]),
            )
            event.update(
                {
                    "attempt": step + 1,
                    "row_hash": selected_hash,
                    "obsolete_static_threshold": static_threshold,
                    "obsolete_static_threshold_exceeded": static_exceeds,
                    "optimizer_update_applied": not event["stop"],
                }
            )
            gradient_tripwire_events.append(event)
            consecutive_relative_exceedances = int(event["consecutive_exceedances"])
            if static_exceeds or event["exceeds"]:
                print(
                    f"a2_gradient_telemetry arm={name} attempt={step + 1} "
                    f"norm={total_norm:.9g} static_exceeds={int(static_exceeds)} "
                    f"relative_armed={int(event['armed'])} "
                    f"relative_exceeds={int(event['exceeds'])} "
                    f"relative_consecutive={event['consecutive_exceedances']}",
                    flush=True,
                )
            if event["stop"]:
                abort_reason = "relative_gradient_explosion_tripwire"
                preupdate_attempts.append(
                    {
                        "row_hash": selected_hash,
                        "abort_reason": abort_reason,
                        "optimizer_update_applied": False,
                        "observed_gradient_norm": total_norm,
                        "reference_median": event["reference_median"],
                        "gradient_norm_threshold": event["threshold"],
                        "consecutive_exceedances": event["consecutive_exceedances"],
                    }
                )
                break
            gradient_norm_history.append(total_norm)
        elif static_exceeds:
            abort_reason = "catastrophe_gradient_norm_tripwire"
            preupdate_attempts.append(
                {
                    "row_hash": selected_hash,
                    "abort_reason": abort_reason,
                    "optimizer_update_applied": False,
                    "observed_gradient_norm": total_norm,
                    "gradient_norm_threshold": static_threshold,
                }
            )
            break
        next_step = step + 1
        learning_rate = 3e-4 * min(1.0, next_step / 100.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        batch_hashes.append(selected_hash)
        step = next_step
        if resume_from_step237 and step == 238:
            if next_batch_assertion is None or not next_batch_assertion.get(
                "actual_selection_passed"
            ):
                raise RuntimeError(f"attempt-238 selection was not asserted for {name}")
            next_batch_assertion["optimizer_update_applied"] = True
            next_batch_assertion["executed_schedule_hash_at_238"] = selected_hash

        if arm == "full_a2" and step % 200 == 0:
            audit = {"step": step, **directional_audit(
                module=module, cache=cache, batches=audit_batches,
                embedding=embedding, weights=all_weights, device=device,
            )}
            audits.append(audit)
            if audit["classification"] == "gross":
                abort_reason = "directional_gross_miss"
            elif audit["classification"] == "marginal":
                repeated = repeated_marginal_bounds(previous_marginal, audit["marginal_bounds"])
                if repeated:
                    audit["repeated_bounds"] = repeated
                    abort_reason = "directional_two_consecutive_marginal_misses"
                previous_marginal = list(audit["marginal_bounds"])
            else:
                previous_marginal = []
            print(
                f"a2_directional arm={name} step={step} class={audit['classification']} "
                f"primary={audit['primary_share']:.6f}",
                flush=True,
            )

        if step % 100 == 0:
            evaluation, rows = evaluate(
                module=module, cache=cache, indices=eval_indices, embedding=embedding,
                teacher_embedding=teacher_embedding, decoder=decoder, decoder_bias=decoder_bias,
                arm=arm, device=device,
            )
            evaluation.update({"step": step, "learning_rate": learning_rate, "train_total_loss": float(total.detach())})
            history.append(evaluation)
            torch.save(rows, arm_private / f"rows_step_{step:04d}.pt")
            quality_event = None
            if resume_from_step200 or resume_from_step237:
                quality_lock = registration["a2_step200_resume_amendment_20260805"]["quality"]
                quality_event = classify_inflight_quality(
                    step_zero_retention=float(history[0]["retention"]),
                    retention=float(evaluation["retention"]),
                    wilson_lower=float(evaluation["retention_wilson_95_lower"]),
                    previous_point_failures=consecutive_quality_failures,
                    point_drop=float(quality_lock["during_training_point_drop_from_step_zero"]),
                    point_failures_to_stop=int(
                        quality_lock["during_training_point_drop_consecutive_evaluations"]
                    ),
                    wilson_floor=float(quality_lock["during_training_wilson_95_lower_floor"]),
                )
                consecutive_quality_failures = int(
                    quality_event["consecutive_point_misses"]
                )
                slope_window = int(quality_lock["negative_slope_window_evaluations"])
                if len(history) >= slope_window:
                    slope = retention_slope_latest(history, evaluations=slope_window)
                    quality_event["retention_slope_latest"] = slope
                    quality_event["negative_slope_warning"] = slope < 0
                    if slope < 0:
                        print(
                            f"a2_quality_warning arm={name} step={step} "
                            f"retention_slope={slope:.9g}",
                            flush=True,
                        )
                evaluation["inflight_quality_tripwire"] = quality_event
                evaluation["endpoint_quality_qualified"] = (
                    bool(evaluation["quality_noninferior"])
                    if step in set(quality_lock["endpoint_steps"])
                    else None
                )
            else:
                consecutive_quality_failures = (
                    consecutive_quality_failures + 1
                    if not evaluation["quality_noninferior"]
                    else 0
                )
            if arm == "draft_only_control" and not evaluation["control_executed_path_bit_exact"]:
                abort_reason = "control_executed_path_changed"
            elif quality_event is not None and quality_event["stop"]:
                abort_reason = quality_event["stop_reason"]
            elif (
                not resume_from_step200
                and not resume_from_step237
                and consecutive_quality_failures >= 2
            ):
                abort_reason = "quality_noninferiority_two_consecutive_evaluations"
            save()
            print(
                f"a2_eval arm={name} step={step} eal={evaluation['mean_accepted_length']:.6f} "
                f"headroom={evaluation['quality_safe_oracle_headroom_relative']:.6f} "
                f"quality={evaluation['retention']:.6f}",
                flush=True,
            )
        if abort_reason is not None:
            save()

    if int(history[-1]["step"]) != step:
        evaluation, rows = evaluate(
            module=module, cache=cache, indices=eval_indices, embedding=embedding,
            teacher_embedding=teacher_embedding, decoder=decoder, decoder_bias=decoder_bias,
            arm=arm, device=device,
        )
        evaluation.update({"step": step, "abort_endpoint": True})
        history.append(evaluation)
        torch.save(rows, arm_private / f"rows_step_{step:04d}.pt")

    frozen_after = _tensor_digest(
        {name_: value for name_, value in module.named_parameters() if not value.requires_grad}
    )
    if frozen_after != frozen_before:
        raise RuntimeError(f"frozen parameter mutation detected for {name}")
    save()
    final = history[-1]
    final_rows_path = arm_private / f"rows_step_{int(final['step']):04d}.pt"
    result = {
        "kind": ARM_KIND,
        "status": "aborted" if abort_reason else "complete",
        "abort_reason": abort_reason,
        "seed": seed,
        "arm": arm,
        "step": step,
        "target_steps": target_steps,
        "active_training_losses": list(active_losses),
        "static_loss_weights": weights,
        "training_row_schedule": {
            "seed": int(lock["common_training_row_seed"]),
            "batches": len(batch_hashes),
            "sha256": hashlib.sha256("\n".join(batch_hashes).encode("ascii")).hexdigest(),
            "definition": "optimizer_updates_only",
        },
        "preupdate_attempts": preupdate_attempts,
        "next_batch_assertion": next_batch_assertion,
        "gradient_norm_telemetry": {
            "observed_optimizer_updates": len(gradient_norm_history),
            "values": gradient_norm_history,
            "events": gradient_tripwire_events,
            "consecutive_relative_exceedances": consecutive_relative_exceedances,
            "obsolete_static_threshold": float(
                lock["catastrophe_gradient_norm_by_seed"][str(seed)]
            ),
            "obsolete_static_threshold_exceedances": static_threshold_exceedances,
            "relative_explosion_contract": (
                registration[STEP237_LOCK_KEY]["relative_explosion"]
                if resume_from_step237
                else None
            ),
        },
        "history": history,
        "directional_audits": audits,
        "final": final,
        "frozen_parameter_hash_before": frozen_before,
        "frozen_parameter_hash_after": frozen_after,
        "source_a1_checkpoint": {"path": str(checkpoint_path), "sha256": expected_checkpoint_sha},
        "checkpoint": {"path": str(resume_path), "sha256": sha256_file(resume_path)},
        "resume_lineage": resume_lineage,
        "guardrail_inventory": (
            registration[STEP237_LOCK_KEY]["rule_inventory"]
            if resume_from_step237
            else None
        ),
        "final_rows": {"path": str(final_rows_path), "sha256": sha256_file(final_rows_path)},
    }
    write_json(output_dir / f"{name}.json", result)
    return result


def _pair_summary(full: dict[str, Any], control: dict[str, Any], private_dir: Path) -> dict[str, Any]:
    if full["training_row_schedule"] != control["training_row_schedule"]:
        raise RuntimeError(
            "full and control arms did not receive the identical executed training rows: "
            f"full={full['training_row_schedule']} control={control['training_row_schedule']}"
        )
    full_rows = torch.load(Path(full["final_rows"]["path"]), map_location="cpu", weights_only=False)
    control_rows = torch.load(Path(control["final_rows"]["path"]), map_location="cpu", weights_only=False)
    difference = full_rows["accepted_length"].float() - control_rows["accepted_length"].float()
    relative = float(full["final"]["quality_safe_oracle_headroom_relative"])
    verdict = paired_verdict(
        relative_headroom=relative,
        full_mean=float(full["final"]["mean_accepted_length"]),
        control_mean=float(control["final"]["mean_accepted_length"]),
        quality_noninferior=bool(full["final"]["quality_noninferior"]),
    )
    metric_verdict = verdict["verdict"]
    if full["status"] != "complete" or control["status"] != "complete":
        verdict["verdict"] = "blocked"
    return {
        "seed": int(full["seed"]),
        "steps": int(full["step"]),
        "full_status": full["status"],
        "control_status": control["status"],
        "full_mean_accepted_length": float(full["final"]["mean_accepted_length"]),
        "control_mean_accepted_length": float(control["final"]["mean_accepted_length"]),
        "paired_difference": {
            "mean": float(difference.mean()),
            "median": float(difference.median()),
            "positive_fraction": float((difference > 0).float().mean()),
            "distribution": _distribution(difference.tolist()),
        },
        "relative_quality_safe_oracle_headroom": relative,
        "metric_verdict_without_completion_status": metric_verdict,
        **verdict,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    registration = _assert_lock(
        root,
        require_resume=args.resume_from_step200 or args.resume_from_step237,
        require_step237_resume=args.resume_from_step237,
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise RuntimeError("A2 matrix requires an A100-class GPU with at least 35 GiB VRAM")
    cache = build_pilot_cache(
        stage0a_summary_path=args.stage0a_summary,
        stage0a_private=args.stage0a_private,
        canonicalizer_path=args.canonicalizer,
        output_path=args.cache,
    )
    if cache["source"]["canonicalizer_sha256"] != registration["canonicalizer"]["sha256"]:
        raise RuntimeError("canonicalizer differs from the A2 lock")
    stage0a_summary = json.loads(args.stage0a_summary.read_text(encoding="utf-8"))
    observed_frozen = {
        "data_sha256": stage0a_summary["config"]["data_sha256"],
        "sample_manifest_sha256": stage0a_summary["manifest"]["sample_manifest_sha256"],
        "position_key_sha256": stage0a_summary["manifest"]["position_key_sha256"],
    }
    for key, observed in observed_frozen.items():
        if observed != registration["frozen_data"][key]:
            raise RuntimeError(f"Stage 0A {key} differs from the A2 lock")
    eval_mask = document_partition(cache["documents"], evaluation_fraction=0.2, seed=20260804)
    eval_indices = torch.where(eval_mask)[0]
    train_indices = torch.where(~eval_mask)[0]
    student_summary = json.loads(
        (args.stage0a_private / "model_cache/student_0p5b/summary.json").read_text(encoding="utf-8")
    )
    teacher_summary = json.loads(
        (args.stage0a_private / "model_cache/teacher_14b/summary.json").read_text(encoding="utf-8")
    )
    student_head = _local_source(student_summary["lm_head"]["path"], args.stage0a_private)
    teacher_head = _local_source(teacher_summary["lm_head"]["path"], args.stage0a_private)
    if sha256_file(student_head) != registration["frozen_data"]["student_lm_head_sha256"]:
        raise RuntimeError("student LM head differs from the A2 lock")
    if sha256_file(teacher_head) != teacher_summary["lm_head"]["sha256"]:
        raise RuntimeError("teacher LM head differs from its Stage 0A receipt")
    embedding_weight = torch.load(student_head, map_location="cpu", weights_only=False)["weight_bfloat16"]
    teacher_weight = torch.load(teacher_head, map_location="cpu", weights_only=False)["weight_bfloat16"]
    teacher_embedding = nn.Embedding.from_pretrained(teacher_weight.float(), freeze=True).to(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    lock = registration["a2_lock_amendment_20260805"]
    checkpoints = {0: args.a1_checkpoint_seed_0, 1: args.a1_checkpoint_seed_1}
    arms = []
    pairs = []
    for seed in (0, 1):
        full = run_arm(
            seed=seed, arm="full_a2", target_steps=1000, cache=cache,
            train_indices=train_indices, eval_indices=eval_indices,
            embedding_weight=embedding_weight, teacher_embedding=teacher_embedding,
            checkpoint_path=checkpoints[seed],
            expected_checkpoint_sha=lock["a1_checkpoint_sha256_by_seed"][str(seed)],
            registration=registration, output_dir=args.output_dir,
            private_dir=args.private_dir, device=args.device,
            resume_from_step200=args.resume_from_step200,
            resume_from_step237=args.resume_from_step237,
        )
        if full["status"] == "complete":
            slope = final_window_slope(full["history"], window=100)
            extension = (
                int(full["step"]) == 1000
                and should_extend(
                    relative_headroom=float(full["final"]["quality_safe_oracle_headroom_relative"]),
                    accepted_length_slope=slope,
                )
            )
            extension_decision = {
                "extend": extension,
                "already_extended": int(full["step"]) > 1000,
                "accepted_length_slope_final_100": slope,
                "relative_oracle_headroom": full["final"]["quality_safe_oracle_headroom_relative"],
            }
            if extension:
                full = run_arm(
                    seed=seed, arm="full_a2", target_steps=2000, cache=cache,
                    train_indices=train_indices, eval_indices=eval_indices,
                    embedding_weight=embedding_weight, teacher_embedding=teacher_embedding,
                    checkpoint_path=checkpoints[seed],
                    expected_checkpoint_sha=lock["a1_checkpoint_sha256_by_seed"][str(seed)],
                    registration=registration, output_dir=args.output_dir,
                    private_dir=args.private_dir, device=args.device,
                    resume_from_step200=args.resume_from_step200,
                    resume_from_step237=args.resume_from_step237,
                )
            full["step_1000_extension_decision"] = extension_decision
            write_json(args.output_dir / f"seed_{seed}_full_a2.json", full)
        matched_budget = int(full["step"])
        control = run_arm(
            seed=seed, arm="draft_only_control", target_steps=matched_budget, cache=cache,
            train_indices=train_indices, eval_indices=eval_indices,
            embedding_weight=embedding_weight, teacher_embedding=teacher_embedding,
            checkpoint_path=checkpoints[seed],
            expected_checkpoint_sha=lock["a1_checkpoint_sha256_by_seed"][str(seed)],
            registration=registration, output_dir=args.output_dir,
            private_dir=args.private_dir, device=args.device,
            resume_from_step200=args.resume_from_step200,
            resume_from_step237=args.resume_from_step237,
        )
        arms.extend([full, control])
        pairs.append(_pair_summary(full, control, args.private_dir))
        del teacher_embedding
        torch.cuda.empty_cache()
        teacher_embedding = nn.Embedding.from_pretrained(teacher_weight.float(), freeze=True).to(args.device)
    all_positive = all(pair["verdict"] == "positive" for pair in pairs)
    summary = {
        "kind": RUN_KIND,
        "status": "complete_positive" if all_positive else "complete_budget_limited_or_blocked",
        "launcher_commit": _git_head(root),
        "lock_commit_ancestry": (
            "continuation launcher committed after the A2 step-237 tripwire amendment"
            if args.resume_from_step237
            else "resume launcher committed after A2 step-200 resume lock b28497ce"
            if args.resume_from_step200
            else "training launcher committed after A2 lock commits b9086cc1 and 9ca0d513"
        ),
        "four_run_matrix": ["seed_0_full_a2", "seed_0_draft_only_control", "seed_1_full_a2", "seed_1_draft_only_control"],
        "train_anchors": int(train_indices.numel()),
        "evaluation_anchors": int(eval_indices.numel()),
        "arms": arms,
        "pairs": pairs,
        "v1d": registration["v1d"],
        "resume_amendment": (
            registration[STEP237_LOCK_KEY]
            if args.resume_from_step237
            else registration["a2_step200_resume_amendment_20260805"]
            if args.resume_from_step200
            else None
        ),
        "guardrail_inventory_validation": (
            validate_guardrail_inventory(registration[STEP237_LOCK_KEY]["rule_inventory"])
            if args.resume_from_step237
            else None
        ),
        "v1d_attachment": (
            registration[STEP237_LOCK_KEY]["v1d_attachment"]
            if args.resume_from_step237
            else None
        ),
        "frozen_confirmatory_partitions_touched": [],
        "do_not_claim": [
            "DEV accepted length is serving throughput",
            "oracle selector headroom is achievable by a deployable router",
            "a budget-limited result is an impossibility result",
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
    parser.add_argument("--a1_checkpoint_seed_0", type=Path, required=True)
    parser.add_argument("--a1_checkpoint_seed_1", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume_from_step200", action="store_true")
    parser.add_argument("--resume_from_step237", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "pairs": result["pairs"]}, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "complete_positive" else 2)


if __name__ == "__main__":
    main()
