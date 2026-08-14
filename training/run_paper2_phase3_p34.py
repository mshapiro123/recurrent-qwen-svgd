"""Run one approved, resumable P3.4 campaign arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import t as student_t

from eval.cache_paper2_phase3_agreement_oracle import _lm_head
from eval.eval_paper2_phase3_p34_share_calibration import _selected_source_cache
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from eval.eval_paper2_phase3_retention_step0 import position_buckets
from training.paper2_phase2_matched_alpha import build_adamw_groups
from training.paper2_phase3_p32 import GateLabel
from training.paper2_phase3_p33 import learning_rate_at_step
from training.paper2_phase3_p33_prep import sha256_file
from training.paper2_phase3_p34 import (
    P34_FLOW_LOOPS,
    P34_LOSS_NAMES,
    P34_SLOT_LOSS_NAMES,
    AnnealingState,
    LossShareBounds,
    SlotSupervisionLift,
    aggregate_postclip_bundle_reads,
    apply_weighted_gradient_bundle,
    classify_loss_shares,
    controller_transition,
    initial_annealing_state,
    loss_gradient_bundle,
    normalize_objective_weights,
    objective_share_controller_update,
    p34_forward_losses,
    postclip_gradient_norms_from_bundle,
    sampled_depth,
    set_p34_trainable,
    slot_supervision_loss,
    solve_static_loss_weights_from_bundles,
)
from training.run_paper2_phase3_p33 import (
    _active_record_pools,
    _direction_lookup,
    atomic_torch_save,
    audit_model,
    load_audit_material,
    read_jsonl,
    restore_rng,
    rng_state,
    tensor_digest,
    write_json,
    write_jsonl,
)


BASE_RUN_KIND = "paper2_phase3_p34_campaign_v1"
RUN_KIND = "paper2_phase3_p34_a2_campaign_v1"
TOTAL_STEPS = 4_000
LOOK_INTERVAL = 200
SHARE_WINDOW_STEPS = 100
POSITIVE_PER_BATCH = 32
NEGATIVE_PER_BATCH = 96
AUDIT_LOOKS = {5, 10, 15, 20}


def _source_batch(
    *,
    rows: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    direction_index: Mapping[str, int],
    directions: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    width = max(int(source["candidate_ids"].shape[-1]) for source in sources.values())
    keys = (
        "hidden4", "candidate_ids", "candidate_mask", "base_candidates", "base_tail",
        "teacher_candidates", "teacher_tail", "position_bucket", "gate_labels",
        "oracle_directions", "kl_mask", "ce_mask",
    )
    selected: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    for row in rows:
        source = sources[str(row["source"])]
        anchor = source["anchor_lookup"][int(row["anchor_index"])]
        horizon = int(row["horizon"]) - 1
        selected["hidden4"].append(source["hidden4"][anchor])
        for key, fill in (
            ("candidate_ids", -1), ("candidate_mask", False),
            ("base_candidates", float("-inf")),
            ("teacher_candidates", float("-inf")),
        ):
            value = source[key][anchor]
            if value.shape[-1] < width:
                grown = torch.full((*value.shape[:-1], width), fill, dtype=value.dtype)
                grown[..., : value.shape[-1]] = value
                value = grown
            selected[key].append(value)
        selected["base_tail"].append(source["base_tail"][anchor])
        selected["teacher_tail"].append(source["teacher_tail"][anchor])
        selected["position_bucket"].append(
            position_buckets(source["positions"][anchor : anchor + 1])[0]
        )
        labels = torch.full((4,), int(GateLabel.IGNORED), dtype=torch.long)
        labels[horizon] = int(row["gate_label"])
        selected["gate_labels"].append(labels)
        oracle = torch.zeros((4, 896), dtype=torch.float32)
        if int(row["gate_label"]) == int(GateLabel.POSITIVE):
            oracle[horizon] = directions[direction_index[str(row["record_id"])]]
        selected["oracle_directions"].append(oracle)
        kl_mask = torch.zeros(4, dtype=torch.bool)
        kl_mask[horizon] = True
        selected["kl_mask"].append(kl_mask)
        ce_mask = torch.zeros(4, dtype=torch.bool)
        ce_mask[horizon] = int(row["gate_label"]) == int(GateLabel.POSITIVE)
        selected["ce_mask"].append(ce_mask)
    batch = {key: torch.stack(values).to(device) for key, values in selected.items()}
    teacher = batch["teacher_candidates"].masked_fill(
        ~batch["candidate_mask"], float("-inf")
    )
    batch["teacher_token_index"] = teacher.argmax(dim=-1)
    batch["teacher_tokens"] = batch["candidate_ids"].gather(
        -1, batch["teacher_token_index"].unsqueeze(-1)
    ).squeeze(-1)
    batch["teacher_mask"] = batch["candidate_mask"].any(dim=-1)
    return batch


def _losses(
    *, module: Any, batch: Mapping[str, torch.Tensor], depth: int,
    slot_lift: SlotSupervisionLift | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    losses, metrics = p34_forward_losses(
        module=module,
        tied_embedding=module.draft.tied_embedding,
        teacher_candidates=batch["teacher_candidates"],
        teacher_tail=batch["teacher_tail"],
        teacher_token_index=batch["teacher_token_index"],
        kl_mask=batch["kl_mask"], ce_mask=batch["ce_mask"], steps=depth,
        hidden4=batch["hidden4"], candidate_ids=batch["candidate_ids"],
        candidate_mask=batch["candidate_mask"], base_candidates=batch["base_candidates"],
        base_tail=batch["base_tail"], gate_labels=batch["gate_labels"],
        oracle_directions=batch["oracle_directions"],
        position_bucket=batch["position_bucket"],
    )
    if slot_lift is not None:
        slot, slot_metrics = slot_supervision_loss(
            lift=slot_lift, flow_states=metrics["flow_states"],
            tied_weight=module.draft.tied_embedding.weight,
            teacher_tokens=batch["teacher_tokens"], teacher_mask=batch["teacher_mask"],
        )
        losses = {**losses, "slot": slot}
        metrics = {**metrics, "slot_metrics": slot_metrics}
    return losses, metrics


def _task_guardrail(rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]) -> dict[str, Any]:
    differences = np.asarray([
        int(bool(row["augmented_correct"])) - int(bool(row["base_correct"]))
        for row in rows
    ], dtype=np.float64)
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    guard = lock["guardrails"]
    critical = float(student_t.ppf(1.0 - float(guard["tier_s_one_sided_alpha"]), len(rows) - 1))
    upper = mean + critical * se
    return {
        "mean_augmented_minus_base": mean, "standard_error": se,
        "upper_bound": upper,
        "tier_s_condition": upper < float(guard["tier_s_decision_margin"]),
        "tier_w_condition": upper < float(guard["tier_w_drop_class"]),
        "rows": len(rows),
    }


def _advance_sequential_rule(
    *, condition: bool, prior_streak: int, required_looks: int
) -> tuple[int, bool]:
    """Advance a consecutive-look rule and emit non-overlapping actions."""

    if required_looks < 2:
        raise ValueError("sequential guardrails require at least two consecutive looks")
    streak = prior_streak + 1 if condition else 0
    event = condition and streak >= required_looks and streak % required_looks == 0
    return streak, event


def _checkpoint_state(module: Any, slot_lift: SlotSupervisionLift | None) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu()
        for name, value in module.named_parameters() if value.requires_grad
    }
    if slot_lift is not None:
        state.update({f"slot_lift.{name}": value.detach().cpu() for name, value in slot_lift.named_parameters()})
    return state


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(20260813 + args.seed)
    np.random.seed(20260813 + args.seed)
    torch.manual_seed(20260813 + args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260813 + args.seed)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if not lock.get("training_authorized") or lock["status"] != "approved_for_training":
        raise RuntimeError("P3.4 base executed lock is not approved")
    if not (
        amendment.get("mark_ratified")
        and amendment.get("locked_before_resumed_training")
        and amendment.get("training_authorized")
    ):
        raise RuntimeError("P3.4 a2 amendment is not ratified, locked, and authorized")
    if args.arm != "main":
        raise ValueError("P3.4 a2 authorizes the two main seeds only")
    expected_endpoint = lock["initialization"][f"seed_{args.seed}"]["sha256"]
    if sha256_file(args.i1) != expected_endpoint:
        raise RuntimeError("P3.4 i1 endpoint SHA mismatch")
    for path in (args.staged_labels, args.positive_audit, args.negative_audit, args.dev_panel):
        if any(token in str(path).casefold() for token in ("confirm", "eval_e")):
            raise RuntimeError("P3.4 sealed-data path contact")

    records = read_jsonl(args.staged_labels)
    positives, negatives = _active_record_pools(records)
    direction_index, directions, direction_receipt = _direction_lookup(args.direction_cache)
    sources_spec = {"old": (args.old_summary, args.old_private), "new": (args.new_summary, args.new_private)}
    embedding, embedding_receipt = _lm_head(sources_spec)
    module, chain = load_condition(
        embedding_weight=embedding, migrated=args.migrated,
        migrated_sha256=args.migrated_sha256, p33=args.p33,
        p33_sha256=args.p33_sha256, i1=args.i1, i1_sha256=expected_endpoint,
    )
    sources = {}
    for source, (summary, private) in sources_spec.items():
        sources[source], _receipt = _selected_source_cache(
            source=source, records=records, summary_path=summary, private_root=private,
        )
    audit_material = load_audit_material(
        positive_path=args.positive_audit, negative_path=args.negative_audit,
        retention_path=args.retention_panel, sources=sources_spec,
    )
    trainable = set_p34_trainable(module)
    slot_lift = None
    parameters = list(trainable.values())
    frozen_before = tensor_digest({name: value for name, value in module.named_parameters() if not value.requires_grad})
    weights_key = f"seed_{args.seed}_main"
    original_weights = dict(lock["loss_share_contract"]["scalar_weights_by_seed"][weights_key])
    output_dir, private_dir = args.output_dir, args.private_dir
    output_dir.mkdir(parents=True, exist_ok=True); private_dir.mkdir(parents=True, exist_ok=True)
    resume_path = private_dir / "resume.pt"
    generator = torch.Generator().manual_seed(20260813 + args.seed)
    step = 0; history: list[dict[str, Any]] = []; schedule_hashes: list[str] = []
    share_window: deque[dict[str, float]] = deque(maxlen=SHARE_WINDOW_STEPS)
    share_contract_events: list[dict[str, Any]] = []
    share_misses = 0; tier_s_streak = 0; tier_w_streak = 0
    controller = initial_annealing_state(
        chi_max_by_rung=tuple(lock["guardrails"]["chi_max_by_rung"])
    )
    last_pi_dep = 0.0; last_chi = 0.0; stop_reason: str | None = None
    objective_weights = normalize_objective_weights(original_weights)
    objective_controller_events: list[dict[str, Any]] = []
    continuation_receipt: dict[str, Any]
    resumed_a2 = resume_path.is_file()
    saved: dict[str, Any]
    source_path = resume_path if resumed_a2 else args.continuation
    expected_source_sha = (
        sha256_file(resume_path)
        if resumed_a2
        else str(amendment["continuations"][f"main_seed_{args.seed}"]["checkpoint_sha256"])
    )
    observed_source_sha = sha256_file(source_path)
    if observed_source_sha != expected_source_sha:
        raise RuntimeError("P3.4 a2 continuation checkpoint SHA mismatch")
    saved = torch.load(source_path, map_location="cpu", weights_only=False)
    expected_kind = RUN_KIND if resumed_a2 else BASE_RUN_KIND
    expected_step = int(
        saved["step"]
        if resumed_a2
        else amendment["continuations"][f"main_seed_{args.seed}"]["step"]
    )
    if (
        saved.get("kind") != expected_kind
        or int(saved.get("seed", -1)) != args.seed
        or saved.get("arm") != "main"
        or int(saved.get("step", -1)) != expected_step
    ):
        raise RuntimeError("P3.4 a2 continuation identity mismatch")
    if saved.get("lock_sha256") != sha256_file(args.lock):
        raise RuntimeError("P3.4 a2 continuation base-lock SHA mismatch")
    if resumed_a2 and saved.get("amendment_sha256") != sha256_file(args.amendment):
        raise RuntimeError("P3.4 a2 resume amendment SHA mismatch")
    current = dict(module.named_parameters())
    for name, value in saved["trainable_state"].items():
        if name.startswith("slot_lift.") or name not in current:
            raise RuntimeError("P3.4 a2 continuation trainable surface changed")
        current[name].data.copy_(value.to(args.device))
    step = int(saved["step"]); history = list(saved["history"])
    schedule_hashes = list(saved["schedule_hashes"]); share_window.extend(saved["share_window"])
    share_contract_events = list(saved.get("share_contract_events", []))
    objective_controller_events = list(saved.get("objective_controller_events", []))
    share_misses = int(saved["share_misses"])
    tier_s_streak = int(saved.get("tier_s_streak", int(saved.get("prior_tier_s", False))))
    tier_w_streak = int(saved.get("tier_w_streak", int(saved.get("prior_tier_w", False))))
    controller = AnnealingState(**saved["controller"])
    last_pi_dep = float(saved["last_pi_dep"]); last_chi = float(saved["last_chi"])
    stop_reason = saved.get("stop_reason"); generator.set_state(saved["generator_state"])
    restore_rng(saved["rng_state"])
    module.bridge.set_gate_ceiling(controller.gate_ceiling)
    if resumed_a2:
        objective_weights = normalize_objective_weights(saved["objective_weights"])
    else:
        # Reconstruct the exact effective vector that produced the pinned
        # checkpoint before applying the a2 controller once.
        objective_weights = dict(original_weights)
        objective_weights["preserve"] *= controller.preservation_weight
        objective_weights = normalize_objective_weights(objective_weights)
    continuation_receipt = {
        "path": str(source_path),
        "sha256": observed_source_sha,
        "source_kind": expected_kind,
        "source_step": step,
        "controlled_lock_migration": not resumed_a2,
        "base_lock_sha256": sha256_file(args.lock),
        "amendment_sha256": sha256_file(args.amendment),
    }
    fixed_rows = read_jsonl(args.share_rows)
    if len(fixed_rows) != 256:
        raise RuntimeError("P3.4 locked share-calibration population changed")
    fixed_batch = _source_batch(
        rows=fixed_rows, sources=sources, direction_index=direction_index,
        directions=directions, device=args.device,
    )
    preflight_bundles: list[dict[str, Any]] = []
    preflight_depths = [1, 2, 3, 4]
    preflight_mass = [0.1, 0.2, 0.3, 0.4]
    for depth in preflight_depths:
        preflight_losses, _ = _losses(
            module=module, batch=fixed_batch, depth=depth, slot_lift=slot_lift
        )
        preflight_bundle = loss_gradient_bundle(
            losses=preflight_losses, module=module, parameters=parameters,
            slot_lift=slot_lift,
        )
        preflight_bundles.extend([preflight_bundle] * depth)
    before_read = aggregate_postclip_bundle_reads(
        preflight_bundles, weights=objective_weights
    )
    target_shares = amendment["rung_targets"][str(controller.rung)]
    if resumed_a2:
        preflight_update = {
            "status": "not_reapplied_on_amended_resume",
            "weights_before": objective_weights,
            "weights_after": objective_weights,
            "observed_shares": before_read["shares"],
            "target_shares": target_shares,
            "reason": "the restart calibration update is applied exactly once at lock migration",
        }
        proposed_weights = objective_weights
    else:
        preflight_update = objective_share_controller_update(
            weights=objective_weights,
            observed_shares=before_read["shares"],
            target_shares=target_shares,
            gain=float(amendment["objective_share_controller"]["gain"]),
            maximum_absolute_log_update=float(
                amendment["objective_share_controller"]["maximum_absolute_log_weight_update"]
            ),
        )
        proposed_weights = preflight_update["weights_after"]
    after_read = aggregate_postclip_bundle_reads(
        preflight_bundles, weights=proposed_weights
    )
    preflight_classification = classify_loss_shares(after_read["shares"])
    arithmetic_residual = max(
        abs(sum(before_read["shares"].values()) - 1.0),
        abs(sum(after_read["shares"].values()) - 1.0),
        abs(float(proposed_weights["kl"]) - 1.0),
    )
    bundle_path = private_dir / "pre_optimizer_gradient_bundles.pt"
    bundle_sha = atomic_torch_save(
        {
            "kind": "paper2_phase3_p34_a2_exact_gradient_bundles_v1",
            "seed": args.seed,
            "arm": args.arm,
            "continuation": continuation_receipt,
            "depth_distribution": dict(zip(preflight_depths, preflight_mass)),
            "bundles": preflight_bundles,
            "weights_before": objective_weights,
            "weights_after": proposed_weights,
        },
        bundle_path,
    )
    preflight_receipt = {
        "kind": "paper2_phase3_p34_a2_exact_preoptimizer_preflight_v1",
        "status": "complete_read_only_no_optimizer",
        "seed": args.seed,
        "arm": args.arm,
        "continuation": continuation_receipt,
        "depth_distribution": dict(zip(preflight_depths, preflight_mass)),
        "rung": controller.rung,
        "target_shares": target_shares,
        "before": before_read,
        "controller_update": preflight_update,
        "after": after_read,
        "loss_share_read": preflight_classification,
        "gradient_bundle_receipt": {"path": str(bundle_path), "sha256": bundle_sha},
        "gradient_bundle_reuse": {
            "same_per_loss_bundles_for_before_and_after": True,
            "reason": (
                "scalar objective weights are applied after per-loss gradients are formed; "
                "combined group clipping is recomputed exactly for each weight vector"
            ),
        },
        "maximum_arithmetic_residual": arithmetic_residual,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(output_dir / "pre_optimizer_estimator_receipt.json", preflight_receipt)
    if args.preflight_only:
        result = {
            **preflight_receipt,
            "status": "complete_preflight_only",
            "training_authorized_by_this_mode": False,
        }
        write_json(output_dir / "summary.json", result)
        return result
    if preflight_classification["classification"] != "pass":
        write_json(output_dir / "blocked_pre_optimizer.json", {
            "kind": RUN_KIND, "status": "blocked_pre_optimizer_estimator_mismatch",
            "seed": args.seed, "arm": args.arm,
            "depth_distribution": dict(zip(preflight_depths, preflight_mass)),
            "loss_share_read": preflight_classification,
            "controller_update": preflight_update,
            "before": before_read,
            "after": after_read,
            "repair_scope": (
                "return to strategy without changing controller constants or targets; "
                "no optimizer was constructed"
            ),
            "optimizer_constructed": False, "optimizer_steps": 0,
        })
        raise RuntimeError("P3.4 a2 exact preflight violates the locked loss-share contract")
    objective_weights = normalize_objective_weights(
        saved["objective_weights"] if resumed_a2 else proposed_weights
    )
    optimizer_groups = build_adamw_groups(module, weight_decay=0.01)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=0.0, betas=(0.9, 0.999))
    optimizer.load_state_dict(saved["optimizer_state"])

    def save(archive: bool) -> tuple[Path, str]:
        payload = {
            "kind": RUN_KIND, "seed": args.seed, "arm": args.arm, "step": step,
            "trainable_state": _checkpoint_state(module, slot_lift),
            "optimizer_state": optimizer.state_dict(), "history": history,
            "schedule_hashes": schedule_hashes, "share_window": list(share_window),
            "share_misses": share_misses, "share_contract_events": share_contract_events,
            "objective_weights": objective_weights,
            "objective_controller_events": objective_controller_events,
            "tier_s_streak": tier_s_streak,
            "tier_w_streak": tier_w_streak, "controller": controller.__dict__,
            "last_pi_dep": last_pi_dep, "last_chi": last_chi,
            "stop_reason": stop_reason, "generator_state": generator.get_state(),
            "rng_state": rng_state(), "lock_sha256": sha256_file(args.lock),
            "amendment_sha256": sha256_file(args.amendment),
            "continuation_receipt": continuation_receipt,
        }
        if archive:
            destination = private_dir / f"checkpoint_step_{step:04d}.pt"
            digest = atomic_torch_save(payload, destination)
            return destination, digest
        digest = atomic_torch_save(payload, resume_path)
        return resume_path, digest

    if not resume_path.is_file():
        save(archive=False)

    while step < TOTAL_STEPS and stop_reason is None:
        pos = torch.randint(len(positives), (POSITIVE_PER_BATCH,), generator=generator)
        neg = torch.randint(len(negatives), (NEGATIVE_PER_BATCH,), generator=generator)
        rows = [positives[int(index)] for index in pos] + [negatives[int(index)] for index in neg]
        rows = [rows[index] for index in torch.randperm(len(rows), generator=generator).tolist()]
        schedule_hashes.append(hashlib.sha256("\n".join(str(row["record_id"]) for row in rows).encode()).hexdigest())
        batch = _source_batch(rows=rows, sources=sources, direction_index=direction_index,
                              directions=directions, device=args.device)
        depth = sampled_depth(generator=generator)
        module.train(); losses, metrics = _losses(module=module, batch=batch, depth=depth, slot_lift=slot_lift)
        if any(not bool(torch.isfinite(value)) for value in losses.values()):
            stop_reason = "non_finite_loss"; break
        effective = dict(objective_weights)
        bundle = loss_gradient_bundle(losses=losses, module=module, parameters=parameters, slot_lift=slot_lift)
        optimizer.zero_grad(set_to_none=True)
        update = apply_weighted_gradient_bundle(bundle=bundle, parameters=parameters, weights=effective)
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients or any(not bool(value.isfinite().all()) for value in gradients):
            stop_reason = "non_finite_gradient"; break
        shares = update["postclip"]["postclip_gradient_norms"]
        denominator = sum(shares.values()); share_window.append({name: value / denominator for name, value in shares.items()})
        next_step = step + 1
        lr = 3e-4 * min(1.0, next_step / 100)
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.step(); step = next_step
        share_transition = None
        classified = None
        if step % SHARE_WINDOW_STEPS == 0:
            if len(share_window) != SHARE_WINDOW_STEPS:
                raise RuntimeError("P3.4 loss-share window is incomplete at its registered read")
            trailing = {
                name: sum(row[name] for row in share_window) / SHARE_WINDOW_STEPS
                for name in share_window[0]
            }
            classified = classify_loss_shares(trailing, prior_consecutive_misses=share_misses)
            share_misses = int(classified["consecutive_misses"])
            objective_update = objective_share_controller_update(
                weights=objective_weights,
                observed_shares=trailing,
                target_shares=amendment["rung_targets"][str(controller.rung)],
                gain=float(amendment["objective_share_controller"]["gain"]),
                maximum_absolute_log_update=float(
                    amendment["objective_share_controller"][
                        "maximum_absolute_log_weight_update"
                    ]
                ),
            )
            objective_weights = objective_update["weights_after"]
            objective_event = {
                "step": step,
                "window_index": step // SHARE_WINDOW_STEPS,
                "rung_used_for_target": controller.rung,
                **objective_update,
            }
            objective_controller_events.append(objective_event)
            if classified["classification"] == "demote":
                before = controller.rung
                controller = AnnealingState(
                    max(0, before - 1), controller.chi_max_by_rung, controller.window + 1
                )
                module.bridge.set_gate_ceiling(controller.gate_ceiling)
                share_transition = {
                    "reason": "loss_share_contract_demote",
                    "rung_before": before,
                    "rung_after": controller.rung,
                    "strategy_review_flagged": True,
                    "at_most_one_rung": before - controller.rung <= 1,
                }
            elif classified["classification"] == "stop":
                stop_reason = "loss_share_contract"
            share_contract_events.append({
                "step": step,
                "window_index": step // SHARE_WINDOW_STEPS,
                "window_steps": SHARE_WINDOW_STEPS,
                "overlap_with_prior_window_steps": 0,
                "read": classified,
                "controller": share_transition,
                "objective_controller": objective_event,
            })
            share_window.clear()
        if step % LOOK_INTERVAL == 0 or stop_reason:
            look = max(1, step // LOOK_INTERVAL)
            checkpoint, checkpoint_sha = save(archive=True)
            task_rows_path = private_dir / f"task_rows_look_{look:02d}.jsonl"
            task_summary_path = output_dir / f"task_summary_look_{look:02d}.json"
            command = [
                sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p34_task_trajectory",
                "--panel", str(args.dev_panel), "--base_scores", str(args.base_scores),
                "--output_jsonl", str(task_rows_path), "--summary", str(task_summary_path),
                "--condition", f"{args.arm}_seed_{args.seed}", "--look", str(look),
                "--seed", str(args.seed), "--migrated", str(args.migrated),
                "--migrated_sha256", args.migrated_sha256, "--p33", str(args.p33),
                "--p33_sha256", args.p33_sha256, "--i1", str(args.i1),
                "--i1_sha256", expected_endpoint, "--p34", str(checkpoint),
                "--p34_sha256", checkpoint_sha,
            ]
            subprocess.run(command, check=True)
            task_rows = read_jsonl(task_rows_path); guardrail = _task_guardrail(task_rows, lock)
            tier_s_streak, tier_s_event = _advance_sequential_rule(
                condition=bool(guardrail["tier_s_condition"]),
                prior_streak=tier_s_streak,
                required_looks=int(lock["guardrails"]["tier_s_consecutive_looks"]),
            )
            tier_w_streak, tier_w_event = _advance_sequential_rule(
                condition=bool(guardrail["tier_w_condition"]),
                prior_streak=tier_w_streak,
                required_looks=int(lock["guardrails"]["tier_w_consecutive_looks"]),
            )
            if tier_s_event: stop_reason = stop_reason or "tier_s_task_collapse"
            controller_transition_blocked = share_transition is not None or stop_reason is not None
            task_tier_w_event = tier_w_event and not controller_transition_blocked
            audit = None
            if look in AUDIT_LOOKS:
                audit, audit_rows = audit_model(
                    module=module, material=audit_material, direction_index=direction_index,
                    directions=directions, seed=args.seed, step=step, device=args.device,
                )
                write_jsonl(private_dir / f"audit_rows_look_{look:02d}.jsonl", audit_rows)
                last_pi_dep = float(audit["pi_dep"]["point"]); last_chi = float(audit["collateral_chi"])
                if controller_transition_blocked:
                    transition = {
                        "reason": "hold_after_share_transition_or_stop",
                        "rung_before": controller.rung,
                        "rung_after": controller.rung,
                        "at_most_one_rung": True,
                    }
                else:
                    controller, transition = controller_transition(
                        controller, pi_dep=last_pi_dep, chi=last_chi,
                        tier_w_event=task_tier_w_event,
                    )
            else:
                transition = {"reason": "hold_until_registered_pi_look", "rung_before": controller.rung,
                              "rung_after": controller.rung}
                if task_tier_w_event and controller.rung > 0:
                    controller = AnnealingState(controller.rung - 1, controller.chi_max_by_rung, controller.window + 1)
                    transition = {"reason": "tier_w_demote", "rung_before": controller.rung + 1,
                                  "rung_after": controller.rung}
            module.bridge.set_gate_ceiling(controller.gate_ceiling)
            history.append({
                "look": look, "step": step, "learning_rate": lr, "depth": depth,
                "losses": {name: float(value.detach()) for name, value in losses.items()},
                "task": json.loads(task_summary_path.read_text(encoding="utf-8")),
                "guardrail": {**guardrail, "tier_s_streak": tier_s_streak,
                              "tier_w_streak": tier_w_streak,
                              "tier_s_event": tier_s_event, "tier_w_event": tier_w_event},
                "audit": audit, "controller": transition,
                "trailing_shares": classified,
                "share_controller": share_transition,
                "objective_share_controller": (
                    objective_controller_events[-1]
                    if objective_controller_events
                    and objective_controller_events[-1]["step"] == step
                    else None
                ),
                "stop_reason": stop_reason,
            })
            save(archive=False)
            print(f"p34_look arm={args.arm} seed={args.seed} look={look} step={step} "
                  f"delta={guardrail['mean_augmented_minus_base']:.6f} rung={controller.rung} stop={stop_reason}", flush=True)

    frozen_after = tensor_digest({name: value for name, value in module.named_parameters() if not value.requires_grad})
    if frozen_after != frozen_before: raise RuntimeError("P3.4 frozen lineage changed")
    checkpoint, checkpoint_sha = save(archive=False)
    result = {
        "kind": RUN_KIND, "status": "stopped" if stop_reason else "complete",
        "seed": args.seed, "arm": args.arm, "step": step, "target_steps": TOTAL_STEPS,
        "stop_reason": stop_reason, "history": history,
        "share_contract_events": share_contract_events,
        "objective_controller_events": objective_controller_events,
        "objective_weights": objective_weights,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "lock_sha256": sha256_file(args.lock), "chain": chain,
        "amendment_sha256": sha256_file(args.amendment),
        "continuation": continuation_receipt,
        "direction_cache": direction_receipt, "embedding": embedding_receipt,
        "pre_optimizer_loss_share": preflight_classification,
        "frozen_digest_before": frozen_before, "frozen_digest_after": frozen_after,
        "schedule_sha256": hashlib.sha256("\n".join(schedule_hashes).encode()).hexdigest(),
        "confirm_scored": False, "eval_e_scored": False,
    }
    write_json(output_dir / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=("main", "slot"), required=True)
    for name in ("old_summary", "old_private", "new_summary", "new_private", "staged_labels",
                 "positive_audit", "negative_audit", "retention_panel", "direction_cache",
                 "dev_panel", "base_scores", "share_rows", "migrated", "p33", "i1",
                 "continuation", "lock", "amendment", "output_dir", "private_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
