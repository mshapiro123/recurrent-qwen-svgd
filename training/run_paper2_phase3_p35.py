"""Run one locked P3.5 stabilized-landing or probe-reader arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from eval.cache_paper2_phase3_agreement_oracle import _lm_head
from eval.eval_paper2_phase3_p34_share_calibration import _selected_source_cache
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from eval.eval_paper2_phase3_retention_step0 import position_buckets
from models.paper2_dc2_student import install_probe_control_reader
from training.paper2_phase2_matched_alpha import build_adamw_groups
from training.paper2_phase3_p34 import (
    AnnealingState,
    classify_loss_shares,
    controller_transition,
    loss_gradient_bundle,
    normalize_objective_weights,
    sampled_depth,
)
from training.paper2_phase3_p35 import (
    P35_EMA_DECAY,
    P35_LANDING_STEPS,
    P35_LOOK_STEPS,
    P35_PRIMARY_EVAL_CEILING,
    P35_SOURCE_STEP,
    initialize_ema,
    landing_learning_rate,
    load_p35_direction_lookup,
    set_p35_trainable,
    update_ema,
)
from training.run_paper2_phase3_p34 import (
    NEGATIVE_PER_BATCH,
    POSITIVE_PER_BATCH,
    SHARE_WINDOW_STEPS,
    _advance_sequential_rule,
    _checkpoint_state,
    _losses,
    _source_batch,
    _task_guardrail,
)
from training.run_paper2_phase3_p33 import (
    _active_record_pools,
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
from training.paper2_phase3_p34 import apply_weighted_gradient_bundle
from training.paper2_phase3_p33_prep import sha256_file


RUN_KIND = "paper2_phase3_p35_landing_v1"


def _adamw_group_names(module: torch.nn.Module) -> list[list[str]]:
    by_id = {id(parameter): name for name, parameter in module.named_parameters()}
    return [
        [by_id[id(parameter)] for parameter in group["params"]]
        for group in build_adamw_groups(module, weight_decay=0.01)
    ]


def _restore_optimizer_by_name(
    *,
    optimizer: torch.optim.Optimizer,
    saved: Mapping[str, Any],
    old_group_names: list[list[str]],
    new_group_names: list[list[str]],
) -> dict[str, Any]:
    if len(saved["param_groups"]) != len(old_group_names):
        raise RuntimeError("P3.5 source optimizer group count changed")
    old_name_to_id: dict[str, int] = {}
    for group, names in zip(saved["param_groups"], old_group_names):
        if len(group["params"]) != len(names):
            raise RuntimeError("P3.5 source optimizer parameter order changed")
        old_name_to_id.update(zip(names, group["params"]))
    current = optimizer.state_dict()
    new_name_to_id: dict[str, int] = {}
    for group, names in zip(current["param_groups"], new_group_names):
        if len(group["params"]) != len(names):
            raise RuntimeError("P3.5 destination optimizer parameter order changed")
        new_name_to_id.update(zip(names, group["params"]))
    restored = 0
    for name, new_id in new_name_to_id.items():
        old_id = old_name_to_id.get(name)
        if old_id is not None and old_id in saved["state"]:
            current["state"][new_id] = saved["state"][old_id]
            restored += 1
    for new_group, old_group in zip(current["param_groups"], saved["param_groups"]):
        for key, value in old_group.items():
            if key != "params":
                new_group[key] = value
    optimizer.load_state_dict(current)
    return {
        "restored_parameter_states": restored,
        "new_parameter_states": len(new_name_to_id) - restored,
        "source_named_parameters": len(old_name_to_id),
        "destination_named_parameters": len(new_name_to_id),
    }


def _save_p35_checkpoint(
    *,
    path: Path,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_state: Mapping[str, torch.Tensor],
    state_variant: str,
    seed: int,
    arm: str,
    step: int,
    history: list[dict[str, Any]],
    share_window: deque[dict[str, float]],
    schedule_hashes: list[str],
    objective_weights: Mapping[str, float],
    generator: torch.Generator,
    lock_sha256: str,
    source_receipt: Mapping[str, Any],
) -> str:
    raw = _checkpoint_state(module, None)
    trainable_state = (
        raw
        if state_variant == "raw"
        else {name: ema_state[name].to(dtype=raw[name].dtype) for name in raw}
    )
    return atomic_torch_save(
        {
            "kind": RUN_KIND,
            "seed": seed,
            "arm": arm,
            "step": step,
            "state_variant": state_variant,
            "control_reader": "probe" if arm == "probe_reader" else "mean",
            "evaluation_gate_ceiling": P35_PRIMARY_EVAL_CEILING,
            "training_rung": 0,
            "trainable_state": trainable_state,
            "raw_trainable_state": raw if state_variant == "ema" else None,
            "ema_state": {name: value.cpu() for name, value in ema_state.items()},
            "ema_decay": P35_EMA_DECAY,
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "share_window": list(share_window),
            "schedule_hashes": schedule_hashes,
            "objective_weights": dict(objective_weights),
            "runtime_controller_frozen": True,
            "generator_state": generator.get_state(),
            "rng_state": rng_state(),
            "lock_sha256": lock_sha256,
            "source_receipt": dict(source_receipt),
        },
        path,
    )


@torch.inference_mode()
def _audit_ema_state(
    *,
    module: torch.nn.Module,
    ema_state: Mapping[str, torch.Tensor],
    material: Mapping[str, Any],
    direction_index: Mapping[str, int],
    directions: torch.Tensor,
    seed: int,
    step: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    named = dict(module.named_parameters())
    if set(ema_state) - set(named):
        raise RuntimeError("P3.5 EMA audit state contains unknown parameters")
    raw = {name: named[name].detach().clone() for name in ema_state}
    try:
        for name, value in ema_state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
        return audit_model(
            module=module,
            material=material,
            direction_index=direction_index,
            directions=directions,
            seed=seed,
            step=step,
            device=device,
        )
    finally:
        for name, value in raw.items():
            named[name].copy_(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if not (
        lock.get("kind") == "paper2_phase3_p35_executed_lock_v1"
        and lock.get("status") == "approved_for_training"
        and lock.get("locked_before_training")
        and lock.get("training_authorized")
        and lock.get("mark_ratified")
    ):
        raise RuntimeError("P3.5 training remains disabled until the executed lock is ratified")
    expected = lock["initialization"][f"seed_{args.seed}"]["sha256"]
    if sha256_file(args.p34) != expected:
        raise RuntimeError("P3.5 source endpoint SHA mismatch")
    if args.arm == "probe_reader" and args.seed != 0:
        raise RuntimeError("Arm R is registered on seed 0 only")
    if args.arm == "stabilized" and args.seed not in (0, 1):
        raise RuntimeError("Arm S is registered on seeds 0 and 1")
    for path in (args.dev_panel, args.staged_labels, args.positive_audit, args.negative_audit):
        if any(token in str(path).casefold() for token in ("confirm", "eval_e")):
            raise RuntimeError("P3.5 sealed-data contact")

    random.seed(20260815 + args.seed)
    np.random.seed(20260815 + args.seed)
    torch.manual_seed(20260815 + args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260815 + args.seed)
    source_payload = torch.load(args.p34, map_location="cpu", weights_only=False)
    if int(source_payload.get("step", -1)) != P35_SOURCE_STEP:
        raise RuntimeError("P3.5 requires the step-4000 P3.4 endpoint")
    if source_payload.get("optimizer_state") is None:
        raise RuntimeError("P3.5 short landing requires resumable optimizer state")
    if source_payload.get("generator_state") is None or source_payload.get("rng_state") is None:
        raise RuntimeError("P3.5 short landing requires resumable RNG state")

    sources_spec = {
        "old": (args.old_summary, args.old_private),
        "new": (args.new_summary, args.new_private),
    }
    embedding, embedding_receipt = _lm_head(sources_spec)
    module, chain = load_condition(
        embedding_weight=embedding,
        migrated=args.migrated,
        migrated_sha256=args.migrated_sha256,
        p33=args.p33,
        p33_sha256=args.p33_sha256,
        i1=args.i1,
        i1_sha256=args.i1_sha256,
        p34=args.p34,
        p34_sha256=expected,
    )
    module.train()
    set_p35_trainable(module, arm="stabilized")
    old_group_names = _adamw_group_names(module)
    if args.arm == "probe_reader":
        install_probe_control_reader(module, n_probes=4)
    trainable = set_p35_trainable(module, arm=args.arm)
    parameters = list(trainable.values())
    new_group_names = _adamw_group_names(module)
    optimizer = torch.optim.AdamW(
        build_adamw_groups(module, weight_decay=0.01), lr=0.0, betas=(0.9, 0.999)
    )
    optimizer_receipt = _restore_optimizer_by_name(
        optimizer=optimizer,
        saved=source_payload["optimizer_state"],
        old_group_names=old_group_names,
        new_group_names=new_group_names,
    )
    objective_weights = normalize_objective_weights(source_payload["objective_weights"])
    module.bridge.set_gate_ceiling(P35_PRIMARY_EVAL_CEILING)
    ema_state = initialize_ema(_checkpoint_state(module, None))

    direction_payload = torch.load(args.direction_cache, map_location="cpu", weights_only=False)
    direction_index, directions = load_p35_direction_lookup(direction_payload)
    direction_receipt = {
        "path": str(args.direction_cache),
        "sha256": sha256_file(args.direction_cache),
        "rows": len(direction_index),
        "source_anchor_identity_rate": direction_payload["source_anchor_identity_rate"],
    }
    records = read_jsonl(args.staged_labels)
    positives, negatives = _active_record_pools(records)
    sources = {}
    for source, (summary, private) in sources_spec.items():
        sources[source], _receipt = _selected_source_cache(
            source=source, records=records, summary_path=summary, private_root=private
        )
    audit_material = load_audit_material(
        positive_path=args.positive_audit,
        negative_path=args.negative_audit,
        retention_path=args.retention_panel,
        sources=sources_spec,
    )

    output_dir, private_dir = args.output_dir, args.private_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    lock_sha = sha256_file(args.lock)
    source_receipt = {"path": str(args.p34), "sha256": expected, "step": 4000}
    history: list[dict[str, Any]] = []
    schedule_hashes: list[str] = []
    share_window: deque[dict[str, float]] = deque(maxlen=SHARE_WINDOW_STEPS)
    generator = torch.Generator()
    generator.set_state(source_payload["generator_state"])
    restore_rng(source_payload["rng_state"])
    stop_reason: str | None = None
    tier_s_streak = 0
    tier_w_streak = 0
    start_step = P35_SOURCE_STEP
    resume_path = private_dir / "resume.pt"
    if resume_path.is_file():
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resumed.get("kind") != RUN_KIND
            or resumed.get("seed") != args.seed
            or resumed.get("arm") != args.arm
            or resumed.get("state_variant") != "raw"
            or resumed.get("lock_sha256") != lock_sha
            or resumed.get("source_receipt", {}).get("sha256") != expected
        ):
            raise RuntimeError("P3.5 durable resume identity changed")
        current = dict(module.named_parameters())
        if set(resumed["trainable_state"]) != set(trainable):
            raise RuntimeError("P3.5 durable resume trainable schema changed")
        with torch.no_grad():
            for name, value in resumed["trainable_state"].items():
                current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
        optimizer.load_state_dict(resumed["optimizer_state"])
        ema_state = {
            name: value.float().clone() for name, value in resumed["ema_state"].items()
        }
        history = list(resumed["history"])
        share_window.extend(resumed["share_window"])
        schedule_hashes = list(resumed["schedule_hashes"])
        objective_weights = normalize_objective_weights(resumed["objective_weights"])
        generator.set_state(resumed["generator_state"])
        restore_rng(resumed["rng_state"])
        start_step = int(resumed["step"])
        if history:
            tier_s_streak = int(history[-1]["guardrail"]["tier_s_streak"])
            tier_w_streak = int(history[-1]["guardrail"]["tier_w_streak"])
    frozen_before = tensor_digest(
        {name: value for name, value in module.named_parameters() if not value.requires_grad}
    )

    for step in range(start_step + 1, P35_SOURCE_STEP + P35_LANDING_STEPS + 1):
        pos = torch.randint(len(positives), (POSITIVE_PER_BATCH,), generator=generator)
        neg = torch.randint(len(negatives), (NEGATIVE_PER_BATCH,), generator=generator)
        rows = [positives[int(index)] for index in pos] + [negatives[int(index)] for index in neg]
        rows = [rows[index] for index in torch.randperm(len(rows), generator=generator).tolist()]
        schedule_hashes.append(
            hashlib.sha256("\n".join(str(row["record_id"]) for row in rows).encode()).hexdigest()
        )
        batch = _source_batch(
            rows=rows,
            sources=sources,
            direction_index=direction_index,
            directions=directions,
            device=args.device,
        )
        depth = sampled_depth(generator=generator)
        module.train()
        losses, _metrics = _losses(module=module, batch=batch, depth=depth, slot_lift=None)
        if any(not bool(torch.isfinite(value)) for value in losses.values()):
            stop_reason = "non_finite_loss"
            break
        bundle = loss_gradient_bundle(
            losses=losses, module=module, parameters=parameters, slot_lift=None
        )
        optimizer.zero_grad(set_to_none=True)
        update = apply_weighted_gradient_bundle(
            bundle=bundle, parameters=parameters, weights=objective_weights
        )
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients or any(not bool(value.isfinite().all()) for value in gradients):
            stop_reason = "non_finite_gradient"
            break
        norms = update["postclip"]["postclip_gradient_norms"]
        denominator = sum(norms.values())
        share_window.append({name: value / denominator for name, value in norms.items()})
        lr = landing_learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        update_ema(ema_state, _checkpoint_state(module, None))

        share_read = None
        if step % SHARE_WINDOW_STEPS == 0:
            trailing = {
                name: sum(row[name] for row in share_window) / len(share_window)
                for name in share_window[0]
            }
            share_read = classify_loss_shares(trailing)
            share_window.clear()

        if step in P35_LOOK_STEPS:
            look = P35_LOOK_STEPS.index(step) + 1
            raw_path = private_dir / f"raw_step_{step}.pt"
            ema_path = private_dir / f"ema_step_{step}.pt"
            raw_sha = _save_p35_checkpoint(
                path=raw_path, module=module, optimizer=optimizer, ema_state=ema_state,
                state_variant="raw", seed=args.seed, arm=args.arm, step=step,
                history=history, share_window=share_window, schedule_hashes=schedule_hashes,
                objective_weights=objective_weights, generator=generator,
                lock_sha256=lock_sha, source_receipt=source_receipt,
            )
            ema_sha = _save_p35_checkpoint(
                path=ema_path, module=module, optimizer=optimizer, ema_state=ema_state,
                state_variant="ema", seed=args.seed, arm=args.arm, step=step,
                history=history, share_window=share_window, schedule_hashes=schedule_hashes,
                objective_weights=objective_weights, generator=generator,
                lock_sha256=lock_sha, source_receipt=source_receipt,
            )
            task_rows_path = private_dir / f"ema_task_rows_step_{step}.jsonl"
            task_summary_path = output_dir / f"ema_task_summary_step_{step}.json"
            command = [
                sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p34_task_trajectory",
                "--panel", str(args.dev_panel), "--base_scores", str(args.base_scores),
                "--output_jsonl", str(task_rows_path), "--summary", str(task_summary_path),
                "--condition", f"p35_{args.arm}_ema_seed_{args.seed}", "--look", str(look),
                "--seed", str(args.seed), "--migrated", str(args.migrated),
                "--migrated_sha256", args.migrated_sha256, "--p33", str(args.p33),
                "--p33_sha256", args.p33_sha256, "--i1", str(args.i1),
                "--i1_sha256", args.i1_sha256, "--p34", str(args.p34),
                "--p34_sha256", expected, "--p35", str(ema_path),
                "--p35_sha256", ema_sha, "--control_reader",
                "probe" if args.arm == "probe_reader" else "mean",
                "--gate_ceiling_override", str(P35_PRIMARY_EVAL_CEILING),
            ]
            subprocess.run(command, check=True)
            task_rows = read_jsonl(task_rows_path)
            guardrail = _task_guardrail(task_rows, lock["p34_lock"])
            tier_s_streak, tier_s_event = _advance_sequential_rule(
                condition=bool(guardrail["tier_s_condition"]),
                prior_streak=tier_s_streak,
                required_looks=4,
            )
            tier_w_streak, tier_w_event = _advance_sequential_rule(
                condition=bool(guardrail["tier_w_condition"]),
                prior_streak=tier_w_streak,
                required_looks=2,
            )
            audit, audit_rows = _audit_ema_state(
                module=module,
                ema_state=ema_state,
                material=audit_material,
                direction_index=direction_index,
                directions=directions,
                seed=args.seed,
                step=step,
                device=args.device,
            )
            write_jsonl(private_dir / f"audit_rows_step_{step}.jsonl", audit_rows)
            counterfactual_state, counterfactual = controller_transition(
                AnnealingState(rung=0),
                pi_dep=float(audit["pi_dep"]["point"]),
                chi=float(audit["collateral_chi"]),
                tier_w_event=bool(tier_w_event),
            )
            history.append(
                {
                    "look": look,
                    "step": step,
                    "learning_rate": lr,
                    "raw_checkpoint": {"path": str(raw_path), "sha256": raw_sha},
                    "ema_checkpoint": {"path": str(ema_path), "sha256": ema_sha},
                    "task": json.loads(task_summary_path.read_text(encoding="utf-8")),
                    "guardrail": {
                        **guardrail,
                        "tier_s_streak": tier_s_streak,
                        "tier_w_streak": tier_w_streak,
                        "tier_s_event": tier_s_event,
                        "tier_w_event": tier_w_event,
                    },
                    "audit": audit,
                    "share_read_observe_only": share_read,
                    "runtime_controller": {
                        "frozen": True,
                        "pinned_rung": 0,
                        "counterfactual": counterfactual,
                        "counterfactual_rung": counterfactual_state.rung,
                    },
                }
            )
            if tier_s_event:
                stop_reason = "tier_s_task_collapse"
            _save_p35_checkpoint(
                path=resume_path, module=module, optimizer=optimizer, ema_state=ema_state,
                state_variant="raw", seed=args.seed, arm=args.arm, step=step,
                history=history, share_window=share_window, schedule_hashes=schedule_hashes,
                objective_weights=objective_weights, generator=generator,
                lock_sha256=lock_sha, source_receipt=source_receipt,
            )
            if stop_reason:
                break
        elif step % 20 == 0:
            _save_p35_checkpoint(
                path=resume_path, module=module, optimizer=optimizer, ema_state=ema_state,
                state_variant="raw", seed=args.seed, arm=args.arm, step=step,
                history=history, share_window=share_window, schedule_hashes=schedule_hashes,
                objective_weights=objective_weights, generator=generator,
                lock_sha256=lock_sha, source_receipt=source_receipt,
            )

    frozen_after = tensor_digest(
        {name: value for name, value in module.named_parameters() if not value.requires_grad}
    )
    if frozen_after != frozen_before:
        raise RuntimeError("P3.5 frozen lineage changed")
    result = {
        "kind": RUN_KIND,
        "status": "stopped" if stop_reason else "complete",
        "seed": args.seed,
        "arm": args.arm,
        "source_step": P35_SOURCE_STEP,
        "step": history[-1]["step"] if history else start_step,
        "target_step": P35_SOURCE_STEP + P35_LANDING_STEPS,
        "stop_reason": stop_reason,
        "history": history,
        "optimizer_restore": optimizer_receipt,
        "objective_weights_frozen": objective_weights,
        "runtime_controller_frozen": True,
        "training_rung": 0,
        "primary_evaluation_ceiling": P35_PRIMARY_EVAL_CEILING,
        "direction_cache": direction_receipt,
        "embedding": embedding_receipt,
        "chain": chain,
        "source": source_receipt,
        "source_rng_restored": True,
        "resume": {
            "path": str(resume_path),
            "sha256": sha256_file(resume_path) if resume_path.is_file() else None,
        },
        "lock_sha256": lock_sha,
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "schedule_sha256": hashlib.sha256("\n".join(schedule_hashes).encode()).hexdigest(),
        "secondary_score_bundle_required": True,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(output_dir / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=("stabilized", "probe_reader"), required=True)
    for name in (
        "old_summary", "old_private", "new_summary", "new_private", "staged_labels",
        "positive_audit", "negative_audit", "retention_panel", "direction_cache",
        "dev_panel", "base_scores", "migrated", "p33", "i1", "p34", "lock",
        "output_dir", "private_dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--i1_sha256", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
