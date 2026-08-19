"""Run one signed, resumable Stage 2B-D seed through its authorized gate."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from eval.eval_paper2_stage2b_campaign import (
    Stage2BTaskInferenceGraph,
    read_jsonl,
    score_dev1,
    score_dev2_margins,
    write_jsonl,
)
from models.lora import apply_loop_scoped_lora_to_recurrent_block
from models.paper2_stage2b_depth import Stage2BDepthAttachment
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from training.paper2_stage2b_depth import (
    DepthObjectiveWeights,
    assert_optimizer_construction_authorized,
    configure_stage2b_trainable_groups,
    depth_objective,
    stage2b_learning_rate,
    stage_for_step,
)
from training.paper2_stage2b_runtime import (
    DeterministicCycleSampler,
    ShardedTeacherCache,
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    collate_teacher_rows,
    restore_rng_state,
    schedule_digest,
    sha256_file,
)
from training.run_paper2_phase3_p33 import tensor_digest


RUN_KIND = "paper2_stage2b_depth_campaign_v1"
AUTHORIZED_FIRST_GATE = 5_000
RETAIN_STEPS = {0, 1_000, 2_500, 5_000, 6_000, 12_000, 18_000, 21_600, 23_000, 24_000}
PEAK_LRS = {"new_modules": 5e-4, "loop_lora": 5e-5, "gates": 2e-4}


def _named_trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _apply_named_state(module: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    current = dict(module.named_parameters())
    expected = {name for name, parameter in current.items() if parameter.requires_grad}
    if set(state) != expected:
        raise RuntimeError("Stage 2B checkpoint trainable schema changed")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(current[name].device, current[name].dtype))


def _update_ema(ema: dict[str, torch.Tensor], module: torch.nn.Module, decay: float) -> None:
    current = _named_trainable_state(module)
    for name, value in current.items():
        ema[name].mul_(decay).add_(value.float(), alpha=1.0 - decay)


def _with_state(module: torch.nn.Module, state: Mapping[str, torch.Tensor]):
    class Context:
        def __enter__(self_nonlocal):
            self_nonlocal.raw = _named_trainable_state(module)
            _apply_named_state(module, state)
            return module

        def __exit__(self_nonlocal, *_args):
            _apply_named_state(module, self_nonlocal.raw)

    return Context()


def _checkpoint_payload(
    *,
    wrapper: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Mapping[str, torch.Tensor],
    sampler: DeterministicCycleSampler,
    amplitude_generator: torch.Generator,
    step: int,
    seed: int,
    lock_sha256: str,
    history: list[dict[str, Any]],
    schedule_hashes: list[str],
    frozen_digest: str,
) -> dict[str, Any]:
    return {
        "kind": RUN_KIND,
        "seed": seed,
        "step": step,
        "raw_trainable_state": _named_trainable_state(wrapper),
        "ema_state": {name: value.cpu() for name, value in ema.items()},
        "optimizer_state": optimizer.state_dict(),
        "sampler_state": sampler.state_dict(),
        "amplitude_generator_state": amplitude_generator.get_state(),
        "rng_state": capture_rng_state(),
        "history": history,
        "schedule_hashes": schedule_hashes,
        "lock_sha256": lock_sha256,
        "frozen_digest": frozen_digest,
        "confirm_scored": False,
        "eval_e_scored": False,
    }


@torch.inference_mode()
def _pass_one_identity(wrapper: Any, row: Mapping[str, Any], device: str) -> float:
    inputs = row["input_ids"].long().unsqueeze(0).to(device)
    attention = torch.ones_like(inputs)
    plain = wrapper(
        input_ids=inputs,
        attention_mask=attention,
        max_loops=1,
        use_cache=False,
        return_dict=True,
    ).logits
    attached = wrapper(
        input_ids=inputs,
        attention_mask=attention,
        max_loops=1,
        stage2b_depth_enabled=True,
        stage2b_stage="M4",
        stage2b_amplitude=0.05,
        use_cache=False,
        return_dict=True,
    ).logits
    difference = float((plain.float() - attached.float()).abs().max().cpu())
    if not torch.equal(plain, attached):
        raise RuntimeError(f"Stage 2B pass-one identity failed: max_abs={difference}")
    return difference


@torch.inference_mode()
def _finite_horizon_diagnostic(attachment: Stage2BDepthAttachment, stage: str) -> dict[str, Any]:
    device = next(attachment.parameters()).device
    generator = torch.Generator(device=device).manual_seed(20260819)
    state = torch.randn(
        (2, 4, 8, attachment.flow.latent_dim), generator=generator, device=device
    )
    context = torch.randn((2, attachment.context_dim), generator=generator, device=device)
    direction = torch.randn(state.shape, generator=generator, device=device)
    direction = direction / direction.float().norm().clamp_min(1e-12)
    epsilon = 0.02

    def apply(value: torch.Tensor, horizon: int) -> torch.Tensor:
        current = value
        for index in range(horizon):
            current = attachment.flow.step(
                current,
                context,
                index,
                prompt_context=context,
                dynamic_routing=stage in {"M3", "M4"},
                constitutive_active=True,
                forced_lane_one=False,
            ).state
        return current

    gains = []
    for horizon in range(1, 5):
        plus = apply(state + epsilon * direction, horizon)
        minus = apply(state - epsilon * direction, horizon)
        gains.append(float(((plus.float() - minus.float()) / (2 * epsilon)).norm().cpu()))
    baseline = attachment.flow.step(
        state,
        context,
        0,
        prompt_context=context,
        dynamic_routing=stage in {"M3", "M4"},
        constitutive_active=True,
    )
    return {
        "kind": "paper2_stage2b_finite_horizon_watch_v1",
        "centered_directional_gains": gains,
        "catastrophe_threshold": 100.0,
        "catastrophe": max(gains) >= 100.0,
        "sinkhorn_row_residual_max": float(baseline.sinkhorn_row_residual.max().cpu()),
        "sinkhorn_column_residual_max": float(baseline.sinkhorn_column_residual.max().cpu()),
        "lambda2_mean": float(baseline.lambda2.mean().cpu()),
        "lane_effective_rank_mean": float(baseline.effective_rank.mean().cpu()),
    }


def _build_model(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]], dict[str, list[torch.nn.Parameter]]]:
    spec = MODEL_SPECS["base"]
    base = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        cache_dir=args.model_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    sidecar, chain = load_condition(
        embedding_weight=base.get_output_embeddings().weight.detach().cpu(),
        migrated=args.migrated,
        migrated_sha256=args.migrated_sha256,
        p33=args.p33,
        p33_sha256=args.p33_sha256,
        i1=args.i1,
        i1_sha256=args.i1_sha256,
        p34=args.p34,
        p34_sha256=args.p34_sha256,
        p35=args.p35,
        p35_sha256=args.p35_sha256,
        control_reader="mean",
    )
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=6, recurrent_end=18)
    ).to(args.device)
    replaced = apply_loop_scoped_lora_to_recurrent_block(
        wrapper, rank=16, alpha=16, adapter_dtype=torch.float32
    )
    if replaced != 48:
        raise RuntimeError(f"Stage 2B expected 48 loop-scoped adapters, observed {replaced}")
    wrapper.install_stage2b_depth_attachment(
        Stage2BDepthAttachment.from_phase3(sidecar).to(args.device)
    )
    groups = configure_stage2b_trainable_groups(wrapper)
    wrapper.eval()
    return wrapper, chain, groups


def _m2_parameter_is_dormant(group: str, name: str) -> bool:
    if group == "loop_lora":
        return True
    if group != "new_modules":
        return False
    return name.startswith(
        (
            "stage2b_depth_attachment.flow.router_norm.",
            "stage2b_depth_attachment.flow.router.",
        )
    ) or name == "stage2b_depth_attachment.flow.rho_logits"


def audit_stage_gradients(
    *,
    groups: dict[str, list[torch.nn.Parameter]],
    parameter_names: dict[int, str],
    stage: str,
) -> dict[str, Any]:
    missing_expected = []
    missing_active = []
    nonfinite = []
    finite = 0
    for group, parameters in groups.items():
        for parameter in parameters:
            name = parameter_names[id(parameter)]
            gradient = parameter.grad
            if gradient is None:
                target = (
                    missing_expected
                    if stage == "M2" and _m2_parameter_is_dormant(group, name)
                    else missing_active
                )
                target.append({"group": group, "name": name})
            elif not bool(torch.isfinite(gradient).all()):
                nonfinite.append({"group": group, "name": name})
            else:
                finite += 1
    return {
        "stage": stage,
        "finite_parameter_tensors": finite,
        "missing_expected": missing_expected,
        "missing_active": missing_active,
        "nonfinite": nonfinite,
        "pass": finite > 0 and not missing_active and not nonfinite,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    assert_optimizer_construction_authorized(args.lock)
    lock_sha = sha256_file(args.lock)
    if args.target_step not in {5_000, 8_000, 24_000}:
        raise RuntimeError("Stage 2B target step is outside an authorized gate")
    if args.target_step > AUTHORIZED_FIRST_GATE and not args.continuation_authority:
        raise RuntimeError("Stage 2B continuation requires a post-gate authority receipt")
    if args.target_step > AUTHORIZED_FIRST_GATE:
        authority = json.loads(args.continuation_authority.read_text(encoding="utf-8"))
        if authority.get("training_authorized") is not True:
            raise RuntimeError("Stage 2B continuation authority is not executable")

    random.seed(20260819 + args.seed)
    np.random.seed(20260819 + args.seed)
    torch.manual_seed(20260819 + args.seed)
    torch.cuda.manual_seed_all(20260819 + args.seed)
    cache = ShardedTeacherCache(args.teacher_cache_index)
    if cache.index["corpus_sha256"] != lock["training"]["data"]["corpus_sha256"]:
        raise RuntimeError("Stage 2B teacher cache and signed corpus disagree")
    wrapper, chain, groups = _build_model(args)
    parameter_names = {id(value): name for name, value in wrapper.named_parameters()}
    trainable_count = {name: sum(value.numel() for value in values) for name, values in groups.items()}
    frozen = {
        name: value for name, value in wrapper.named_parameters() if not value.requires_grad
    }
    frozen_digest = tensor_digest(frozen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "kind": "paper2_stage2b_campaign_preoptimizer_v1",
        "seed": args.seed,
        "lock_sha256": lock_sha,
        "teacher_cache_index_sha256": sha256_file(args.teacher_cache_index),
        "teacher_rows": len(cache),
        "trainable_parameter_count": trainable_count,
        "checkpoint_chain": chain,
        "target_step": args.target_step,
        "start_step": 0,
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "preoptimizer_receipt.json", preflight)
    if args.preflight_only:
        return preflight

    optimizer_groups = [
        {"params": values, "lr": 0.0, "name": name, "weight_decay": 0.01}
        for name, values in groups.items()
    ]
    # This assertion must remain immediately before optimizer construction.
    assert_optimizer_construction_authorized(args.lock)
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999))
    ema = {name: value.float() for name, value in _named_trainable_state(wrapper).items()}
    sampler = DeterministicCycleSampler.create(len(cache), 20260819 + args.seed)
    amplitude_generator = torch.Generator(device="cpu").manual_seed(2026081900 + args.seed)
    history: list[dict[str, Any]] = []
    schedule_hashes: list[str] = []
    start_step = 0
    resume_path = args.private_dir / "resume.pt"
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume.get("kind") != RUN_KIND
            or int(resume.get("seed", -1)) != args.seed
            or resume.get("lock_sha256") != lock_sha
            or resume.get("frozen_digest") != frozen_digest
        ):
            raise RuntimeError("Stage 2B durable resume identity changed")
        _apply_named_state(wrapper, resume["raw_trainable_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        ema = {name: value.float() for name, value in resume["ema_state"].items()}
        sampler = DeterministicCycleSampler.restore(resume["sampler_state"])
        amplitude_generator.set_state(resume["amplitude_generator_state"])
        restore_rng_state(resume["rng_state"])
        history = list(resume["history"])
        schedule_hashes = list(resume["schedule_hashes"])
        start_step = int(resume["step"])
        preflight["start_step"] = start_step
        atomic_json(args.output_dir / "preoptimizer_receipt.json", preflight)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS["base"]["model"], revision=MODEL_SPECS["base"]["revision"]
    )
    dev1 = read_jsonl(args.dev1_panel)
    dev2_manifest = read_jsonl(args.dev2_manifest)
    reference = {str(row["item_id"]): row for row in read_jsonl(args.reference_rows)}
    dev2 = [reference[str(row["item_id"])] for row in dev2_manifest]
    base_rows = {str(row["item_id"]): row for row in read_jsonl(args.base_scores)}
    initialization_rows = {
        str(row["item_id"]): row for row in read_jsonl(args.initialization_scores)
    }
    if len(dev1) != 1_024 or len(dev2) != 2_048:
        raise RuntimeError("Stage 2B DEV panel sizes changed")

    weights_row = lock["training"]["objective"]["weights_by_seed"][str(args.seed)]
    weights = DepthObjectiveWeights(
        ce=float(weights_row["ce"]),
        kl=float(weights_row["kl"]),
        monotonicity=float(weights_row["monotonicity"]),
    )
    stop_reason = None
    last_gradient_audit: dict[str, Any] | None = None
    started = time.time()
    for step in range(start_step + 1, args.target_step + 1):
        batch_indexes = sampler.take(128)
        schedule_hashes.append(schedule_digest(batch_indexes))
        amplitude = 0.02 + (0.11 - 0.02) * float(
            torch.rand((), generator=amplitude_generator)
        )
        stage = stage_for_step(step)
        optimizer.zero_grad(set_to_none=True)
        component_sums = {"ce": 0.0, "kl": 0.0, "monotonicity": 0.0}
        for offset in range(0, 128, args.microbatch_size):
            indexes = batch_indexes[offset : offset + args.microbatch_size]
            rows = [cache[index] for index in indexes]
            batch = collate_teacher_rows(rows, device=args.device)
            output = wrapper(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_loops=4,
                stage2b_depth_enabled=True,
                stage2b_stage=stage,
                stage2b_amplitude=amplitude,
                return_loop_logits=True,
                use_cache=False,
                return_dict=True,
            )
            loops = output.loop_logits[:, 0]
            loop_logits = [loops[:, index, :-1] for index in range(4)]
            total, components = depth_objective(
                loop_logits=loop_logits,
                teacher_topk_token_ids=batch["teacher_topk_token_ids"],
                teacher_topk_logits=batch["teacher_topk_logits"],
                teacher_tokens=batch["teacher_tokens"],
                loss_mask=batch["loss_mask"],
                weights=weights,
                hinge_delta=0.01,
            )
            scale = len(indexes) / 128.0
            (total * scale).backward()
            for name in component_sums:
                component_sums[name] += float(components[name].detach().cpu()) * scale
            del output, loops, loop_logits, total, components, batch
        last_gradient_audit = audit_stage_gradients(
            groups=groups, parameter_names=parameter_names, stage=stage
        )
        if last_gradient_audit["nonfinite"]:
            stop_reason = "nonfinite_gradient"
        elif last_gradient_audit["missing_active"] or not last_gradient_audit["finite_parameter_tensors"]:
            stop_reason = "missing_active_gradient"
        if stop_reason is None:
            for group in optimizer.param_groups:
                group["lr"] = stage2b_learning_rate(step, peak=PEAK_LRS[group["name"]])
            optimizer.step()
            _update_ema(ema, wrapper, 0.999)
        if stop_reason is not None:
            break

        if step % args.resume_interval == 0 and step % 1000 != 0:
            atomic_torch_save(
                resume_path,
                _checkpoint_payload(
                    wrapper=wrapper,
                    optimizer=optimizer,
                    ema=ema,
                    sampler=sampler,
                    amplitude_generator=amplitude_generator,
                    step=step,
                    seed=args.seed,
                    lock_sha256=lock_sha,
                    history=history,
                    schedule_hashes=schedule_hashes,
                    frozen_digest=frozen_digest,
                ),
            )

        if step % 1000 == 0:
            look = step // 1000
            raw_state = _named_trainable_state(wrapper)
            ema_state = {name: value.clone() for name, value in ema.items()}
            archive = args.private_dir / f"ema_step_{step:05d}.pt"
            archive_sha = atomic_torch_save(
                archive,
                {
                    "kind": RUN_KIND,
                    "seed": args.seed,
                    "step": step,
                    "state_variant": "ema",
                    "trainable_state": ema_state,
                    "raw_trainable_state": raw_state,
                    "lock_sha256": lock_sha,
                },
            )
            with _with_state(wrapper, ema_state):
                identity = _pass_one_identity(wrapper, cache[0], args.device)
                graph = Stage2BTaskInferenceGraph(
                    wrapper=wrapper, stage=stage, amplitude=0.05
                )
                dev1_rows, dev1_summary = score_dev1(
                    graph=graph,
                    tokenizer=tokenizer,
                    panel=dev1,
                    base_rows=base_rows,
                    initialization_rows=initialization_rows,
                    seed=args.seed,
                    look=look,
                    mcq_batch_size=args.eval_mcq_batch_size,
                    generation_batch_size=args.eval_generation_batch_size,
                )
                dev2_rows, dev2_summary = score_dev2_margins(
                    graph=graph,
                    tokenizer=tokenizer,
                    rows=dev2,
                    seed=args.seed,
                    look=look,
                    batch_size=args.eval_margin_batch_size,
                )
                diagnostic = _finite_horizon_diagnostic(
                    wrapper.stage2b_depth_attachment, stage
                )
            write_jsonl(args.private_dir / f"dev1_rows_look_{look}.jsonl", dev1_rows)
            write_jsonl(args.private_dir / f"dev2_margin_rows_look_{look}.jsonl", dev2_rows)
            look_receipt = {
                "kind": "paper2_stage2b_campaign_look_v1",
                "seed": args.seed,
                "look": look,
                "step": step,
                "stage": stage,
                "ema_checkpoint": {"path": str(archive), "sha256": archive_sha},
                "objective_components": component_sums,
                "loop1_kl": component_sums["kl"],
                "pass_one_max_abs_difference": identity,
                "dev1": dev1_summary,
                "dev2": dev2_summary,
                "finite_horizon": diagnostic,
                "r2_desk_read": {
                    "loop1_kl": component_sums["kl"],
                    "finite_horizon_gains": diagnostic["centered_directional_gains"],
                    "lane_effective_rank_mean": diagnostic["lane_effective_rank_mean"],
                    "lambda2_mean": diagnostic["lambda2_mean"],
                },
                "confirm_scored": False,
                "eval_e_scored": False,
            }
            atomic_json(args.output_dir / f"look_{look}.json", look_receipt)
            history.append(look_receipt)
            if not bool(dev1_summary["safety"]["pass"]):
                stop_reason = "dev1_hard_floor"
            if bool(diagnostic["catastrophe"]):
                stop_reason = "finite_horizon_catastrophe"
            atomic_torch_save(
                resume_path,
                _checkpoint_payload(
                    wrapper=wrapper,
                    optimizer=optimizer,
                    ema=ema,
                    sampler=sampler,
                    amplitude_generator=amplitude_generator,
                    step=step,
                    seed=args.seed,
                    lock_sha256=lock_sha,
                    history=history,
                    schedule_hashes=schedule_hashes,
                    frozen_digest=frozen_digest,
                ),
            )
            if stop_reason:
                break
        if step % 20 == 0:
            elapsed = time.time() - started
            print(
                f"stage2b_train_progress seed={args.seed} step={step}/{args.target_step} "
                f"stage={stage} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    completed_step = int(history[-1]["step"]) if history else start_step
    if tensor_digest({name: value for name, value in wrapper.named_parameters() if not value.requires_grad}) != frozen_digest:
        raise RuntimeError("Stage 2B frozen substrate changed")
    status = (
        "awaiting_step_5000_strategy_adjudication"
        if completed_step == 5_000 and stop_reason is None
        else "stopped" if stop_reason else "complete"
    )
    summary = {
        "kind": RUN_KIND,
        "status": status,
        "seed": args.seed,
        "step": completed_step,
        "target_step": args.target_step,
        "stop_reason": stop_reason,
        "history": history,
        "lock_sha256": lock_sha,
        "teacher_cache_index_sha256": sha256_file(args.teacher_cache_index),
        "schedule_hashes": schedule_hashes,
        "frozen_digest": frozen_digest,
        "last_gradient_audit": last_gradient_audit,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    for name in (
        "lock", "teacher_cache_index", "dev1_panel", "dev2_manifest",
        "reference_rows", "base_scores", "initialization_scores", "migrated",
        "p33", "i1", "p34", "p35", "model_cache", "output_dir", "private_dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("migrated_sha256", "p33_sha256", "i1_sha256", "p34_sha256", "p35_sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--target_step", type=int, default=5_000)
    parser.add_argument("--continuation_authority", type=Path)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--resume_interval", type=int, default=20)
    parser.add_argument("--eval_mcq_batch_size", type=int, default=8)
    parser.add_argument("--eval_generation_batch_size", type=int, default=2)
    parser.add_argument("--eval_margin_batch_size", type=int, default=2)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
