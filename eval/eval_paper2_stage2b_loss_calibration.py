"""No-optimizer Stage 2B gradient-share calibration on the sparse estimator."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from models.lora import apply_loop_scoped_lora_to_recurrent_block
from models.paper2_stage2b_depth import Stage2BDepthAttachment
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from training.paper2_stage2b_depth import (
    DepthObjectiveWeights,
    calibrated_gradient_share_weights,
    configure_stage2b_trainable_groups,
    depth_objective,
)


def _accumulate(target: list[torch.Tensor], gradients: tuple[torch.Tensor | None, ...]) -> None:
    for index, gradient in enumerate(gradients):
        if gradient is not None:
            target[index].add_(gradient.detach().float().cpu())


def _vector_norm(values: list[torch.Tensor], denominator: int) -> float:
    total = sum(float((value / denominator).double().square().sum()) for value in values)
    return total**0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated-sha256", required=True)
    parser.add_argument("--p33", type=Path, required=True)
    parser.add_argument("--p33-sha256", required=True)
    parser.add_argument("--i1", type=Path, required=True)
    parser.add_argument("--i1-sha256", required=True)
    parser.add_argument("--p34", type=Path, required=True)
    parser.add_argument("--p34-sha256", required=True)
    parser.add_argument("--p35", type=Path, required=True)
    parser.add_argument("--p35-sha256", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    teacher = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    if teacher.get("kind") != "paper2_stage2b_calibration_teacher_cache_v1":
        raise RuntimeError("wrong Stage 2B teacher calibration cache")
    model_spec = MODEL_SPECS["base"]
    base = AutoModelForCausalLM.from_pretrained(
        model_spec["model"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
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
    ).to("cuda")
    apply_loop_scoped_lora_to_recurrent_block(
        wrapper, rank=16, alpha=16, adapter_dtype=torch.float32
    )
    wrapper.install_stage2b_depth_attachment(Stage2BDepthAttachment.from_phase3(sidecar).to("cuda"))
    groups = configure_stage2b_trainable_groups(wrapper)
    # Calibration needs gradients but must retain deterministic serving behavior.
    wrapper.eval()
    parameters = [parameter for values in groups.values() for parameter in values]
    accumulators = {
        name: [torch.zeros_like(parameter, dtype=torch.float32, device="cpu") for parameter in parameters]
        for name in ("ce", "kl", "monotonicity")
    }
    component_values = {name: [] for name in accumulators}
    for index, row in enumerate(teacher["rows"]):
        input_ids = row["input_ids"].long().unsqueeze(0).to("cuda")
        attention = torch.ones_like(input_ids)
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M4",
            stage2b_amplitude=0.05,
            return_loop_logits=True,
            use_cache=False,
            return_dict=True,
        )
        if output.loop_logits is None:
            raise RuntimeError("Stage 2B calibration requires per-loop logits")
        loops = output.loop_logits[:, 0]
        loop_logits = [loops[:, loop, :-1] for loop in range(4)]
        top_ids = row["teacher_topk_token_ids"].long().unsqueeze(0).to("cuda")
        top_logits = row["teacher_topk_logits"].unsqueeze(0).to("cuda")
        targets = top_ids[..., 0]
        mask = torch.ones(targets.shape, dtype=torch.bool, device="cuda")
        _total, components = depth_objective(
            loop_logits=loop_logits,
            teacher_topk_token_ids=top_ids,
            teacher_topk_logits=top_logits,
            teacher_tokens=targets,
            loss_mask=mask,
            weights=DepthObjectiveWeights(ce=0.3, kl=0.5, monotonicity=0.2),
            hinge_delta=0.01,
        )
        for component_index, name in enumerate(("ce", "kl", "monotonicity")):
            gradients = torch.autograd.grad(
                components[name],
                parameters,
                retain_graph=component_index < 2,
                allow_unused=True,
            )
            _accumulate(accumulators[name], gradients)
            component_values[name].append(float(components[name].detach().cpu()))
        print(f"stage2b_loss_calibration_progress seed={args.seed} row={index + 1}/{len(teacher['rows'])}", flush=True)
        del output, loops, loop_logits, components

    raw_norms = {
        name: _vector_norm(values, len(teacher["rows"])) for name, values in accumulators.items()
    }
    weights = calibrated_gradient_share_weights(raw_norms)
    weighted = {
        "ce": weights.ce * raw_norms["ce"],
        "kl": weights.kl * raw_norms["kl"],
        "monotonicity": weights.monotonicity * raw_norms["monotonicity"],
    }
    weighted_total = sum(weighted.values())
    receipt: dict[str, Any] = {
        "kind": "paper2_stage2b_loss_calibration_v1",
        "status": "complete_no_optimizer",
        "seed": args.seed,
        "teacher_cache_manifest_sha256": teacher["manifest_sha256"],
        "rows": len(teacher["rows"]),
        "next_token_positions": sum(row["teacher_topk_token_ids"].shape[0] for row in teacher["rows"]),
        "estimator": "mean full-sequence per-example gradient vector over all M4 trainable parameters",
        "raw_gradient_norms": raw_norms,
        "weights": {
            "ce": weights.ce,
            "kl": weights.kl,
            "monotonicity": weights.monotonicity,
            "verified_depth": 0.0,
        },
        "realized_independent_gradient_shares": {
            name: value / weighted_total for name, value in weighted.items()
        },
        "component_means": {
            name: sum(values) / len(values) for name, values in component_values.items()
        },
        "parameter_counts": {
            name: sum(parameter.numel() for parameter in values) for name, values in groups.items()
        },
        "checkpoint_chain": chain,
        "p35_checkpoint_sha256": args.p35_sha256,
        "runtime": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "weights_dtype": "bfloat16",
            "attention_backend": "sdpa",
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
