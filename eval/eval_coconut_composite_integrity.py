"""Run the no-training RG-1 through RG-11 composite integrity battery."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from models.coconut_composite import (  # noqa: E402
    CoconutRecurrentQwen,
    assert_parameter_group_coverage,
    configure_composite_trainable_set,
)
from models.lora import LoRALinear, apply_lora_to_recurrent_block  # noqa: E402
from training.internal_think_token_runtime import (  # noqa: E402
    install_internal_control_tokens,
    split_internal_control_token_rows,
)
from training.run_internal_think_token_t1_lite import DeviceEMA  # noqa: E402


def loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        checkpoint=None,
        split=args.split,
        bridge_projection_mode="split",
        dtype="float32",
        attn_implementation="default",
        device=args.device,
        lora_rank=0,
        lora_alpha=16,
        adapter_dtype="float32",
        base_lora_layer_range="all",
    )


def make_batch(
    tokenizer: Any,
    *,
    latent_token_id: int,
    horizontal_steps: int,
    device: str,
) -> dict[str, torch.Tensor]:
    prefix = list(
        tokenizer(
            "Follow the transition table and return the final symbol. Input: A.",
            add_special_tokens=True,
        )["input_ids"]
    )[:24]
    suffix = list(tokenizer(" Answer: A", add_special_tokens=False)["input_ids"])
    if not suffix:
        raise AssertionError("The integrity prompt produced no answer token")
    ids = prefix + [int(latent_token_id)] * int(horizontal_steps) + suffix
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.detach().float().flatten(),
            right.detach().float().flatten(),
            dim=0,
        ).item()
    )


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def clear_gradients(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def optimizer_parameter_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> set[str]:
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names: set[str] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in by_id:
                raise AssertionError("optimizer contains a parameter outside the composite model")
            names.add(by_id[id(parameter)])
    return names


def freeze_for_activation_audit(model: CoconutRecurrentQwen) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.horizontal_bridge.delta.weight.requires_grad_(True)


def horizontal_delta_gradient(
    model: CoconutRecurrentQwen,
    inputs: dict[str, torch.Tensor],
    *,
    mode: str = "recompute",
    checkpointing: bool = False,
) -> tuple[Any, torch.Tensor]:
    model.train()
    model.recurrent.qwen.gradient_checkpointing = bool(checkpointing)
    output = model(
        **inputs,
        horizontal_steps=1,
        max_loops=1,
        execution_mode=mode,
    )
    gradient = torch.autograd.grad(output.loss, model.horizontal_bridge.delta.weight)[0]
    model.recurrent.qwen.gradient_checkpointing = False
    return output, gradient


def write_receipt(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# COCONUT Composite Integrity Receipt",
        "",
        f"- Status: `{summary['status']}`",
        f"- Model: `{summary['model_name']}`",
        f"- Training performed: `{summary['training_performed']}`",
        f"- RG-12 authorized or run: `{summary['rg12']['run']}`",
        "",
        "| Check | Passed | Key observation |",
        "|---|---:|---|",
    ]
    for name, result in summary["contracts"].items():
        observation = result.get("observation", "")
        lines.append(f"| {name.upper()} | `{result['passed']}` | {observation} |")
    lines.extend(
        [
            "",
            "The reference recompute path is the authorized future training path. ",
            "Sliced cache is limited to L=1 and is forbidden with active gradient checkpointing.",
            "RG-12 remains unrun pending the T1-lite verdict and a locked null-calibrated KL floor.",
            "",
        ]
    )
    (output_dir / "receipt.md").write_text("\n".join(lines), encoding="utf-8")


def run_battery(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(loader_args(args), None)
    resize = install_internal_control_tokens(tokenizer, wrapper.base_model)
    split_rows = split_internal_control_token_rows(
        wrapper.base_model,
        original_vocab_size=resize.original_vocab_size,
    )
    latent_token_id = int(resize.control_token_ids[2])
    model = CoconutRecurrentQwen(wrapper, latent_token_id=latent_token_id).to(args.device)
    model.eval()
    contracts: dict[str, dict[str, Any]] = {}

    h0 = make_batch(tokenizer, latent_token_id=latent_token_id, horizontal_steps=0, device=args.device)
    h1 = make_batch(tokenizer, latent_token_id=latent_token_id, horizontal_steps=1, device=args.device)
    h2 = make_batch(tokenizer, latent_token_id=latent_token_id, horizontal_steps=2, device=args.device)

    with torch.no_grad():
        identity_diffs: dict[str, float] = {}
        for loops in (1, 2):
            direct = wrapper(**h0, max_loops=loops, use_cache=False, return_dict=True)
            composite = model(**h0, horizontal_steps=0, max_loops=loops)
            identity_diffs[f"full_block_l{loops}"] = maximum_difference(direct.logits, composite.logits)
    contracts["rg1"] = {
        "passed": max(identity_diffs.values()) < 1e-3,
        "maximum_logit_difference": identity_diffs,
        "observation": f"full-block H=0 max diff {max(identity_diffs.values()):.3g}",
    }

    freeze_for_activation_audit(model)
    bridge_output = model(**h1, horizontal_steps=1, max_loops=1)
    raw_output = model(**h1, horizontal_steps=1, max_loops=1, raw_feedback=True)
    rg2_difference = maximum_difference(bridge_output.logits, raw_output.logits)
    contracts["rg2"] = {
        "passed": rg2_difference == 0.0,
        "maximum_logit_difference": rg2_difference,
        "observation": f"identity bridge versus raw feedback max diff {rg2_difference:.3g}",
    }

    reachability = model(**h1, horizontal_steps=1, max_loops=1)
    fed_gradient, embedding_gradient = torch.autograd.grad(
        reachability.loss,
        (reachability.horizontal_fed_states[0], reachability.input_embeddings),
    )
    fed_norm = float(fed_gradient.float().norm().item())
    first_latent_position = int(
        h1["input_ids"].eq(latent_token_id).nonzero(as_tuple=False)[0, 1].item()
    )
    prompt_norm = float(
        embedding_gradient[:, :first_latent_position].float().norm().item()
    )
    finite_reachability = bool(torch.isfinite(fed_gradient).all() and torch.isfinite(embedding_gradient).all())
    contracts["rg3"] = {
        "passed": finite_reachability and fed_norm > 0.0 and prompt_norm > 0.0,
        "fed_state_gradient_norm": fed_norm,
        "prompt_activation_gradient_norm": prompt_norm,
        "observation": f"fed={fed_norm:.3e}, prompt={prompt_norm:.3e}",
    }

    torch.manual_seed(args.seed)
    direction = torch.randn(1, wrapper.bridge.hidden_size, device=args.device)
    direction = direction / direction.norm()
    epsilon = torch.tensor(0.0, device=args.device, requires_grad=True)
    analytic_output = model(
        **h1,
        horizontal_steps=1,
        max_loops=1,
        horizontal_state_additions={1: epsilon * direction},
    )
    analytic = float(torch.autograd.grad(analytic_output.loss, epsilon)[0].item())
    finite_step = 1e-3
    with torch.no_grad():
        plus = model(
            **h1,
            horizontal_steps=1,
            max_loops=1,
            horizontal_state_additions={1: finite_step * direction},
        ).loss
        minus = model(
            **h1,
            horizontal_steps=1,
            max_loops=1,
            horizontal_state_additions={1: -finite_step * direction},
        ).loss
    finite_difference = float(((plus - minus) / (2.0 * finite_step)).item())
    fd_error = abs(analytic - finite_difference)
    fd_tolerance = max(5e-4, 0.1 * abs(finite_difference))
    contracts["rg4"] = {
        "passed": fd_error <= fd_tolerance,
        "analytic_directional_derivative": analytic,
        "finite_difference_directional_derivative": finite_difference,
        "absolute_error": fd_error,
        "tolerance": fd_tolerance,
        "observation": f"directional derivative abs error {fd_error:.3e}",
    }

    recompute, recompute_gradient = horizontal_delta_gradient(model, h1, mode="recompute")
    cached, cached_gradient = horizontal_delta_gradient(model, h1, mode="sliced_cache")
    cache_logit_difference = maximum_difference(recompute.logits, cached.logits)
    cache_gradient_difference = maximum_difference(recompute_gradient, cached_gradient)
    cache_gradient_cosine = gradient_cosine(recompute_gradient, cached_gradient)
    contracts["rg5"] = {
        "passed": cache_logit_difference <= 2e-5
        and cache_gradient_difference <= 2e-5
        and cache_gradient_cosine >= 0.9999,
        "logit_max_abs_difference": cache_logit_difference,
        "probe_gradient_max_abs_difference": cache_gradient_difference,
        "probe_gradient_cosine": cache_gradient_cosine,
        "cache_prefix_lengths": list(cached.cache_prefix_lengths),
        "observation": f"cache/recompute grad cosine {cache_gradient_cosine:.6f}",
    }

    plain, plain_gradient = horizontal_delta_gradient(model, h1, checkpointing=False)
    checkpointed, checkpointed_gradient = horizontal_delta_gradient(model, h1, checkpointing=True)
    checkpoint_logit_difference = maximum_difference(plain.logits, checkpointed.logits)
    checkpoint_gradient_cosine = gradient_cosine(plain_gradient, checkpointed_gradient)
    contracts["rg10"] = {
        "passed": checkpoint_logit_difference <= 2e-5 and checkpoint_gradient_cosine >= 0.9999,
        "logit_max_abs_difference": checkpoint_logit_difference,
        "probe_gradient_cosine": checkpoint_gradient_cosine,
        "observation": f"checkpoint gradient cosine {checkpoint_gradient_cosine:.6f}",
    }

    model.eval()
    with torch.autograd.detect_anomaly():
        anomaly_output = model(**h1, horizontal_steps=1, max_loops=1)
        anomaly_gradient = torch.autograd.grad(
            anomaly_output.loss,
            model.horizontal_bridge.delta.weight,
        )[0]
    contracts["rg9"] = {
        "passed": bool(torch.isfinite(anomaly_output.loss) and torch.isfinite(anomaly_gradient).all()),
        "observation": "one full forward/backward completed under detect_anomaly",
    }

    fp32_output = model(**h1, horizontal_steps=1, max_loops=1)
    fp32_fed_gradient = torch.autograd.grad(
        fp32_output.loss,
        fp32_output.horizontal_fed_states[0],
    )[0]
    model.to(dtype=torch.bfloat16)
    bf16_output = model(**h1, horizontal_steps=1, max_loops=1)
    bf16_fed_gradient = torch.autograd.grad(
        bf16_output.loss,
        bf16_output.horizontal_fed_states[0],
    )[0]
    precision_cosine = gradient_cosine(fp32_fed_gradient, bf16_fed_gradient)
    boundary_finite = all(torch.isfinite(state).all() for state in bf16_output.horizontal_fed_states)
    contracts["rg11"] = {
        "passed": boundary_finite and precision_cosine >= 0.99,
        "fed_state_gradient_cosine": precision_cosine,
        "all_feedback_boundaries_finite": boundary_finite,
        "observation": f"bf16/fp32 fed-gradient cosine {precision_cosine:.6f}",
    }

    replaced = apply_lora_to_recurrent_block(
        wrapper,
        rank=16,
        alpha=16,
        adapter_dtype=torch.float32,
    )
    trainable_names = configure_composite_trainable_set(
        model,
        budget="adapter_r16",
        horizontal_bridge_trainable=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
    )
    ema = DeviceEMA(model.named_parameters(), decay=0.999)
    coverage = assert_parameter_group_coverage(
        trainable_names,
        optimizer_parameter_names(model, optimizer),
        ema.shadow.keys(),
    )
    with torch.no_grad():
        direct_adapter = wrapper(**h0, max_loops=2, use_cache=False, return_dict=True)
        composite_adapter = model(**h0, horizontal_steps=0, max_loops=2)
    adapter_identity_difference = maximum_difference(direct_adapter.logits, composite_adapter.logits)
    contracts["rg1"]["adapter_l2_maximum_logit_difference"] = adapter_identity_difference
    contracts["rg1"]["passed"] = contracts["rg1"]["passed"] and adapter_identity_difference < 1e-3

    clear_gradients(model)
    adapter_output = model(**h1, horizontal_steps=1, max_loops=1)
    adapter_output.loss.backward()
    frozen_base_grads = [
        parameter.grad
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in module.base.parameters()
    ]
    lora_grad_norms = [
        float(parameter.grad.float().norm().item())
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in module.lora_parameters()
        if parameter.grad is not None
    ]
    fed_live = all(
        state.grad is not None and float(state.grad.float().norm().item()) > 0.0
        for state in adapter_output.horizontal_fed_states
    )
    contracts["rg6"] = {
        "passed": all(gradient is None or int(torch.count_nonzero(gradient)) == 0 for gradient in frozen_base_grads)
        and any(norm > 0.0 for norm in lora_grad_norms)
        and fed_live,
        "lora_module_count": replaced,
        "maximum_lora_gradient_norm": max(lora_grad_norms, default=0.0),
        "earlier_feedback_activation_live": fed_live,
        "observation": f"{replaced} LoRA modules; frozen base transparent",
    }

    clear_gradients(model)
    forward_calls = 0
    backward_calls = 0

    def count_forward(_module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
        nonlocal forward_calls
        forward_calls += 1

    def count_backward(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal backward_calls
        backward_calls += 1
        return gradient

    representative_layer = wrapper.qwen.layers[wrapper.layer_split.prelude_end]
    forward_handle = representative_layer.register_forward_hook(count_forward)
    grid_output = model(**h2, horizontal_steps=2, max_loops=1)
    forward_handle.remove()
    grid_states = tuple(
        state for column in grid_output.recurrent_application_states for state in column
    )
    for state in grid_states:
        state.retain_grad()
    backward_handles = [state.register_hook(count_backward) for state in grid_states]
    grid_output.loss.backward()
    for handle in backward_handles:
        handle.remove()
    grid_norms = [
        0.0 if state.grad is None else float(state.grad.float().norm().item())
        for state in grid_states
    ]
    contracts["rg7"] = {
        "passed": grid_output.feedback_grid_applications == 2
        and grid_output.total_grid_applications == 3
        and forward_calls == 3
        and backward_calls == 3
        and all(norm > 0.0 for norm in grid_norms),
        "feedback_producing_applications": grid_output.feedback_grid_applications,
        "total_recurrent_block_applications": grid_output.total_grid_applications,
        "independent_forward_hook_count": forward_calls,
        "independent_backward_hook_count": backward_calls,
        "gradient_norms": grid_norms,
        "observation": "H*L=2 feedback cells; (H+1)*L=3 total cells",
    }

    input_gradient = grid_output.input_embeddings.grad
    if input_gradient is None:
        raise AssertionError("input activation gradient was not retained")
    latent_mask = h2["input_ids"].eq(latent_token_id).unsqueeze(-1).expand_as(input_gradient)
    latent_input_gradient = input_gradient.masked_select(latent_mask)
    contracts["rg8"] = {
        "passed": int(torch.count_nonzero(latent_input_gradient)) == 0 and coverage["passed"],
        "replaced_input_activation_nonzero_count": int(torch.count_nonzero(latent_input_gradient)),
        "parameter_group_coverage": coverage,
        "observation": "replaced input-slot gradient zero; parameter-name hashes exact",
    }

    all_passed = all(bool(result["passed"]) for result in contracts.values())
    return {
        "kind": "coconut_composite_rg1_rg11_integrity",
        "status": "passed_rg1_through_rg11" if all_passed else "failed_integrity_contract",
        "training_performed": False,
        "checkpoint_written": False,
        "model_name": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "latent_token": "<|recur_readout|>",
        "latent_token_id": latent_token_id,
        "control_token_resize": resize.to_dict(),
        "split_control_rows": split_rows.to_dict(),
        "contracts": contracts,
        "rg12": {
            "run": False,
            "authorized": False,
            "horizontal_bridge_frozen_identity_required": True,
            "kl_floor": None,
            "reason": "waits for T1-lite verdict and null-calibrated corruption design",
        },
        "execution_contract": {
            "reference_training_mode": "recompute",
            "sliced_cache_max_vertical_loops": 1,
            "sliced_cache_with_gradient_checkpointing": "forbidden",
            "first_integration_pilot_vertical_loops": 1,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_battery(args)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_receipt(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "passed_rg1_through_rg11" else 2


if __name__ == "__main__":
    raise SystemExit(main())
