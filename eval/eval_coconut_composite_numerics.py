"""Bounded RG-4/RG-11 numerical follow-up for the COCONUT composite graph."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_coconut_composite_integrity import gradient_cosine, loader_args  # noqa: E402
from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from models.coconut_composite import CoconutRecurrentQwen  # noqa: E402
from training.internal_think_token_runtime import (  # noqa: E402
    install_internal_control_tokens,
    split_internal_control_token_rows,
)


EPSILONS = (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)
FIXED_PROMPTS = (
    "Follow the transition table and return the final symbol. Input: A.",
    "Apply the mapping twice and return only the resulting letter. Start: B.",
    "Trace the pointer chain until the requested step. Initial value: C.",
    "Execute the deterministic update and give the final symbol. Seed: D.",
)


def adjacent_finite_difference_pass(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for left, right in zip(rows, rows[1:]):
        if left.get("passes_original_criterion") and right.get("passes_original_criterion"):
            return {
                "passed": True,
                "adjacent_epsilons": [float(left["epsilon"]), float(right["epsilon"])],
            }
    return {"passed": False, "adjacent_epsilons": None}


def select_precision_policy(policies: dict[str, dict[str, Any]]) -> str | None:
    for name in ("fp32_master_bf16_autocast", "full_fp32"):
        if policies.get(name, {}).get("all_examples_pass") is True:
            return name
    return None


def make_prompt_batch(
    tokenizer: Any,
    *,
    prompt: str,
    latent_token_id: int,
    device: str,
) -> dict[str, torch.Tensor]:
    prefix = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])[:48]
    suffix = list(tokenizer(" Answer: A", add_special_tokens=False)["input_ids"])
    ids = prefix + [int(latent_token_id)] + suffix
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def load_composite(args: argparse.Namespace) -> tuple[Any, CoconutRecurrentQwen, int]:
    torch.manual_seed(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(loader_args(args), None)
    resize = install_internal_control_tokens(tokenizer, wrapper.base_model)
    split_internal_control_token_rows(wrapper.base_model, original_vocab_size=resize.original_vocab_size)
    latent_token_id = int(resize.control_token_ids[2])
    model = CoconutRecurrentQwen(wrapper, latent_token_id=latent_token_id).to(args.device).eval()
    return tokenizer, model, latent_token_id


def fed_gradient(
    model: CoconutRecurrentQwen,
    inputs: dict[str, torch.Tensor],
    *,
    autocast_bf16: bool,
) -> tuple[torch.Tensor, bool]:
    device_type = inputs["input_ids"].device.type
    context = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if autocast_bf16
        else nullcontext()
    )
    with context:
        output = model(**inputs, horizontal_steps=1, max_loops=1, execution_mode="recompute")
        gradient = torch.autograd.grad(output.loss, output.horizontal_fed_states[0])[0]
    finite = bool(
        torch.isfinite(output.loss)
        and torch.isfinite(gradient).all()
        and all(torch.isfinite(state).all() for state in output.horizontal_fed_states)
    )
    return gradient.detach().float().cpu(), finite


def finite_difference_sweep(
    model: CoconutRecurrentQwen,
    inputs: dict[str, torch.Tensor],
    *,
    hidden_size: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    direction = torch.randn(1, int(hidden_size), device=inputs["input_ids"].device)
    direction = direction / direction.norm()
    epsilon = torch.tensor(0.0, device=inputs["input_ids"].device, requires_grad=True)
    output = model(
        **inputs,
        horizontal_steps=1,
        max_loops=1,
        execution_mode="recompute",
        horizontal_state_additions={1: epsilon * direction},
    )
    analytic = float(torch.autograd.grad(output.loss, epsilon)[0].item())
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for value in EPSILONS:
            plus = model(
                **inputs,
                horizontal_steps=1,
                max_loops=1,
                execution_mode="recompute",
                horizontal_state_additions={1: float(value) * direction},
            ).loss
            minus = model(
                **inputs,
                horizontal_steps=1,
                max_loops=1,
                execution_mode="recompute",
                horizontal_state_additions={1: -float(value) * direction},
            ).loss
            finite = float(((plus - minus) / (2.0 * float(value))).item())
            error = abs(analytic - finite)
            tolerance = max(5e-4, 0.1 * abs(finite))
            rows.append(
                {
                    "epsilon": float(value),
                    "analytic_directional_derivative": analytic,
                    "finite_difference_directional_derivative": finite,
                    "absolute_error": error,
                    "tolerance": tolerance,
                    "sign_matches": analytic == 0.0 or finite == 0.0 or (analytic > 0) == (finite > 0),
                    "passes_original_criterion": error <= tolerance,
                }
            )
    adjacency = adjacent_finite_difference_pass(rows)
    return {
        "analytic_directional_derivative": analytic,
        "epsilons": rows,
        "adjacent_stability": adjacency,
        "passed": adjacency["passed"],
    }


def run_followup(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer, fp32_model, latent_token_id = load_composite(args)
    batches = [
        make_prompt_batch(
            tokenizer,
            prompt=prompt,
            latent_token_id=latent_token_id,
            device=args.device,
        )
        for prompt in FIXED_PROMPTS
    ]
    fd = finite_difference_sweep(
        fp32_model,
        batches[0],
        hidden_size=fp32_model.recurrent.bridge.hidden_size,
        seed=args.seed,
    )
    fp32_gradients: list[torch.Tensor] = []
    autocast_gradients: list[torch.Tensor] = []
    fp32_finite: list[bool] = []
    autocast_finite: list[bool] = []
    for batch in batches:
        gradient, finite = fed_gradient(fp32_model, batch, autocast_bf16=False)
        fp32_gradients.append(gradient)
        fp32_finite.append(finite)
        gradient, finite = fed_gradient(fp32_model, batch, autocast_bf16=True)
        autocast_gradients.append(gradient)
        autocast_finite.append(finite)

    del fp32_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer_bf16, bf16_model, latent_token_id_bf16 = load_composite(args)
    bf16_model.to(dtype=torch.bfloat16)
    bf16_gradients: list[torch.Tensor] = []
    bf16_finite: list[bool] = []
    for prompt in FIXED_PROMPTS:
        batch = make_prompt_batch(
            tokenizer_bf16,
            prompt=prompt,
            latent_token_id=latent_token_id_bf16,
            device=args.device,
        )
        gradient, finite = fed_gradient(bf16_model, batch, autocast_bf16=False)
        bf16_gradients.append(gradient)
        bf16_finite.append(finite)

    def policy_receipt(
        gradients: list[torch.Tensor],
        finite: list[bool],
    ) -> dict[str, Any]:
        examples = [
            {
                "index": index,
                "gradient_cosine_to_fp32": gradient_cosine(fp32_gradients[index], gradient),
                "finite": bool(finite[index]),
            }
            for index, gradient in enumerate(gradients)
        ]
        for row in examples:
            row["passes"] = row["finite"] and row["gradient_cosine_to_fp32"] >= 0.99
        return {
            "threshold": 0.99,
            "examples": examples,
            "minimum_cosine": min(row["gradient_cosine_to_fp32"] for row in examples),
            "all_examples_pass": all(row["passes"] for row in examples),
        }

    policies = {
        "full_fp32": policy_receipt(fp32_gradients, fp32_finite),
        "fp32_master_bf16_autocast": policy_receipt(autocast_gradients, autocast_finite),
        "full_bf16": policy_receipt(bf16_gradients, bf16_finite),
    }
    selected = select_precision_policy(policies)
    ready = bool(fd["passed"] and selected is not None)
    return {
        "kind": "coconut_composite_numerical_followup",
        "status": "engineering_preflight_passed" if ready else "engineering_preflight_needs_review",
        "model_name": args.model_name,
        "training_performed": False,
        "checkpoint_written": False,
        "execution_mode": "recompute_only",
        "sliced_cache": {
            "authorized": False,
            "disposition": "retired_after_rg5_strict_equivalence_failure",
        },
        "rg4_epsilon_stability": fd,
        "rg11_precision_policies": policies,
        "declared_future_precision_policy": selected,
        "rg12": {
            "authorized": False,
            "run": False,
            "reason": "C track remains design-stage; engineering follow-up cannot authorize training",
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
    summary = run_followup(args)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "engineering_preflight_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
