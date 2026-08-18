"""Pre-training M0 stability and identity battery for Stage 2B-D."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import (
    MODEL_SPECS,
    _chat_prompt,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
)
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition, sha256_file
from models.lora import (
    LoopScopedLoRALinear,
    apply_loop_scoped_lora_to_recurrent_block,
    set_loop_scoped_lora_index,
)
from models.paper2_stage2b_depth import Stage2BDepthAttachment
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def centered_gain(
    function: Any, state: torch.Tensor, direction: torch.Tensor, *, epsilon: float
) -> float:
    direction = direction.float() / _rms(direction).clamp_min(1e-12)
    serving_direction = direction.to(state.dtype)
    # This is a score-only finite-difference probe. Building one autograd graph
    # per direction needlessly retains full-model activations across the sweep.
    with torch.inference_mode():
        plus = function(state + float(epsilon) * serving_direction)
        minus = function(state - float(epsilon) * serving_direction)
    numerator = _rms(plus - minus)
    denominator = 2.0 * float(epsilon) * _rms(serving_direction)
    return float((numerator / denominator.clamp_min(1e-12)).detach().cpu())


def stability_verdict(receipt: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if float(receipt["identity"]["maximum_absolute_logit_difference"]) != 0.0:
        failures.append("one-pass identity changed")
    if float(receipt["identity"]["trained_adapter_maximum_absolute_difference"]) != 0.0:
        failures.append("trained loop adapter leaked into pass one")
    if not bool(receipt["m1"]["single_lane_bit_exact"]):
        failures.append("M1 failed single-lane reproduction")
    if float(receipt["routing"]["row_residual_maximum"]) > 1e-5:
        failures.append("Sinkhorn row residual exceeded 1e-5")
    if float(receipt["routing"]["column_residual_maximum"]) > 1e-5:
        failures.append("Sinkhorn column residual exceeded 1e-5")
    gains = [float(value) for value in receipt["finite_horizon"]["centered_rms_gains"]]
    if not gains or not all(torch.isfinite(torch.tensor(gains)).tolist()):
        failures.append("finite-horizon gain was non-finite")
    elif max(gains) > 100.0:
        failures.append("finite-horizon gain crossed catastrophe tripwire 100")
    return not failures, failures


def _render_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}:
        question, choices, _answer = _mcq(row)
        content = _mcq_prompt(question, choices)
    else:
        content, _cap = _generation_prompt(row)
    return _chat_prompt(tokenizer, content)


def _select_rows(rows: Sequence[Mapping[str, Any]], count: int = 8) -> list[Mapping[str, Any]]:
    selected = []
    seen = set()
    for row in rows:
        battery = str(row["battery"])
        if battery not in seen:
            selected.append(row)
            seen.add(battery)
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= count:
            break
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33", type=Path, required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--i1", type=Path, required=True)
    parser.add_argument("--i1_sha256", required=True)
    parser.add_argument("--p34", type=Path, required=True)
    parser.add_argument("--p34_sha256", required=True)
    parser.add_argument("--p35", type=Path, required=True)
    parser.add_argument("--p35_sha256", required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = read_jsonl(args.panel)
    selected = _select_rows(panel)
    model_spec = MODEL_SPECS["base"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model"], revision=model_spec["revision"], cache_dir=args.model_cache
    )
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
    sidecar.to("cuda").eval()
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=6, recurrent_end=18)
    ).to("cuda").eval()
    installed = apply_loop_scoped_lora_to_recurrent_block(
        wrapper, rank=16, alpha=16, adapter_dtype=torch.float32
    )
    if installed <= 0:
        raise RuntimeError("M0 failed to install loop-scoped attention adapters")
    recurrent_projection_dtype = next(
        module.base.weight.dtype
        for module in wrapper.modules()
        if isinstance(module, LoopScopedLoRALinear)
    )
    attachment = Stage2BDepthAttachment.from_phase3(sidecar).to("cuda").eval()
    wrapper.install_stage2b_depth_attachment(attachment)

    prompt = _render_prompt(tokenizer, selected[0])
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    with torch.inference_mode():
        set_loop_scoped_lora_index(wrapper, 0)
        direct = base(**encoded, use_cache=False, return_dict=True).logits
        attached = wrapper(
            **encoded,
            max_loops=1,
            stage2b_depth_enabled=True,
            stage2b_stage="M1",
            use_cache=False,
            return_dict=True,
        ).logits
    identity_max = float((direct.float() - attached.float()).abs().max().cpu())

    saved_b = []
    with torch.no_grad():
        for module in wrapper.modules():
            if isinstance(module, LoopScopedLoRALinear):
                saved_b.append((module, module.lora_b.weight.detach().clone()))
                module.lora_b.weight.normal_(std=0.02)
        trained_attached = wrapper(
            **encoded,
            max_loops=1,
            stage2b_depth_enabled=True,
            stage2b_stage="M4",
            use_cache=False,
            return_dict=True,
        ).logits
        for module, value in saved_b:
            module.lora_b.weight.copy_(value)
    trained_identity_max = float((direct.float() - trained_attached.float()).abs().max().cpu())

    with torch.inference_mode():
        base_output = base(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = base_output.hidden_states[-1]
        scratch = attachment.initializer(hidden, encoded["attention_mask"].bool())
        context = attachment._masked_mean(hidden.float(), encoded["attention_mask"])
        single = attachment.flow.base_flow(scratch, context, steps=3).state
        multi = attachment.flow(
            attachment.flow.replicate(scratch),
            context,
            steps=3,
            dynamic_routing=False,
            constitutive_active=False,
            forced_lane_one=True,
        ).read_state

    gains = []
    routing_rows = []
    routing_columns = []
    lambda2 = []
    ranks = []
    generator = torch.Generator(device="cuda").manual_seed(20260818)
    for row in selected:
        encoded_row = tokenizer(
            _render_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=True
        ).to("cuda")
        embeds = (
            base.get_input_embeddings()(encoded_row["input_ids"])
            .detach()
            .to(recurrent_projection_dtype)
        )
        direction = torch.randn(embeds.shape, generator=generator, device="cuda", dtype=torch.float32)
        for loops in (1, 2, 3, 4):
            def forward(candidate: torch.Tensor, depth: int = loops) -> torch.Tensor:
                output = wrapper(
                    inputs_embeds=candidate,
                    attention_mask=encoded_row["attention_mask"],
                    max_loops=depth,
                    stage2b_depth_enabled=True,
                    stage2b_stage="M3",
                    stage2b_amplitude=0.05,
                    use_cache=False,
                    return_dict=True,
                )
                return output.logits[:, -1].float()

            gains.append(centered_gain(forward, embeds, direction, epsilon=0.02))
        with torch.inference_mode():
            telemetry = wrapper(
                **encoded_row,
                max_loops=4,
                stage2b_depth_enabled=True,
                stage2b_stage="M3",
                stage2b_amplitude=0.05,
                use_cache=False,
                return_dict=True,
            ).metrics
        routing_rows.append(float(telemetry["stage2b_sinkhorn_row_residual_max"].cpu()))
        routing_columns.append(float(telemetry["stage2b_sinkhorn_column_residual_max"].cpu()))
        lambda2.append(float(telemetry["stage2b_lambda2_mean"].cpu()))
        ranks.append(float(telemetry["stage2b_lane_effective_rank_mean"].cpu()))

    receipt = {
        "kind": "paper2_stage2b_m0_stability_v1",
        "status": "pending_verdict",
        "panel_sha256": sha256_file(args.panel),
        "sample_item_ids": [str(row["item_id"]) for row in selected],
        "checkpoint_chain": chain,
        "p35_checkpoint_sha256": args.p35_sha256,
        "loop_scoped_lora_modules": installed,
        "identity": {
            "maximum_absolute_logit_difference": identity_max,
            "trained_adapter_maximum_absolute_difference": trained_identity_max,
            "requirement": "exact zero",
        },
        "m1": {
            "single_lane_bit_exact": bool(torch.equal(single, multi)),
            "maximum_absolute_state_difference": float((single.float() - multi.float()).abs().max().cpu()),
        },
        "finite_horizon": {
            "estimator": "centered finite difference, RMS output delta over RMS input delta",
            "epsilon": 0.02,
            "centered_rms_gains": gains,
            "catastrophe_tripwire": 100.0,
        },
        "routing": {
            "row_residual_maximum": max(routing_rows),
            "column_residual_maximum": max(routing_columns),
            "lambda2_mean": sum(lambda2) / len(lambda2),
            "lane_effective_rank_mean": sum(ranks) / len(ranks),
        },
        "runtime": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "attention_backend": "sdpa",
            "weights_dtype": "bfloat16",
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    passed, failures = stability_verdict(receipt)
    receipt["passed"] = passed
    receipt["failures"] = failures
    receipt["status"] = "passed" if passed else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"M0 stability battery failed: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
