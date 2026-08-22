"""Trace the P3.5 and Stage 2B serving graphs from one immutable initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS, _chat_prompt, _generation_prompt
from eval.eval_paper2_phase3_p34_task_inference import (
    P34TaskInferenceGraph,
    current_position_mask,
    position_buckets,
)
from eval.eval_paper2_stage2b_autopsy import _build_model, _named_trainable_state, _state_digest
from eval.eval_paper2_stage2b_campaign import Stage2BTaskInferenceGraph, read_jsonl


RUN_KIND = "paper2_stage2bs_reconciliation_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def tensor_comparison(left: torch.Tensor, right: torch.Tensor, *, rank_k: int = 32) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "comparable": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "reason": "shape_mismatch",
        }
    a = left.detach().float().reshape(-1).cpu()
    b = right.detach().float().reshape(-1).cpu()
    maximum = float((a - b).abs().max()) if a.numel() else 0.0
    denominator = float(a.norm() * b.norm())
    cosine = float(torch.dot(a, b) / denominator) if denominator else float(a.equal(b))
    k = min(int(rank_k), int(a.numel()))
    if k:
        left_top = set(torch.topk(a.abs(), k).indices.tolist())
        right_top = set(torch.topk(b.abs(), k).indices.tolist())
        rank_agreement = len(left_top & right_top) / k
    else:
        rank_agreement = 1.0
    return {
        "comparable": True,
        "shape": list(left.shape),
        "bit_exact": bool(torch.equal(left, right)),
        "max_abs_delta": maximum,
        "cosine": cosine,
        "top_abs_coordinate_agreement_at_k": rank_agreement,
        "rank_k": k,
    }


def logit_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    result = tensor_comparison(left, right, rank_k=128)
    if result["comparable"]:
        left_ids = torch.topk(left.detach().float().reshape(-1), min(128, left.numel())).indices
        right_ids = torch.topk(right.detach().float().reshape(-1), min(128, right.numel())).indices
        result.update(
            {
                "top1_equal": bool(left_ids[0] == right_ids[0]),
                "top128_token_set_agreement": len(set(left_ids.tolist()) & set(right_ids.tolist()))
                / len(left_ids),
                "left_top1": int(left_ids[0]),
                "right_top1": int(right_ids[0]),
            }
        )
    return result


@torch.inference_mode()
def trace_p35_graph(
    *, wrapper: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reproduce P34TaskInferenceGraph._augment while retaining every state."""

    wrapper._set_loop_scoped_adapter_index(0)
    output = wrapper.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.hidden_states[-1]
    positions = current_position_mask(attention_mask)[1]
    batch = torch.arange(input_ids.shape[0], device=input_ids.device)
    attachment = wrapper.stage2b_depth_attachment
    scratch = attachment.initializer(hidden, attention_mask.bool())
    context = hidden.float().mean(dim=1)
    trace: dict[str, torch.Tensor] = {
        "tokenized_inputs": _cpu(input_ids),
        "prefix_output": _cpu(output.hidden_states[6]),
        "full_base_hidden": _cpu(hidden),
        "initializer_scratch": _cpu(scratch),
        "context": _cpu(context),
        "base_next_token_logits": _cpu(output.logits[batch, positions]),
    }
    current = scratch
    updates = []
    for index in range(4):
        trace[f"loop_{index + 1}_pre_state"] = _cpu(current)
        current, update, _magnitude, _ratio = attachment.flow.base_flow.step(
            current, context, index
        )
        updates.append(update)
        trace[f"loop_{index + 1}_post_state"] = _cpu(current)
        trace[f"loop_{index + 1}_flow_update"] = _cpu(update)
    innovation_norm = updates[-1].float().square().mean(dim=-1).sqrt().mean(dim=1)
    control = attachment.control(
        scratch=current,
        previous=None,
        innovation_norm=innovation_norm,
        student_entropy=hidden.new_zeros((hidden.shape[0],)),
        top2_margin=hidden.new_zeros((hidden.shape[0],)),
        position_bucket=position_buckets(positions),
    )
    token_hidden = hidden[batch, positions]
    compact_hidden = torch.stack([torch.zeros_like(token_hidden), token_hidden], dim=1)
    compact_mask = torch.zeros((hidden.shape[0], 2, 1), dtype=torch.bool, device=hidden.device)
    compact_mask[:, 1] = True
    attachment.bridge.set_gate_ceiling(0.05)
    bridge = attachment.bridge(
        h0=compact_hidden,
        previous=compact_hidden,
        scratch=current,
        control_state=control,
        loop_index=3,
        active=True,
        write_position_mask=compact_mask,
    )
    writeback = bridge.position_gate * bridge.delta
    head_input = bridge.hidden[:, 1]
    head = wrapper.base_model.get_output_embeddings()
    logits = head(head_input.to(head.weight.dtype))
    trace.update(
        {
            "loop_4_uncapped_write": _cpu(bridge.delta[:, 1]),
            "loop_4_capped_write": _cpu(writeback[:, 1]),
            "loop_4_position_gate": _cpu(bridge.position_gate[:, 1]),
            "loop_4_suffix_head_input": _cpu(head_input),
            "final_next_token_logits": _cpu(logits),
        }
    )
    return trace, logits


@torch.inference_mode()
def trace_stage2b_graph(
    *, wrapper: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    attachment = wrapper.stage2b_depth_attachment
    trace: dict[str, torch.Tensor] = {
        "tokenized_inputs": _cpu(input_ids),
    }
    bridge_calls: list[dict[str, torch.Tensor]] = []
    original_begin = attachment.begin
    original_observe = attachment.observe
    original_reenter = attachment.reenter

    def begin(**kwargs: Any):
        trace["prefix_output"] = _cpu(kwargs["prelude_hidden"])
        return original_begin(**kwargs)

    def observe(**kwargs: Any):
        result = original_observe(**kwargs)
        recurrent_pass = int(kwargs["loop_index"])
        if recurrent_pass == 0:
            trace["full_base_hidden"] = _cpu(kwargs["coda_hidden"])
            trace["initializer_scratch"] = _cpu(result.lane_state[:, 0])
            trace["base_suffix_head_input"] = _cpu(kwargs["coda_hidden"])
        else:
            # Re-entry i feeds recurrent pass i+1. Name the resulting suffix
            # state by the sidecar update index so it aligns with P3.5 flow i.
            trace[f"loop_{recurrent_pass}_suffix_head_input"] = _cpu(
                kwargs["coda_hidden"]
            )
        return result

    def bridge_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any], output: Any) -> None:
        bridge_calls.append(
            {
                "uncapped": _cpu(output.delta),
                "capped": _cpu(output.position_gate * output.delta),
                "gate": _cpu(output.position_gate),
                "hidden": _cpu(output.hidden),
                "h0": _cpu(kwargs["h0"]),
                "previous": _cpu(kwargs["previous"]),
            }
        )

    def reenter(**kwargs: Any):
        loop = int(kwargs["loop_index"])
        lane_pre = kwargs["trace"].lane_state
        trace[f"loop_{loop}_pre_state"] = _cpu(lane_pre[:, 0])
        trace[f"loop_{loop}_context"] = _cpu(
            attachment._masked_mean(kwargs["recurrent_hidden"].float(), kwargs["attention_mask"])
        )
        result = original_reenter(**kwargs)
        step = result.trace.routing_steps[-1]
        trace[f"loop_{loop}_post_state"] = _cpu(step.read_state)
        trace[f"loop_{loop}_flow_update"] = _cpu(step.flow_update[:, 0])
        trace[f"loop_{loop}_constitutive_update"] = _cpu(step.constitutive_update[:, 0])
        call = bridge_calls[-1]
        trace[f"loop_{loop}_uncapped_write"] = call["uncapped"][:, -1]
        trace[f"loop_{loop}_capped_write"] = call["capped"][:, -1]
        trace[f"loop_{loop}_position_gate"] = call["gate"][:, -1]
        trace[f"loop_{loop}_reentry_hidden"] = call["hidden"][:, -1]
        return result

    attachment.begin = begin
    attachment.observe = observe
    attachment.reenter = reenter
    hook = attachment.bridge.register_forward_hook(bridge_hook, with_kwargs=True)
    try:
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.05,
            stage2b_score_only_sparse_logits=True,
            return_loop_logits=True,
            use_cache=False,
            return_dict=True,
        )
    finally:
        hook.remove()
        attachment.begin = original_begin
        attachment.observe = original_observe
        attachment.reenter = original_reenter
    positions = current_position_mask(attention_mask)[1]
    batch = torch.arange(input_ids.shape[0], device=input_ids.device)
    loops = output.loop_logits[:, 0]
    selected = loops[batch, -1, positions]
    trace["base_next_token_logits"] = _cpu(loops[batch, 0, positions])
    trace["final_next_token_logits"] = _cpu(selected)
    return trace, selected


def _stage_table(p35: Mapping[str, torch.Tensor], stage2b: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    ordered = ["tokenized_inputs", "prefix_output", "full_base_hidden", "initializer_scratch"]
    for loop in range(1, 5):
        ordered.extend(
            [
                f"loop_{loop}_pre_state",
                f"loop_{loop}_post_state",
                f"loop_{loop}_uncapped_write",
                f"loop_{loop}_capped_write",
                f"loop_{loop}_suffix_head_input",
            ]
        )
    ordered.append("final_next_token_logits")
    rows = []
    for index, name in enumerate(ordered):
        left = p35.get(name)
        right = stage2b.get(name)
        if left is None or right is None:
            comparison = {
                "comparable": False,
                "reason": "operation_absent_in_" + ("p35" if left is None else "stage2b"),
            }
        elif name.endswith("logits"):
            comparison = logit_comparison(left, right)
        else:
            comparison = tensor_comparison(left, right)
        rows.append({"order": index, "stage": name, **comparison})
    return rows


def _first_divergence(table: list[Mapping[str, Any]]) -> dict[str, Any]:
    for row in table:
        if not row.get("comparable") or not row.get("bit_exact", False):
            stage = str(row["stage"])
            if stage in {"tokenized_inputs", "prefix_output", "full_base_hidden", "initializer_scratch"}:
                category = "prompt_or_prefix_semantics"
            elif "write" in stage:
                category = "bridge_or_amplitude_semantics"
            else:
                category = "effective_k_and_one_shot_vs_full_recurrent_iteration"
            return {"stage": stage, "classification": category, "comparison": dict(row)}
    return {"stage": None, "classification": "no_divergence", "comparison": {}}


def _source_provenance(root: Path) -> dict[str, Any]:
    training_path = root / "training/run_paper2_stage2b_depth.py"
    serving_path = root / "eval/eval_paper2_stage2b_campaign.py"
    p35_path = root / "eval/eval_paper2_phase3_p34_task_inference.py"
    training = training_path.read_text(encoding="utf-8")
    serving = serving_path.read_text(encoding="utf-8")
    p35 = p35_path.read_text(encoding="utf-8")
    checks = {
        "training_uses_stage2b_wrapper": all(
            marker in training
            for marker in ("stage2b_depth_enabled=True", "depth_objective(", "Stage2BTaskInferenceGraph(")
        ),
        "dev_floors_use_stage2b_graph": "graph = Stage2BTaskInferenceGraph(" in training,
        "stage2b_serving_runs_recurrent_wrapper": "max_loops=self.flow_loops" in serving,
        "p35_serving_runs_flow_then_one_bridge": (
            "self.sidecar.flow(scratch0, context, steps=self.flow_loops)" in p35
            and "bridge = self.sidecar.bridge(" in p35
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"source provenance contract changed: {checks}")
    return {
        "checks": checks,
        "files": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (training_path, serving_path, p35_path)
        },
        "success_defining_graph": "Stage2BTaskInferenceGraph_over_RecurrentQwenForCausalLM",
        "historical_p35_graph": "P34TaskInferenceGraph_one_base_forward_four_sidecar_steps_one_bridge_write",
        "same_graph": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("kind") != "paper2_stage2bs_reconciliation_lock_v1" or not lock.get("locked_before_trace"):
        raise RuntimeError("Stage 2B-S reconciliation lacks the locked identity contract")
    if any(term in str(value).casefold() for value in vars(args).values() for term in ("confirm", "eval_e")):
        raise RuntimeError("sealed partition contact")
    wrapper, checkpoint_chain, _groups = _build_model(args)
    before = _state_digest(_named_trainable_state(wrapper))
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS["base"]["model"], revision=MODEL_SPECS["base"]["revision"]
    )
    panel = read_jsonl(args.dev1_panel)
    generation = [row for row in panel if str(row["battery"]) in {"tier1", "gsm8k", "mbpp"}]
    row = next((row for row in generation if str(row["battery"]) == "gsm8k"), generation[0])
    prompt = _chat_prompt(tokenizer, _generation_prompt(row)[0])
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
    p35_trace, p35_logits = trace_p35_graph(
        wrapper=wrapper, input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    )
    sidecar = SimpleNamespace(
        initializer=wrapper.stage2b_depth_attachment.initializer,
        flow=wrapper.stage2b_depth_attachment.flow.base_flow,
        control=wrapper.stage2b_depth_attachment.control,
        bridge=wrapper.stage2b_depth_attachment.bridge,
    )
    p35_graph = P34TaskInferenceGraph(base_model=wrapper.base_model, sidecar=sidecar, flow_loops=4)
    p35_production = p35_graph.next_token(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    ).augmented_logits
    p35_trace_validation = logit_comparison(_cpu(p35_logits), _cpu(p35_production))
    if not p35_trace_validation["bit_exact"]:
        raise RuntimeError("manual P3.5 trace changed production evaluator output")

    stage2b_trace, stage2b_logits = trace_stage2b_graph(
        wrapper=wrapper, input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    )
    stage2b_graph = Stage2BTaskInferenceGraph(
        wrapper=wrapper, stage="M2", amplitude=0.05, flow_loops=4
    )
    stage2b_production = stage2b_graph.next_token(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    ).augmented_logits
    stage2b_trace_validation = logit_comparison(_cpu(stage2b_logits), _cpu(stage2b_production))
    if not stage2b_trace_validation["bit_exact"]:
        raise RuntimeError("manual Stage 2B trace changed production evaluator output")
    after = _state_digest(_named_trainable_state(wrapper))
    if before != after:
        raise RuntimeError("score-only reconciliation mutated the initialization state")

    table = _stage_table(p35_trace, stage2b_trace)
    first = _first_divergence(table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.private_dir / f"seed_{args.seed}_paired_trace_tensors.pt"
    torch.save({"p35": p35_trace, "stage2b": stage2b_trace}, tensor_path)
    table_path = args.output_dir / f"seed_{args.seed}_stage_table.json"
    atomic_json(table_path, {"seed": args.seed, "rows": table})
    source = _source_provenance(args.repo_root)
    result = {
        "kind": RUN_KIND,
        "status": "complete_score_only",
        "seed": args.seed,
        "authority": lock["authority"],
        "lock_sha256": sha256_file(args.lock),
        "runtime": {
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "weights_dtype": "bfloat16",
            "attention_backend": "sdpa",
        },
        "checkpoint_chain": checkpoint_chain,
        "initialization_state_digest_before": before,
        "initialization_state_digest_after": after,
        "initialization_immutable": before == after,
        "matched_row": {
            "item_id": str(row["item_id"]),
            "battery": str(row["battery"]),
            "prompt_token_sha256": hashlib.sha256(encoded["input_ids"].cpu().numpy().tobytes()).hexdigest(),
            "prompt_tokens": int(encoded["input_ids"].numel()),
        },
        "trace_semantics": {
            "p35": "four sidecar flow updates followed by one loop-4 bridge write and one LM-head projection",
            "stage2b": "one identity recurrent pass followed by three sidecar updates, three bridge writes, and three recurrent re-entries",
            "stage2b_loop_labels": "loop_i denotes sidecar update i; the initial identity pass is recorded under base_*",
        },
        "trace_validation": {"p35": p35_trace_validation, "stage2b": stage2b_trace_validation},
        "first_divergence": first,
        "registered_prediction_supported": first["classification"] == "bridge_or_amplitude_semantics",
        "final_logits": logit_comparison(_cpu(p35_logits), _cpu(stage2b_logits)),
        "source_provenance": source,
        "decision_mapping": (
            "SCORER_ARTIFACT__BANK_NATIVE_STAGE2B_K4_AND_FOOTNOTE_P35_AMPLITUDE"
            if source["success_defining_graph"].startswith("Stage2B") and not source["same_graph"]
            else "ESCALATE"
        ),
        "artifacts": {
            "paired_trace_tensors": {"path": str(tensor_path), "sha256": sha256_file(tensor_path)},
            "stage_table": {"path": str(table_path), "sha256": sha256_file(table_path)},
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / f"seed_{args.seed}_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--dev1_panel", type=Path, required=True)
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
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
