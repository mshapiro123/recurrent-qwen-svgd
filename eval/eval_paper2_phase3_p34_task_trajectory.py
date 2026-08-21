"""Score one P3.4 calibration checkpoint on the frozen DEV task panel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.cache_paper2_phase3_agreement_oracle import _load_phase3_module
from eval.eval_paper2_phase3_p31_references import (
    MODEL_SPECS,
    _chat_prompt,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
    score_generated,
)
from eval.eval_paper2_phase3_p34_task_inference import P34TaskInferenceGraph
from models.paper2_dc2_student import install_probe_control_reader
from training.paper2_phase3_p34 import P34_GATE_CEILINGS
from training.paper2_phase3_p34_lock import panel_identity_sha256
from training.paper2_phase3_p35 import margin_summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _telemetry_summary(
    gates: Sequence[float],
    ratios: Sequence[float],
    *,
    memory_gates: Sequence[float] = (),
    memory_scores: Sequence[float] = (),
    memory_entropies: Sequence[float] = (),
    memory_slot_ids: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    if len(gates) != len(ratios) or not gates:
        raise ValueError("P3.4 DEV telemetry requires paired nonempty gate/write reads")
    result = {
        "telemetry_positions": len(gates),
        "position_gate_mean": sum(gates) / len(gates),
        "position_gate_max": max(gates),
        "realized_writeback_ratio_mean": sum(ratios) / len(ratios),
        "realized_writeback_ratio_max": max(ratios),
    }
    if memory_gates:
        result.update(
            {
                "memory_compatibility_gate_mean": sum(memory_gates) / len(memory_gates),
                "memory_retrieval_score_mean": sum(memory_scores) / len(memory_scores),
                "memory_retrieval_entropy_mean": sum(memory_entropies)
                / len(memory_entropies),
                "memory_top_k_slot_ids": list(memory_slot_ids),
            }
        )
    return result


def _append_memory_telemetry(
    destination: dict[str, list[Any]], output: Any, index: int
) -> None:
    if output.memory_compatibility_gate is None:
        return
    destination.setdefault("memory_gates", []).append(
        float(output.memory_compatibility_gate[index].cpu())
    )
    destination.setdefault("memory_scores", []).append(
        float(output.memory_slot_scores[index, 0].cpu())
    )
    weights = output.memory_slot_weights[index].float()
    destination.setdefault("memory_entropies", []).append(
        float((-(weights * weights.clamp_min(1e-12).log()).sum()).cpu())
    )
    destination.setdefault("memory_slot_ids", []).append(
        [int(value) for value in output.memory_slot_indices[index].cpu().tolist()]
    )


def resolve_evaluation_gate_ceiling(
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    override: float | None,
    *,
    authorized_overrides: Sequence[float] = (0.02, 0.08),
) -> tuple[float, str]:
    """Resolve the score-only ruler independently of training controller state."""

    campaign = [item for item in checkpoint_receipts if item.get("label") == "p34"]
    p35 = [item for item in checkpoint_receipts if item.get("label") == "p35"]
    if p35 and override is None:
        return float(p35[-1]["evaluation_gate_ceiling"]), "p35_checkpoint_contract"
    registered = 0.08 if not campaign else P34_GATE_CEILINGS[int(campaign[-1]["controller_rung"])]
    if override is None:
        return float(registered), "checkpoint_controller_rung"
    value = float(override)
    authorized = tuple(float(item) for item in authorized_overrides)
    if value not in authorized:
        raise ValueError(
            "score-only fixed-ceiling override is not authorized: "
            f"value={value} authorized={authorized}"
        )
    return value, "score_only_fixed_ceiling_override"


def _apply_state(module: Any, path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"P3.4 {label} checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"P3.4 {label} checkpoint lacks trainable_state")
    current = dict(module.named_parameters())
    unknown = sorted(
        name for name in set(state) - set(current) if not name.startswith("slot_lift.")
    )
    if unknown:
        raise RuntimeError(f"P3.4 {label} state contains unknown tensors: {unknown}")
    with torch.no_grad():
        for name, value in state.items():
            if name.startswith("slot_lift."):
                continue
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    receipt = {
        "label": label,
        "path": str(path),
        "sha256": observed,
        "step": int(payload["step"]),
        "state_keys": sorted(state),
    }
    if label == "p34":
        controller = payload.get("controller")
        if not isinstance(controller, dict) or "rung" not in controller:
            raise RuntimeError("P3.4 campaign checkpoint lacks controller state")
        receipt["controller_rung"] = int(controller["rung"])
    if label == "p35":
        receipt["evaluation_gate_ceiling"] = float(
            payload["evaluation_gate_ceiling"]
        )
        receipt["control_reader"] = str(payload["control_reader"])
    return receipt


def load_condition(
    *,
    embedding_weight: torch.Tensor,
    migrated: Path,
    migrated_sha256: str,
    p33: Path | None,
    p33_sha256: str | None,
    i1: Path | None,
    i1_sha256: str | None,
    p34: Path | None = None,
    p34_sha256: str | None = None,
    p35: Path | None = None,
    p35_sha256: str | None = None,
    control_reader: str = "mean",
) -> tuple[Any, list[dict[str, Any]]]:
    if sha256_file(migrated) != migrated_sha256:
        raise RuntimeError("P3.4 migrated checkpoint SHA mismatch")
    module, migrated_receipt = _load_phase3_module(
        checkpoint=migrated,
        embedding_weight=embedding_weight,
        device="cuda",
    )
    receipts = [{"label": "migrated", **migrated_receipt}]
    if p33 is not None:
        if p33_sha256 is None:
            raise ValueError("P3.4 P3.3 checkpoint requires an expected SHA")
        receipts.append(_apply_state(module, p33, expected_sha256=p33_sha256, label="p33"))
    if i1 is not None:
        if p33 is None or i1_sha256 is None:
            raise ValueError("P3.4 i1 condition requires P3.3 state and an expected SHA")
        receipts.append(_apply_state(module, i1, expected_sha256=i1_sha256, label="i1"))
    if p34 is not None:
        if i1 is None or p34_sha256 is None:
            raise ValueError("P3.4 campaign state requires i1 state and an expected SHA")
        receipts.append(_apply_state(module, p34, expected_sha256=p34_sha256, label="p34"))
    if p35 is not None:
        if p34 is None or p35_sha256 is None:
            raise ValueError("P3.5 state requires P3.4 state and an expected SHA")
        if control_reader == "probe":
            install_probe_control_reader(module, n_probes=4)
        elif control_reader != "mean":
            raise ValueError("P3.5 control reader must be mean or probe")
        receipts.append(_apply_state(module, p35, expected_sha256=p35_sha256, label="p35"))
    if p35 is not None:
        module.bridge.set_gate_ceiling(float(receipts[-1]["evaluation_gate_ceiling"]))
    elif p34 is None:
        module.bridge.set_gate_ceiling(0.08)
    else:
        module.bridge.set_gate_ceiling((0.02, 0.08, 0.20, 0.50)[int(receipts[-1]["controller_rung"])])
    return module.eval(), receipts


@torch.inference_mode()
def score_mcq(
    graph: P34TaskInferenceGraph,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    prompts: list[str] = []
    prompt_candidates: list[list[int]] = []
    candidate_metadata: list[tuple[str, str]] = []
    candidate_tokens: list[list[int]] = []
    verified_label_suffixes: dict[str, list[int]] = {}
    answers: dict[str, str] = {}
    prompt_item_ids: list[str] = []
    telemetry: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"gates": [], "ratios": []}
    )
    for row in rows:
        question, choices, answer = _mcq(row)
        item_id = str(row["item_id"])
        answers[item_id] = answer
        labels = [label for label, _text in choices]
        for shift in range(len(choices)):
            permuted = []
            label_map = {}
            for new_index, new_label in enumerate(labels):
                original_label, original_text = choices[(new_index - shift) % len(choices)]
                permuted.append((new_label, original_text))
                label_map[new_label] = original_label
            prompt = _mcq_prompt(question, permuted)
            local_candidates = []
            for new_label in labels:
                continuation = verified_label_suffixes.get(new_label)
                if continuation is None:
                    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                    complete = tokenizer(
                        prompt + f" {new_label}", add_special_tokens=True
                    )["input_ids"]
                    suffix = tokenizer(
                        f" {new_label}", add_special_tokens=False
                    )["input_ids"]
                    if complete[: len(prompt_ids)] != prompt_ids or complete[
                        len(prompt_ids) :
                    ] != suffix:
                        raise RuntimeError(
                            f"P3.4 MCQ suffix tokenization is not stable for label {new_label}"
                        )
                    continuation = [int(token) for token in suffix]
                    verified_label_suffixes[new_label] = continuation
                if not continuation:
                    raise RuntimeError(f"P3.4 MCQ candidate has no continuation: {new_label}")
                local_candidates.append(len(candidate_metadata))
                candidate_metadata.append((item_id, label_map[new_label]))
                candidate_tokens.append(continuation)
            prompts.append(prompt)
            prompt_candidates.append(local_candidates)
            prompt_item_ids.append(item_id)

    option_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    candidate_scores: list[list[float]] = [[] for _row in candidate_metadata]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for start in range(0, len(prompts), batch_size):
        stop = min(len(prompts), start + batch_size)
        encoded = tokenizer(
            prompts[start:stop],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(graph.device)
        _state, output = graph.prefill_cached(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )
        log_probabilities = torch.log_softmax(output.augmented_logits.float(), dim=-1)
        for local, item_id in enumerate(prompt_item_ids[start:stop]):
            telemetry[item_id]["gates"].append(float(output.position_gate[local].cpu()))
            telemetry[item_id]["ratios"].append(float(output.writeback_ratio[local].cpu()))
            _append_memory_telemetry(telemetry[item_id], output, local)
        for local, candidate_indexes in enumerate(prompt_candidates[start:stop]):
            for candidate_index in candidate_indexes:
                token = candidate_tokens[candidate_index][0]
                candidate_scores[candidate_index].append(
                    float(log_probabilities[local, token].cpu())
                )
        print(
            f"p34_mcq_root_progress prompts={stop}/{len(prompts)}", flush=True
        )

    branch_lookup: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    prompt_ids = [tokenizer(prompt, add_special_tokens=True)["input_ids"] for prompt in prompts]
    for prompt_index, candidate_indexes in enumerate(prompt_candidates):
        for candidate_index in candidate_indexes:
            tokens = candidate_tokens[candidate_index]
            for depth in range(1, len(tokens)):
                branch_lookup[(prompt_index, tuple(tokens[:depth]))].append(
                    (candidate_index, tokens[depth])
                )
    branches = [
        ([*prompt_ids[prompt_index], *prefix], targets, prompt_index)
        for (prompt_index, prefix), targets in branch_lookup.items()
    ]
    for start in range(0, len(branches), batch_size):
        stop = min(len(branches), start + batch_size)
        batch = branches[start:stop]
        width = max(len(prefix) for prefix, _targets, _prompt_index in batch)
        input_ids = torch.full(
            (len(batch), width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=graph.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for local, (prefix, _targets, _prompt_index) in enumerate(batch):
            input_ids[local, -len(prefix) :] = torch.tensor(
                prefix, device=graph.device, dtype=torch.long
            )
            attention_mask[local, -len(prefix) :] = 1
        _state, output = graph.prefill_cached(
            input_ids=input_ids, attention_mask=attention_mask
        )
        log_probabilities = torch.log_softmax(output.augmented_logits.float(), dim=-1)
        for local, (_prefix, targets, prompt_index) in enumerate(batch):
            item_id = prompt_item_ids[prompt_index]
            telemetry[item_id]["gates"].append(float(output.position_gate[local].cpu()))
            telemetry[item_id]["ratios"].append(float(output.writeback_ratio[local].cpu()))
            _append_memory_telemetry(telemetry[item_id], output, local)
            for candidate_index, target in targets:
                candidate_scores[candidate_index].append(
                    float(log_probabilities[local, target].cpu())
                )
        print(
            f"p34_mcq_branch_progress branches={stop}/{len(branches)}", flush=True
        )

    for (item_id, original_label), scores in zip(
        candidate_metadata, candidate_scores
    ):
        option_scores[item_id][original_label].append(sum(scores) / len(scores))

    results = []
    for row in rows:
        item_id = str(row["item_id"])
        means = {
            label: sum(values) / len(values)
            for label, values in option_scores[item_id].items()
        }
        prediction = max(means, key=means.get)
        ordered_scores = sorted(means.values(), reverse=True)
        decision_margin = float(ordered_scores[0] - ordered_scores[1])
        results.append(
            {
                "item_id": item_id,
                "battery": str(row["battery"]),
                "prediction": prediction,
                "augmented_correct": prediction == answers[item_id],
                "option_scores": means,
                "reader": "cyclic_label_aggregated_permutation_mean_v1",
                "answer_token_margins": [decision_margin],
                "answer_token_margin_minimum": decision_margin,
                **_telemetry_summary(
                    telemetry[item_id]["gates"],
                    telemetry[item_id]["ratios"],
                    memory_gates=telemetry[item_id].get("memory_gates", ()),
                    memory_scores=telemetry[item_id].get("memory_scores", ()),
                    memory_entropies=telemetry[item_id].get("memory_entropies", ()),
                    memory_slot_ids=telemetry[item_id].get("memory_slot_ids", ()),
                ),
            }
        )
    return results


@torch.inference_mode()
def score_generation(
    graph: P34TaskInferenceGraph,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    emit_batch: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    by_cap: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _content, cap = _generation_prompt(row)
        by_cap[cap].append(row)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    results = []
    for cap, selected in sorted(by_cap.items()):
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            prepare_probe_batch = getattr(graph, "prepare_probe_batch", None)
            if prepare_probe_batch is not None:
                prepare_probe_batch(batch)
            prompts = [
                _chat_prompt(tokenizer, _generation_prompt(row)[0]) for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(graph.device)
            state, output = graph.prefill_cached(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )
            generated: list[list[int]] = [[] for _row in batch]
            finished = [False for _row in batch]
            gates: list[list[float]] = [[] for _row in batch]
            ratios: list[list[float]] = [[] for _row in batch]
            margins: list[list[float]] = [[] for _row in batch]
            memory: list[dict[str, list[Any]]] = [{} for _row in batch]
            generated_steps: list[torch.Tensor] = []
            gate_steps: list[torch.Tensor] = []
            ratio_steps: list[torch.Tensor] = []
            margin_steps: list[torch.Tensor] = []
            vectorized_telemetry = output.memory_compatibility_gate is None
            finished_mask = torch.zeros(
                len(batch), dtype=torch.bool, device=graph.device
            )
            for token_index in range(cap):
                selected_tokens = output.augmented_logits.argmax(dim=-1)
                if vectorized_telemetry:
                    generated_steps.append(selected_tokens.detach())
                    gate_steps.append(output.position_gate.detach())
                    ratio_steps.append(output.writeback_ratio.detach())
                    margin_steps.append(output.answer_token_margin.detach())
                    if tokenizer.eos_token_id is not None:
                        finished_mask |= selected_tokens.eq(int(tokenizer.eos_token_id))
                else:
                    for index, token in enumerate(selected_tokens.tolist()):
                        if not finished[index]:
                            gates[index].append(float(output.position_gate[index].cpu()))
                            ratios[index].append(float(output.writeback_ratio[index].cpu()))
                            margins[index].append(float(output.answer_token_margin[index].cpu()))
                            _append_memory_telemetry(memory[index], output, index)
                            generated[index].append(int(token))
                            if (
                                tokenizer.eos_token_id is not None
                                and int(token) == int(tokenizer.eos_token_id)
                            ):
                                finished[index] = True
                # Running the vectorized path to the registered cap avoids a
                # CPU-GPU synchronization on every token. Rows are truncated
                # at their first EOS below, so predictions and telemetry stay
                # identical to the early-exit path.
                if (
                    not vectorized_telemetry
                    and all(finished)
                ) or token_index + 1 == cap:
                    break
                state, output = graph.advance_cached(
                    state=state, selected_tokens=selected_tokens
                )
            if vectorized_telemetry:
                token_rows = torch.stack(generated_steps, dim=1).cpu().tolist()
                gate_rows = torch.stack(gate_steps, dim=1).float().cpu().tolist()
                ratio_rows = torch.stack(ratio_steps, dim=1).float().cpu().tolist()
                margin_rows = torch.stack(margin_steps, dim=1).float().cpu().tolist()
                for index, token_ids in enumerate(token_rows):
                    length = len(token_ids)
                    if (
                        tokenizer.eos_token_id is not None
                        and int(tokenizer.eos_token_id) in token_ids
                    ):
                        length = token_ids.index(int(tokenizer.eos_token_id)) + 1
                    generated[index] = [int(token) for token in token_ids[:length]]
                    gates[index] = [float(value) for value in gate_rows[index][:length]]
                    ratios[index] = [float(value) for value in ratio_rows[index][:length]]
                    margins[index] = [float(value) for value in margin_rows[index][:length]]
            batch_results = []
            for row, token_ids, row_gates, row_ratios, row_margins, row_memory in zip(
                batch, generated, gates, ratios, margins, memory
            ):
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                correct, prediction = score_generated(row, text)
                batch_results.append(
                    {
                        "item_id": str(row["item_id"]),
                        "battery": str(row["battery"]),
                        "prediction": prediction,
                        "generated_text": text,
                        "generated_token_ids": token_ids,
                        "generated_tokens": len(token_ids),
                        "augmented_correct": bool(correct),
                        "reader": str(row["reader"]),
                        "answer_token_margins": row_margins,
                        "answer_token_margin_minimum": min(row_margins),
                        **_telemetry_summary(
                            row_gates,
                            row_ratios,
                            memory_gates=row_memory.get("memory_gates", ()),
                            memory_scores=row_memory.get("memory_scores", ()),
                            memory_entropies=row_memory.get("memory_entropies", ()),
                            memory_slot_ids=row_memory.get("memory_slot_ids", ()),
                        ),
                    }
                )
            results.extend(batch_results)
            if emit_batch is not None:
                emit_batch(batch_results)
            print(
                f"p34_generation_progress battery={batch[0]['battery']} "
                f"rows={min(start + len(batch), len(selected))}/{len(selected)} cap={cap}",
                flush=True,
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--look", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33", type=Path)
    parser.add_argument("--p33_sha256")
    parser.add_argument("--i1", type=Path)
    parser.add_argument("--i1_sha256")
    parser.add_argument("--p34", type=Path)
    parser.add_argument("--p34_sha256")
    parser.add_argument("--p35", type=Path)
    parser.add_argument("--p35_sha256")
    parser.add_argument("--control_reader", choices=("mean", "probe"), default="mean")
    parser.add_argument("--mcq_batch_size", type=int, default=32)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--flow_loops", type=int, default=4)
    parser.add_argument("--allow_clamped_extension", action="store_true")
    parser.add_argument("--gate_ceiling_override", type=float)
    parser.add_argument(
        "--authorized_gate_ceiling_override",
        type=float,
        action="append",
        help="Explicit score-only authorization; defaults to the legacy 0.02/0.08 pair.",
    )
    args = parser.parse_args()

    panel = read_jsonl(args.panel)
    if len(panel) != 1_024 or any(str(row["partition"]) != "dev" for row in panel):
        raise RuntimeError("P3.4 trajectory scorer requires the frozen 1,024-row DEV panel")
    panel_sha256 = panel_identity_sha256(panel)
    base_rows = {str(row["item_id"]): row for row in read_jsonl(args.base_scores)}
    if any(str(row["item_id"]) not in base_rows for row in panel):
        raise RuntimeError("P3.4 base-score coverage is incomplete")
    existing = read_jsonl(args.output_jsonl) if args.output_jsonl.exists() else []
    existing_lookup = {str(row["item_id"]): row for row in existing}
    if len(existing_lookup) != len(existing):
        raise RuntimeError("P3.4 resumable trajectory output contains duplicate items")
    for row in existing:
        if (
            row.get("condition") != args.condition
            or int(row.get("look", -1)) != args.look
            or int(row.get("seed", -1)) != args.seed
            or str(row.get("partition")) != "dev"
            or int(row.get("flow_loops", -1)) != int(args.flow_loops)
            or bool(row.get("clamped_extension", False))
            != bool(args.allow_clamped_extension)
            or (
                args.gate_ceiling_override is not None
                and float(row.get("evaluation_gate_ceiling", -1.0))
                != float(args.gate_ceiling_override)
            )
        ):
            raise RuntimeError("P3.4 resumable trajectory identity changed")
    pending = [row for row in panel if str(row["item_id"]) not in existing_lookup]

    spec = MODEL_SPECS["base"]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    sidecar, checkpoint_receipts = load_condition(
        embedding_weight=model.get_output_embeddings().weight.detach().cpu(),
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
        control_reader=args.control_reader,
    )
    evaluation_gate_ceiling, evaluation_gate_ceiling_source = resolve_evaluation_gate_ceiling(
        checkpoint_receipts,
        args.gate_ceiling_override,
        authorized_overrides=(
            args.authorized_gate_ceiling_override
            if args.authorized_gate_ceiling_override is not None
            else (0.02, 0.08)
        ),
    )
    sidecar.bridge.set_gate_ceiling(evaluation_gate_ceiling)
    graph = P34TaskInferenceGraph(
        base_model=model,
        sidecar=sidecar,
        flow_loops=args.flow_loops,
        allow_clamped_extension=args.allow_clamped_extension,
    )

    # The cache changes transport only.  Require identical selected tokens on a
    # real panel prompt before using it for the expensive trajectory pass.
    first_prompt = _mcq_prompt(*_mcq(next(row for row in panel if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}))[:2])
    probe = tokenizer(first_prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
    _cache, cached_probe = graph.prefill_cached(
        input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]
    )
    uncached_probe = graph.next_token(
        input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]
    )
    cache_max_abs = float(
        (cached_probe.augmented_logits - uncached_probe.augmented_logits).abs().max().cpu()
    )
    cache_argmax_equal = bool(
        torch.equal(
            cached_probe.augmented_logits.argmax(dim=-1),
            uncached_probe.augmented_logits.argmax(dim=-1),
        )
    )
    if not cache_argmax_equal:
        raise RuntimeError("P3.4 cached and uncached serving paths choose different tokens")

    source_lookup = {str(row["item_id"]): row for row in panel}

    def emit(scored_rows: list[dict[str, Any]]) -> None:
        enriched = []
        for augmented in scored_rows:
            item_id = str(augmented["item_id"])
            row = source_lookup[item_id]
            enriched.append({
                "kind": "paper2_phase3_p34_task_trajectory_row_v2",
                "condition": args.condition,
                "look": args.look,
                "seed": args.seed,
                "partition": "dev",
                "battery": row["battery"],
                "battery_role": row["battery_role"],
                "panel_group": row["p34_panel_group"],
                "document_id": row["document_id"],
                "item_id": item_id,
                "base_correct": bool(base_rows[item_id]["correct"]),
                "evaluation_gate_ceiling": evaluation_gate_ceiling,
                "flow_loops": int(args.flow_loops),
                "clamped_extension": bool(args.allow_clamped_extension),
                **augmented,
            })
        append_jsonl(args.output_jsonl, enriched)

    mcq_rows = [row for row in pending if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}]
    generated_rows = [row for row in pending if row["battery"] in {"gsm8k", "mbpp", "tier1"}]
    if mcq_rows:
        emit(score_mcq(graph, tokenizer, mcq_rows, batch_size=args.mcq_batch_size))
    if generated_rows:
        score_generation(
            graph,
            tokenizer,
            generated_rows,
            batch_size=args.generation_batch_size,
            emit_batch=emit,
        )
    output_rows = read_jsonl(args.output_jsonl)
    if len(output_rows) != 1_024 or {
        str(row["item_id"]) for row in output_rows
    } != set(source_lookup):
        raise RuntimeError("P3.4 augmented score coverage changed")
    telemetry_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        if not bool(row["base_correct"]) and bool(row["augmented_correct"]):
            outcome = "fix"
        elif bool(row["base_correct"]) and not bool(row["augmented_correct"]):
            outcome = "regression"
        else:
            outcome = "unchanged"
        telemetry_groups[f"battery:{row['battery']}"].append(row)
        telemetry_groups[f"outcome:{outcome}"].append(row)

    telemetry_summary = {}
    for key, selected in sorted(telemetry_groups.items()):
        telemetry_summary[key] = {
            "rows": len(selected),
            "position_gate_mean": sum(float(row["position_gate_mean"]) for row in selected)
            / len(selected),
            "realized_writeback_ratio_mean": sum(
                float(row["realized_writeback_ratio_mean"]) for row in selected
            )
            / len(selected),
            "telemetry_positions": sum(int(row["telemetry_positions"]) for row in selected),
        }
    summary = {
        "kind": "paper2_phase3_p34_task_trajectory_condition_v2",
        "status": "complete_dev_only",
        "condition": args.condition,
        "look": args.look,
        "seed": args.seed,
        "rows": len(output_rows),
        "panel_sha256": panel_sha256,
        "output_sha256": sha256_file(args.output_jsonl),
        "checkpoint_receipts": checkpoint_receipts,
        "evaluation_gate_ceiling": evaluation_gate_ceiling,
        "evaluation_gate_ceiling_source": evaluation_gate_ceiling_source,
        "flow_loops": int(args.flow_loops),
        "clamped_extension": bool(args.allow_clamped_extension),
        "depth_scope": (
            "registered_trained_support"
            if int(args.flow_loops) <= 4
            else "exploratory_last_step_clamped_off_contract"
        ),
        "depth_parameter_indices": {
            "flow_step_embedding": [min(index, 3) for index in range(int(args.flow_loops))],
            "bridge_gate_and_rho": min(int(args.flow_loops) - 1, 3),
        },
        "cache_transport": {
            "argmax_equal_on_real_prompt": cache_argmax_equal,
            "maximum_absolute_logit_difference": cache_max_abs,
            "base_kv_cache_only": True,
            "full_hidden_prefix_retained_for_sidecar": True,
        },
        "base_accuracy": sum(row["base_correct"] for row in output_rows) / len(output_rows),
        "augmented_accuracy": sum(row["augmented_correct"] for row in output_rows) / len(output_rows),
        "score_preserving_telemetry": {
            "instrumentation_only": True,
            "used_by_scoring": False,
            "used_by_controller": False,
            "by_battery_and_outcome": telemetry_summary,
        },
        "answer_token_margins": margin_summary(output_rows),
        "confirm_scored": False,
        "eval_e_scored": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del graph, sidecar, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
