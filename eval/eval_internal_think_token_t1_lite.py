"""Evaluate T1-lite forced, self-halted, baseline, extrapolation, and causal gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs
from training.internal_think_token_runtime import install_internal_control_tokens, split_internal_control_token_rows
from training.internal_think_token_t1 import (
    build_candidate_trie_contract,
    causal_override_schedule,
    locate_readout_positions,
    phase_t1_gate_verdict,
    score_control_predictions,
)
from training.internal_think_token_t1_spec import phase_t1_locked, validate_locked_phase_t1
from training.run_internal_think_token_p0_cell import PilotDataset, candidate_values_from_rows, collate_pilot, read_jsonl
from training.train_unfrozen_recurrent import prepare_wrapper


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("instance_id") or "")


def manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "row_id_sha256": sha256_lines([row_id(row) for row in rows]),
        "row_sha256": sha256_lines([canonical_row(row) for row in rows]),
        "by_depth": {
            str(depth): sum(int(row["depth"]) == depth for row in rows)
            for depth in sorted({int(row["depth"]) for row in rows})
        },
    }


def restore_checkpoint(wrapper: Any, checkpoint: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["trainable_state_dict"]
    current = dict(wrapper.named_parameters())
    missing = sorted(set(state) - set(current))
    if missing:
        raise RuntimeError(f"T1 evaluation checkpoint keys are unavailable: {missing[:8]}")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    return payload


def eval_row_to_control_sft(row: dict[str, Any]) -> dict[str, Any]:
    """Convert frozen MCQ metadata to the registered question-only symbol reader."""

    converted = dict(row)
    depth = int(row["depth"])
    question = str(row.get("question") or row.get("prompt") or "").rstrip()
    if not question:
        raise ValueError("T1 frozen evaluation row has no question")
    converted["prompt"] = question + "\n<|recur_readout|>\nAnswer:"
    converted["completion"] = " " + str(row["target"])
    orbit = [str(value) for value in row.get("orbit") or []]
    if len(orbit) < depth + 1:
        raise ValueError("T1 frozen evaluation row lacks the full target orbit")
    converted["loop_completions"] = [" " + orbit[index] for index in range(1, depth + 1)]
    converted["target_loop_count"] = depth
    converted["control_active"] = True
    converted["control_targets"] = [0] * (depth - 1) + [1]
    converted["score_target"] = "full_symbols"
    converted["prompt_style"] = "question_only"
    return converted


def prepare_control_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    augmented = [eval_row_to_control_sft(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in augmented), encoding="utf-8")
    return path


def _first_stop(predictions: list[int], max_loops: int) -> tuple[int, bool]:
    selected = next((index + 1 for index, value in enumerate(predictions) if value == 1), None)
    return (int(selected), False) if selected is not None else (int(max_loops), True)


@torch.no_grad()
def evaluate_rows(
    wrapper: Any,
    tokenizer: Any,
    data_path: Path,
    *,
    device: str,
    max_loops: int,
    batch_size: int,
    continue_id: int,
    stop_id: int,
    readout_id: int,
    include_features: bool,
) -> dict[str, Any]:
    dataset = PilotDataset(data_path, tokenizer, max_length=512, max_loops=max_loops)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=partial(collate_pilot, pad_token_id=tokenizer.pad_token_id),
    )
    candidate_values = candidate_values_from_rows(dataset.base.rows)
    contract = build_candidate_trie_contract(
        tokenizer,
        prompt=str(dataset.base.rows[0]["prompt"]),
        candidate_values=candidate_values,
    )
    if any(len(tokens) != 1 for tokens in contract.candidate_token_ids):
        raise AssertionError("registered A-P same-reader candidates must each be one token")
    candidate_ids = torch.tensor(
        [tokens[0] for tokens in contract.candidate_token_ids], device=device, dtype=torch.long
    )
    token_to_value = {tokens[0]: value for value, tokens in zip(contract.candidate_values, contract.candidate_token_ids)}
    rows_out: list[dict[str, Any]] = []
    wrapper.eval()
    offset = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = wrapper(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
            max_loops=max_loops,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            return_loop_recurrent_states=include_features,
        )
        logits = output.loop_logits
        if logits is None:
            raise AssertionError("T1 evaluation requires loop logits")
        positions = locate_readout_positions(
            batch["input_ids"], readout_token_id=readout_id, control_active=batch["control_active"]
        )
        for row_index in range(batch["input_ids"].shape[0]):
            source = dataset.base.rows[offset + row_index]
            depth = int(source["depth"])
            active = batch["labels"][row_index].ne(-100).nonzero(as_tuple=False).view(-1)
            if active.numel() != 1:
                raise AssertionError("registered letter evaluator requires one answer token")
            answer_position = int(active[0].item())
            target_id = int(batch["labels"][row_index, answer_position].item())
            if target_id not in token_to_value:
                raise AssertionError("answer target is outside the registered A-P candidates")
            answer_scores = logits[row_index, 0, :, answer_position - 1].index_select(-1, candidate_ids).float()
            answer_predictions = candidate_ids[answer_scores.argmax(dim=-1)].tolist()
            control_scores = logits[row_index, 0, :, int(positions[row_index])].index_select(
                -1, torch.tensor([continue_id, stop_id], device=device, dtype=torch.long)
            ).float()
            predictions = control_scores.argmax(dim=-1).tolist()
            selected, exhausted = _first_stop(predictions, max_loops)
            forced_index = depth - 1
            self_index = selected - 1
            feature_payload: dict[str, Any] = {}
            if include_features:
                probabilities = torch.softmax(answer_scores, dim=-1)
                top2 = answer_scores.topk(k=2, dim=-1).values
                margins = (top2[:, 0] - top2[:, 1]).tolist()
                successive_kl = [None]
                for loop_index in range(1, max_loops):
                    previous = probabilities[loop_index - 1].clamp_min(1e-8)
                    current = probabilities[loop_index].clamp_min(1e-8)
                    successive_kl.append(float((current * (current.log() - previous.log())).sum().item()))
                states = output.loop_recurrent_states
                if states is None:
                    raise AssertionError("feature evaluation requires recurrent states")
                state_updates: list[float | None] = [None]
                position = int(positions[row_index])
                for loop_index in range(1, max_loops):
                    delta = states[row_index, 0, loop_index, position].float() - states[row_index, 0, loop_index - 1, position].float()
                    state_updates.append(float(delta.square().mean().sqrt().item()))
                feature_payload = {
                    "answer_margins": margins,
                    "successive_output_kl": successive_kl,
                    "hidden_update_rms": state_updates,
                }
            rows_out.append(
                {
                    "row_id": row_id(source),
                    "depth": depth,
                    "control_predictions": predictions,
                    "selected_loop": selected,
                    "exhausted": exhausted,
                    "forced_answer_correct": int(answer_predictions[forced_index] == target_id),
                    "self_halted_answer_correct": int(answer_predictions[self_index] == target_id),
                    "answer_predictions": [token_to_value[int(value)] for value in answer_predictions],
                    **feature_payload,
                }
            )
        offset += int(batch["input_ids"].shape[0])
        print(f"t1_eval_progress={offset}/{len(dataset)}", flush=True)
    control = score_control_predictions(
        [{"row_id": row["row_id"], "depth": row["depth"], "predictions": row["control_predictions"]} for row in rows_out],
        max_loops=max_loops,
    )
    return {
        "rows": rows_out,
        "control": control,
        "forced_correct": sum(row["forced_answer_correct"] for row in rows_out),
        "self_halted_correct": sum(row["self_halted_answer_correct"] for row in rows_out),
        "total": len(rows_out),
        "exhausted": sum(row["exhausted"] for row in rows_out),
    }


def select_by_threshold(values: list[float | None], threshold: float, *, mode: str) -> int:
    for loop_index, value in enumerate(values, start=1):
        if value is None:
            continue
        if (mode == "ge" and float(value) >= threshold) or (mode == "le" and float(value) <= threshold):
            return loop_index
    return len(values)


def fit_baseline(rows: list[dict[str, Any]], *, field: str, mode: str) -> dict[str, Any]:
    values = sorted({float(value) for row in rows for value in row[field] if value is not None})
    if not values:
        raise ValueError(f"baseline feature {field} has no finite values")
    span = max(1.0, abs(values[0]), abs(values[-1]))
    candidates = [values[0] - span, values[-1] + span] + values
    best = None
    for threshold in candidates:
        selected = [select_by_threshold(row[field], threshold, mode=mode) for row in rows]
        correct = sum(value == int(row["depth"]) for value, row in zip(selected, rows))
        mean_loops = sum(selected) / len(selected)
        candidate = (correct, -mean_loops, -float(threshold) if math.isfinite(threshold) else 0.0, threshold)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return {"field": field, "mode": mode, "threshold": best[3], "calibration_correct": best[0], "calibration_total": len(rows)}


def score_baselines(calibration_rows: list[dict[str, Any]], gated_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = max(
        (
            sum(int(row["depth"]) == depth for row in calibration_rows),
            -depth,
            depth,
        )
        for depth in range(1, 9)
    )
    fitted = {
        "fixed_depth": {"selected_depth": fixed[2], "calibration_correct": fixed[0], "calibration_total": len(calibration_rows)},
        "answer_logit_margin": fit_baseline(calibration_rows, field="answer_margins", mode="ge"),
        "successive_output_kl": fit_baseline(calibration_rows, field="successive_output_kl", mode="le"),
        "hidden_state_update_norm": fit_baseline(calibration_rows, field="hidden_update_rms", mode="le"),
    }
    for name, fit in fitted.items():
        if name == "fixed_depth":
            selected = [int(fit["selected_depth"])] * len(gated_rows)
        else:
            selected = [select_by_threshold(row[fit["field"]], float(fit["threshold"]), mode=str(fit["mode"])) for row in gated_rows]
        fit["gated_exact_correct"] = sum(value == int(row["depth"]) for value, row in zip(selected, gated_rows))
        fit["gated_total"] = len(gated_rows)
        fit["gated_mean_selected_loops"] = sum(selected) / len(selected)
    return fitted


@torch.no_grad()
def run_causal_sweep(
    wrapper: Any,
    tokenizer: Any,
    data_path: Path,
    *,
    device: str,
    continue_id: int,
    stop_id: int,
    readout_id: int,
    progress_path: Path,
) -> dict[str, Any]:
    dataset = PilotDataset(data_path, tokenizer, max_length=512, max_loops=9)
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[str(row["intervention_id"])] = row
    exact = sum(int(row["exact"]) for row in completed.values())
    handle = progress_path.open("a", encoding="utf-8")
    try:
        for index in range(len(dataset)):
            item = collate_pilot([dataset[index]], pad_token_id=tokenizer.pad_token_id)
            item = {key: value.to(device) for key, value in item.items()}
            position = locate_readout_positions(
                item["input_ids"], readout_token_id=readout_id, control_active=item["control_active"]
            )
            source = dataset.base.rows[index]
            depth = int(source["depth"])
            schedule = causal_override_schedule(depth)
            interventions = [
                (f"{row_id(source)}:stop:{stop_index}", payload, max(depth, stop_index))
                for stop_index, payload in enumerate(schedule["forced_stops"], start=1)
            ]
            interventions.append((f"{row_id(source)}:continue:{depth}", schedule["forced_continue"], depth + 1))
            for intervention_id, payload, maximum in interventions:
                if intervention_id in completed:
                    continue
                output = wrapper(
                    input_ids=item["input_ids"],
                    attention_mask=item["attention_mask"],
                    labels=None,
                    max_loops=maximum,
                    use_cache=False,
                    return_dict=True,
                    internal_control_enabled=True,
                    internal_control_token_ids=(continue_id, stop_id),
                    internal_control_readout_positions=position,
                    internal_control_overrides={int(key): value for key, value in payload["overrides"].items()},
                )
                observed = int(output.executed_loops.item())
                expected = int(payload["expected_executed_loops"])
                record = {
                    "intervention_id": intervention_id,
                    "row_id": row_id(source),
                    "depth": depth,
                    "expected_executed_loops": expected,
                    "observed_executed_loops": observed,
                    "exact": observed == expected,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                completed[intervention_id] = record
                exact += int(record["exact"])
            if (index + 1) % 16 == 0:
                print(f"causal_sweep_rows={index + 1}/{len(dataset)} interventions={len(completed)}/5632 exact={exact}", flush=True)
    finally:
        handle.close()
    return {"exact": exact, "total": len(completed), "required_total": 5632, "passed": exact == 5632 and len(completed) == 5632}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gated_jsonl", required=True)
    parser.add_argument("--calibration_jsonl", required=True)
    parser.add_argument("--extrapolation_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--run_causal_sweep", action="store_true")
    parser.add_argument("--causal_progress_path")
    args = parser.parse_args()

    prereg = phase_t1_locked()
    validate_locked_phase_t1(prereg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_load_kwargs(args.dtype, args.attn_implementation)
    ).to(args.device)
    resize = install_internal_control_tokens(tokenizer, model)
    split_internal_control_token_rows(model, original_vocab_size=resize.original_vocab_size)
    config = {
        "layer_split": "6,18",
        "initial_halt_prob": 0.15,
        "bridge_projection_mode": "split",
        "adapter_dtype": "float32",
        "training_mode": "full_block",
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "train_auxiliary": {"bridge": True, "halting": False, "reentry_adapter": False, "latent": False},
    }
    wrapper, _ = prepare_wrapper(model, config, device=args.device)
    wrapper.base_model.get_input_embeddings().control_rows.requires_grad_(True)
    checkpoint = restore_checkpoint(wrapper, args.checkpoint)
    continue_id, stop_id, readout_id = (int(value) for value in resize.control_token_ids)

    gated_source = [row for row in read_jsonl(args.gated_jsonl) if 1 <= int(row["depth"]) <= 8]
    extrap_source = [row for row in read_jsonl(args.extrapolation_jsonl) if 9 <= int(row["depth"]) <= 14]
    calibration_source = read_jsonl(args.calibration_jsonl)
    for label, rows, contract in (
        ("gated", gated_source, prereg["evaluation"]["gated"]),
        ("extrapolation", extrap_source, prereg["evaluation"]["extrapolation"]),
        ("calibration", calibration_source, prereg["evaluation"]["calibration"]),
    ):
        observed = manifest(rows)
        for key in ("row_id_sha256", "row_sha256"):
            if observed[key] != contract[key]:
                raise RuntimeError(f"{label} frozen manifest mismatch for {key}: {observed[key]} != {contract[key]}")
        write_json(output_dir / f"{label}_manifest.json", observed)
    gated_path = prepare_control_rows(gated_source, output_dir / "data" / "gated_control.jsonl")
    extrap_path = prepare_control_rows(extrap_source, output_dir / "data" / "extrap_control.jsonl")
    calibration_path = prepare_control_rows(calibration_source, output_dir / "data" / "calibration_control.jsonl")

    precausal_cache = output_dir / "precausal_cache.json"
    if precausal_cache.exists():
        cached = json.loads(precausal_cache.read_text(encoding="utf-8"))
        if cached.get("checkpoint") != str(args.checkpoint):
            raise RuntimeError("T1-lite precausal cache belongs to a different checkpoint")
        gated = cached["gated"]
        calibration = cached["calibration"]
        extrapolation = cached["extrapolation"]
        baselines = cached["descriptive_baselines"]
        print(f"resumed_t1_precausal_cache={precausal_cache}", flush=True)
    else:
        gated = evaluate_rows(
            wrapper, tokenizer, gated_path, device=args.device, max_loops=12, batch_size=args.batch_size,
            continue_id=continue_id, stop_id=stop_id, readout_id=readout_id, include_features=True,
        )
        calibration = evaluate_rows(
            wrapper, tokenizer, calibration_path, device=args.device, max_loops=12, batch_size=args.batch_size,
            continue_id=continue_id, stop_id=stop_id, readout_id=readout_id, include_features=True,
        )
        extrapolation = evaluate_rows(
            wrapper, tokenizer, extrap_path, device=args.device, max_loops=16, batch_size=args.batch_size,
            continue_id=continue_id, stop_id=stop_id, readout_id=readout_id, include_features=False,
        )
        baselines = score_baselines(calibration["rows"], gated["rows"])
        write_json(
            precausal_cache,
            {
                "checkpoint": str(args.checkpoint),
                "gated": gated,
                "calibration": calibration,
                "extrapolation": extrapolation,
                "descriptive_baselines": baselines,
            },
        )
    causal = {"status": "not_requested"}
    if args.run_causal_sweep:
        causal = run_causal_sweep(
            wrapper, tokenizer, gated_path, device=args.device, continue_id=continue_id,
            stop_id=stop_id, readout_id=readout_id,
            progress_path=(
                Path(args.causal_progress_path)
                if args.causal_progress_path
                else output_dir / "causal_override_progress.jsonl"
            ),
        )
    selection_by_depth = gated["control"]["by_depth"]
    gates = None
    if args.run_causal_sweep:
        gates = phase_t1_gate_verdict(
            forced_correct=gated["forced_correct"], forced_total=gated["total"],
            self_halted_correct=gated["self_halted_correct"], self_halted_total=gated["total"],
            selection_by_depth=selection_by_depth,
            causal_exact=int(causal["exact"]), causal_total=int(causal["total"]),
        )
    write_json(output_dir / "gated_rows.json", gated["rows"])
    write_json(output_dir / "calibration_rows.json", calibration["rows"])
    write_json(output_dir / "extrapolation_rows.json", extrapolation["rows"])
    summary = {
        "kind": "paper2_t1_lite_evaluation",
        "status": "finished",
        "checkpoint": str(args.checkpoint),
        "checkpoint_kind": checkpoint.get("kind"),
        "final_step": checkpoint.get("step"),
        "gated": {key: value for key, value in gated.items() if key != "rows"},
        "calibration": {key: value for key, value in calibration.items() if key != "rows"},
        "extrapolation": {key: value for key, value in extrapolation.items() if key != "rows"},
        "descriptive_baselines": baselines,
        "causal_override": causal,
        "registered_gates": gates,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
