"""Run the authorized Stage 2A CV-1 crossed-value audit on DEV only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_inference import P34TaskInferenceGraph
from eval.eval_paper2_phase3_p34_task_trajectory import (
    load_condition,
    score_generation,
    score_mcq,
)
from eval.eval_paper2_stage2a import (
    exact_sign_test,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from training.paper2_phase3_p31_completion import sha256_file
from training.paper2_stage2a_lock import assert_stage2a_training_authorized
from training.paper2_stage2a_runtime import (
    Stage2AMemorySystem,
    frozen_sidecar_digest,
    tensor_digest,
)


VALUE_CONDITIONS = ("correct", "shuffled", "random")
DOSE_MULTIPLIERS = (0.0, 0.5, 1.0)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _value_parameters(memory: Stage2AMemorySystem) -> list[tuple[str, torch.nn.Parameter]]:
    if memory.arm == "t3a":
        return [("reader.values", memory.reader.values)]
    if memory.arm == "t3b":
        return [
            (f"reader.tables.{index}.weight", table.weight)
            for index, table in enumerate(memory.reader.tables)
        ]
    raise ValueError(f"CV-1 host must be t3a or t3b, got {memory.arm}")


def value_bank(memory: Stage2AMemorySystem) -> torch.Tensor:
    return torch.cat([parameter.detach().float().cpu() for _, parameter in _value_parameters(memory)])


def fixed_map_digest(memory: Stage2AMemorySystem) -> str:
    excluded = {name for name, _ in _value_parameters(memory)}
    return tensor_digest(
        {
            name: value
            for name, value in memory.state_dict().items()
            if name not in excluded
        }
    )


def moment_matched_random(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("moment matching requires at least two value rows")
    source = values.detach().float().cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(source.shape, generator=generator, dtype=torch.float32)
    noise = noise - noise.mean(dim=0, keepdim=True)
    noise_std = noise.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
    noise = noise / noise_std
    mean = source.mean(dim=0, keepdim=True)
    std = (source - mean).square().mean(dim=0, keepdim=True).sqrt()
    return noise * std + mean


@torch.no_grad()
def apply_value_condition(
    memory: Stage2AMemorySystem,
    condition: str,
    *,
    shuffle_seed: int,
    random_seed: int,
) -> dict[str, Any]:
    if condition not in VALUE_CONDITIONS:
        raise ValueError(f"unknown CV-1 value condition: {condition}")
    before_fixed = fixed_map_digest(memory)
    source = value_bank(memory)
    if condition == "correct":
        transformed = source.clone()
    elif condition == "shuffled":
        generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
        transformed = source[torch.randperm(source.shape[0], generator=generator)]
    else:
        transformed = moment_matched_random(source, seed=random_seed)

    cursor = 0
    for _name, parameter in _value_parameters(memory):
        stop = cursor + parameter.shape[0]
        parameter.copy_(transformed[cursor:stop].to(dtype=parameter.dtype, device=parameter.device))
        cursor = stop
    if cursor != transformed.shape[0]:
        raise RuntimeError("CV-1 value-bank restoration did not consume every row")
    after_fixed = fixed_map_digest(memory)
    if before_fixed != after_fixed:
        raise RuntimeError("CV-1 value swap changed the trained gate, map, or addressing state")

    target = value_bank(memory)
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_std = (source - source_mean).square().mean(dim=0).sqrt()
    target_std = (target - target_mean).square().mean(dim=0).sqrt()
    return {
        "condition": condition,
        "rows": int(source.shape[0]),
        "width": int(source.shape[1]),
        "source_bank_digest": tensor_digest({"values": source}),
        "conditioned_bank_digest": tensor_digest({"values": target}),
        "fixed_map_digest": after_fixed,
        "maximum_mean_error": float((source_mean - target_mean).abs().max()),
        "maximum_std_error": float((source_std - target_std).abs().max()),
    }


def _load_host_memory(
    *,
    host: str,
    checkpoint: Mapping[str, Any],
    geometry: Mapping[str, Any],
    memory_slots: int,
) -> Stage2AMemorySystem:
    memory = Stage2AMemorySystem(
        arm=host,
        memory_slots=memory_slots,
        memory_keys=geometry["memory_keys"].float(),
        teacher_values=geometry["teacher_values"].float(),
        seed=0,
    )
    memory.load_state_dict(checkpoint["raw_state"])
    named = dict(memory.named_parameters())
    with torch.no_grad():
        for name, value in checkpoint["ema_state"].items():
            if name not in named:
                raise RuntimeError(f"Stage 2A EMA contains unknown parameter: {name}")
            named[name].copy_(value.to(dtype=named[name].dtype))
    return memory.eval()


def _serving_signature(row: Mapping[str, Any]) -> str:
    ignored = {
        "host_arm",
        "value_condition",
        "dose_multiplier",
        "kind",
        "memory_compatibility_gate_mean",
        "memory_retrieval_entropy_mean",
        "memory_retrieval_score_mean",
        "memory_top_k_slot_ids",
    }
    return canonical_sha256({key: value for key, value in row.items() if key not in ignored})


def summarize_rows(
    rows: Iterable[Mapping[str, Any]],
    initialization: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = list(rows)
    base_fixes = sum(not row["base_correct"] and row["augmented_correct"] for row in selected)
    base_regressions = sum(row["base_correct"] and not row["augmented_correct"] for row in selected)
    init_fixes = sum(
        not initialization[str(row["item_id"])]["augmented_correct"] and row["augmented_correct"]
        for row in selected
    )
    init_regressions = sum(
        initialization[str(row["item_id"])]["augmented_correct"] and not row["augmented_correct"]
        for row in selected
    )
    base_correct = sum(bool(row["base_correct"]) for row in selected)
    init_correct = sum(
        bool(initialization[str(row["item_id"])]["augmented_correct"]) for row in selected
    )
    observed = sum(bool(row["augmented_correct"]) for row in selected)
    return {
        "rows": len(selected),
        "observed_correct": observed,
        "base": {
            "correct": base_correct,
            "delta_rows": observed - base_correct,
            "fixes": base_fixes,
            "regressions": base_regressions,
            "paired_sign_test_p_two_sided": exact_sign_test(base_fixes, base_regressions),
        },
        "initialization": {
            "correct": init_correct,
            "delta_rows": observed - init_correct,
            "fixes": init_fixes,
            "regressions": init_regressions,
            "paired_sign_test_p_two_sided": exact_sign_test(init_fixes, init_regressions),
        },
    }


def _score_cell(
    *,
    host: str,
    condition: str,
    dose: float,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    geometry: Mapping[str, Any],
    memory_slots: int,
    graph_inputs: Mapping[str, Any],
    tokenizer: Any,
    panel: list[dict[str, Any]],
    base: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    shuffle_seed: int,
    random_seed: int,
    mcq_batch_size: int,
    generation_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_path = output_dir / "dev_rows.jsonl"
    cell_path = output_dir / "physical_summary.json"
    if rows_path.is_file() and cell_path.is_file():
        cached = read_json(cell_path)
        if (
            cached.get("status") == "complete_dev_only"
            and cached.get("host_arm") == host
            and cached.get("value_condition") == condition
            and abs(float(cached.get("dose_multiplier", -1)) - float(dose)) <= 1e-12
            and cached.get("checkpoint_sha256") == sha256_file(checkpoint_path)
            and cached.get("dev_rows_sha256") == sha256_file(rows_path)
        ):
            return read_jsonl(rows_path), cached

    memory = _load_host_memory(
        host=host,
        checkpoint=checkpoint,
        geometry=geometry,
        memory_slots=memory_slots,
    )
    transform_audit = apply_value_condition(
        memory,
        condition,
        shuffle_seed=shuffle_seed,
        random_seed=random_seed,
    )
    memory.to("cuda").eval()
    graph = P34TaskInferenceGraph(
        base_model=graph_inputs["model"],
        sidecar=graph_inputs["sidecar"],
        flow_loops=4,
        stage2a_memory_system=memory,
        stage2a_geometry=geometry,
        stage2a_amplitude=0.05,
        stage2a_value_scale=dose,
        stage2a_diagnostic_value_scale_authorized=True,
    )
    mcq = [row for row in panel if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}]
    generated = [row for row in panel if row["battery"] in {"gsm8k", "mbpp", "tier1"}]
    with torch.inference_mode():
        scored = score_mcq(graph, tokenizer, mcq, batch_size=mcq_batch_size)
        scored.extend(
            score_generation(
                graph,
                tokenizer,
                generated,
                batch_size=generation_batch_size,
            )
        )
    by_id = {str(row["item_id"]): row for row in scored}
    if set(by_id) != set(base):
        raise RuntimeError("CV-1 DEV output coverage changed")
    source = {str(row["item_id"]): row for row in panel}
    rows = [
        {
            "kind": "paper2_stage2a_cv1_dev_row_v1",
            "host_arm": host,
            "value_condition": condition,
            "dose_multiplier": dose,
            "partition": "dev",
            "item_id": item_id,
            "battery": source[item_id]["battery"],
            "battery_role": source[item_id]["battery_role"],
            "base_correct": bool(base[item_id]["correct"]),
            **by_id[item_id],
        }
        for item_id in sorted(by_id)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows_path, rows)
    physical = {
        "kind": "paper2_stage2a_cv1_physical_cell_v1",
        "status": "complete_dev_only",
        "host_arm": host,
        "value_condition": condition,
        "dose_multiplier": dose,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "value_transform_audit": transform_audit,
        "dev_rows_sha256": sha256_file(rows_path),
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(cell_path, physical)
    return rows, physical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--t3a_checkpoint", type=Path, required=True)
    parser.add_argument("--t3b_checkpoint", type=Path, required=True)
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
    parser.add_argument("--mcq_batch_size", type=int, default=32)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    args = parser.parse_args()

    spec = read_json(args.spec)
    lock = read_json(args.lock)
    assert_stage2a_training_authorized(lock)
    if spec.get("kind") != "paper2_stage2a_cv1_d5_score_only_spec_v1":
        raise RuntimeError("CV-1 score-only specification identity changed")
    if spec["boundaries"] != {
        "confirm_scored": False,
        "eval_e_scored": False,
        "optimizer_constructed": False,
        "partition": "dev",
        "training_authorized": False,
    }:
        raise RuntimeError("CV-1 score-only boundaries changed")
    panel = read_jsonl(args.panel)
    if len(panel) != 1_024 or any(str(row.get("partition")) != "dev" for row in panel):
        raise RuntimeError("CV-1 requires the frozen 1,024-row DEV panel")
    if sha256_file(args.panel) != lock["data_separation"]["panel_manifest_sha256"]:
        raise RuntimeError("CV-1 DEV panel SHA changed")
    base = {str(row["item_id"]): row for row in read_jsonl(args.base_scores)}
    if set(base) != {str(row["item_id"]) for row in panel}:
        raise RuntimeError("CV-1 base-score coverage changed")
    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    if sha256_file(args.geometry) != lock["geometry_fit"]["artifact_sha256"]:
        raise RuntimeError("CV-1 geometry SHA changed")

    checkpoint_paths = {"t3a": args.t3a_checkpoint, "t3b": args.t3b_checkpoint}
    checkpoints = {}
    for host, path in checkpoint_paths.items():
        observed = sha256_file(path)
        if observed != spec["checkpoint_sha256"][host]:
            raise RuntimeError(f"CV-1 {host} checkpoint SHA changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("arm") != host or int(checkpoint.get("seed", -1)) != 0:
            raise RuntimeError(f"CV-1 {host} endpoint identity changed")
        if int(checkpoint.get("step", -1)) != 1_200:
            raise RuntimeError(f"CV-1 {host} endpoint is not step 1200")
        checkpoints[host] = checkpoint

    model_spec = MODEL_SPECS["base"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model"], revision=model_spec["revision"], cache_dir=args.model_cache
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["model"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
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
        control_reader="mean",
    )
    sidecar.bridge.set_gate_ceiling(0.02)
    sidecar.to("cuda").eval()
    frozen_before = frozen_sidecar_digest(sidecar)
    graph_inputs = {"model": model, "sidecar": sidecar}

    physical: dict[str, dict[str, Any]] = {}
    physical_rows: dict[str, list[dict[str, Any]]] = {}
    for host in ("t3a", "t3b"):
        key = f"{host}__correct__dose_0"
        rows, receipt = _score_cell(
            host=host,
            condition="correct",
            dose=0.0,
            checkpoint=checkpoints[host],
            checkpoint_path=checkpoint_paths[host],
            geometry=geometry,
            memory_slots=int(lock["data_separation"]["memory_slots"]),
            graph_inputs=graph_inputs,
            tokenizer=tokenizer,
            panel=panel,
            base=base,
            output_dir=args.output_dir / "cells" / key,
            shuffle_seed=int(spec["shuffle_seed"]),
            random_seed=int(spec["random_seed"]),
            mcq_batch_size=args.mcq_batch_size,
            generation_batch_size=args.generation_batch_size,
        )
        physical[key] = receipt
        physical_rows[key] = rows

    first_zero = {str(row["item_id"]): _serving_signature(row) for row in physical_rows["t3a__correct__dose_0"]}
    second_zero = {str(row["item_id"]): _serving_signature(row) for row in physical_rows["t3b__correct__dose_0"]}
    if first_zero != second_zero:
        raise RuntimeError("CV-1 dose-zero host outputs are not bit exact")
    initialization = {
        str(row["item_id"]): row for row in physical_rows["t3a__correct__dose_0"]
    }

    for host in ("t3a", "t3b"):
        for condition in VALUE_CONDITIONS:
            for dose in (0.5, 1.0):
                key = f"{host}__{condition}__dose_{str(dose).replace('.', 'p')}"
                rows, receipt = _score_cell(
                    host=host,
                    condition=condition,
                    dose=dose,
                    checkpoint=checkpoints[host],
                    checkpoint_path=checkpoint_paths[host],
                    geometry=geometry,
                    memory_slots=int(lock["data_separation"]["memory_slots"]),
                    graph_inputs=graph_inputs,
                    tokenizer=tokenizer,
                    panel=panel,
                    base=base,
                    output_dir=args.output_dir / "cells" / key,
                    shuffle_seed=int(spec["shuffle_seed"]),
                    random_seed=int(spec["random_seed"]),
                    mcq_batch_size=args.mcq_batch_size,
                    generation_batch_size=args.generation_batch_size,
                )
                physical[key] = receipt
                physical_rows[key] = rows

    if frozen_sidecar_digest(sidecar) != frozen_before:
        raise RuntimeError("CV-1 changed the frozen sidecar")

    logical_cells: dict[str, Any] = {}
    for host in ("t3a", "t3b"):
        zero_rows = physical_rows[f"{host}__correct__dose_0"]
        for condition in VALUE_CONDITIONS:
            for dose in DOSE_MULTIPLIERS:
                physical_key = (
                    f"{host}__correct__dose_0"
                    if dose == 0.0
                    else f"{host}__{condition}__dose_{str(dose).replace('.', 'p')}"
                )
                rows = zero_rows if dose == 0.0 else physical_rows[physical_key]
                by_battery: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    by_battery[str(row["battery"])].append(row)
                logical_key = f"{host}__{condition}__dose_{str(dose).replace('.', 'p')}"
                logical_cells[logical_key] = {
                    "host_arm": host,
                    "value_condition": condition,
                    "dose_multiplier": dose,
                    "physical_cell": physical_key,
                    "reused_zero_dose_identity": dose == 0.0 and condition != "correct",
                    "pooled": summarize_rows(rows, initialization),
                    "by_battery": {
                        battery: summarize_rows(selected, initialization)
                        for battery, selected in sorted(by_battery.items())
                    },
                }

    aggregate = {
        "kind": "paper2_stage2a_cv1_summary_v1",
        "status": "complete_dev_score_only",
        "spec_sha256": sha256_file(args.spec),
        "panel_sha256": sha256_file(args.panel),
        "geometry_sha256": sha256_file(args.geometry),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "dose_zero_host_outputs_bit_exact": True,
        "initialization_correct": sum(
            bool(row["augmented_correct"]) for row in initialization.values()
        ),
        "base_correct": sum(bool(row["correct"]) for row in base.values()),
        "logical_cells": logical_cells,
        "physical_cells": physical,
        "checkpoint_receipts": checkpoint_receipts,
        "frozen_sidecar_digest": frozen_before,
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(args.output_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
