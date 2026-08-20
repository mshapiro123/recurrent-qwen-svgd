"""Run the signed, score-only Stage 2B-A stop autopsy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import score_generation, score_mcq
from eval.eval_paper2_stage2b_campaign import (
    Stage2BTaskInferenceGraph,
    _forced_target,
    read_jsonl,
    score_dev2_margins,
    write_jsonl,
)
from training.paper2_stage2b_autopsy import (
    battery_counts,
    discrete_mutual_information,
    load_and_validate_autopsy_lock,
    margin_correlation_receipt,
    normalized_gram_eigengap,
    sha256_file,
    spherical_kmeans,
    stable_dev2_subsample,
)
from training.paper2_stage2b_depth import (
    monotonicity_hinge,
    sparse_forward_kl_per_example,
)
from training.run_paper2_stage2b_depth import (
    _apply_named_state,
    _build_model,
    _named_trainable_state,
)
from training.run_paper2_phase3_p33 import tensor_digest


RUN_KIND = "paper2_stage2b_autopsy_v1"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_state(path: Path, *, expected_sha256: str | None = None) -> dict[str, torch.Tensor]:
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Stage 2B-A checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state", payload.get("ema_state"))
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Stage 2B-A checkpoint lacks an EMA state: {path}")
    return {str(name): value.detach().cpu() for name, value in state.items()}


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_digest({name: value for name, value in state.items()})


def _apply_state(wrapper: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    _apply_named_state(wrapper, state)
    wrapper.eval()


def _condition_name(prefix: str, *, gamma: float | None = None, mode: str | None = None) -> str:
    parts = [prefix]
    if gamma is not None:
        parts.append(f"gamma_{gamma:.2f}".replace(".", "p"))
    if mode is not None:
        parts.append(mode)
    return "__".join(parts)


def _score_dev1_condition(
    *,
    wrapper: Any,
    tokenizer: Any,
    panel: Sequence[Mapping[str, Any]],
    base_rows: Mapping[str, Mapping[str, Any]],
    initialization_rows: Mapping[str, Mapping[str, Any]],
    seed: int,
    gamma: float,
    mode: str,
    condition: str,
    private_dir: Path,
    mcq_batch_size: int,
    generation_batch_size: int,
    precomputed_rows: Sequence[Mapping[str, Any]] | None = None,
    precomputed_source: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_path = private_dir / f"dev1__{condition}.jsonl"
    partial_path = private_dir / f"dev1__{condition}.partial.jsonl"
    source = {str(row["item_id"]): row for row in panel}
    expected_ids = list(source)

    def enrich(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for row in scored:
            item_id = str(row["item_id"])
            if item_id not in source or item_id not in base_rows or item_id not in initialization_rows:
                raise RuntimeError("Stage 2B-A DEV-1 comparator coverage is incomplete")
            enriched.append(
                {
                    "kind": "paper2_stage2b_dev1_row_v1",
                    "seed": seed,
                    "look": 1000,
                    "item_id": item_id,
                    "battery": str(source[item_id]["battery"]),
                    "current_correct": bool(row["augmented_correct"]),
                    "base_correct": bool(
                        base_rows[item_id].get(
                            "correct", base_rows[item_id].get("augmented_correct")
                        )
                    ),
                    "initialization_correct": bool(
                        initialization_rows[item_id].get(
                            "augmented_correct", initialization_rows[item_id].get("correct")
                        )
                    ),
                    **dict(row),
                    "autopsy_condition": condition,
                }
            )
        return enriched

    def validate_cached(rows: Sequence[Mapping[str, Any]], *, complete: bool) -> None:
        ids = [str(row["item_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Stage 2B-A duplicate resumable DEV-1 rows: {condition}")
        if not set(ids).issubset(source):
            raise RuntimeError(f"Stage 2B-A foreign resumable DEV-1 rows: {condition}")
        if complete and set(ids) != set(expected_ids):
            raise RuntimeError(f"Stage 2B-A complete DEV-1 cache is incomplete: {condition}")
        for row in rows:
            if (
                int(row.get("seed", -1)) != seed
                or int(row.get("look", -1)) != 1000
                or str(row.get("autopsy_condition")) != condition
            ):
                raise RuntimeError(f"Stage 2B-A resumable DEV-1 metadata changed: {condition}")

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_battery: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_battery[str(row["battery"])].append(row)
        battery = {}
        for name, selected in sorted(by_battery.items()):
            current = sum(bool(row["current_correct"]) for row in selected)
            base = sum(bool(row["base_correct"]) for row in selected)
            initialization = sum(bool(row["initialization_correct"]) for row in selected)
            battery[name] = {
                "rows": len(selected),
                "current_correct": current,
                "base_correct": base,
                "initialization_correct": initialization,
                "delta_vs_base_rows": current - base,
                "delta_vs_initialization_rows": current - initialization,
            }
        tier1 = int(battery["tier1"]["current_correct"])
        gsm8k = int(battery["gsm8k"]["current_correct"])
        tier1_floor = 19
        gsm8k_floor = 91 if seed == 0 else 94
        summary = {
            "kind": "paper2_stage2b_dev1_summary_v1",
            "seed": seed,
            "look": 1000,
            "rows": len(rows),
            "battery": battery,
            "safety": {
                "tier1_floor": tier1_floor,
                "tier1_correct": tier1,
                "tier1_pass": tier1 >= tier1_floor,
                "gsm8k_floor": gsm8k_floor,
                "gsm8k_correct": gsm8k,
                "gsm8k_pass": gsm8k >= gsm8k_floor,
            },
            "both_comparators_reported": True,
            "autopsy_condition": condition,
            "gamma": gamma,
            "diagnostic_mode": mode,
        }
        summary["safety"]["pass"] = bool(
            summary["safety"]["tier1_pass"] and summary["safety"]["gsm8k_pass"]
        )
        return summary

    if final_path.exists():
        rows = read_jsonl(final_path)
        validate_cached(rows, complete=True)
        return rows, summarize(rows)

    if precomputed_rows is not None:
        rows = enrich(precomputed_rows)
        validate_cached(rows, complete=True)
        summary = summarize(rows)
        summary["reused_precomputed_rows"] = dict(precomputed_source or {})
        write_jsonl(final_path, rows)
        partial_path.unlink(missing_ok=True)
        return rows, summary

    rows = read_jsonl(partial_path) if partial_path.exists() else []
    validate_cached(rows, complete=False)
    row_by_id = {str(row["item_id"]): dict(row) for row in rows}

    graph = Stage2BTaskInferenceGraph(
        wrapper=wrapper,
        stage="M2",
        amplitude=gamma,
        flow_loops=4,
        diagnostic_mode=mode,
    )
    mcq = [
        row
        for row in panel
        if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}
        and str(row["item_id"]) not in row_by_id
    ]
    if mcq:
        for row in enrich(score_mcq(graph, tokenizer, mcq, batch_size=mcq_batch_size)):
            row_by_id[str(row["item_id"])] = row
        write_jsonl(partial_path, [row_by_id[item_id] for item_id in expected_ids if item_id in row_by_id])

    generation = [
        row
        for row in panel
        if row["battery"] in {"gsm8k", "mbpp", "tier1"}
        and str(row["item_id"]) not in row_by_id
    ]
    rows_since_persist = 0

    def emit_batch(scored: list[dict[str, Any]]) -> None:
        nonlocal rows_since_persist
        for row in enrich(scored):
            item_id = str(row["item_id"])
            if item_id in row_by_id:
                raise RuntimeError(f"Stage 2B-A duplicate generated row during resume: {item_id}")
            row_by_id[item_id] = row
        rows_since_persist += len(scored)
        if rows_since_persist >= 8:
            ordered = [row_by_id[item_id] for item_id in expected_ids if item_id in row_by_id]
            write_jsonl(partial_path, ordered)
            print(
                f"stage2b_dev1_resume condition={condition} rows={len(ordered)}/{len(expected_ids)}",
                flush=True,
            )
            rows_since_persist = 0

    if generation:
        score_generation(
            graph,
            tokenizer,
            generation,
            batch_size=generation_batch_size,
            emit_batch=emit_batch,
        )

    rows = [row_by_id[item_id] for item_id in expected_ids if item_id in row_by_id]
    validate_cached(rows, complete=True)
    summary = summarize(rows)
    write_jsonl(final_path, rows)
    partial_path.unlink(missing_ok=True)
    return rows, summary


def _score_dev2_condition(
    *,
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    gamma: float,
    mode: str,
    condition: str,
    private_dir: Path,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = Stage2BTaskInferenceGraph(
        wrapper=wrapper,
        stage="M2",
        amplitude=gamma,
        flow_loops=4,
        diagnostic_mode=mode,
    )
    scored, summary = score_dev2_margins(
        graph=graph,
        tokenizer=tokenizer,
        rows=rows,
        seed=seed,
        look=1000,
        batch_size=batch_size,
    )
    for row in scored:
        row["autopsy_condition"] = condition
    summary["autopsy_condition"] = condition
    summary["gamma"] = gamma
    summary["diagnostic_mode"] = mode
    write_jsonl(private_dir / f"dev2__{condition}.jsonl", scored)
    return scored, summary


def _same_predictions(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    left_rows = {str(row["item_id"]): row for row in left}
    right_rows = {str(row["item_id"]): row for row in right}
    if set(left_rows) != set(right_rows):
        return False
    keys = ("prediction", "augmented_correct", "generated_token_ids")
    return all(
        all(left_rows[item].get(key) == right_rows[item].get(key) for key in keys)
        for item in left_rows
    )


@torch.inference_mode()
def _component_pass_one_receipt(
    wrapper: Any, tokenizer: Any, row: Mapping[str, Any]
) -> dict[str, Any]:
    prompt, target = _forced_target(tokenizer, row)
    tokens = torch.tensor([prompt + target], dtype=torch.long, device="cuda")
    attention = torch.ones_like(tokens)
    logits = {}
    for mode in ("standard", "constitutive_off", "fresh_state_each_loop", "inherited_flow_off"):
        output = wrapper(
            input_ids=tokens,
            attention_mask=attention,
            max_loops=1,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.05,
            stage2b_diagnostic_mode=mode,
            return_loop_logits=True,
            use_cache=False,
            return_dict=True,
        )
        logits[mode] = output.loop_logits.detach().cpu()
    standard = logits["standard"]
    cells = {}
    for mode, value in logits.items():
        cells[mode] = {
            "bit_exact_against_standard": torch.equal(value, standard),
            "max_abs_difference": float((value.float() - standard.float()).abs().max()),
        }
        if not cells[mode]["bit_exact_against_standard"]:
            raise RuntimeError(f"Stage 2B-A component pass-one identity failed: {mode}")
    return {"row_id": str(row["item_id"]), "cells": cells, "all_pass": True}


def _component_activation_receipt(component: Mapping[str, Any]) -> dict[str, Any]:
    metric = {
        "constitutive_off": "stage2b_constitutive_update_loop_{loop}_max_abs",
        "fresh_state_each_loop": "stage2b_carry_contribution_loop_{loop}_max_abs",
        "inherited_flow_off": "stage2b_flow_update_loop_{loop}_max_abs",
    }
    cells = {}
    for mode, template in metric.items():
        maxima = component[mode]["dev2"]["activation_maxima"]
        values = {str(loop): float(maxima[template.format(loop=loop)]) for loop in range(1, 5)}
        exact = all(value == 0.0 for value in values.values())
        if not exact:
            raise RuntimeError(f"Stage 2B-A disabled component remained active: {mode} {values}")
        cells[mode] = {"metric": template, "per_loop_max_abs": values, "exact_zero": True}
    return {"cells": cells, "all_pass": True}


def _unit(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-12)


def _extract_correction_field(
    *,
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    """Read per-row local correction and deployed-write directions without mutation."""

    state_before = _state_digest(_named_trainable_state(wrapper))
    versions_before = {name: parameter._version for name, parameter in wrapper.named_parameters()}
    corrections = {loop: [] for loop in (2, 3, 4)}
    writes = {loop: [] for loop in (2, 3, 4)}
    batteries = []
    item_ids = []
    zero_correction_rows = {loop: 0 for loop in (2, 3, 4)}
    prepared = [(row, *_forced_target(tokenizer, row)) for row in rows]
    bridge = wrapper.stage2b_depth_attachment.bridge
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        width = max(len(prompt) + len(target) for _row, prompt, target in batch)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        attention = torch.zeros_like(input_ids)
        spans = []
        for index, (row, prompt, target) in enumerate(batch):
            tokens = prompt + target
            input_ids[index, : len(tokens)] = torch.tensor(tokens, device="cuda")
            attention[index, : len(tokens)] = 1
            spans.append((len(prompt) - 1, target))
            batteries.append(str(row["battery"]))
            item_ids.append(str(row["item_id"]))

        captured = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            output.hidden.retain_grad()
            captured.append(output)

        wrapper.zero_grad(set_to_none=True)
        handle = bridge.register_forward_hook(capture)
        try:
            output = wrapper(
                input_ids=input_ids,
                attention_mask=attention,
                max_loops=4,
                stage2b_depth_enabled=True,
                stage2b_stage="M2",
                stage2b_amplitude=0.05,
                stage2b_diagnostic_mode="standard",
                return_loop_logits=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        if len(captured) != 3:
            raise RuntimeError(f"Stage 2B-A expected three bridge reentries, observed {len(captured)}")
        final_logits = output.loop_logits[:, 0, -1].float()
        losses = []
        for index, (first, targets) in enumerate(spans):
            target_ids = torch.tensor(targets, device="cuda")
            losses.append(
                F.cross_entropy(final_logits[index, first : first + len(targets)], target_ids)
            )
        torch.stack(losses).sum().backward()
        writable = attention.bool()
        writable[:, 0] = False
        for loop, bridge_output in zip((2, 3, 4), captured):
            if bridge_output.hidden.grad is None or bridge_output.position_gate is None:
                raise RuntimeError("Stage 2B-A correction-field hook did not receive gradients")
            writeback = bridge_output.delta * bridge_output.position_gate
            for index in range(len(batch)):
                mask = writable[index]
                correction = -bridge_output.hidden.grad[index, mask].float().mean(dim=0)
                write = writeback[index, mask].detach().float().mean(dim=0)
                if float(correction.norm()) == 0.0:
                    zero_correction_rows[loop] += 1
                corrections[loop].append(_unit(correction).cpu())
                writes[loop].append(_unit(write).cpu())
        wrapper.zero_grad(set_to_none=True)
        print(
            f"stage2b_arm6_progress rows={min(start + batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )
    versions_after = {name: parameter._version for name, parameter in wrapper.named_parameters()}
    state_after = _state_digest(_named_trainable_state(wrapper))
    if versions_after != versions_before or state_after != state_before:
        raise RuntimeError("Stage 2B-A Arm 6 mutated model parameters")
    return {
        "corrections": {loop: torch.stack(values) for loop, values in corrections.items()},
        "writes": {loop: torch.stack(values) for loop, values in writes.items()},
        "batteries": batteries,
        "item_ids": item_ids,
        "zero_correction_rows": zero_correction_rows,
        "parameter_state_digest_before": state_before,
        "parameter_state_digest_after": state_after,
        "parameter_versions_unchanged": True,
    }


def _empirical_upper_p(observed: float, null: Sequence[float]) -> float:
    return (1.0 + sum(value >= observed for value in null)) / (len(null) + 1.0)


def _clusterability_receipt(
    directions: torch.Tensor, batteries: Sequence[str], *, seed: int
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = _unit(directions.to(device))
    selected = None
    by_k = {}
    for clusters in range(2, 9):
        labels, silhouette = spherical_kmeans(
            x, clusters=clusters, restarts=8, iterations=50, seed=seed + clusters
        )
        by_k[str(clusters)] = silhouette
        if selected is None or silhouette > selected[2]:
            selected = (clusters, labels, silhouette)
    assert selected is not None
    clusters, labels, silhouette = selected
    eigengap = normalized_gram_eigengap(x, max_rank=8)

    null_silhouettes = []
    null_eigengaps = []
    generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    for replicate in range(128):
        null = _unit(torch.randn(x.shape, generator=generator, device=device))
        maximum = -float("inf")
        for null_clusters in range(2, 9):
            _labels, value = spherical_kmeans(
                null,
                clusters=null_clusters,
                restarts=8,
                iterations=50,
                seed=seed + 100_000 + replicate * 10 + null_clusters,
            )
            maximum = max(maximum, value)
        null_silhouettes.append(maximum)
        null_eigengaps.append(normalized_gram_eigengap(null, max_rank=8)["maximum"])

    permutation = torch.randperm(x.shape[1], generator=generator, device=device)
    signs = torch.where(
        torch.rand((x.shape[1],), generator=generator, device=device) < 0.5,
        x.new_tensor(-1.0),
        x.new_tensor(1.0),
    )
    transformed = x[:, permutation] * signs
    gram_invariance = float(((x @ x.T) - (transformed @ transformed.T)).abs().max().cpu())

    label_list = [int(value) for value in labels.cpu().tolist()]
    association = discrete_mutual_information(label_list, list(batteries))
    rng = random.Random(seed + 20_000)
    permuted_mi = []
    shuffled = list(batteries)
    for _ in range(4096):
        rng.shuffle(shuffled)
        permuted_mi.append(discrete_mutual_information(label_list, shuffled)["nats"])
    association["permutation_replicates"] = 4096
    association["upper_tail_p"] = _empirical_upper_p(association["nats"], permuted_mi)
    return {
        "rows": int(x.shape[0]),
        "dimensions": int(x.shape[1]),
        "spherical_kmeans": {
            "candidate_clusters": list(range(2, 9)),
            "silhouette_by_k": by_k,
            "selected_clusters": clusters,
            "selected_silhouette": silhouette,
            "isotropic_row_direction_null_replicates": 128,
            "null_mean": sum(null_silhouettes) / len(null_silhouettes),
            "upper_tail_p": _empirical_upper_p(silhouette, null_silhouettes),
        },
        "gram_eigengap": {
            **eigengap,
            "isotropic_row_direction_null_replicates": 128,
            "null_mean": sum(null_eigengaps) / len(null_eigengaps),
            "upper_tail_p": _empirical_upper_p(eigengap["maximum"], null_eigengaps),
        },
        "cluster_battery_mutual_information": association,
        "shared_signed_permutation_gram_max_abs_difference": gram_invariance,
        "null_semantics": {
            "clusterability": "independent isotropic row directions preserving unit row norms",
            "battery_association": "battery-label permutation with cluster labels fixed",
            "shared_signed_permutation": "Gram-invariance pipeline sanity check, not a null",
        },
    }


@torch.inference_mode()
def _zero_write_logits(wrapper: Any, tokenizer: Any, row: Mapping[str, Any]) -> torch.Tensor:
    prompt, target = _forced_target(tokenizer, row)
    tokens = torch.tensor([prompt + target], dtype=torch.long, device="cuda")
    attention = torch.ones_like(tokens)
    output = wrapper(
        input_ids=tokens,
        attention_mask=attention,
        max_loops=4,
        stage2b_depth_enabled=True,
        stage2b_stage="M2",
        stage2b_amplitude=0.0,
        stage2b_diagnostic_mode="zero_write",
        return_loop_logits=True,
        use_cache=False,
        return_dict=True,
    )
    return output.loop_logits.detach().float().cpu()


@torch.inference_mode()
def _sparse_loop_projection_receipt(
    wrapper: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Prove that skipping unused loop projections preserves the registered read."""

    prepared = [(row, *_forced_target(tokenizer, row)) for row in rows]
    width = max(len(prompt) + len(target) for _row, prompt, target in prepared)
    input_ids = torch.zeros((len(prepared), width), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for index, (_row, prompt, target) in enumerate(prepared):
        tokens = prompt + target
        input_ids[index, : len(tokens)] = torch.tensor(tokens, device="cuda")
        attention[index, : len(tokens)] = 1

    common = {
        "wrapper": wrapper,
        "stage": "M2",
        "amplitude": 0.05,
        "flow_loops": 4,
        "diagnostic_mode": "standard",
    }
    full = Stage2BTaskInferenceGraph(
        **common,
        last_token_projection=False,
        sparse_loop_projection=False,
    ).next_token(
        input_ids=input_ids, attention_mask=attention
    )
    sparse = Stage2BTaskInferenceGraph(
        **common,
        last_token_projection=False,
        sparse_loop_projection=True,
    ).next_token(
        input_ids=input_ids, attention_mask=attention
    )
    cells = {}
    for name in ("augmented_logits", "base_logits", "position_gate", "writeback_ratio"):
        left = getattr(full, name).detach().float().cpu()
        right = getattr(sparse, name).detach().float().cpu()
        cells[name] = {
            "bit_exact": torch.equal(left, right),
            "max_abs_difference": float((left - right).abs().max()),
        }
    all_pass = all(cell["bit_exact"] for cell in cells.values())
    return {
        "rows": len(rows),
        "cells": cells,
        "all_pass": all_pass,
        "optimization": "project full-sequence LM-head logits on loops 1 and 4 only",
        "estimator_change": False,
    }


def _offdiagonal_cosine(states: torch.Tensor, *, center: bool) -> dict[str, float]:
    values = states.float()
    if center:
        values = values - values.mean(dim=0, keepdim=True)
    values = F.normalize(values, dim=-1, eps=1e-12)
    matrix = values @ values.transpose(0, 1)
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    off = matrix[mask]
    return {
        "mean": float(off.mean()),
        "median": float(off.median()),
        "p95": float(torch.quantile(off, 0.95)),
    }


@torch.inference_mode()
def _state_similarity(
    *, wrapper: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], batch_size: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    prepared = [(row, *_forced_target(tokenizer, row)) for row in rows]
    loop1_states = []
    loop4_states = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        widths = [len(prompt) + len(target) for _row, prompt, target in batch]
        width = max(widths)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        attention = torch.zeros_like(input_ids)
        for index, (_row, prompt, target) in enumerate(batch):
            tokens = prompt + target
            input_ids[index, : len(tokens)] = torch.tensor(tokens, device="cuda")
            attention[index, : len(tokens)] = 1
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.05,
            stage2b_diagnostic_mode="standard",
            return_loop_recurrent_states=True,
            use_cache=False,
            return_dict=True,
        )
        states = output.loop_recurrent_states[:, 0].float()
        weights = attention.float().unsqueeze(1).unsqueeze(-1)
        pooled = (states * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        loop1_states.append(pooled[:, 0].cpu())
        loop4_states.append(pooled[:, 3].cpu())
    loop1 = torch.cat(loop1_states)
    loop4 = torch.cat(loop4_states)
    direction = loop4 - loop1
    receipt = {
        "rows": len(rows),
        "raw_loop4_offdiagonal_cosine": _offdiagonal_cosine(loop4, center=False),
        "centered_loop4_offdiagonal_cosine": _offdiagonal_cosine(loop4, center=True),
        "loop4_minus_loop1_direction_offdiagonal_cosine": _offdiagonal_cosine(
            direction, center=False
        ),
        "loop4_state_variance": float(loop4.var(dim=0, unbiased=False).mean()),
        "loop1_state_variance": float(loop1.var(dim=0, unbiased=False).mean()),
    }
    return receipt, {"loop1": loop1, "loop4": loop4, "direction": direction}


@torch.inference_mode()
def _objective_read(wrapper: Any, teacher_cache: Path) -> dict[str, Any]:
    teacher = torch.load(teacher_cache, map_location="cpu", weights_only=False)
    if teacher.get("kind") != "paper2_stage2b_calibration_teacher_cache_v1":
        raise RuntimeError("wrong Stage 2B-A heldout teacher cache")
    ce_by_loop: list[list[float]] = [[] for _ in range(4)]
    kl_by_loop: list[list[float]] = [[] for _ in range(4)]
    mono = []
    for row in teacher["rows"]:
        input_ids = row["input_ids"].long().unsqueeze(0).to("cuda")
        attention = torch.ones_like(input_ids)
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.05,
            return_loop_logits=True,
            use_cache=False,
            return_dict=True,
        )
        loops = output.loop_logits[:, 0, :, :-1]
        top_ids = row["teacher_topk_token_ids"].long().unsqueeze(0).to("cuda")
        top_logits = row["teacher_topk_logits"].unsqueeze(0).to("cuda")
        targets = top_ids[..., 0]
        mask = torch.ones(targets.shape, dtype=torch.bool, device="cuda")
        row_kls = []
        for index in range(4):
            kl = sparse_forward_kl_per_example(
                loops[:, index], top_ids, top_logits, mask
            )
            ce = F.cross_entropy(
                loops[:, index].float().reshape(-1, loops.shape[-1]),
                targets.reshape(-1),
            )
            kl_by_loop[index].append(float(kl.mean().cpu()))
            ce_by_loop[index].append(float(ce.cpu()))
            row_kls.append(kl)
        mono.append(float(monotonicity_hinge(row_kls, delta=0.01).mean().cpu()))
    mean = lambda values: sum(values) / len(values)
    return {
        "rows": len(teacher["rows"]),
        "next_token_positions": sum(
            int(row["teacher_topk_token_ids"].shape[0]) for row in teacher["rows"]
        ),
        "per_loop_ce": [mean(values) for values in ce_by_loop],
        "per_loop_forward_kl": [mean(values) for values in kl_by_loop],
        "monotonicity_component": mean(mono),
        "teacher_manifest_sha256": teacher["manifest_sha256"],
    }


def _k_sweep(
    *,
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    condition: str,
    private_dir: Path,
    batch_size: int,
    precomputed_k4: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = {}
    for loops in (1, 2, 3, 4):
        if loops == 4 and precomputed_k4 is not None:
            expected_ids = {str(row["item_id"]) for row in rows}
            scored = [
                dict(row)
                for row in precomputed_k4
                if str(row["item_id"]) in expected_ids
            ]
            if {str(row["item_id"]) for row in scored} != expected_ids:
                raise RuntimeError("Stage 2B-A reused K=4 row coverage changed")
        else:
            graph = Stage2BTaskInferenceGraph(
                wrapper=wrapper,
                stage="M2",
                amplitude=0.05,
                flow_loops=loops,
            )
            scored = score_generation(graph, tokenizer, rows, batch_size=batch_size)
        for row in scored:
            row["seed"] = seed
            row["flow_loops"] = loops
            row["autopsy_condition"] = condition
        write_jsonl(private_dir / f"k_sweep__{condition}__k{loops}.jsonl", scored)
        summary[str(loops)] = {
            "rows": len(scored),
            "correct": sum(bool(row["augmented_correct"]) for row in scored),
            "battery_counts": battery_counts(scored),
            "reused_identical_amplitude_cell": loops == 4 and precomputed_k4 is not None,
        }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_and_validate_autopsy_lock(args.lock, require_signature=True)
    if args.dev2_subsample_manifest is None:
        raise RuntimeError("signed Stage 2B-A run requires a frozen DEV-2 subsample")
    if sha256_file(args.dev2_subsample_manifest) != lock["dev2_subsample"]["manifest_sha256"]:
        raise RuntimeError("Stage 2B-A DEV-2 subsample hash changed")
    if sha256_file(args.heldout_teacher_cache) != lock["heldout_training_slice"]["teacher_cache_sha256"]:
        raise RuntimeError("Stage 2B-A heldout teacher cache hash changed")
    random.seed(20260819 + args.seed)
    np.random.seed(20260819 + args.seed)
    torch.manual_seed(20260819 + args.seed)
    torch.cuda.manual_seed_all(20260819 + args.seed)

    wrapper, chain, _groups = _build_model(args)
    initialization_state = _named_trainable_state(wrapper)
    stop_spec = lock["stop_checkpoints"][str(args.seed)]
    stop_state = _checkpoint_state(
        args.stop_checkpoint, expected_sha256=stop_spec["sha256"]
    )
    dev1 = read_jsonl(args.dev1_panel)
    reference = {str(row["item_id"]): row for row in read_jsonl(args.reference_rows)}
    manifest = read_jsonl(args.dev2_subsample_manifest)
    dev2_subsample = [reference[str(row["item_id"])] for row in manifest]
    base_rows = {str(row["item_id"]): row for row in read_jsonl(args.base_scores)}
    initialization_rows = {
        str(row["item_id"]): row for row in read_jsonl(args.initialization_scores)
    }
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS["base"]["model"], revision=MODEL_SPECS["base"]["revision"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)

    receipt: dict[str, Any] = {
        "kind": RUN_KIND,
        "status": "running",
        "seed": args.seed,
        "lock_sha256": sha256_file(args.lock),
        "checkpoint_chain": chain,
        "state_digests": {
            "initialization": _state_digest(initialization_state),
            "stop_ema": _state_digest(stop_state),
        },
        "dev2_subsample": {
            "rows": len(dev2_subsample),
            "manifest_sha256": sha256_file(args.dev2_subsample_manifest),
            "battery_counts": battery_counts(dev2_subsample),
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "status.json", receipt)

    projection = {}
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        projection[state_name] = _sparse_loop_projection_receipt(
            wrapper, tokenizer, dev2_subsample[:6]
        )
    receipt["sparse_loop_projection_equivalence"] = {
        "cells": projection,
        "all_pass": all(cell["all_pass"] for cell in projection.values()),
    }
    atomic_json(args.output_dir / "status.json", receipt)
    if not receipt["sparse_loop_projection_equivalence"]["all_pass"]:
        raise RuntimeError("Stage 2B-A sparse-loop projection equivalence failed")

    _apply_state(wrapper, initialization_state)
    zero_init_logits = _zero_write_logits(wrapper, tokenizer, dev2_subsample[0])
    _apply_state(wrapper, stop_state)
    zero_stop_logits = _zero_write_logits(wrapper, tokenizer, dev2_subsample[0])
    zero_logit_max_abs = float((zero_init_logits - zero_stop_logits).abs().max())
    zero_logit_exact = torch.equal(zero_init_logits, zero_stop_logits)
    if not zero_logit_exact:
        raise RuntimeError(
            f"Stage 2B-A zero-write logit identity failed: max_abs={zero_logit_max_abs}"
        )

    amplitude = defaultdict(dict)
    amplitude_rows: dict[str, dict[float, list[dict[str, Any]]]] = defaultdict(dict)
    initialization_precomputed = {
        0.02: read_jsonl(args.initialization_scores_0p02),
        0.05: read_jsonl(args.initialization_scores),
    }
    initialization_sources = {
        0.02: {
            "path": str(args.initialization_scores_0p02),
            "sha256": sha256_file(args.initialization_scores_0p02),
            "checkpoint": "p35_ema_step_4400",
            "scorer_path": "registered_p35_amplitude_surface",
        },
        0.05: {
            "path": str(args.initialization_scores),
            "sha256": sha256_file(args.initialization_scores),
            "checkpoint": "p35_ema_step_4400",
            "scorer_path": "registered_p35_amplitude_surface",
        },
    }
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        for gamma in (0.0, 0.01, 0.02, 0.05):
            mode = "zero_write" if gamma == 0.0 else "standard"
            condition = _condition_name(state_name, gamma=gamma)
            rows, summary = _score_dev1_condition(
                wrapper=wrapper,
                tokenizer=tokenizer,
                panel=dev1,
                base_rows=base_rows,
                initialization_rows=initialization_rows,
                seed=args.seed,
                gamma=gamma,
                mode=mode,
                condition=condition,
                private_dir=args.private_dir,
                mcq_batch_size=args.mcq_batch_size,
                generation_batch_size=args.generation_batch_size,
                precomputed_rows=(
                    initialization_precomputed.get(gamma)
                    if state_name == "initialization"
                    else None
                ),
                precomputed_source=(
                    initialization_sources.get(gamma)
                    if state_name == "initialization"
                    else None
                ),
            )
            amplitude[state_name][str(gamma)] = summary
            amplitude_rows[state_name][gamma] = rows
    zero_identity = _same_predictions(
        amplitude_rows["initialization"][0.0], amplitude_rows["stop"][0.0]
    )
    if not zero_identity:
        raise RuntimeError("Stage 2B-A zero-write identity gate failed")
    receipt["amplitude_response"] = {
        "cells": dict(amplitude),
        "zero_write_checkpoint_independent": zero_identity,
        "zero_write_full_logit_bit_exact": zero_logit_exact,
        "zero_write_full_logit_max_abs_difference": zero_logit_max_abs,
    }
    atomic_json(args.output_dir / "status.json", receipt)

    _apply_state(wrapper, stop_state)
    component = {}
    component_pass_one = _component_pass_one_receipt(
        wrapper, tokenizer, dev2_subsample[0]
    )
    for mode in ("standard", "constitutive_off", "fresh_state_each_loop", "inherited_flow_off"):
        condition = _condition_name("stop", gamma=0.05, mode=mode)
        if mode == "standard":
            source_condition = _condition_name("stop", gamma=0.05)
            source_rows = amplitude_rows["stop"][0.05]
            copied_rows = []
            for row in source_rows:
                copied = dict(row)
                copied["autopsy_condition"] = condition
                copied_rows.append(copied)
            write_jsonl(args.private_dir / f"dev1__{condition}.jsonl", copied_rows)
            dev1_summary = copy.deepcopy(amplitude["stop"]["0.05"])
            dev1_summary.update(
                {
                    "autopsy_condition": condition,
                    "diagnostic_mode": mode,
                    "reused_identical_condition": source_condition,
                }
            )
        else:
            _rows, dev1_summary = _score_dev1_condition(
                wrapper=wrapper,
                tokenizer=tokenizer,
                panel=dev1,
                base_rows=base_rows,
                initialization_rows=initialization_rows,
                seed=args.seed,
                gamma=0.05,
                mode=mode,
                condition=condition,
                private_dir=args.private_dir,
                mcq_batch_size=args.mcq_batch_size,
                generation_batch_size=args.generation_batch_size,
            )
        _margin_rows, dev2_summary = _score_dev2_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            seed=args.seed,
            gamma=0.05,
            mode=mode,
            condition=condition,
            private_dir=args.private_dir,
            batch_size=args.margin_batch_size,
        )
        component[mode] = {"dev1": dev1_summary, "dev2": dev2_summary}
    receipt["component_attribution"] = {
        "cells": component,
        "pass_one_identity": component_pass_one,
        "disabled_component_activation": _component_activation_receipt(component),
    }
    atomic_json(args.output_dir / "status.json", receipt)

    arm6 = {}
    arm6_private = {}
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        extracted = _extract_correction_field(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            batch_size=args.margin_batch_size,
        )
        private_path = args.private_dir / f"correction_field__{state_name}.pt"
        torch.save(extracted, private_path)
        arm6_private[state_name] = extracted
        arm6[state_name] = {
            "correction_field_clusterability_loop4": _clusterability_receipt(
                extracted["corrections"][4],
                extracted["batteries"],
                seed=20260819 + args.seed + (0 if state_name == "initialization" else 1000),
            ),
            "descriptive_correction_geometry": {
                str(loop): normalized_gram_eigengap(extracted["corrections"][loop], max_rank=8)
                for loop in (2, 3, 4)
            },
            "zero_correction_rows": extracted["zero_correction_rows"],
            "parameter_state_digest_before": extracted["parameter_state_digest_before"],
            "parameter_state_digest_after": extracted["parameter_state_digest_after"],
            "parameter_versions_unchanged": extracted["parameter_versions_unchanged"],
            "private_artifact": {"path": str(private_path), "sha256": sha256_file(private_path)},
        }
    mean_field = {}
    for loop in (2, 3, 4):
        correction_mean = arm6_private["initialization"]["corrections"][loop].mean(dim=0)
        trained_write_mean = arm6_private["stop"]["writes"][loop].mean(dim=0)
        mean_field[str(loop)] = float(
            F.cosine_similarity(correction_mean, trained_write_mean, dim=0, eps=1e-12)
        )
    arm6["mean_field_confirmation"] = {
        "primary_loop": 4,
        "cosine_by_loop": mean_field,
        "primary_cosine": mean_field["4"],
        "definition": "cosine(mean normalized stop writebacks, mean normalized init correction directions)",
    }
    arm6["optimizer_constructed"] = False
    arm6["parameter_mutation"] = False
    receipt["correction_field_clusterability"] = arm6
    atomic_json(args.output_dir / "status.json", receipt)

    attractor = {}
    margin_rows_by_state = {}
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        condition = _condition_name(state_name, gamma=0.05, mode="standard")
        margin_rows, margin_summary = _score_dev2_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            seed=args.seed,
            gamma=0.05,
            mode="standard",
            condition=condition,
            private_dir=args.private_dir,
            batch_size=args.margin_batch_size,
        )
        margin_rows_by_state[state_name] = margin_rows
        state_summary, features = _state_similarity(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            batch_size=args.margin_batch_size,
        )
        feature_path = args.private_dir / f"state_features__{state_name}.pt"
        torch.save(features, feature_path)
        attractor[state_name] = {
            "margin_summary": margin_summary,
            "k1_k4_margin_correlation": margin_correlation_receipt(margin_rows),
            "state_similarity": state_summary,
            "state_feature_sha256": sha256_file(feature_path),
        }
    generative = [row for row in dev1 if row["battery"] in {"gsm8k", "mbpp", "tier1"}]
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        attractor[state_name]["generative_k_sweep"] = _k_sweep(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=generative,
            seed=args.seed,
            condition=state_name,
            private_dir=args.private_dir,
            batch_size=args.generation_batch_size,
            precomputed_k4=amplitude_rows[state_name][0.05],
        )
    receipt["attractor_discriminators"] = attractor
    atomic_json(args.output_dir / "status.json", receipt)

    objective = {}
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        objective[state_name] = _objective_read(wrapper, args.heldout_teacher_cache)
    receipt["objective_task_divergence"] = objective

    onset = {}
    for step, state in ((0, initialization_state), (1000, stop_state)):
        _apply_state(wrapper, state)
        condition = f"onset_step_{step:05d}"
        _rows, summary = _score_dev2_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            seed=args.seed,
            gamma=0.05,
            mode="standard",
            condition=condition,
            private_dir=args.private_dir,
            batch_size=args.margin_batch_size,
        )
        onset[str(step)] = summary
    if set(map(int, onset)) != set(lock["onset_trajectory"]["steps"]):
        raise RuntimeError("Stage 2B-A onset checkpoint set changed")
    training_summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    if int(training_summary.get("seed", -1)) != args.seed:
        raise RuntimeError("Stage 2B-A contemporaneous training summary seed changed")
    receipt["onset_trajectory"] = {
        "checkpointed_score_endpoints": onset,
        "contemporaneous_training_telemetry": training_summary.get("history", []),
        "telemetry_role": "training_process_telemetry_not_checkpointed_score_trajectory",
        "claim_boundary": lock["onset_trajectory"]["claim_boundary"],
    }
    receipt["runtime"] = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "weights_dtype": "bfloat16",
        "attention_backend": "sdpa",
    }
    receipt["status"] = "complete_score_only"
    atomic_json(args.output_dir / "summary.json", receipt)
    atomic_json(args.output_dir / "status.json", receipt)
    return receipt


def freeze_dev2_subsample(args: argparse.Namespace) -> dict[str, Any]:
    source = read_jsonl(args.dev2_manifest)
    selected = stable_dev2_subsample(source, size=256, seed=20260819)
    manifest = [
        {"item_id": str(row["item_id"]), "battery": str(row["battery"])}
        for row in selected
    ]
    write_jsonl(args.dev2_subsample_manifest, manifest)
    receipt = {
        "kind": "paper2_stage2b_autopsy_dev2_subsample_v1",
        "status": "frozen_before_autopsy_model_contact",
        "rows": len(manifest),
        "seed": 20260819,
        "source_manifest_sha256": sha256_file(args.dev2_manifest),
        "manifest_sha256": sha256_file(args.dev2_subsample_manifest),
        "battery_counts": battery_counts(manifest),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.dev2_subsample_manifest.with_suffix(".receipt.json"), receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dev2-subsample", action="store_true")
    parser.add_argument("--seed", type=int, choices=(0, 1))
    for name in (
        "lock", "dev1_panel", "dev2_manifest", "dev2_subsample_manifest", "reference_rows", "base_scores",
        "initialization_scores", "initialization_scores_0p02", "heldout_teacher_cache", "stop_checkpoint", "training_summary", "migrated",
        "p33", "i1", "p34", "p35", "model_cache", "output_dir", "private_dir",
    ):
        parser.add_argument(f"--{name}", type=Path)
    for name in ("migrated_sha256", "p33_sha256", "i1_sha256", "p34_sha256", "p35_sha256"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mcq_batch_size", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=2)
    parser.add_argument("--margin_batch_size", type=int, default=2)
    args = parser.parse_args()
    required = ["dev2_manifest", "dev2_subsample_manifest"]
    if not args.freeze_dev2_subsample:
        required.extend(
            [
                "seed", "lock", "dev1_panel", "reference_rows", "base_scores", "initialization_scores",
                "initialization_scores_0p02",
                "heldout_teacher_cache", "stop_checkpoint", "training_summary", "migrated", "p33", "i1",
                "p34", "p35", "model_cache", "output_dir", "private_dir",
                "migrated_sha256", "p33_sha256", "i1_sha256", "p34_sha256", "p35_sha256",
            ]
        )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {missing}")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = freeze_dev2_subsample(args) if args.freeze_dev2_subsample else run(args)
    except Exception as error:
        if not args.freeze_dev2_subsample and args.output_dir is not None:
            status_path = args.output_dir / "status.json"
            failure = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.is_file()
                else {"kind": RUN_KIND, "seed": args.seed}
            )
            failure.update(
                {
                    "status": "failed_evaluator",
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                    "optimizer_constructed": False,
                    "optimizer_steps": 0,
                    "confirm_scored": False,
                    "eval_e_scored": False,
                }
            )
            atomic_json(status_path, failure)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
