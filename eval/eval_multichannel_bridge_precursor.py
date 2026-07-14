"""Eval-only precursor battery for a learned multi-channel re-entry bridge.

The battery asks whether the 14 query-head write subspaces of the final
recurrent attention layer explain loop drift, table retrieval, or prelude
injection sensitivity better than dimension-matched random bases.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from eval.eval_synthetic_depth_active_labels import (  # noqa: E402
    active_target_for_loop,
    candidates_for_row,
    prompt_for_row,
    read_jsonl,
    single_token_candidate_ids,
)
from models.halting import masked_mean  # noqa: E402


def _validate_partition_shape(hidden_size: int, num_heads: int) -> int:
    if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
        raise ValueError("hidden_size must be positive and divisible by num_heads")
    return hidden_size // num_heads


def output_projection_head_subspaces(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Return orthonormal bases for query-head write subspaces of ``o_proj``.

    Qwen concatenates query-head outputs before applying ``o_proj``. Therefore
    each input-column block of ``o_proj`` maps one query head into the residual
    stream. The blocks can overlap, so each is orthonormalized independently.
    """

    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("o_proj weight must be a square [hidden_size, hidden_size] tensor")
    hidden_size = int(weight.shape[0])
    head_dim = _validate_partition_shape(hidden_size, int(num_heads))
    work = weight.detach().float()
    bases = []
    for head_index in range(num_heads):
        block = work[:, head_index * head_dim : (head_index + 1) * head_dim]
        basis, _ = torch.linalg.qr(block, mode="reduced")
        bases.append(basis)
    return torch.stack(bases, dim=0)


def random_orthogonal_partitions(
    *,
    hidden_size: int,
    num_heads: int,
    draws: int,
    seed: int,
) -> torch.Tensor:
    """Generate reproducible, dimension-matched random orthogonal partitions."""

    head_dim = _validate_partition_shape(int(hidden_size), int(num_heads))
    if draws < 1:
        raise ValueError("draws must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    partitions = []
    for _ in range(int(draws)):
        matrix = torch.randn(hidden_size, hidden_size, generator=generator)
        orthogonal, _ = torch.linalg.qr(matrix)
        partitions.append(
            torch.stack(
                [
                    orthogonal[:, index * head_dim : (index + 1) * head_dim]
                    for index in range(num_heads)
                ],
                dim=0,
            )
        )
    return torch.stack(partitions, dim=0)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    return float(torch.quantile(torch.tensor(list(values), dtype=torch.float64), probability).item())


def classify_subspace_drift(
    *,
    head_top3_by_loop: dict[str, float],
    random_top3_mean_by_loop: dict[str, float],
    minimum_loop: int = 6,
    advantage_ratio: float = 2.0,
    consistency_fraction: float = 0.75,
) -> dict[str, Any]:
    """Apply the locked M1 rule to already-aggregated loop statistics."""

    loops = sorted(
        int(loop)
        for loop in set(head_top3_by_loop) & set(random_top3_mean_by_loop)
        if int(loop) >= minimum_loop
    )
    ratios: dict[str, float] = {}
    qualifying = 0
    for loop in loops:
        random_mean = float(random_top3_mean_by_loop[str(loop)])
        ratio = math.inf if random_mean == 0.0 else float(head_top3_by_loop[str(loop)]) / random_mean
        ratios[str(loop)] = ratio
        qualifying += int(ratio >= advantage_ratio)
    required = math.ceil(consistency_fraction * len(loops)) if loops else 1
    return {
        "confirmed": bool(loops and qualifying >= required),
        "reading": "specialization" if loops and qualifying >= required else "smeared",
        "eligible_loops": len(loops),
        "qualifying_loops": qualifying,
        "required_qualifying_loops": required,
        "advantage_ratio_by_loop": ratios,
        "minimum_loop": minimum_loop,
        "locked_advantage_ratio": advantage_ratio,
        "locked_consistency_fraction": consistency_fraction,
    }


def classify_retrieval_heads(
    *,
    head_ratio_by_id: dict[str, float],
    stable_fraction_by_id: dict[str, float],
    actual_concentration: float,
    random_concentrations: Sequence[float],
    minimum_ratio: float = 3.0,
    minimum_stable_fraction: float = 0.50,
) -> dict[str, Any]:
    """Apply the locked M2 rule plus its random-rotation control."""

    qualifying = sorted(
        head_id
        for head_id, ratio in head_ratio_by_id.items()
        if float(ratio) >= minimum_ratio
        and float(stable_fraction_by_id.get(head_id, 0.0)) >= minimum_stable_fraction
    )
    random_p95 = _quantile(random_concentrations, 0.95)
    random_null_win = float(actual_concentration) > random_p95
    confirmed = len(qualifying) >= 2 and random_null_win
    return {
        "confirmed": confirmed,
        "reading": "retrieval_heads_exist" if confirmed else "retrieval_heads_not_established",
        "qualifying_heads": qualifying,
        "qualifying_head_count": len(qualifying),
        "actual_concentration": float(actual_concentration),
        "random_concentration_p95": random_p95,
        "random_null_win": random_null_win,
        "locked_minimum_ratio": minimum_ratio,
        "locked_minimum_stable_fraction": minimum_stable_fraction,
    }


def classify_injection_heterogeneity(
    *,
    head_damage: Sequence[float],
    random_damage: Sequence[float],
    minimum_max_to_median_ratio: float = 5.0,
) -> dict[str, Any]:
    """Apply the locked M3 max/median and matched-random criteria."""

    if not head_damage:
        raise ValueError("head_damage must not be empty")
    damage = torch.tensor([max(0.0, float(value)) for value in head_damage], dtype=torch.float64)
    maximum = float(damage.max().item())
    median = float(damage.median().item())
    ratio = math.inf if median == 0.0 and maximum > 0.0 else (maximum / median if median else 0.0)
    random_p95 = _quantile([max(0.0, float(value)) for value in random_damage], 0.95)
    random_null_win = maximum > random_p95
    confirmed = ratio >= minimum_max_to_median_ratio and random_null_win
    return {
        "confirmed": confirmed,
        "reading": "heterogeneous" if confirmed else "homogeneous_or_unresolved",
        "max_damage": maximum,
        "median_damage": median,
        "max_to_median_ratio": ratio,
        "random_damage_p95": random_p95,
        "random_null_win": random_null_win,
        "locked_minimum_max_to_median_ratio": minimum_max_to_median_ratio,
    }


def projected_energy_shares(vectors: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """Return per-subspace energy shares for vectors ``[..., hidden]``."""

    if vectors.shape[-1] != bases.shape[-2]:
        raise ValueError("vectors and bases disagree on hidden size")
    coefficients = torch.einsum("...h,khd->...kd", vectors.float(), bases.float())
    energy = coefficients.square().sum(dim=-1)
    denominator = energy.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(energy.dtype).eps)
    return energy / denominator


def table_character_span(prompt: str) -> tuple[int, int]:
    """Locate the rendered table in forward or handoff-style prompts."""

    marker = "Function table:\n"
    start = prompt.find(marker)
    if start >= 0:
        start += len(marker)
    else:
        start = 0
    end = prompt.find("\n\n", start)
    if end < 0 or end <= start:
        raise ValueError("Could not locate a non-empty rendered table block in prompt")
    return start, end


def table_token_mask(tokenizer: Any, prompt: str, *, max_length: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0]
    start, end = table_character_span(prompt)
    mask = torch.tensor(
        [int(left) < end and int(right) > start for left, right in offsets.tolist()],
        dtype=torch.bool,
    )
    if not bool(mask.any()):
        raise ValueError("Rendered table has no tokenizer-aligned tokens after truncation")
    return encoded, mask


def select_rows_by_depth(
    rows: Sequence[dict[str, Any]],
    *,
    max_depth: int,
    rows_per_depth: int,
    deepest_first: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for row in rows:
        depth = int(row["depth"])
        if depth < 1 or depth > max_depth or counts.get(depth, 0) >= rows_per_depth:
            continue
        selected.append(row)
        counts[depth] = counts.get(depth, 0) + 1
    missing = {depth: rows_per_depth - counts.get(depth, 0) for depth in range(1, max_depth + 1) if counts.get(depth, 0) < rows_per_depth}
    if missing:
        raise ValueError(f"Frozen data lacks requested rows by depth: {missing}")
    if deepest_first:
        # A pilot must exercise its expensive depth-14 path before spending time
        # on shallow rows. The chosen rows and analysis population stay unchanged.
        selected.sort(key=lambda row: (-int(row["depth"]), str(row.get("id") or row.get("instance_id"))))
    return selected


def wrapper_head_subspaces(wrapper: Any) -> tuple[torch.Tensor, dict[str, int]]:
    layer_index = int(wrapper.layer_split.recurrent_end) - 1
    attention = wrapper.qwen.layers[layer_index].self_attn
    weight = attention.o_proj.weight.detach().float().cpu()
    num_heads = int(wrapper.config.num_attention_heads)
    num_kv_heads = int(getattr(wrapper.config, "num_key_value_heads", num_heads))
    bases = output_projection_head_subspaces(weight, num_heads=num_heads)
    return bases, {
        "source_layer_index": layer_index,
        "num_query_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": int(weight.shape[1] // num_heads),
        "hidden_size": int(weight.shape[0]),
    }


def _prepare_with_attentions(
    wrapper: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, Any]:
    prepared = wrapper._prepare_inputs(  # noqa: SLF001
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
    )
    inputs_embeds = prepared["inputs_embeds"]
    position_ids = prepared["position_ids"]
    cache_position = prepared["cache_position"]
    causal_mask = wrapper._update_causal_mask(  # noqa: SLF001
        prepared["attention_mask"],
        inputs_embeds,
        cache_position,
        past_key_values=None,
        output_attentions=True,
    )
    position_embeddings = wrapper._rotary_embeddings(inputs_embeds, position_ids)  # noqa: SLF001
    hidden, _ = wrapper._run_layer_range(  # noqa: SLF001
        start=0,
        end=wrapper.layer_split.prelude_end,
        hidden_states=inputs_embeds,
        causal_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        collect_hidden=False,
        hidden_history=None,
    )
    return hidden, causal_mask, position_ids, cache_position, position_embeddings


def collect_row_dynamics(
    wrapper: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_loops: int,
    max_length: int,
    on_loop_complete: Callable[[int, float], None] | None = None,
    max_seconds: float = 0.0,
) -> dict[str, Any]:
    """Collect pooled carried states and recurrent-layer table attention."""

    prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
    encoded, rendered_table_mask = table_token_mask(tokenizer, prompt, max_length=max_length)
    model_device = next(wrapper.parameters()).device
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(model_device)
    table_mask = rendered_table_mask.to(model_device)
    started = time.monotonic()
    with torch.no_grad():
        entry, causal_mask, position_ids, cache_position, position_embeddings = _prepare_with_attentions(
            wrapper,
            input_ids,
            attention_mask,
        )
        recurrent_state = entry
        pooled_states: list[torch.Tensor] = []
        table_mass_by_loop: list[torch.Tensor] = []
        answer_position = int(attention_mask[0].long().sum().item()) - 1
        for loop_index in range(max_loops):
            loop_input = recurrent_state
            if loop_index > 0:
                loop_input = wrapper.bridge(loop_input, prelude_hidden=entry)
            recurrent_state, attentions = wrapper._run_layer_range(  # noqa: SLF001
                start=wrapper.layer_split.prelude_end,
                end=wrapper.layer_split.recurrent_end,
                hidden_states=loop_input,
                causal_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                output_attentions=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                collect_hidden=False,
                hidden_history=None,
            )
            if len(attentions) != wrapper.layer_split.recurrent_end - wrapper.layer_split.prelude_end:
                raise RuntimeError(
                    "Attention capture did not return one tensor per recurrent layer; "
                    "use --attn_implementation eager"
                )
            pooled_states.append(masked_mean(recurrent_state, attention_mask)[0].detach().float().cpu())
            layer_mass = []
            for attention in attentions:
                if attention.ndim != 4:
                    raise RuntimeError(f"Unexpected attention shape: {tuple(attention.shape)}")
                mass = attention[0, :, answer_position, :][:, table_mask].float().sum(dim=-1)
                layer_mass.append(mass.detach().cpu())
            table_mass_by_loop.append(torch.stack(layer_mass, dim=0))
            elapsed = time.monotonic() - started
            if on_loop_complete is not None:
                on_loop_complete(loop_index + 1, elapsed)
            if max_seconds > 0.0 and elapsed > max_seconds:
                raise TimeoutError(
                    f"Dynamics collection exceeded {max_seconds:.1f}s for row "
                    f"{row.get('id') or row.get('instance_id')} after loop {loop_index + 1}"
                )
    return {
        "id": str(row.get("id") or row.get("instance_id")),
        "depth": int(row["depth"]),
        "pooled_states": torch.stack(pooled_states, dim=0),
        "table_mass": torch.stack(table_mass_by_loop, dim=0),
        "table_tokens": int(rendered_table_mask.sum().item()),
        "prompt_tokens": int(attention_mask.sum().item()),
    }


def collect_dynamics(
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    max_loops: int,
    max_length: int,
    cache_path: Path,
    progress_every: int,
    status_path: Path,
    max_seconds_per_row: float,
) -> list[dict[str, Any]]:
    signature = {
        "row_ids": [str(row.get("id") or row.get("instance_id")) for row in rows],
        "max_loops": int(max_loops),
        "max_length": int(max_length),
    }
    records: list[dict[str, Any]] = []
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("signature") != signature or not isinstance(payload.get("records"), list):
            raise TypeError(f"Unexpected collector cache payload: {cache_path}")
        records = payload["records"]
    completed = {str(record["id"]) for record in records}
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or row.get("instance_id"))
        if row_id in completed:
            continue
        row_started = time.monotonic()
        write_json(
            status_path,
            {
                "kind": "multichannel_dynamics_progress",
                "status": "collecting_row",
                "row_index": index,
                "total_rows": len(rows),
                "completed_rows": len(records),
                "row_id": row_id,
                "depth": int(row["depth"]),
                "loop_complete": 0,
                "max_loops": int(max_loops),
                "max_seconds_per_row": float(max_seconds_per_row),
            },
        )

        def on_loop_complete(loop_index: int, elapsed: float) -> None:
            write_json(
                status_path,
                {
                    "kind": "multichannel_dynamics_progress",
                    "status": "collecting_row",
                    "row_index": index,
                    "total_rows": len(rows),
                    "completed_rows": len(records),
                    "row_id": row_id,
                    "depth": int(row["depth"]),
                    "loop_complete": int(loop_index),
                    "max_loops": int(max_loops),
                    "elapsed_seconds": round(float(elapsed), 3),
                    "max_seconds_per_row": float(max_seconds_per_row),
                },
            )
            print(
                f"multichannel_dynamics_loop row={index}/{len(rows)} depth={row['depth']} "
                f"loop={loop_index}/{max_loops} elapsed_s={elapsed:.1f}",
                flush=True,
            )

        try:
            record = collect_row_dynamics(
                wrapper,
                tokenizer,
                row,
                max_loops=max_loops,
                max_length=max_length,
                on_loop_complete=on_loop_complete,
                max_seconds=max_seconds_per_row,
            )
        except Exception as exc:
            write_json(
                status_path,
                {
                    "kind": "multichannel_dynamics_progress",
                    "status": "failed",
                    "row_index": index,
                    "total_rows": len(rows),
                    "completed_rows": len(records),
                    "row_id": row_id,
                    "depth": int(row["depth"]),
                    "elapsed_seconds": round(time.monotonic() - row_started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        records.append(record)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Persist every row. A CUDA/runtime interruption must cost at most one row.
        torch.save({"signature": signature, "records": records}, cache_path)
        write_json(
            status_path,
            {
                "kind": "multichannel_dynamics_progress",
                "status": "collecting",
                "row_index": index,
                "total_rows": len(rows),
                "completed_rows": len(records),
                "last_row_id": row_id,
                "last_row_elapsed_seconds": round(time.monotonic() - row_started, 3),
            },
        )
        if index == 1 or index % progress_every == 0 or index == len(rows):
            print(f"multichannel_dynamics_progress row={index}/{len(rows)} cached={len(records)}", flush=True)
    torch.save({"signature": signature, "records": records}, cache_path)
    write_json(
        status_path,
        {
            "kind": "multichannel_dynamics_progress",
            "status": "finished",
            "total_rows": len(rows),
            "completed_rows": len(records),
        },
    )
    return records


def _energy_share_aggregate(vectors: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    coefficients = torch.einsum("nh,khd->nkd", vectors.float(), bases.float())
    energy = coefficients.square().sum(dim=(0, 2))
    return energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)


def analyze_m1(
    records: Sequence[dict[str, Any]],
    head_bases: torch.Tensor,
    random_partitions: torch.Tensor,
    *,
    envelope_rank: int,
) -> dict[str, Any]:
    state_tensor = torch.stack([record["pooled_states"] for record in records], dim=0).float()
    loop1 = state_tensor[:, 0]
    mean = loop1.mean(dim=0, keepdim=True)
    centered_loop1 = loop1 - mean
    rank = min(int(envelope_rank), max(1, centered_loop1.shape[0] - 1), centered_loop1.shape[1])
    _u, _s, vh = torch.linalg.svd(centered_loop1, full_matrices=False)
    envelope = vh[:rank].T.contiguous()
    loops: dict[str, Any] = {}
    head_top3: dict[str, float] = {}
    random_mean: dict[str, float] = {}
    random_p95: dict[str, float] = {}
    for loop_index in range(state_tensor.shape[1]):
        states = state_tensor[:, loop_index]
        centered = states - mean
        residual = centered - (centered @ envelope) @ envelope.T
        norm_share = _energy_share_aggregate(states, head_bases)
        drift_share = _energy_share_aggregate(residual, head_bases)
        actual_top3 = float(torch.topk(drift_share, k=min(3, drift_share.numel())).values.sum().item())
        null_top3 = []
        for partition in random_partitions:
            shares = _energy_share_aggregate(residual, partition)
            null_top3.append(float(torch.topk(shares, k=min(3, shares.numel())).values.sum().item()))
        loop = str(loop_index + 1)
        head_top3[loop] = actual_top3
        random_mean[loop] = sum(null_top3) / len(null_top3)
        random_p95[loop] = _quantile(null_top3, 0.95)
        loops[loop] = {
            "head_norm_share": norm_share.tolist(),
            "head_off_manifold_share": drift_share.tolist(),
            "head_top3_off_manifold_share": actual_top3,
            "random_top3_mean": random_mean[loop],
            "random_top3_p95": random_p95[loop],
            "head_outside_random_p95": actual_top3 > random_p95[loop],
            "off_manifold_rms": float(residual.square().mean().sqrt().item()),
        }
    classification = classify_subspace_drift(
        head_top3_by_loop=head_top3,
        random_top3_mean_by_loop=random_mean,
    )
    eligible = [loop for loop in classification["advantage_ratio_by_loop"]]
    outside_p95 = sum(bool(loops[loop]["head_outside_random_p95"]) for loop in eligible)
    required = classification["required_qualifying_loops"]
    classification["outside_random_p95_loops"] = outside_p95
    classification["required_outside_random_p95_loops"] = required
    classification["confirmed"] = bool(classification["confirmed"] and outside_p95 >= required)
    classification["reading"] = "specialization" if classification["confirmed"] else "smeared"
    return {
        "kind": "multichannel_bridge_m1_subspace_drift",
        "rows": len(records),
        "loops": loops,
        "loop1_envelope_rank": rank,
        "classification": classification,
    }


def _top2_energy_concentration(vectors: torch.Tensor) -> torch.Tensor:
    energy = vectors.float().square()
    return torch.topk(energy, k=min(2, energy.shape[-1]), dim=-1).values.sum(dim=-1) / energy.sum(dim=-1).clamp_min(1e-12)


def analyze_m2(records: Sequence[dict[str, Any]], *, random_draws: int, seed: int) -> dict[str, Any]:
    masses = torch.stack([record["table_mass"] for record in records], dim=0).float()
    # [rows, loops, recurrent_layers, query_heads]
    row_layer_head = masses.mean(dim=1)
    mean_layer_head = row_layer_head.mean(dim=0)
    ratios: dict[str, float] = {}
    stability: dict[str, float] = {}
    for layer in range(mean_layer_head.shape[0]):
        layer_median = float(mean_layer_head[layer].median().item())
        row_median = row_layer_head[:, layer].median(dim=-1).values
        for head in range(mean_layer_head.shape[1]):
            key = f"L{layer}H{head}"
            ratios[key] = float(mean_layer_head[layer, head].item()) / max(layer_median, 1e-12)
            stability[key] = float(
                (row_layer_head[:, layer, head] >= 3.0 * row_median.clamp_min(1e-12)).float().mean().item()
            )
    actual_concentration = float(_top2_energy_concentration(masses).mean().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_concentrations = []
    for _ in range(int(random_draws)):
        matrix = torch.randn(masses.shape[-1], masses.shape[-1], generator=generator)
        rotation, _ = torch.linalg.qr(matrix)
        rotated = masses @ rotation
        random_concentrations.append(float(_top2_energy_concentration(rotated).mean().item()))
    classification = classify_retrieval_heads(
        head_ratio_by_id=ratios,
        stable_fraction_by_id=stability,
        actual_concentration=actual_concentration,
        random_concentrations=random_concentrations,
    )
    per_loop = {
        str(loop + 1): {
            "mean_table_mass_by_layer_head": masses[:, loop].mean(dim=0).tolist(),
            "top2_energy_concentration": float(_top2_energy_concentration(masses[:, loop]).mean().item()),
        }
        for loop in range(masses.shape[1])
    }
    return {
        "kind": "multichannel_bridge_m2_retrieval_head_census",
        "rows": len(records),
        "head_ratio_by_id": ratios,
        "stable_fraction_by_id": stability,
        "per_loop": per_loop,
        "random_rotation_concentrations": random_concentrations,
        "classification": classification,
        "null_definition": "independent random orthogonal rotation of each 14-head attention-mass vector",
    }


def assert_flag_off_equivalence(wrapper: Any, tokenizer: Any, row: dict[str, Any], *, max_loops: int, device: str) -> dict[str, Any]:
    prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    kwargs = dict(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_loops=max_loops,
        num_trajectories=1,
        particle_update_mode="none",
        use_cache=False,
        return_dict=True,
        return_loop_logits=True,
        logits_to_keep=1,
    )
    with torch.no_grad():
        baseline = wrapper(**kwargs).loop_logits
        explicit_off = wrapper(**kwargs, bridge_prelude_ablation_basis=None).loop_logits
    if baseline is None or explicit_off is None:
        raise RuntimeError("Flag-off equivalence requires loop logits")
    exact = torch.equal(baseline, explicit_off)
    max_abs = float((baseline.float() - explicit_off.float()).abs().max().item())
    if not exact:
        raise AssertionError(f"Bridge intervention flag-off equivalence failed: max_abs={max_abs}")
    return {"exact": exact, "max_abs_diff": max_abs, "shape": list(baseline.shape)}


def score_active_labels_batched(
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    max_loops: int,
    batch_size: int,
    device: str,
    value_prefix: str,
    ablation_basis: torch.Tensor | None,
    on_batch_complete: Callable[[int, int, float], None] | None = None,
    max_seconds: float = 0.0,
) -> dict[str, Any]:
    correct_by_loop = {loop: 0 for loop in range(1, max_loops + 1)}
    total_by_loop = {loop: 0 for loop in range(1, max_loops + 1)}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start : start + batch_size])
        prompts = [prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only") for row in chunk]
        token_ids_by_row = []
        for row, prompt in zip(chunk, prompts):
            candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix=value_prefix)
            token_ids = single_token_candidate_ids(tokenizer, prompt, candidates)
            if token_ids is None:
                raise RuntimeError("M3 requires exact single-token full-symbol candidates")
            token_ids_by_row.append(token_ids)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_loops=max_loops,
                num_trajectories=1,
                particle_update_mode="none",
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
                logits_to_keep=1,
                bridge_prelude_ablation_basis=ablation_basis,
            )
        if output.loop_logits is None:
            raise RuntimeError("M3 requires return_loop_logits")
        for row_index, (row, token_ids) in enumerate(zip(chunk, token_ids_by_row)):
            for loop in range(1, min(int(row["depth"]), max_loops) + 1):
                logits = output.loop_logits[row_index, 0, loop - 1, -1]
                prediction = max(token_ids, key=lambda name: float(logits[token_ids[name]].item()))
                target = active_target_for_loop(
                    row,
                    loop,
                    prediction_space="full_symbols",
                    value_prefix=value_prefix,
                )
                correct_by_loop[loop] += int(prediction == target)
                total_by_loop[loop] += 1
        print(
            f"m3_eval_progress rows={min(start + batch_size, len(rows))}/{len(rows)} "
            f"ablation={'none' if ablation_basis is None else 'active'}",
            flush=True,
        )
        elapsed = time.monotonic() - started
        if on_batch_complete is not None:
            on_batch_complete(min(start + batch_size, len(rows)), len(rows), elapsed)
        if max_seconds > 0.0 and elapsed > max_seconds:
            raise TimeoutError(
                f"M3 ablation pass exceeded {max_seconds:.1f}s after "
                f"{min(start + batch_size, len(rows))}/{len(rows)} rows"
            )
    total = sum(total_by_loop.values())
    correct = sum(correct_by_loop.values())
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "by_loop": {
            str(loop): {
                "correct": correct_by_loop[loop],
                "total": total_by_loop[loop],
                "accuracy": correct_by_loop[loop] / total_by_loop[loop] if total_by_loop[loop] else 0.0,
            }
            for loop in range(1, max_loops + 1)
        },
    }


def run_m3(
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    head_bases: torch.Tensor,
    random_partitions: torch.Tensor,
    *,
    max_loops: int,
    batch_size: int,
    device: str,
    value_prefix: str,
    progress_path: Path,
    cache_signature: dict[str, Any],
    max_seconds_per_pass: float,
) -> dict[str, Any]:
    flag_off = assert_flag_off_equivalence(wrapper, tokenizer, rows[0], max_loops=max_loops, device=device)
    progress: dict[str, Any] = {"signature": cache_signature, "head": {}, "random": {}}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature") != cache_signature:
            raise RuntimeError(f"M3 resume cache signature mismatch: {progress_path}")
    def score_pass(label: str, basis: torch.Tensor | None) -> dict[str, Any]:
        progress["active_pass"] = {"label": label, "completed_batches": 0, "total_rows": len(rows)}
        write_json(progress_path, progress)

        def on_batch_complete(done: int, total: int, elapsed: float) -> None:
            progress["active_pass"] = {
                "label": label,
                "completed_rows": int(done),
                "total_rows": int(total),
                "elapsed_seconds": round(float(elapsed), 3),
            }
            write_json(progress_path, progress)

        try:
            score = score_active_labels_batched(
                wrapper,
                tokenizer,
                rows,
                max_loops=max_loops,
                batch_size=batch_size,
                device=device,
                value_prefix=value_prefix,
                ablation_basis=basis,
                on_batch_complete=on_batch_complete,
                max_seconds=max_seconds_per_pass,
            )
        except Exception as exc:
            progress["active_pass"] = {"label": label, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            write_json(progress_path, progress)
            raise
        progress.pop("active_pass", None)
        return score

    if "baseline" not in progress:
        progress["baseline"] = score_pass("baseline", None)
        write_json(progress_path, progress)
    baseline_accuracy = float(progress["baseline"]["accuracy"])
    for index, basis in enumerate(head_bases):
        key = str(index)
        if key not in progress["head"]:
            score = score_pass(f"head:{key}", basis)
            score["damage"] = baseline_accuracy - float(score["accuracy"])
            progress["head"][key] = score
            write_json(progress_path, progress)
    for index, partition in enumerate(random_partitions):
        key = str(index)
        if key not in progress["random"]:
            score = score_pass(f"random:{key}", partition[0])
            score["damage"] = baseline_accuracy - float(score["accuracy"])
            progress["random"][key] = score
            write_json(progress_path, progress)
    head_damage = [float(progress["head"][str(index)]["damage"]) for index in range(head_bases.shape[0])]
    random_damage = [float(progress["random"][str(index)]["damage"]) for index in range(random_partitions.shape[0])]
    return {
        "kind": "multichannel_bridge_m3_injection_sensitivity",
        "rows": len(rows),
        "flag_off_equivalence": flag_off,
        "baseline": progress["baseline"],
        "head_ablations": progress["head"],
        "random_ablations": progress["random"],
        "classification": classify_injection_heterogeneity(
            head_damage=head_damage,
            random_damage=random_damage,
        ),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def write_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Multi-Channel Bridge Precursor - {payload.get('condition', 'condition')}",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Checkpoint SHA256: `{payload.get('checkpoint_sha256')}`",
        f"- Rows: `{payload.get('rows')}`",
        f"- Max loops: `{payload.get('max_loops')}`",
        f"- Random controls: `{payload.get('random_draws')}`",
        f"- Query-head write basis: `{payload.get('basis_definition')}`",
        "",
    ]
    for name, result in payload.get("measurements", {}).items():
        classification = result.get("classification", {})
        lines.extend(
            [
                f"## {name.upper()}",
                f"- Confirmed: `{classification.get('confirmed')}`",
                f"- Reading: `{classification.get('reading')}`",
                f"- Classification: `{classification}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            f"- Measurements confirmed: `{payload.get('confirmed_measurements')}`",
            f"- Battery specialization criterion: `{payload.get('battery_specialization_confirmed')}`",
            f"- Staircase reading one: `{payload.get('staircase_reading_one')}`",
            f"- Architecture activation eligible: `{payload.get('architecture_activation_eligible')}`",
            "",
            "The battery does not change the experiment queue. Activation requires both at least two "
            "positive measurements and a staircase reading of per-position installation cost.",
            "",
        ]
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument(
        "--classify_payload",
        help="Optional JSON containing pre-aggregated m1/m2/m3 inputs; useful for CPU receipt checks.",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint_sha256", default="")
    parser.add_argument("--data_jsonl")
    parser.add_argument("--condition", default="unnamed")
    parser.add_argument("--resume_cache_dir", default="")
    parser.add_argument("--measurements", default="m1,m2,m3")
    parser.add_argument("--max_depth", type=int, default=14)
    parser.add_argument("--max_loops", type=int, default=14)
    parser.add_argument("--rows_per_depth", type=int, default=64)
    parser.add_argument("--random_draws", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=20260713)
    parser.add_argument("--envelope_rank", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--m3_batch_size", type=int, default=8)
    parser.add_argument("--value_prefix", default="letter:")
    parser.add_argument("--progress_every", type=int, default=8)
    parser.add_argument("--dynamics_row_timeout_seconds", type=float, default=600.0)
    parser.add_argument("--m3_pass_timeout_seconds", type=float, default=1800.0)
    parser.add_argument("--dynamics_deepest_first", action="store_true")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--staircase_reading_one", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload: dict[str, Any] = {
        "kind": "multichannel_bridge_precursor",
        "status": "schema_only",
    }
    if args.classify_payload:
        source = json.loads(Path(args.classify_payload).read_text(encoding="utf-8"))
        results: dict[str, Any] = {}
        if "m1" in source:
            results["m1"] = classify_subspace_drift(**source["m1"])
        if "m2" in source:
            results["m2"] = classify_retrieval_heads(**source["m2"])
        if "m3" in source:
            results["m3"] = classify_injection_heterogeneity(**source["m3"])
        confirmed = sum(bool(result["confirmed"]) for result in results.values())
        payload.update(
            status="classified",
            measurements=results,
            confirmed_measurements=confirmed,
            battery_specialization_confirmed=confirmed >= 2,
        )
        write_json(args.output_summary, payload)
        return 0

    if not args.checkpoint or not args.data_jsonl:
        parser.error("--checkpoint and --data_jsonl are required unless --classify_payload is used")
    if args.random_draws < 20:
        raise ValueError("The locked battery requires at least 20 random control draws")
    requested = {item.strip().lower() for item in args.measurements.split(",") if item.strip()}
    unknown = requested - {"m1", "m2", "m3"}
    if unknown:
        raise ValueError(f"Unknown measurements: {sorted(unknown)}")

    from transformers import AutoTokenizer

    output_summary = Path(args.output_summary)
    output_dir = output_summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_cache_dir = Path(args.resume_cache_dir) if args.resume_cache_dir else output_dir
    resume_cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        resume_cache_dir / "dynamics_status.json",
        {
            "kind": "multichannel_dynamics_progress",
            "status": "starting",
            "condition": args.condition,
            "checkpoint": args.checkpoint,
            "max_depth": args.max_depth,
            "max_loops": args.max_loops,
            "rows_per_depth": args.rows_per_depth,
        },
    )
    rows = select_rows_by_depth(
        read_jsonl(args.data_jsonl),
        max_depth=args.max_depth,
        rows_per_depth=args.rows_per_depth,
        deepest_first=args.dynamics_deepest_first,
    )
    if "m3" in requested and any("orbit" not in row for row in rows):
        raise ValueError("M3 active-label scoring currently requires forward rows with an orbit field")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    head_bases, basis_metadata = wrapper_head_subspaces(wrapper)
    random_partitions = random_orthogonal_partitions(
        hidden_size=basis_metadata["hidden_size"],
        num_heads=basis_metadata["num_query_heads"],
        draws=args.random_draws,
        seed=args.random_seed,
    )
    results: dict[str, Any] = {}
    if requested & {"m1", "m2"}:
        records = collect_dynamics(
            wrapper,
            tokenizer,
            rows,
            max_loops=args.max_loops,
            max_length=args.max_length,
            cache_path=resume_cache_dir / "dynamics_cache.pt",
            progress_every=args.progress_every,
            status_path=resume_cache_dir / "dynamics_status.json",
            max_seconds_per_row=args.dynamics_row_timeout_seconds,
        )
        if "m1" in requested:
            results["m1"] = analyze_m1(
                records,
                head_bases,
                random_partitions,
                envelope_rank=args.envelope_rank,
            )
            write_json(output_dir / "m1_subspace_drift.json", results["m1"])
        if "m2" in requested:
            results["m2"] = analyze_m2(records, random_draws=args.random_draws, seed=args.random_seed + 1)
            write_json(output_dir / "m2_retrieval_heads.json", results["m2"])
    if "m3" in requested:
        results["m3"] = run_m3(
            wrapper,
            tokenizer,
            rows,
            head_bases,
            random_partitions,
            max_loops=args.max_loops,
            batch_size=args.m3_batch_size,
            device=args.device,
            value_prefix=args.value_prefix,
            progress_path=resume_cache_dir / "m3_progress.json",
            cache_signature={
                "checkpoint_sha256": args.checkpoint_sha256,
                "row_ids": [str(row.get("id") or row.get("instance_id")) for row in rows],
                "max_loops": args.max_loops,
                "random_draws": args.random_draws,
                "random_seed": args.random_seed,
                "basis": "final_recurrent_o_proj_query_head_blocks",
            },
            max_seconds_per_pass=args.m3_pass_timeout_seconds,
        )
        write_json(output_dir / "m3_injection_sensitivity.json", results["m3"])

    confirmed = sum(bool(result.get("classification", {}).get("confirmed")) for result in results.values())
    payload = {
        "kind": "multichannel_bridge_precursor",
        "status": "finished",
        "condition": args.condition,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256 or None,
        "data_jsonl": args.data_jsonl,
        "rows": len(rows),
        "max_depth": args.max_depth,
        "max_loops": args.max_loops,
        "rows_per_depth": args.rows_per_depth,
        "random_draws": args.random_draws,
        "basis_definition": "final recurrent layer o_proj query-head input-column blocks, independently orthonormalized",
        "basis_metadata": basis_metadata,
        "measurements": results,
        "confirmed_measurements": confirmed,
        "battery_specialization_confirmed": confirmed >= 2,
        "staircase_reading_one": bool(args.staircase_reading_one),
        "architecture_activation_eligible": bool(confirmed >= 2 and args.staircase_reading_one),
        "queue_effect": "none; eval-only banked architecture precursor",
    }
    write_json(output_summary, payload)
    write_markdown(output_dir / "summary.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
