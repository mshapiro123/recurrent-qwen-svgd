"""Run the DEV-only Phase-2 V1b finite-perturbation causal check."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl  # noqa: E402
from eval.eval_paper2_dc0_depth_by_append import group_batches  # noqa: E402
from eval.eval_reentry_drift import prepare_recurrent_inputs, run_recurrent_block  # noqa: E402
from eval.eval_paper2_phase2_v1_v2 import quantile_summary  # noqa: E402
from eval.eval_speculative_depth_d0_floor import load_partition_cache  # noqa: E402
from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402


C_VALUES = (0.01, 0.02, 0.05)


def tube_radius(
    *,
    c_value: float,
    state_rms: float,
    hidden_size: int,
    gamma: float,
    rho: float,
) -> float:
    if not 0 <= float(rho) < 1:
        raise ValueError("rho must be in [0, 1)")
    if min(float(c_value), float(state_rms), float(gamma)) < 0:
        raise ValueError("c, RMS, and gamma must be nonnegative")
    return (
        float(gamma)
        * float(c_value)
        * float(state_rms)
        * math.sqrt(int(hidden_size))
        / (1.0 - float(rho))
    )


def deterministic_position_sample(
    records: Sequence[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    cohort: str,
) -> list[dict[str, Any]]:
    if int(sample_size) < 1:
        raise ValueError("sample_size must be positive")
    if len(records) < int(sample_size):
        raise ValueError(
            f"{cohort} has {len(records)} positions, fewer than requested {sample_size}"
        )

    def key(record: dict[str, Any]) -> str:
        material = (
            f"phase2-v1b|{int(seed)}|{cohort}|{record['row_id']}|"
            f"{int(record['position'])}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    return sorted((dict(record) for record in records), key=key)[: int(sample_size)]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compare_paired_predictions(
    *,
    registered: torch.Tensor,
    neutral: torch.Tensor,
    perturbed: torch.Tensor,
    teacher: torch.Tensor,
    position: int,
) -> dict[str, Any]:
    """Compare a perturbation with an unmodified, batch-shape-matched forward."""
    lengths = {int(value.numel()) for value in (registered, neutral, perturbed, teacher)}
    if len(lengths) != 1:
        raise ValueError("paired prediction tensors must have equal lengths")
    if not 0 <= int(position) < int(teacher.numel()):
        raise ValueError("intervention position is outside the scored sequence")

    position = int(position)
    neutral_correct = neutral.eq(teacher)
    perturbed_correct = perturbed.eq(teacher)
    causal_prefix_changes = int(perturbed[:position].ne(neutral[:position]).sum())
    if causal_prefix_changes:
        raise RuntimeError(
            "V1b causal contract failed against the batch-matched neutral: "
            "a perturbation changed a prior position"
        )

    collateral = torch.ones(int(teacher.numel()), dtype=torch.bool)
    collateral[position] = False
    future = torch.arange(int(teacher.numel())) > position
    before_other = neutral_correct[collateral]
    after_other = perturbed_correct[collateral]
    before_future = neutral_correct[future]
    after_future = perturbed_correct[future]
    return {
        "neutral_vs_registered_prediction_changes": int(neutral.ne(registered).sum()),
        "neutral_vs_registered_prefix_changes": int(
            neutral[:position].ne(registered[:position]).sum()
        ),
        "neutral_vs_registered_target_changed": bool(
            neutral[position] != registered[position]
        ),
        "target_correct_before": bool(neutral_correct[position]),
        "target_correct_after": bool(perturbed_correct[position]),
        "collateral_positions": int(collateral.sum()),
        "collateral_helps": int((~before_other & after_other).sum()),
        "collateral_hurts": int((before_other & ~after_other).sum()),
        "collateral_prediction_changes": int(
            perturbed[collateral].ne(neutral[collateral]).sum()
        ),
        "causal_prefix_prediction_changes": causal_prefix_changes,
        "future_positions": int(future.sum()),
        "future_helps": int((~before_future & after_future).sum()),
        "future_hurts": int((before_future & ~after_future).sum()),
        "future_prediction_changes": int(
            perturbed[future].ne(neutral[future]).sum()
        ),
    }


def aggregate_intervention_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["cohort"]), f"{float(record['c_value']):g}")].append(
            record
        )
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (cohort, c_key), rows in sorted(grouped.items()):
        positions = len(rows)
        first_order = sum(bool(row["first_order_predicted_pair_cross"]) for row in rows)
        pair_cross = sum(bool(row["realized_pair_cross"]) for row in rows)
        target_before = sum(bool(row["target_correct_before"]) for row in rows)
        target_after = sum(bool(row["target_correct_after"]) for row in rows)
        collateral_positions = sum(int(row["collateral_positions"]) for row in rows)
        collateral_helps = sum(int(row["collateral_helps"]) for row in rows)
        collateral_hurts = sum(int(row["collateral_hurts"]) for row in rows)
        collateral_changes = sum(
            int(row["collateral_prediction_changes"]) for row in rows
        )
        future_positions = sum(int(row.get("future_positions", 0)) for row in rows)
        future_helps = sum(int(row.get("future_helps", 0)) for row in rows)
        future_hurts = sum(int(row.get("future_hurts", 0)) for row in rows)
        future_changes = sum(
            int(row.get("future_prediction_changes", 0)) for row in rows
        )
        neutral_registered_changes = sum(
            int(row.get("neutral_vs_registered_prediction_changes", 0))
            for row in rows
        )
        neutral_registered_positions = sum(
            int(row.get("scored_positions", 0)) for row in rows
        )
        neutral_registered_prefix_changes = sum(
            int(row.get("neutral_vs_registered_prefix_changes", 0))
            for row in rows
        )
        neutral_registered_prefix_positions = sum(
            int(row.get("position", 0)) for row in rows
        )
        first_order_rate = _ratio(first_order, positions)
        pair_cross_rate = _ratio(pair_cross, positions)
        target_after_rate = _ratio(target_after, positions)
        cell = {
            "positions": positions,
            "first_order_predicted_pair_crosses": first_order,
            "first_order_predicted_pair_cross_rate": first_order_rate,
            "realized_pair_crosses": pair_cross,
            "realized_pair_cross_rate": pair_cross_rate,
            "pair_cross_prediction_gap": (
                pair_cross_rate - first_order_rate
                if pair_cross_rate is not None and first_order_rate is not None
                else None
            ),
            "target_correct_before": target_before,
            "target_correct_after": target_after,
            "target_correct_after_rate": target_after_rate,
            "collateral_positions": collateral_positions,
            "collateral_helps": collateral_helps,
            "collateral_hurts": collateral_hurts,
            "collateral_net": collateral_helps - collateral_hurts,
            "collateral_help_rate": _ratio(collateral_helps, collateral_positions),
            "collateral_hurt_rate": _ratio(collateral_hurts, collateral_positions),
            "collateral_prediction_changes": collateral_changes,
            "collateral_prediction_change_rate": _ratio(
                collateral_changes, collateral_positions
            ),
            "rows_with_any_collateral_hurt": sum(
                int(row["collateral_hurts"]) > 0 for row in rows
            ),
            "causal_prefix_prediction_changes": sum(
                int(row.get("causal_prefix_prediction_changes", 0)) for row in rows
            ),
            "neutral_vs_registered_prediction_changes": neutral_registered_changes,
            "neutral_vs_registered_prediction_change_rate": _ratio(
                neutral_registered_changes,
                neutral_registered_positions,
            ),
            "neutral_vs_registered_prefix_changes": neutral_registered_prefix_changes,
            "neutral_vs_registered_prefix_change_rate": _ratio(
                neutral_registered_prefix_changes,
                neutral_registered_prefix_positions,
            ),
            "neutral_vs_registered_target_changes": sum(
                bool(row.get("neutral_vs_registered_target_changed", False))
                for row in rows
            ),
            "future_positions": future_positions,
            "future_helps": future_helps,
            "future_hurts": future_hurts,
            "future_net": future_helps - future_hurts,
            "future_help_rate": _ratio(future_helps, future_positions),
            "future_hurt_rate": _ratio(future_hurts, future_positions),
            "future_prediction_changes": future_changes,
            "future_prediction_change_rate": _ratio(
                future_changes, future_positions
            ),
            "radius": quantile_summary(
                [float(row["radius"]) for row in rows if "radius" in row]
            ),
            "state_rms": quantile_summary(
                [float(row["state_rms"]) for row in rows if "state_rms" in row]
            ),
            "gradient_l2": quantile_summary(
                [float(row["gradient_l2"]) for row in rows if "gradient_l2" in row]
            ),
            "margin_before": quantile_summary(
                [float(row["margin_before"]) for row in rows if "margin_before" in row]
            ),
            "margin_after": quantile_summary(
                [float(row["margin_after"]) for row in rows if "margin_after" in row]
            ),
        }
        if cohort == "oracle_help":
            cell["realized_teacher_flips"] = target_after
            cell["realized_teacher_flip_rate"] = target_after_rate
            cell["teacher_flip_minus_first_order_gap"] = (
                target_after_rate - first_order_rate
                if target_after_rate is not None and first_order_rate is not None
                else None
            )
        else:
            cell["target_preservation_rate"] = target_after_rate
        result[cohort][c_key] = cell
    return dict(result)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _manifest_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _expected_append_cache_paths(
    *, cache_dir: Path, rows: Sequence[dict[str, Any]], batch_size: int
) -> list[tuple[Path, list[int]]]:
    batches = group_batches(rows, int(batch_size))
    return [
        (cache_dir / f"batch_{batch_number:06d}.pt", indices)
        for batch_number, indices in enumerate(batches, start=1)
    ]


def _load_append_predictions(
    *, cache_dir: Path, rows: Sequence[dict[str, Any]], batch_size: int
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor | None] = [None] * len(rows)
    expected = _expected_append_cache_paths(
        cache_dir=cache_dir, rows=rows, batch_size=batch_size
    )
    paths = [
        path for path, _indices in expected
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "V1 append cache is missing locked-prefix batches: " + json.dumps(missing)
        )
    extras = sorted(set(cache_dir.glob("batch_*.pt")) - set(paths))
    print(
        f"phase2_v1b_append_cache locked_batches={len(paths)} "
        f"ignored_trailing_batches={len(extras)}",
        flush=True,
    )
    for path, expected_indices in expected:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        observed_indices = [int(index) for index in payload["indices"]]
        if observed_indices != [int(index) for index in expected_indices]:
            raise RuntimeError(
                f"V1 append cache batch {path.name} indices differ from the "
                f"length-grouped contract: observed={observed_indices}, "
                f"expected={expected_indices}"
            )
        for local, row_index in enumerate(payload["indices"]):
            outputs[int(row_index)] = payload["predictions"][local]
    if any(value is None for value in outputs):
        raise RuntimeError("V1 append cache left selected rows missing")
    return [value for value in outputs if value is not None]


def _position_sets(
    *,
    wrapper: Any,
    rows: Sequence[dict[str, Any]],
    teacher_rows: Sequence[torch.Tensor],
    append_grids: Sequence[torch.Tensor],
    vocab_size: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    helps: list[dict[str, Any]] = []
    preserves: list[dict[str, Any]] = []
    for row_index, (row, target, grid) in enumerate(
        zip(rows, teacher_rows, append_grids)
    ):
        values = torch.tensor([row["input_ids"]], device=device)
        with torch.inference_mode():
            output = wrapper(
                input_ids=values,
                attention_mask=torch.ones_like(values),
                max_loops=1,
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[0, :-1, :vocab_size].float().cpu()
        target = target.long()
        baseline = logits.argmax(dim=-1)
        candidate = grid[:, 1].long()
        if not (baseline.shape == candidate.shape == target.shape):
            raise ValueError("V1b baseline, append, and teacher positions misalign")
        help_mask = baseline.ne(target) & candidate.eq(target)
        preserve_mask = baseline.eq(target) & candidate.eq(target)
        top_two = logits.topk(k=2, dim=-1).indices
        strongest_non_teacher = torch.where(
            top_two[:, 0].eq(target), top_two[:, 1], top_two[:, 0]
        )
        for cohort, mask in (
            ("oracle_help", help_mask),
            ("preserve_control", preserve_mask),
        ):
            destination = helps if cohort == "oracle_help" else preserves
            for position in torch.where(mask)[0].tolist():
                wrong_id = (
                    int(baseline[position])
                    if cohort == "oracle_help"
                    else int(strongest_non_teacher[position])
                )
                teacher_id = int(target[position])
                destination.append(
                    {
                        "cohort": cohort,
                        "row_id": str(row["row_id"]),
                        "row_index": row_index,
                        "position": int(position),
                        "stratum": str(row["stratum"]),
                        "wrong_token_id": wrong_id,
                        "teacher_token_id": teacher_id,
                        "margin": float(
                            logits[position, wrong_id] - logits[position, teacher_id]
                        ),
                    }
                )
        if row_index == 0 or (row_index + 1) % 16 == 0:
            print(
                f"phase2_v1b_scan rows={row_index + 1}/{len(rows)} "
                f"helps={len(helps)} preserves={len(preserves)}",
                flush=True,
            )
    return helps, preserves


def _upper_stack_hidden(
    wrapper: Any,
    hidden: torch.Tensor,
    *,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
) -> torch.Tensor:
    block = run_recurrent_block(
        wrapper,
        hidden,
        causal_mask,
        position_ids,
        cache_position,
        position_embeddings,
    )
    coda, _ = wrapper._run_layer_range(  # noqa: SLF001
        start=wrapper.layer_split.recurrent_end,
        end=len(wrapper.qwen.layers),
        hidden_states=block,
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
    return coda


def _batch_context(
    wrapper: Any,
    *,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    base_position_ids: torch.Tensor,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, Any]:
    batch = hidden.shape[0]
    expanded_attention = attention_mask.expand(batch, -1)
    position_ids = base_position_ids.expand(batch, -1)
    causal_mask = wrapper._update_causal_mask(  # noqa: SLF001
        expanded_attention,
        hidden,
        cache_position,
        past_key_values=None,
        output_attentions=False,
    )
    rotary = wrapper._rotary_embeddings(hidden, position_ids)  # noqa: SLF001
    return causal_mask, position_ids, rotary


def _predictions_from_coda(
    wrapper: Any,
    coda: torch.Tensor,
    *,
    scored_positions: int,
    vocab_size: int,
    position_chunk: int,
) -> torch.Tensor:
    pieces = []
    for start in range(0, int(scored_positions), int(position_chunk)):
        stop = min(int(scored_positions), start + int(position_chunk))
        logits = wrapper.lm_head(wrapper.qwen.norm(coda[:, start:stop]))[
            :, :, :vocab_size
        ].float()
        pieces.append(logits.argmax(dim=-1).cpu())
    return torch.cat(pieces, dim=1)


def _row_interventions(
    *,
    wrapper: Any,
    row: dict[str, Any],
    teacher: torch.Tensor,
    records: Sequence[dict[str, Any]],
    vocab_size: int,
    gamma: float,
    rho: float,
    c_values: Sequence[float],
    perturbation_batch: int,
    logit_position_chunk: int,
) -> list[dict[str, Any]]:
    device = next(wrapper.parameters()).device
    values = torch.tensor([row["input_ids"]], device=device)
    attention = torch.ones_like(values)
    hidden, _mask, causal_mask, position_ids, cache_position, rotary = (
        prepare_recurrent_inputs(wrapper, values, attention)
    )
    hidden = hidden.detach()
    differentiable = hidden.detach().requires_grad_(True)
    coda = _upper_stack_hidden(
        wrapper,
        differentiable,
        causal_mask=causal_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        position_embeddings=rotary,
    )
    positions = torch.tensor(
        [int(record["position"]) for record in records], device=device
    )
    target_hidden = coda[0, positions]
    target_logits = wrapper.lm_head(wrapper.qwen.norm(target_hidden))[
        :, :vocab_size
    ].float()
    margins = torch.stack(
        [
            target_logits[index, int(record["wrong_token_id"])]
            - target_logits[index, int(record["teacher_token_id"])]
            for index, record in enumerate(records)
        ]
    )
    computed_margins = [float(value) for value in margins.detach().cpu()]
    gradients = []
    for index, margin in enumerate(margins):
        gradient = torch.autograd.grad(
            margin,
            differentiable,
            retain_graph=index + 1 < len(records),
        )[0][0, int(records[index]["position"])].detach().float()
        gradients.append(gradient)
    del coda, target_hidden, target_logits, margins, differentiable

    with torch.no_grad():
        baseline_coda = _upper_stack_hidden(
            wrapper,
            hidden,
            causal_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=rotary,
        )
        baseline_predictions = _predictions_from_coda(
            wrapper,
            baseline_coda,
            scored_positions=len(teacher),
            vocab_size=vocab_size,
            position_chunk=logit_position_chunk,
        )[0]
    teacher = teacher.long().cpu()
    hidden_size = int(hidden.shape[-1])
    prepared: list[dict[str, Any]] = []
    for record, gradient, computed_margin in zip(
        records, gradients, computed_margins
    ):
        norm = float(gradient.norm().cpu())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("V1b encountered a zero or non-finite margin gradient")
        position = int(record["position"])
        state_rms = float(hidden[0, position].float().square().mean().sqrt().cpu())
        observed_margin = float(record["margin"])
        if not math.isfinite(computed_margin):
            raise RuntimeError("V1b margin is non-finite")
        if abs(computed_margin - observed_margin) > 1e-3:
            raise RuntimeError(
                "V1b decomposed upper path disagrees with registered one-loop margin: "
                f"observed={observed_margin}, decomposed={computed_margin}"
            )
        unit = gradient / gradient.norm().clamp_min(1e-12)
        for c_value in c_values:
            radius = tube_radius(
                c_value=float(c_value),
                state_rms=state_rms,
                hidden_size=hidden_size,
                gamma=gamma,
                rho=rho,
            )
            prepared.append(
                {
                    "record": record,
                    "gradient_l2": norm,
                    "state_rms": state_rms,
                    "radius": radius,
                    "delta": -float(radius) * unit,
                    "first_order_predicted_pair_cross": (
                        str(record["cohort"]) == "oracle_help"
                        and observed_margin - float(radius) * norm <= 0.0
                    ),
                    "c_value": float(c_value),
                }
            )

    neutral_predictions_by_size: dict[int, torch.Tensor] = {}

    def neutral_predictions(batch_size: int) -> torch.Tensor:
        cached = neutral_predictions_by_size.get(int(batch_size))
        if cached is not None:
            return cached
        neutral_states = hidden.expand(int(batch_size), -1, -1).clone()
        neutral_mask, neutral_positions, neutral_rotary = _batch_context(
            wrapper,
            hidden=neutral_states,
            attention_mask=attention,
            base_position_ids=position_ids,
            cache_position=cache_position,
        )
        with torch.no_grad():
            neutral_coda = _upper_stack_hidden(
                wrapper,
                neutral_states,
                causal_mask=neutral_mask,
                position_ids=neutral_positions,
                cache_position=cache_position,
                position_embeddings=neutral_rotary,
            )
            observed = _predictions_from_coda(
                wrapper,
                neutral_coda,
                scored_positions=len(teacher),
                vocab_size=vocab_size,
                position_chunk=logit_position_chunk,
            )
        if int(batch_size) > 1 and bool(observed[1:].ne(observed[:1]).any()):
            raise RuntimeError(
                "V1b batch-matched neutral controls disagree across identical rows"
            )
        neutral_predictions_by_size[int(batch_size)] = observed
        return observed

    outputs: list[dict[str, Any]] = []
    for start in range(0, len(prepared), int(perturbation_batch)):
        batch_rows = prepared[start : start + int(perturbation_batch)]
        neutral_batch = neutral_predictions(len(batch_rows))
        states = hidden.expand(len(batch_rows), -1, -1).clone()
        for local, item in enumerate(batch_rows):
            position = int(item["record"]["position"])
            states[local, position] = states[local, position] + item["delta"].to(
                device=states.device, dtype=states.dtype
            )
        batch_mask, batch_positions, batch_rotary = _batch_context(
            wrapper,
            hidden=states,
            attention_mask=attention,
            base_position_ids=position_ids,
            cache_position=cache_position,
        )
        with torch.no_grad():
            perturbed_coda = _upper_stack_hidden(
                wrapper,
                states,
                causal_mask=batch_mask,
                position_ids=batch_positions,
                cache_position=cache_position,
                position_embeddings=batch_rotary,
            )
            predictions = _predictions_from_coda(
                wrapper,
                perturbed_coda,
                scored_positions=len(teacher),
                vocab_size=vocab_size,
                position_chunk=logit_position_chunk,
            )
            selected_hidden = torch.stack(
                [
                    perturbed_coda[local, int(item["record"]["position"])]
                    for local, item in enumerate(batch_rows)
                ]
            )
            selected_logits = wrapper.lm_head(wrapper.qwen.norm(selected_hidden))[
                :, :vocab_size
            ].float().cpu()
        for local, item in enumerate(batch_rows):
            record = item["record"]
            position = int(record["position"])
            predicted = predictions[local]
            comparison = compare_paired_predictions(
                registered=baseline_predictions,
                neutral=neutral_batch[local],
                perturbed=predicted,
                teacher=teacher,
                position=position,
            )
            wrong_id = int(record["wrong_token_id"])
            teacher_id = int(record["teacher_token_id"])
            realized_margin = float(
                selected_logits[local, wrong_id]
                - selected_logits[local, teacher_id]
            )
            outputs.append(
                {
                    "cohort": str(record["cohort"]),
                    "row_id": str(record["row_id"]),
                    "row_index": int(record["row_index"]),
                    "position": position,
                    "stratum": str(record["stratum"]),
                    "c_value": float(item["c_value"]),
                    "radius": float(item["radius"]),
                    "state_rms": float(item["state_rms"]),
                    "gradient_l2": float(item["gradient_l2"]),
                    "margin_before": float(record["margin"]),
                    "margin_after": realized_margin,
                    "first_order_predicted_pair_cross": bool(
                        item["first_order_predicted_pair_cross"]
                    ),
                    "realized_pair_cross": (
                        float(record["margin"]) > 0.0 and realized_margin <= 0.0
                    ),
                    "scored_positions": len(teacher),
                    **comparison,
                }
            )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--v1_summary", required=True)
    parser.add_argument("--v1_private_dir", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--sample_seed", type=int, default=20260731)
    parser.add_argument("--perturbation_batch", type=int, default=8)
    parser.add_argument("--logit_position_chunk", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.8)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("V1b post-D0 checkpoint SHA-256 mismatch")
    v1_summary = json.loads(Path(args.v1_summary).read_text(encoding="utf-8"))
    if v1_summary.get("status") != "complete_no_training_dev_only":
        raise RuntimeError("V1b requires a completed V1/V2 receipt")
    first_order_source_key = (
        "first_order_compatibility_using_margin_gradient_norm"
        if "first_order_compatibility_using_margin_gradient_norm"
        in v1_summary["v1"]
        else "first_order_margin_compatible_fraction"
    )
    first_order_compatibility = v1_summary["v1"][first_order_source_key]
    v1_private = Path(args.v1_private_dir)
    v1_config = json.loads((v1_private / "config.json").read_text(encoding="utf-8"))
    all_rows = read_jsonl(args.data_jsonl)
    selected_indices = sorted(
        range(len(all_rows)),
        key=lambda index: hashlib.sha256(
            f"phase2:{all_rows[index]['row_id']}:-1".encode("utf-8")
        ).hexdigest(),
    )[: int(v1_config["selected_rows"])]
    observed_indices_sha = hashlib.sha256(
        json.dumps(selected_indices, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if observed_indices_sha != v1_config["selected_indices_sha256"]:
        raise RuntimeError("V1b cannot reproduce V1 selected DEV-C rows")
    rows = [all_rows[index] for index in selected_indices]
    teacher_summary = json.loads(
        Path(args.teacher_cache_summary).read_text(encoding="utf-8")
    )
    teacher_cache = load_partition_cache(teacher_summary, "teacher_7b", "dev_c")
    teacher_rows = [
        teacher_cache[index]["teacher_greedy_token_id"].long()
        for index in selected_indices
    ]
    append_dir = v1_private / "v1/append_predictions/trained_append_k1"
    expected_append_paths = [
        path
        for path, _indices in _expected_append_cache_paths(
            cache_dir=append_dir,
            rows=rows,
            batch_size=int(v1_config["append_batch_size"]),
        )
    ]
    append_grids = _load_append_predictions(
        cache_dir=append_dir,
        rows=rows,
        batch_size=int(v1_config["append_batch_size"]),
    )

    _tokenizer, wrapper, resize, _vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype="float32",
        attn_implementation="sdpa",
    )
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    wrapper.eval()
    helps, preserves = _position_sets(
        wrapper=wrapper,
        rows=rows,
        teacher_rows=teacher_rows,
        append_grids=append_grids,
        vocab_size=resize.original_tokenizer_size,
        device=args.device,
    )
    registered_help_path = v1_private / "v1_help_records.json"
    registered_helps = json.loads(registered_help_path.read_text(encoding="utf-8"))
    registered_keys = {
        (
            int(record["row_index"]),
            int(record["position"]),
            int(record["baseline_token_id"]),
            int(record["teacher_token_id"]),
        )
        for record in registered_helps
    }
    observed_keys = {
        (
            int(record["row_index"]),
            int(record["position"]),
            int(record["wrong_token_id"]),
            int(record["teacher_token_id"]),
        )
        for record in helps
    }
    if registered_keys != observed_keys:
        raise RuntimeError("V1b oracle-help reconstruction differs from V1")
    sampled_help = deterministic_position_sample(
        helps,
        sample_size=args.sample_size,
        seed=args.sample_seed,
        cohort="oracle_help",
    )
    sampled_preserve = deterministic_position_sample(
        preserves,
        sample_size=args.sample_size,
        seed=args.sample_seed,
        cohort="preserve_control",
    )
    selected = sampled_help + sampled_preserve
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        by_row[int(record["row_index"])].append(record)

    private = Path(args.private_dir)
    private.mkdir(parents=True, exist_ok=True)
    config = {
        "kind": "paper2_phase2_v1b_private_config",
        "data_jsonl_sha256": sha256_file(args.data_jsonl),
        "teacher_cache_summary_sha256": sha256_file(args.teacher_cache_summary),
        "checkpoint_sha256": args.checkpoint_sha256,
        "v1_summary_sha256": sha256_file(args.v1_summary),
        "v1_config_sha256": sha256_file(v1_private / "config.json"),
        "v1_help_records_sha256": sha256_file(registered_help_path),
        "v1_append_cache_manifest_sha256": _manifest_sha256(
            expected_append_paths
        ),
        "sample_size_per_cohort": args.sample_size,
        "sample_seed": args.sample_seed,
        "c_values": list(C_VALUES),
        "gamma": args.gamma,
        "rho": args.rho,
        "perturbation_batch": args.perturbation_batch,
        "logit_position_chunk": args.logit_position_chunk,
        "comparison_baseline": "same_shape_same_batch_index_neutral_v2",
    }
    config_path = private / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("V1b resume configuration differs from private lock")
    else:
        _write_json(config_path, config)

    row_dir = private / "rows"
    outputs: list[dict[str, Any]] = []
    for number, row_index in enumerate(sorted(by_row), start=1):
        row_path = row_dir / f"row_{row_index:05d}.json"
        if row_path.exists():
            row_outputs = json.loads(row_path.read_text(encoding="utf-8"))
            expected = len(by_row[row_index]) * len(C_VALUES)
            if len(row_outputs) != expected:
                raise RuntimeError(
                    f"V1b cached row {row_index} has {len(row_outputs)} records; "
                    f"expected {expected}"
                )
        else:
            row_outputs = _row_interventions(
                wrapper=wrapper,
                row=rows[row_index],
                teacher=teacher_rows[row_index],
                records=by_row[row_index],
                vocab_size=resize.original_tokenizer_size,
                gamma=args.gamma,
                rho=args.rho,
                c_values=C_VALUES,
                perturbation_batch=args.perturbation_batch,
                logit_position_chunk=args.logit_position_chunk,
            )
            _write_json(row_path, row_outputs)
        outputs.extend(row_outputs)
        print(
            f"phase2_v1b_progress rows={number}/{len(by_row)} "
            f"records={len(outputs)}/{len(selected) * len(C_VALUES)}",
            flush=True,
        )
    expected_records = len(selected) * len(C_VALUES)
    if len(outputs) != expected_records:
        raise RuntimeError(
            f"V1b record count {len(outputs)} != expected {expected_records}"
        )
    if any(parameter.grad is not None for parameter in wrapper.parameters()):
        raise RuntimeError("V1b unexpectedly populated a model-parameter gradient")

    public = {
        "kind": "paper2_phase2_v1b_finite_perturbation",
        "status": "complete_no_training_dev_only",
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "sample": {
            "seed": args.sample_seed,
            "oracle_help_positions_available": len(helps),
            "preserve_positions_available": len(preserves),
            "positions_per_cohort": args.sample_size,
            "strata": {
                cohort: dict(
                    Counter(
                        str(row["stratum"])
                        for row in selected
                        if row["cohort"] == cohort
                    )
                )
                for cohort in ("oracle_help", "preserve_control")
            },
        },
        "constants": {
            "c_values": list(C_VALUES),
            "gamma": args.gamma,
            "rho": args.rho,
            "radius_formula": "gamma*c*RMS(h0)*sqrt(d)/(1-rho)",
        },
        "v1_terminology_reconciliation": {
            "source_key": first_order_source_key,
            "governing_name": "first_order compatibility using the margin-gradient norm",
            "values_unchanged": True,
            "first_order_compatibility_using_margin_gradient_norm": first_order_compatibility,
        },
        "results": {
            "pooled": aggregate_intervention_records(outputs),
            "by_stratum": {
                stratum: aggregate_intervention_records(
                    [row for row in outputs if str(row["stratum"]) == stratum]
                )
                for stratum in sorted({str(row["stratum"]) for row in outputs})
            },
        },
        "sources": {
            key: value
            for key, value in config.items()
            if key.endswith("sha256")
        },
        "method_notes": [
            "First-order prediction is crossing of the original top-1 wrong token versus teacher-token margin.",
            "Realized pair crossing and realized teacher-token top-1 flip are reported separately.",
            "Collateral counts cover every other scored causal position on the same row.",
            "Causally exposed future-position collateral is reported separately so structurally unchanged prefix positions do not dilute the rate.",
            "Perturbed predictions are compared with an unmodified forward at the same batch size and batch index; differences from the registered batch-1 path are reported separately as numerical-kernel sensitivity.",
            "Preserve controls are baseline-and-trained-append correct and receive a teacher-favoring perturbation against their strongest non-teacher competitor.",
        ],
        "do_not_claim": [
            "V1b perturbations are a deployable controller",
            "pair crossing guarantees the teacher token is top-1",
            "local finite perturbations establish global reachability",
            "DEV-only results are frozen-slice confirmation",
        ],
    }
    _write_json(Path(args.output_summary), public)
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
