"""Read-only deployable-feature probes for D0 depth routing."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_speculative_depth_router_feasibility import (
    INTERIOR_BUDGET_FRACTIONS,
    ROUTER_AUDIT_SEED,
    binary_auroc,
    budget_policy_curve,
    deterministic_group_split,
    oracle_depth_profile,
    router_verdict,
)
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_spec import DRAFTER_CHECKPOINT_SHA256


PROJECTION_DIM = 128
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
BOOTSTRAP_DRAWS = 500
SEQUENTIAL_BUDGETS = (1.25, 1.50, 2.0, 3.0)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixed_projection(hidden_size: int, projection_dim: int, seed: int) -> torch.Tensor:
    if projection_dim < 1 or projection_dim > hidden_size:
        raise ValueError("projection dimension must be within the hidden width")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(hidden_size, projection_dim, generator=generator, dtype=torch.float32)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.contiguous()


def frozen_parameter_fingerprint(module: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = parameter.detach().float()
            payload = (
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                float(values.sum().item()),
                float(values.square().sum().item()),
            )
            digest.update(repr(payload).encode("utf-8"))
    return digest.hexdigest()


def row_cache_valid(path: Path, signature: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return payload.get("signature") == signature


@torch.inference_mode()
def extract_feature_cache(
    *,
    data_jsonl: Path,
    private_rows_path: Path,
    checkpoint: Path,
    cache_path: Path,
    row_cache_dir: Path,
    expected_private_sha256: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    projection_dim: int,
    projection_seed: int,
) -> dict[str, Any]:
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError("router probe checkpoint SHA-256 mismatch")
    private_sha = sha256_file(private_rows_path)
    if private_sha != expected_private_sha256:
        raise RuntimeError("router probe private-floor SHA-256 mismatch")
    runtime_hardware = (
        torch.cuda.get_device_name(torch.device(device))
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else str(device)
    )
    signature = {
        "kind": "paper2_d0_router_feature_cache_v3_cross_hardware_sensitivity",
        "checkpoint_sha256": checkpoint_sha,
        "private_rows_sha256": private_sha,
        "data_jsonl_sha256": sha256_file(data_jsonl),
        "projection_dim": int(projection_dim),
        "projection_seed": int(projection_seed),
        "runtime_hardware": runtime_hardware,
        "torch_version": str(torch.__version__),
        "forced_depths": [1, 2, 3, 4, 5, 6],
        "scalar_feature_schema": [
            "answer_top1_top2_margin",
            "answer_top1_logprob",
            "answer_top1_logit",
            "answer_prediction_changed",
            "state_rms",
            "state_update_rms",
            "state_previous_cosine",
            "control_stop_minus_continue_margin",
        ],
        "teacher_features_excluded": True,
    }
    if row_cache_valid(cache_path, signature):
        print(f"router_feature_cache_reused={cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    sources = read_jsonl(data_jsonl)
    private = read_json(private_rows_path)
    floor_rows = list(private["rows"])
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in floor_rows:
        by_source[int(row["row_index"])].append(row)
    for values in by_source.values():
        values.sort(key=lambda row: int(row["local_position"]))

    tokenizer, wrapper, resize, _ = load_drafter(
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    del tokenizer
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in wrapper.parameters()):
        raise AssertionError("router feature extraction requires an entirely frozen wrapper")
    wrapper.eval()
    before = frozen_parameter_fingerprint(wrapper)
    hidden_size = int(wrapper.base_model.config.hidden_size)
    projection = fixed_projection(hidden_size, projection_dim, projection_seed).to(device)

    metadata: list[dict[str, Any]] = []
    prelude_chunks: list[torch.Tensor] = []
    state_chunks: list[torch.Tensor] = []
    scalar_chunks: list[torch.Tensor] = []
    match_chunks: list[torch.Tensor] = []
    runtime_match_chunks: list[torch.Tensor] = []
    mismatch_chunks: list[torch.Tensor] = []
    mismatch_gap_chunks: list[torch.Tensor] = []
    row_cache_dir.mkdir(parents=True, exist_ok=True)

    processed_positions = 0
    processed_rows = 0
    for row_index, source in enumerate(sources):
        records = by_source.get(row_index)
        if not records:
            continue
        row_signature = {**signature, "row_index": row_index}
        row_cache = row_cache_dir / f"row_{row_index:06d}.pt"
        if row_cache_valid(row_cache, row_signature):
            result = torch.load(row_cache, map_location="cpu", weights_only=False)
        else:
            values = [int(value) for value in source["input_ids"]]
            input_ids = torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)
            attention = torch.ones_like(input_ids)
            prelude_captures: list[torch.Tensor] = []
            recurrent_states: list[torch.Tensor] = []

            def capture_prelude(
                _module: Any,
                _inputs: tuple[Any, ...],
                layer_output: Any,
            ) -> None:
                state = layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output
                prelude_captures.append(state)

            prelude_layer = wrapper.qwen.layers[int(wrapper.layer_split.prelude_end) - 1]
            hook = prelude_layer.register_forward_hook(capture_prelude)
            try:
                # Keep this call equivalent to eval_speculative_depth_d0_floor.forced_predictions.
                # Hooks and the recurrent-state sink observe tensors without selecting a different
                # model output path. Frozen-floor predictions remain the primary labels;
                # any cross-hardware bfloat16 argmax drift is measured separately below.
                output = wrapper(
                    input_ids=input_ids,
                    attention_mask=attention,
                    labels=None,
                    max_loops=6,
                    use_cache=False,
                    return_dict=True,
                    return_loop_logits=True,
                    recurrent_application_sink=recurrent_states,
                )
            finally:
                hook.remove()
            if output.loop_logits is None or len(recurrent_states) != 6:
                raise RuntimeError("router probe requires loop logits and six recurrent states")
            if len(prelude_captures) != 1:
                raise RuntimeError("router probe did not capture exactly one Prelude output")
            prelude = prelude_captures[0][0]
            states = torch.stack(recurrent_states, dim=1)[0]
            positions = torch.tensor(
                [int(record["local_position"]) for record in records],
                device=device,
                dtype=torch.long,
            )
            if int(positions.max().item()) >= len(values) - 1:
                raise RuntimeError("router probe local position exceeds the next-token range")
            prelude_selected = prelude.index_select(0, positions).float()
            states_selected = states.index_select(1, positions).transpose(0, 1).float()
            prelude_projected = prelude_selected @ projection
            states_projected = states_selected @ projection

            all_logits = output.loop_logits[0, 0, :, : len(values) - 1]
            logits = all_logits[..., : resize.original_tokenizer_size]
            selected_logits = logits.index_select(1, positions)
            control_ids = torch.tensor(
                resize.control_token_ids[:2],
                device=device,
                dtype=torch.long,
            )
            selected_control = all_logits.index_select(1, positions).index_select(
                -1, control_ids
            )
            predictions: list[torch.Tensor] = []
            scalar_by_loop: list[torch.Tensor] = []
            previous_state = prelude_selected
            previous_prediction: torch.Tensor | None = None
            for loop_index in range(6):
                loop_scores = selected_logits[loop_index]
                control_scores = selected_control[loop_index].float()
                top = loop_scores.topk(k=2, dim=-1)
                prediction = top.indices[:, 0]
                predictions.append(prediction)
                log_normalizer = torch.logsumexp(loop_scores.float(), dim=-1)
                top1 = top.values[:, 0].float()
                top2 = top.values[:, 1].float()
                current_state = states_selected[:, loop_index]
                delta = current_state - previous_state
                cosine = torch.nn.functional.cosine_similarity(
                    current_state,
                    previous_state,
                    dim=-1,
                    eps=1e-8,
                )
                changed = (
                    torch.zeros_like(prediction, dtype=torch.float32)
                    if previous_prediction is None
                    else prediction.ne(previous_prediction).float()
                )
                scalar_by_loop.append(
                    torch.stack(
                        [
                            top1 - top2,
                            top1 - log_normalizer,
                            top1,
                            changed,
                            current_state.square().mean(dim=-1).sqrt(),
                            delta.square().mean(dim=-1).sqrt(),
                            cosine,
                            control_scores[:, 1] - control_scores[:, 0],
                        ],
                        dim=-1,
                    )
                )
                previous_state = current_state
                previous_prediction = prediction
            predicted = torch.stack(predictions, dim=1).cpu()
            expected = torch.tensor(
                [[int(value) for value in record["predictions"]] for record in records],
                dtype=torch.long,
            )
            teacher_7b = torch.tensor(
                [int(record["teacher_7b"]) for record in records],
                dtype=torch.long,
            )
            prediction_mismatch = predicted.ne(expected)
            runtime_matches = predicted.eq(teacher_7b.unsqueeze(1))
            scores_by_position = selected_logits.transpose(0, 1).float()
            runtime_logit = scores_by_position.gather(
                -1, predicted.to(device=device).unsqueeze(-1)
            ).squeeze(-1)
            reference_logit = scores_by_position.gather(
                -1, expected.to(device=device).unsqueeze(-1)
            ).squeeze(-1)
            runtime_over_reference_logit_gap = (runtime_logit - reference_logit).cpu()
            result = {
                "signature": row_signature,
                "metadata": [
                    {
                        "row_index": row_index,
                        "local_position": int(record["local_position"]),
                        "sequence_length": len(values),
                        "stratum": str(record["stratum"]),
                    }
                    for record in records
                ],
                "prelude_projection": prelude_projected.to(device="cpu", dtype=torch.bfloat16),
                "state_projection": states_projected.to(device="cpu", dtype=torch.bfloat16),
                "scalars": torch.stack(scalar_by_loop, dim=1).to(device="cpu", dtype=torch.float32),
                "matches": torch.tensor(
                    [[bool(value) for value in record["matches_teacher_7b"]] for record in records],
                    dtype=torch.bool,
                ),
                # The frozen floor remains the primary outcome source. Runtime labels are
                # retained only for a complete cross-hardware sensitivity analysis because
                # bfloat16 SDPA argmax is not guaranteed bit-identical across GPU classes.
                "runtime_matches": runtime_matches,
                "prediction_mismatch": prediction_mismatch,
                "runtime_over_reference_logit_gap": runtime_over_reference_logit_gap,
            }
            temporary = row_cache.with_suffix(".pt.tmp")
            torch.save(result, temporary)
            os.replace(temporary, row_cache)
            del output, all_logits, logits, selected_logits, selected_control, states, prelude
        metadata.extend(result["metadata"])
        prelude_chunks.append(result["prelude_projection"])
        state_chunks.append(result["state_projection"])
        scalar_chunks.append(result["scalars"])
        match_chunks.append(result["matches"])
        runtime_match_chunks.append(result["runtime_matches"])
        mismatch_chunks.append(result["prediction_mismatch"])
        mismatch_gap_chunks.append(result["runtime_over_reference_logit_gap"])
        processed_positions += len(result["metadata"])
        processed_rows += 1
        if processed_rows == 1 or processed_rows % 16 == 0 or processed_positions == len(floor_rows):
            print(
                f"router_feature_progress source_rows={processed_rows} positions={processed_positions}/{len(floor_rows)} "
                f"reference_mismatches={sum(int(chunk.sum().item()) for chunk in mismatch_chunks)}",
                flush=True,
            )

    after = frozen_parameter_fingerprint(wrapper)
    if before != after:
        raise RuntimeError("frozen wrapper fingerprint changed during router feature extraction")
    payload = {
        "signature": signature,
        "frozen_parameter_fingerprint_before": before,
        "frozen_parameter_fingerprint_after": after,
        "metadata": metadata,
        "prelude_projection": torch.cat(prelude_chunks, dim=0),
        "state_projection": torch.cat(state_chunks, dim=0),
        "scalars": torch.cat(scalar_chunks, dim=0),
        "matches": torch.cat(match_chunks, dim=0),
        "runtime_matches": torch.cat(runtime_match_chunks, dim=0),
        "prediction_mismatch": torch.cat(mismatch_chunks, dim=0),
        "runtime_over_reference_logit_gap": torch.cat(mismatch_gap_chunks, dim=0),
    }
    if len(metadata) != len(floor_rows):
        raise RuntimeError("router feature cache does not cover every private floor position")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"router_feature_cache_saved={cache_path}", flush=True)
    return payload


def prediction_equivalence_summary(cache: dict[str, Any]) -> dict[str, Any]:
    mismatch = cache["prediction_mismatch"].bool()
    logit_gap = cache["runtime_over_reference_logit_gap"].float()
    metadata = list(cache["metadata"])
    if mismatch.dim() != 2 or mismatch.shape[1] != 6:
        raise ValueError("prediction mismatch matrix must be [positions, six loops]")
    affected_positions = mismatch.any(dim=1)
    affected_source_rows = {
        (int(metadata[index]["row_index"]), str(metadata[index]["stratum"]))
        for index in affected_positions.nonzero(as_tuple=False).flatten().tolist()
    }
    mismatches = int(mismatch.sum().item())
    total = int(mismatch.numel())
    mismatch_gaps = logit_gap[mismatch]
    return {
        "reference_label_source": "frozen_floor_receipt",
        "runtime_label_source": "current_feature_extraction_forward",
        "runtime_hardware": cache["signature"]["runtime_hardware"],
        "exact_match_cells": total - mismatches,
        "mismatch_cells": mismatches,
        "total_cells": total,
        "mismatch_rate": mismatches / total if total else 0.0,
        "mismatch_runtime_over_reference_logit_gap": {
            "median": float(mismatch_gaps.median().item()) if mismatches else 0.0,
            "p95": float(torch.quantile(mismatch_gaps, 0.95).item()) if mismatches else 0.0,
            "maximum": float(mismatch_gaps.max().item()) if mismatches else 0.0,
        },
        "affected_positions": int(affected_positions.sum().item()),
        "total_positions": int(mismatch.shape[0]),
        "affected_source_rows": len(affected_source_rows),
        "per_loop_mismatch_cells": {
            str(loop + 1): int(mismatch[:, loop].sum().item()) for loop in range(6)
        },
        "primary_analysis_uses_frozen_floor": True,
        "runtime_analysis_is_sensitivity_only": True,
    }


def standardize_from_train(
    features: torch.Tensor, train_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = features.index_select(0, train_indices).float()
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (features.float() - mean) / scale, mean, scale


def weighted_ridge_fit(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    x = features.index_select(0, indices).float()
    y_bool = labels.index_select(0, indices).bool()
    positives = int(y_bool.sum().item())
    negatives = len(y_bool) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ridge probe needs both label classes")
    y = y_bool.float().mul(2.0).sub(1.0)
    weights = torch.where(
        y_bool,
        torch.full_like(y, len(y) / (2.0 * positives)),
        torch.full_like(y, len(y) / (2.0 * negatives)),
    )
    design = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
    weighted = design * weights.sqrt().unsqueeze(1)
    gram = (weighted.T @ weighted) / len(x)
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype) * float(ridge)
    penalty[-1, -1] = 0.0
    target = (design.T @ (weights * y)) / len(x)
    return torch.linalg.solve(gram + penalty, target)


def ridge_scores(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    design = torch.cat(
        [features.float(), torch.ones(len(features), 1, dtype=features.dtype)], dim=1
    )
    return design @ weights


def fit_ridge_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    *,
    ridge_grid: Sequence[float] = RIDGE_GRID,
) -> dict[str, Any]:
    standardized, _, _ = standardize_from_train(features, train_indices)
    best: tuple[float, float, torch.Tensor] | None = None
    validation_labels = labels.index_select(0, validation_indices).tolist()
    for ridge in ridge_grid:
        try:
            weights = weighted_ridge_fit(standardized, labels, train_indices, float(ridge))
        except RuntimeError:
            continue
        scores = ridge_scores(standardized.index_select(0, validation_indices), weights)
        auc = binary_auroc(scores.tolist(), validation_labels)
        value = float(auc) if auc is not None else float("-inf")
        candidate = (value, -float(ridge), weights)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("ridge grid produced no candidate")
    weights = best[2]
    validation_scores = ridge_scores(
        standardized.index_select(0, validation_indices), weights
    )
    test_scores = ridge_scores(standardized.index_select(0, test_indices), weights)
    return {
        "ridge": -best[1],
        "validation_auroc": best[0],
        "test_auroc": binary_auroc(
            test_scores.tolist(), labels.index_select(0, test_indices).tolist()
        ),
        "validation_scores": validation_scores,
        "test_scores": test_scores,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    location = (len(ordered) - 1) * float(q)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_budget_lower_bounds(
    rows: list[dict[str, Any]],
    *,
    fractions: Sequence[float],
    draws: int,
    seed: int,
) -> dict[float, float]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["row_index"]), str(row["stratum"]))].append(row)
    keys = sorted(groups)
    generator = random.Random(int(seed))
    samples: dict[float, list[float]] = {float(value): [] for value in fractions}
    for _ in range(int(draws)):
        boot: list[dict[str, Any]] = []
        for _group_index in range(len(keys)):
            key = keys[generator.randrange(len(keys))]
            boot.extend(groups[key])
        points = budget_policy_curve(boot, score_field="score", fractions=fractions)
        for point in points:
            samples[float(point["fraction"])].append(float(point["uplift_vs_random"]))
    return {fraction: percentile(values, 0.025) for fraction, values in samples.items()}


def simulate_sequential_policy(
    scores: torch.Tensor, matches: torch.Tensor, threshold: float
) -> dict[str, Any]:
    if scores.shape[1] != 5 or matches.shape[1] != 6 or len(scores) != len(matches):
        raise ValueError("sequential policy requires five decisions and six outcomes")
    selected: list[int] = []
    correct = 0
    for row_index in range(len(scores)):
        depth = 6
        for loop_index in range(5):
            if float(scores[row_index, loop_index]) < float(threshold):
                depth = loop_index + 1
                break
        selected.append(depth)
        correct += int(bool(matches[row_index, depth - 1]))
    return {
        "threshold": float(threshold),
        "correct": correct,
        "total": len(matches),
        "accuracy": correct / len(matches),
        "mean_loops": sum(selected) / len(selected),
        "selected_depth_counts": {
            str(depth): selected.count(depth) for depth in range(1, 7)
        },
    }


def threshold_candidates(scores: torch.Tensor, count: int = 101) -> list[float]:
    values = sorted(float(value) for value in scores.flatten().tolist())
    candidates = [values[0] - 1.0]
    candidates.extend(percentile(values, index / (count - 1)) for index in range(count))
    candidates.append(values[-1] + 1.0)
    return sorted(set(candidates))


def select_sequential_frontier(
    validation_scores: torch.Tensor,
    validation_matches: torch.Tensor,
    test_scores: torch.Tensor,
    test_matches: torch.Tensor,
    budgets: Sequence[float] = SEQUENTIAL_BUDGETS,
) -> list[dict[str, Any]]:
    candidates = [
        simulate_sequential_policy(validation_scores, validation_matches, threshold)
        for threshold in threshold_candidates(validation_scores)
    ]
    output: list[dict[str, Any]] = []
    for budget in budgets:
        feasible = [point for point in candidates if point["mean_loops"] <= float(budget)]
        if not feasible:
            feasible = [min(candidates, key=lambda point: point["mean_loops"])]
        selected = max(
            feasible,
            key=lambda point: (point["accuracy"], -point["mean_loops"]),
        )
        test = simulate_sequential_policy(
            test_scores, test_matches, float(selected["threshold"])
        )
        output.append(
            {
                "target_mean_loop_budget": float(budget),
                "validation": selected,
                "test": test,
            }
        )
    return output


def analyze_feature_cache(cache: dict[str, Any], *, seed: int) -> dict[str, Any]:
    metadata = list(cache["metadata"])
    prelude = cache["prelude_projection"].float()
    states = cache["state_projection"].float()
    scalars = cache["scalars"].float()
    matches = cache["matches"].bool()
    mapping = deterministic_group_split(metadata, seed=seed)
    split_names = [mapping[(int(row["row_index"]), str(row["stratum"]))] for row in metadata]
    indices = {
        name: torch.tensor(
            [index for index, value in enumerate(split_names) if value == name],
            dtype=torch.long,
        )
        for name in ("train", "validation", "test")
    }
    structural = torch.tensor(
        [
            [
                int(row["local_position"]) / max(1, int(row["sequence_length"]) - 1),
                math.log(max(2, int(row["sequence_length"]))),
                float(str(row["stratum"]) == "code"),
            ]
            for row in metadata
        ],
        dtype=torch.float32,
    )
    any_extra = torch.tensor(
        [not bool(row[0]) and any(bool(value) for value in row[1:]) for row in matches],
        dtype=torch.bool,
    )
    loop2_benefit = torch.tensor(
        [not bool(row[0]) and bool(row[1]) for row in matches],
        dtype=torch.bool,
    )
    any_extra_hidden = fit_ridge_probe(
        torch.cat([prelude, structural], dim=1),
        any_extra,
        indices["train"],
        indices["validation"],
        indices["test"],
    )
    any_extra_structural = fit_ridge_probe(
        structural,
        any_extra,
        indices["train"],
        indices["validation"],
        indices["test"],
    )
    loop2_hidden = fit_ridge_probe(
        torch.cat([prelude, structural], dim=1),
        loop2_benefit,
        indices["train"],
        indices["validation"],
        indices["test"],
    )
    loop2_structural = fit_ridge_probe(
        structural,
        loop2_benefit,
        indices["train"],
        indices["validation"],
        indices["test"],
    )
    test_indices = indices["test"]
    test_rows: list[dict[str, Any]] = []
    test_scores = loop2_hidden["test_scores"].tolist()
    for local_index, global_index in enumerate(test_indices.tolist()):
        test_rows.append(
            {
                **metadata[global_index],
                "loop1_correct": bool(matches[global_index, 0]),
                "loop2_correct": bool(matches[global_index, 1]),
                "score": float(test_scores[local_index]),
            }
        )
    budget_points = budget_policy_curve(
        test_rows,
        score_field="score",
        fractions=INTERIOR_BUDGET_FRACTIONS,
    )
    lower = cluster_bootstrap_budget_lower_bounds(
        test_rows,
        fractions=INTERIOR_BUDGET_FRACTIONS,
        draws=BOOTSTRAP_DRAWS,
        seed=seed,
    )
    for point in budget_points:
        point["bootstrap_low"] = lower[float(point["fraction"])]
    preloop_verdict = router_verdict(
        auroc=float(loop2_hidden["test_auroc"]),
        budget_points=budget_points,
    )

    sequential: dict[str, Any] = {}
    combined_validation: list[torch.Tensor] = []
    combined_test: list[torch.Tensor] = []
    previous = prelude
    for loop_index in range(5):
        current = states[:, loop_index]
        labels = torch.tensor(
            [
                not bool(row[loop_index])
                and any(bool(value) for value in row[loop_index + 1 :])
                for row in matches
            ],
            dtype=torch.bool,
        )
        combined_features = torch.cat(
            [prelude, current, current - previous, scalars[:, loop_index], structural],
            dim=1,
        )
        scalar_features = torch.cat([scalars[:, loop_index], structural], dim=1)
        combined_probe = fit_ridge_probe(
            combined_features,
            labels,
            indices["train"],
            indices["validation"],
            indices["test"],
        )
        scalar_probe = fit_ridge_probe(
            scalar_features,
            labels,
            indices["train"],
            indices["validation"],
            indices["test"],
        )
        combined_validation.append(combined_probe["validation_scores"])
        combined_test.append(combined_probe["test_scores"])
        sequential[str(loop_index + 1)] = {
            "positive_total": int(labels.sum().item()),
            "positive_test": int(labels.index_select(0, test_indices).sum().item()),
            "combined_validation_auroc": combined_probe["validation_auroc"],
            "combined_test_auroc": combined_probe["test_auroc"],
            "scalar_validation_auroc": scalar_probe["validation_auroc"],
            "scalar_test_auroc": scalar_probe["test_auroc"],
            "combined_ridge": combined_probe["ridge"],
            "scalar_ridge": scalar_probe["ridge"],
        }
        previous = current
    validation_scores = torch.stack(combined_validation, dim=1)
    test_sequential_scores = torch.stack(combined_test, dim=1)
    validation_matches = matches.index_select(0, indices["validation"])
    test_matches = matches.index_select(0, test_indices)
    sequential_frontier = select_sequential_frontier(
        validation_scores,
        validation_matches,
        test_sequential_scores,
        test_matches,
    )
    fixed_test = [
        float(test_matches[:, depth].float().mean().item()) for depth in range(6)
    ]
    oracle_test = sum(any(bool(value) for value in row) for row in test_matches) / len(test_matches)
    return {
        "split": {
            name: {
                "positions": len(value),
                "source_rows": len(
                    {
                        (int(metadata[index]["row_index"]), str(metadata[index]["stratum"]))
                        for index in value.tolist()
                    }
                ),
            }
            for name, value in indices.items()
        },
        "preloop": {
            "any_extra_depth": {
                "label": "loop1_wrong_and_any_later_loop_correct",
                "positive_total": int(any_extra.sum().item()),
                "hidden_plus_structure": {
                    "validation_auroc": any_extra_hidden["validation_auroc"],
                    "test_auroc": any_extra_hidden["test_auroc"],
                    "ridge": any_extra_hidden["ridge"],
                },
                "structure_only": {
                    "validation_auroc": any_extra_structural["validation_auroc"],
                    "test_auroc": any_extra_structural["test_auroc"],
                    "ridge": any_extra_structural["ridge"],
                },
            },
            "loop2_decision": {
                "label": "loop1_wrong_and_loop2_correct",
                "positive_total": int(loop2_benefit.sum().item()),
                "hidden_plus_structure": {
                    "validation_auroc": loop2_hidden["validation_auroc"],
                    "test_auroc": loop2_hidden["test_auroc"],
                    "ridge": loop2_hidden["ridge"],
                },
                "structure_only": {
                    "validation_auroc": loop2_structural["validation_auroc"],
                    "test_auroc": loop2_structural["test_auroc"],
                    "ridge": loop2_structural["ridge"],
                },
            },
            "loop2_budget_points": budget_points,
            "verdict": preloop_verdict,
        },
        "sequential": {
            "label": "currently_wrong_and_any_later_loop_correct",
            "per_loop": sequential,
            "frontier": sequential_frontier,
            "test_fixed_depth_accuracy": {
                str(depth): fixed_test[depth - 1] for depth in range(1, 7)
            },
            "test_best_fixed_accuracy": max(fixed_test),
            "test_oracle_any_depth_accuracy": oracle_test,
        },
        "teacher_features_used": False,
        "evaluation_partition_touched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--floor_private_rows", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--row_cache_dir", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--expected_private_rows_sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--projection_dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--seed", type=int, default=ROUTER_AUDIT_SEED)
    args = parser.parse_args()
    cache = extract_feature_cache(
        data_jsonl=Path(args.data_jsonl),
        private_rows_path=Path(args.floor_private_rows),
        checkpoint=Path(args.checkpoint),
        cache_path=Path(args.feature_cache),
        row_cache_dir=Path(args.row_cache_dir),
        expected_private_sha256=args.expected_private_rows_sha256,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        projection_dim=args.projection_dim,
        projection_seed=args.seed,
    )
    analysis = analyze_feature_cache(cache, seed=args.seed)
    runtime_cache = {**cache, "matches": cache["runtime_matches"]}
    runtime_analysis = analyze_feature_cache(runtime_cache, seed=args.seed)
    summary = {
        "kind": "paper2_d0_deployable_router_probe",
        "status": "complete",
        "scope": "locked_calibration_7b_rejected_positions_grouped_source_row_split",
        "checkpoint_sha256": cache["signature"]["checkpoint_sha256"],
        "floor_private_rows_sha256": cache["signature"]["private_rows_sha256"],
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "projection_dim": args.projection_dim,
        "seed": args.seed,
        "no_model_training": True,
        "no_model_mutation": (
            cache["frozen_parameter_fingerprint_before"]
            == cache["frozen_parameter_fingerprint_after"]
        ),
        "primary_label_source": "frozen_floor_receipt",
        "prediction_equivalence": prediction_equivalence_summary(cache),
        "runtime_hardware_sensitivity": runtime_analysis,
        **analysis,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
