"""Tail-convergence selector diagnostic for forced-depth recurrent sweeps.

This is a narrow follow-up to the rescue-selector transfer analysis. It adds
one dynamic feature family: how much the example's low-rank tail state moves
between loop 1->2 and 2->3. The feature is computed from hidden states, then
joined to already-landed forced-depth correctness rows so we do not rescore
MCQ answers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_depth_sweep import (  # noqa: E402
    joined_examples,
    load_loop_payloads,
    path_for_cli,
    read_json,
    read_jsonl,
    resolve_path,
    row_file_for,
)
from eval.analyze_rescue_predictability import (  # noqa: E402
    auc_for_binary,
    fisher_ratio,
    label_examples,
    mean,
    safe_metric,
)
from eval.eval_identity import resolve_dtype  # noqa: E402
from eval.eval_mcq import MCQExample, format_prompt  # noqa: E402
from eval.eval_reentry_drift import (  # noqa: E402
    load_wrapper,
    masked_token_matrix,
    prepare_recurrent_inputs,
    run_recurrent_block,
    rms,
)
from eval.eval_reentry_tail_diagnostic import centered_covariance, eig_desc  # noqa: E402
from eval.evaluate_rescue_selector_transfer import (  # noqa: E402
    best_result,
    diverse_probe_detectability as telemetry_probe_detectability,
    evaluate_score_gate,
    fit_feature_stats,
    fit_subsample_directions,
    pairwise_alignment,
    percentile,
    score_gate_sweep,
    spectral_localization,
    transferred_curve,
)
from eval.prepare_arc_mcq import row_to_mcq  # noqa: E402


EPS = 1e-12
TAIL_FEATURES = [
    "tail_rel_disp_12",
    "tail_rel_disp_23",
    "tail_cos_12",
    "tail_cos_23",
    "tail_deceleration_12_minus_23",
    "tail_disp_ratio_23_over_12",
]


@dataclass(frozen=True)
class PromptExample:
    id: str
    prompt: str


def finite_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = float(value.detach().double().cpu())
    value = float(value)
    return value if math.isfinite(value) else None


def median_or_zero(values: list[float]) -> float:
    kept = sorted(value for value in values if math.isfinite(float(value)))
    if not kept:
        return 0.0
    return kept[len(kept) // 2]


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), EPS)


def tail_convergence_features(loop_tail_vectors: dict[int, torch.Tensor]) -> dict[str, float]:
    """Return cross-loop convergence features for one example.

    ``loop_tail_vectors`` maps loop number to a single tail-coordinate vector.
    Positive ``tail_deceleration_12_minus_23`` means the loop-2->3 movement is
    smaller than loop-1->2.
    """

    required = [1, 2, 3]
    if any(loop not in loop_tail_vectors for loop in required):
        missing = [loop for loop in required if loop not in loop_tail_vectors]
        raise KeyError(f"Missing loop tail vectors: {missing}")
    v1 = loop_tail_vectors[1].double().flatten()
    v2 = loop_tail_vectors[2].double().flatten()
    v3 = loop_tail_vectors[3].double().flatten()
    d12 = torch.linalg.vector_norm(v2 - v1)
    d23 = torch.linalg.vector_norm(v3 - v2)
    n1 = torch.linalg.vector_norm(v1).clamp_min(EPS)
    n2 = torch.linalg.vector_norm(v2).clamp_min(EPS)
    rel12 = finite_float(d12 / n1) or 0.0
    rel23 = finite_float(d23 / n2) or 0.0
    cos12 = finite_float(F.cosine_similarity(v1.view(1, -1), v2.view(1, -1), dim=-1).squeeze(0)) or 0.0
    cos23 = finite_float(F.cosine_similarity(v2.view(1, -1), v3.view(1, -1), dim=-1).squeeze(0)) or 0.0
    return {
        "tail_rel_disp_12": rel12,
        "tail_rel_disp_23": rel23,
        "tail_cos_12": cos12,
        "tail_cos_23": cos23,
        "tail_deceleration_12_minus_23": rel12 - rel23,
        "tail_disp_ratio_23_over_12": safe_divide(rel23, rel12),
    }


def pca_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    matrix = matrix.detach().double().cpu()
    if matrix.numel() == 0:
        return matrix.new_zeros((matrix.shape[-1] if matrix.dim() else 0, 0))
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    max_rank = min(max(int(rank), 1), centered.shape[0], centered.shape[1])
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:max_rank].T.contiguous()


def principal_subspace_rotation(loop_vectors: dict[int, list[torch.Tensor]], *, rank: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for left, right in [(1, 2), (2, 3)]:
        if left not in loop_vectors or right not in loop_vectors:
            continue
        x = torch.stack(loop_vectors[left]).double()
        y = torch.stack(loop_vectors[right]).double()
        basis_x = pca_basis(x, rank=rank)
        basis_y = pca_basis(y, rank=rank)
        if basis_x.numel() == 0 or basis_y.numel() == 0:
            cosines = torch.empty(0, dtype=torch.float64)
        else:
            cosines = torch.linalg.svdvals(basis_x.T @ basis_y).clamp(0.0, 1.0)
        out[f"{left}_to_{right}"] = {
            "rank": int(min(basis_x.shape[1], basis_y.shape[1])),
            "principal_cosines": [float(value) for value in cosines.tolist()],
            "mean_squared_cosine": float(cosines.square().mean().item()) if cosines.numel() else 0.0,
            "min_cosine": float(cosines.min().item()) if cosines.numel() else 0.0,
        }
    return out


def first_recurrent_row(sweep_summary: Path, *, benchmark: str, score_target: str, aggregate: str) -> dict[str, Any]:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    first_loop = min(loop_payloads)
    path = row_file_for(loop_payloads[first_loop], benchmark, "recurrent", score_target)
    for row in read_jsonl(path):
        if str(row.get("aggregate") or "mean") == aggregate:
            return row
    raise ValueError(f"No row with aggregate={aggregate!r} in {path}")


def examples_for_sweep(
    sweep_summary: Path,
    *,
    benchmark: str,
    score_target: str,
    aggregate: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    examples = joined_examples(loop_payloads, benchmark, score_target, aggregate)
    return loops, label_examples(examples, loops)


def available_benchmarks(sweep_summary: Path) -> list[str]:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    first_loop = min(loop_payloads)
    return list(loop_payloads[first_loop].get("benchmarks", []))


def arc_prompt_examples(ids: list[str], *, benchmark: str, prompt_style: str) -> list[PromptExample]:
    if benchmark == "arc_easy":
        config = "ARC-Easy"
        split = "validation"
    elif benchmark == "open_hard_arc_challenge":
        config = "ARC-Challenge"
        split = "test"
    elif benchmark == "arc_challenge":
        config = "ARC-Challenge"
        split = "validation"
    else:
        raise ValueError(f"Tail-convergence prompt reconstruction only supports ARC benchmarks, got {benchmark!r}")

    wanted = set(ids)
    by_id: dict[str, PromptExample] = {}
    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    for idx, row in enumerate(dataset):
        prepared = row_to_mcq(dict(row), index=idx, seed=0, shuffle_choices=True)
        row_id = str(prepared["id"])
        if row_id not in wanted:
            continue
        example = MCQExample(
            id=row_id,
            question=str(prepared["question"]),
            choices=[(str(label), str(text)) for label, text in dict(prepared["choices"]).items()],
            answer=str(prepared["answer"]),
        )
        by_id[row_id] = PromptExample(id=row_id, prompt=format_prompt(example, prompt_style))
        if len(by_id) == len(wanted):
            break
    missing = [row_id for row_id in ids if row_id not in by_id]
    if missing:
        raise KeyError(f"Could not reconstruct {len(missing)} prompts for benchmark={benchmark}: {missing[:5]}")
    return [by_id[row_id] for row_id in ids]


def token_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0)).squeeze(0).float().cpu()


def collect_tail_inputs(
    wrapper: Any,
    tokenizer: Any,
    prompts: list[PromptExample],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]:
    entry_tokens: list[torch.Tensor] = []
    pooled_by_id: dict[str, dict[int, torch.Tensor]] = {}
    max_loop = max(args.loops)
    damper = wrapper._load_reentry_tail_damper(args.reentry_tail_damper_path)  # noqa: SLF001
    for item in prompts:
        encoded = tokenizer(item.prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
        with torch.no_grad():
            entry_state, mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
                wrapper,
                encoded["input_ids"],
                attention_mask,
            )
            entry_tokens.append(masked_token_matrix(entry_state, mask).cpu())
            entry_rms = rms(entry_state, mask).clamp_min(1e-8)
            recurrent_state = entry_state
            pooled_by_id[item.id] = {}
            for loop_idx in range(max_loop):
                loop_number = loop_idx + 1
                loop_input = recurrent_state if loop_idx == 0 else wrapper.bridge(recurrent_state)
                if loop_idx > 0 and args.reentry_rescale_mode == "entry_rms":
                    current_rms = rms(loop_input, mask).clamp_min(1e-8)
                    loop_input = loop_input * (entry_rms / current_rms).to(dtype=loop_input.dtype)
                if loop_idx > 0 and args.use_reentry_adapter:
                    loop_input = wrapper.reentry_adapter(loop_input, loop_idx=loop_idx, mode=args.reentry_adapter_mode)
                if loop_idx > 0 and damper is not None and args.reentry_tail_damper_strength > 0:
                    from models.reentry_tail_damper import apply_tail_damper

                    loop_input = apply_tail_damper(
                        loop_input,
                        mean=damper["mean"],
                        basis=damper["basis"],
                        damper_scale=damper["damper_scale"],
                        strength=args.reentry_tail_damper_strength,
                    )
                loop_output = run_recurrent_block(
                    wrapper,
                    loop_input,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                )
                if loop_number in args.loops:
                    pooled_by_id[item.id][loop_number] = token_mean(loop_output, mask)
                recurrent_state = loop_output
    return torch.cat(entry_tokens, dim=0), pooled_by_id


def calibrate_tail_basis(entry_tokens: torch.Tensor, *, n_tail: int, drop_top: int) -> tuple[torch.Tensor, torch.Tensor]:
    cov, mean_vec = centered_covariance(entry_tokens)
    _evals, evecs = eig_desc(cov)
    start = min(max(drop_top, 0), evecs.shape[1] - 1)
    stop = min(start + max(1, n_tail), evecs.shape[1])
    return mean_vec.double(), evecs[:, start:stop].double().contiguous()


def project_tail_vector(vector: torch.Tensor, mean_vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (vector.double().cpu() - mean_vec) @ basis


def attach_tail_features(
    examples: list[dict[str, Any]],
    pooled_by_id: dict[str, dict[int, torch.Tensor]],
    *,
    mean_vec: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[int, list[torch.Tensor]]]:
    enriched: list[dict[str, Any]] = []
    loop_tail_vectors: dict[int, list[torch.Tensor]] = {}
    for example in examples:
        row_id = str(example["id"])
        loop_vectors = {
            loop: project_tail_vector(vector, mean_vec, basis)
            for loop, vector in pooled_by_id[row_id].items()
        }
        for loop, vector in loop_vectors.items():
            loop_tail_vectors.setdefault(loop, []).append(vector)
        enriched.append({**example, "tail_convergence": tail_convergence_features(loop_vectors)})
    return enriched, loop_tail_vectors


def feature_scores(examples: list[dict[str, Any]], feature: str) -> list[float]:
    scores = []
    for example in examples:
        value = (example.get("tail_convergence") or {}).get(feature)
        scores.append(float(value) if value is not None and math.isfinite(float(value)) else float("nan"))
    return scores


def score_discrimination(examples: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    pairs: list[tuple[float, bool]] = []
    for example in examples:
        value = (example.get("tail_convergence") or {}).get(feature)
        if value is None or not math.isfinite(float(value)):
            continue
        pairs.append((float(value), bool(example.get("rescuable"))))
    if not pairs:
        return {"feature": feature, "n": 0}
    scores = [score for score, _label in pairs]
    labels = [label for _score, label in pairs]
    pos = [score for score, label in pairs if label]
    neg = [score for score, label in pairs if not label]
    auc = auc_for_binary(scores, labels)
    return {
        "feature": feature,
        "n": len(pairs),
        "positive": len(pos),
        "negative": len(neg),
        "auc": safe_metric(auc),
        "oriented_auc": None if auc is None else max(auc, 1.0 - auc),
        "direction": None if auc is None else ("high_predicts_positive" if auc >= 0.5 else "low_predicts_positive"),
        "fisher_ratio": safe_metric(fisher_ratio(pos, neg)),
        "positive_mean": safe_metric(mean(pos)),
        "negative_mean": safe_metric(mean(neg)),
    }


def matrix_from_tail_features(examples: list[dict[str, Any]], features: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    raw = torch.tensor(
        [
            [
                float((example.get("tail_convergence") or {}).get(feature, float("nan")))
                for feature in features
            ]
            for example in examples
        ],
        dtype=torch.float64,
    )
    means = []
    stds = []
    for col in range(raw.shape[1]):
        values = raw[:, col]
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            means.append(0.0)
            stds.append(1.0)
            continue
        kept = values[finite]
        col_mean = float(kept.mean().item())
        col_std = float(kept.std(unbiased=False).item())
        means.append(col_mean)
        stds.append(col_std if col_std > EPS else 1.0)
        raw[:, col] = torch.where(finite, values, torch.full_like(values, col_mean))
    means_tensor = torch.tensor(means, dtype=torch.float64)
    stds_tensor = torch.tensor(stds, dtype=torch.float64).clamp_min(EPS)
    return (raw - means_tensor) / stds_tensor, {"features": features, "means": means, "stds": stds}


def tail_probe_detectability(examples: list[dict[str, Any]], *, features: list[str], seed: int = 13) -> dict[str, Any]:
    x, stats = matrix_from_tail_features(examples, features)
    labels = torch.tensor([example["category"] == "rescuable" for example in examples], dtype=torch.bool)
    if int(labels.sum().item()) < 2 or int((~labels).sum().item()) < 2:
        return {
            "available": False,
            "reason": "insufficient_positive_or_negative_examples",
            "positive": int(labels.sum().item()),
            "negative": int((~labels).sum().item()),
        }
    observed_directions = fit_subsample_directions(
        x,
        labels,
        shrinkage=1.0,
        repeats=48,
        sample_fraction=0.7,
        seed=seed,
    )
    observed = pairwise_alignment(observed_directions)
    generator = torch.Generator().manual_seed(seed + 1)
    null_values: list[float] = []
    for index in range(64):
        permuted = labels[torch.randperm(labels.numel(), generator=generator)]
        directions = fit_subsample_directions(
            x,
            permuted,
            shrinkage=1.0,
            repeats=16,
            sample_fraction=0.7,
            seed=seed + 101 + index,
        )
        value = pairwise_alignment(directions)
        if value is not None:
            null_values.append(value)
    null_p95 = percentile(null_values, 0.95)
    return {
        "available": observed is not None and null_p95 is not None,
        "feature_space": "tail_convergence",
        "features": features,
        "positive": int(labels.sum().item()),
        "negative": int((~labels).sum().item()),
        "stats": stats,
        "observed_alignment": observed,
        "null_mean_alignment": (sum(null_values) / len(null_values)) if null_values else None,
        "null_p95_alignment": null_p95,
        "clears_null_p95": bool(observed is not None and null_p95 is not None and observed > null_p95),
    }


def policies_from_feature_curve(examples: list[dict[str, Any]], loops: list[int], feature: str) -> list[dict[str, Any]]:
    rows = score_gate_sweep(examples, feature_scores(examples, feature), loops, feature=feature)
    policies = []
    seen = set()
    for index, row in enumerate(rows):
        key = (row["feature"], row["threshold"], row["direction"], row["fallback_loop"])
        if key in seen:
            continue
        seen.add(key)
        policies.append({"label": f"{feature}_curve_{index:04d}", "discovery_result": row, **row})
    return policies


def apply_feature_policies(
    examples: list[dict[str, Any]],
    loops: list[int],
    feature: str,
    policies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scores = feature_scores(examples, feature)
    rows = []
    available_loops = set(loops)
    for policy in policies:
        fallback_loop = int(policy["fallback_loop"])
        if fallback_loop not in available_loops:
            continue
        result = evaluate_score_gate(
            examples,
            scores,
            feature=feature,
            threshold=float(policy["threshold"]),
            direction=str(policy["direction"]),
            fallback_loop=fallback_loop,
        )
        discovery = policy.get("discovery_result") or {}
        rows.append(
            {
                **result,
                "policy_label": policy["label"],
                "discovery_delta_vs_loop1": discovery.get("delta_vs_loop1"),
                "discovery_rescue_captured": discovery.get("rescue_captured"),
                "discovery_harm_triggered": discovery.get("harm_triggered"),
                "discovery_routed_deep": discovery.get("routed_deep"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("delta_vs_loop1", 0)),
            int(row.get("rescue_captured", 0)),
            -int(row.get("harm_triggered", 0)),
            -int(row.get("routed_deep", 0)),
        ),
        reverse=True,
    )


def curve_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "points": len(rows),
        "max_net": best_result(rows, label="max_net"),
        "zero_harm": best_result(rows, label="zero_harm", harm_budget=0),
        "harm_budget_1": best_result(rows, label="harm_budget_1", harm_budget=1),
        "harm_budget_2": best_result(rows, label="harm_budget_2", harm_budget=2),
    }


def checkpoint_from_sweep(sweep_summary: Path, *, benchmark: str, score_target: str, aggregate: str) -> Path:
    row = first_recurrent_row(sweep_summary, benchmark=benchmark, score_target=score_target, aggregate=aggregate)
    value = row.get("checkpoint")
    if not value:
        raise ValueError(f"Could not infer checkpoint from {sweep_summary}")
    return resolve_path(str(value))


def run_feature_collection(
    *,
    wrapper: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    benchmark: str,
    prompt_style: str,
    args: argparse.Namespace,
    mean_vec: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, dict[str, Any]]:
    ids = [str(example["id"]) for example in examples]
    if args.max_examples_per_benchmark > 0:
        ids = ids[: args.max_examples_per_benchmark]
        keep = set(ids)
        examples = [example for example in examples if str(example["id"]) in keep]
    prompts = arc_prompt_examples(ids, benchmark=benchmark, prompt_style=prompt_style)
    entry_tokens, pooled_by_id = collect_tail_inputs(wrapper, tokenizer, prompts, args)
    if mean_vec is None or basis is None:
        mean_vec, basis = calibrate_tail_basis(entry_tokens, n_tail=args.n_tail, drop_top=args.drop_top)
    enriched, loop_tail_vectors = attach_tail_features(examples, pooled_by_id, mean_vec=mean_vec, basis=basis)
    companion = {
        "population_subspace_rotation": principal_subspace_rotation(loop_tail_vectors, rank=min(args.n_tail, 4)),
        "median_feature_values": {
            feature: median_or_zero(feature_scores(enriched, feature))
            for feature in TAIL_FEATURES
        },
    }
    return enriched, mean_vec, basis, companion


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    discovery_sweep = resolve_path(args.discovery_sweep_summary)
    heldout_sweep = resolve_path(args.heldout_sweep_summary)
    score_target = args.score_target
    aggregate = args.aggregate
    discovery_benchmark = args.discovery_benchmark

    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else checkpoint_from_sweep(
        discovery_sweep,
        benchmark=discovery_benchmark,
        score_target=score_target,
        aggregate=aggregate,
    )
    args.checkpoint = str(checkpoint)
    args.loops = [int(item) for item in str(args.loops).split(",") if item.strip()]

    wrapper = load_wrapper(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    discovery_loops, discovery_examples = examples_for_sweep(
        discovery_sweep,
        benchmark=discovery_benchmark,
        score_target=score_target,
        aggregate=aggregate,
    )
    prompt_style = str(first_recurrent_row(discovery_sweep, benchmark=discovery_benchmark, score_target=score_target, aggregate=aggregate).get("prompt_style") or "question_only")
    discovery_enriched, mean_vec, basis, discovery_companion = run_feature_collection(
        wrapper=wrapper,
        tokenizer=tokenizer,
        examples=discovery_examples,
        benchmark=discovery_benchmark,
        prompt_style=prompt_style,
        args=args,
    )

    policies_by_feature = {
        feature: policies_from_feature_curve(discovery_enriched, discovery_loops, feature)
        for feature in TAIL_FEATURES
    }
    discovery_curves = {
        feature: [policy["discovery_result"] for policy in policies]
        for feature, policies in policies_by_feature.items()
    }
    discovery = {
        "benchmark": discovery_benchmark,
        "total": len(discovery_enriched),
        "loops": discovery_loops,
        "category_counts": {
            category: sum(1 for example in discovery_enriched if example["category"] == category)
            for category in sorted({str(example["category"]) for example in discovery_enriched})
        },
        "tail_feature_discrimination": [score_discrimination(discovery_enriched, feature) for feature in TAIL_FEATURES],
        "tail_probe_detectability": tail_probe_detectability(discovery_enriched, features=TAIL_FEATURES),
        "telemetry_probe_detectability_reference": telemetry_probe_detectability(discovery_enriched, features=[
            "base_predicted_margin",
            "loop1_predicted_margin",
            "loop1_margin_minus_base_margin",
            "loop1_score_entropy",
            "loop1_mean_expected_loops",
            "loop1_mean_halt_entropy",
            "loop1_prediction_expected_loops",
            "loop1_prediction_halt_entropy",
        ]),
        "curve_summary_by_feature": {
            feature: curve_summary(rows)
            for feature, rows in discovery_curves.items()
        },
        "curve_top10_by_feature": {
            feature: rows[:10]
            for feature, rows in discovery_curves.items()
        },
        **discovery_companion,
    }

    heldout: dict[str, Any] = {}
    for benchmark in available_benchmarks(heldout_sweep):
        if benchmark not in {"arc_easy", "arc_challenge", "open_hard_arc_challenge"}:
            continue
        loops, examples = examples_for_sweep(
            heldout_sweep,
            benchmark=benchmark,
            score_target=score_target,
            aggregate=aggregate,
        )
        if not examples:
            continue
        prompt_style = str(first_recurrent_row(heldout_sweep, benchmark=benchmark, score_target=score_target, aggregate=aggregate).get("prompt_style") or "question_only")
        enriched, _mean, _basis, companion = run_feature_collection(
            wrapper=wrapper,
            tokenizer=tokenizer,
            examples=examples,
            benchmark=benchmark,
            prompt_style=prompt_style,
            args=args,
            mean_vec=mean_vec,
            basis=basis,
        )
        curves = {
            feature: apply_feature_policies(enriched, loops, feature, policies_by_feature[feature])
            for feature in TAIL_FEATURES
        }
        heldout[benchmark] = {
            "total": len(enriched),
            "loops": loops,
            "category_counts": {
                category: sum(1 for example in enriched if example["category"] == category)
                for category in sorted({str(example["category"]) for example in enriched})
            },
            "tail_feature_discrimination": [score_discrimination(enriched, feature) for feature in TAIL_FEATURES],
            "transferred_curve_summary_by_feature": {
                feature: curve_summary(rows)
                for feature, rows in curves.items()
            },
            "transferred_curve_top10_by_feature": {
                feature: rows[:10]
                for feature, rows in curves.items()
            },
            **companion,
        }

    return {
        "kind": "stage5_tail_convergence_selector",
        "run_id": args.run_id or "stage5_tail_convergence_selector_" + time.strftime("%Y%m%d_%H%M%S"),
        "discovery_sweep_summary": path_for_cli(discovery_sweep),
        "heldout_sweep_summary": path_for_cli(heldout_sweep),
        "checkpoint": path_for_cli(checkpoint),
        "model_name": args.model_name,
        "score_target": score_target,
        "aggregate": aggregate,
        "n_tail": args.n_tail,
        "drop_top": args.drop_top,
        "tail_features": TAIL_FEATURES,
        "max_examples_per_benchmark": args.max_examples_per_benchmark,
        "discovery": discovery,
        "heldout": heldout,
    }


def markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage 5 Tail-Convergence Selector Diagnostic",
        "",
        f"- Run: `{payload['run_id']}`",
        f"- Discovery sweep: `{payload['discovery_sweep_summary']}`",
        f"- Held-out sweep: `{payload['heldout_sweep_summary']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Tail basis: drop top `{payload['drop_top']}`, keep `{payload['n_tail']}`",
        "",
        "## Discovery",
        "",
    ]
    detect = payload["discovery"]["tail_probe_detectability"]
    lines.extend(
        [
            f"- Tail probe clears permutation null p95: `{detect.get('clears_null_p95')}`",
            f"- Observed/null p95 alignment: `{detect.get('observed_alignment')}` / `{detect.get('null_p95_alignment')}`",
            "",
            "| feature | oriented AUC | pos mean | neg mean | best zero-harm delta | best max-net delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    discrimination = {row["feature"]: row for row in payload["discovery"]["tail_feature_discrimination"]}
    curves = payload["discovery"]["curve_summary_by_feature"]
    for feature in payload["tail_features"]:
        disc = discrimination.get(feature, {})
        summary = curves.get(feature, {})
        zero = summary.get("zero_harm") or {}
        max_net = summary.get("max_net") or {}
        lines.append(
            f"| `{feature}` | {disc.get('oriented_auc')} | {disc.get('positive_mean')} | "
            f"{disc.get('negative_mean')} | {zero.get('delta_vs_loop1')} | {max_net.get('delta_vs_loop1')} |"
        )
    lines.extend(["", "## Held-Out Transfer", ""])
    for benchmark, result in payload["heldout"].items():
        lines.extend([f"### {benchmark}", "", "| feature | zero-harm delta | max-net delta | max-net W/L |", "|---|---:|---:|---:|"])
        summaries = result["transferred_curve_summary_by_feature"]
        for feature in payload["tail_features"]:
            summary = summaries.get(feature, {})
            zero = summary.get("zero_harm") or {}
            max_net = summary.get("max_net") or {}
            lines.append(
                f"| `{feature}` | {zero.get('delta_vs_loop1')} | {max_net.get('delta_vs_loop1')} | "
                f"{max_net.get('wins_vs_loop1')}/{max_net.get('losses_vs_loop1')} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery_sweep_summary", required=True)
    parser.add_argument("--heldout_sweep_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--discovery_benchmark", default="arc_challenge")
    parser.add_argument("--score_target", default="content_question_only")
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--loops", default="1,2,3")
    parser.add_argument("--n_tail", type=int, default=7)
    parser.add_argument("--drop_top", type=int, default=1)
    parser.add_argument("--max_examples_per_benchmark", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--reentry_rescale_mode", default="none")
    parser.add_argument("--use_reentry_adapter", action="store_true")
    parser.add_argument("--reentry_adapter_mode", default="affine")
    parser.add_argument("--reentry_tail_damper_path", default="")
    parser.add_argument("--reentry_tail_damper_strength", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    # Validate dtype names early; load_wrapper will use them again.
    resolve_dtype(args.dtype)
    resolve_dtype(args.adapter_dtype)

    payload = analyze(args)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    print(f"saved_summary={path_for_cli(output_dir / 'summary.json')}", flush=True)
    print(markdown_summary(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
