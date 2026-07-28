"""Forward-only DC0 comparison of in-place depth and transient depth-by-append."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_speculative_depth_d0_floor import load_partition_cache
from models.coconut_composite import CoconutRecurrentQwen
from training.paper2_dc0 import layer_application_costs
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_spec import deterministic_argmax_fp32


AUDIT_HELPS = 8564
AUDIT_HURTS = 30008
MAX_APPEND = 3
MAX_INPLACE = 4
BOOTSTRAP_SEED = 20260728


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def parameter_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = parameter.detach().float()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(
                repr((float(value.sum().cpu()), float(value.square().sum().cpu()))).encode(
                    "ascii"
                )
            )
    return digest.hexdigest()


def transition_counts(before: torch.Tensor, after: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    before_match = before.eq(teacher)
    after_match = after.eq(teacher)
    helps = int((~before_match & after_match).sum())
    hurts = int((before_match & ~after_match).sum())
    neutral = len(teacher) - helps - hurts
    total = len(teacher)
    return {
        "helps": helps,
        "hurts": hurts,
        "neutral": neutral,
        "total": total,
        "net_correct_delta": helps - hurts,
        "before_correct": int(before_match.sum()),
        "after_correct": int(after_match.sum()),
        "before_accuracy": float(before_match.float().mean()) if total else None,
        "after_accuracy": float(after_match.float().mean()) if total else None,
        "harm_to_help_ratio": hurts / helps if helps else None,
    }


def anchor_registered_k0(
    cached_grid: torch.Tensor,
    registered_k0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Use the registered full-sequence baseline while retaining cache-path drift."""

    if cached_grid.ndim != 2 or cached_grid.shape[1] < 2:
        raise ValueError("cached append grid must have shape [positions, 2+]")
    if registered_k0.ndim != 1 or len(registered_k0) != len(cached_grid):
        raise ValueError("registered k0 must align one-to-one with append positions")
    cached_k0 = cached_grid[:, 0].clone()
    anchored = cached_grid.clone()
    anchored[:, 0] = registered_k0
    disagreements = int(cached_k0.ne(registered_k0).sum())
    positions = len(registered_k0)
    return anchored, cached_k0, {
        "positions": positions,
        "prediction_disagreements": disagreements,
        "prediction_disagreement_rate": disagreements / positions if positions else 0.0,
        "primary_k0_source": "registered_full_sequence_depth_1",
        "append_k_positive_source": "incremental_cache_append",
    }


def _row_transition_counts(
    before: torch.Tensor,
    after: torch.Tensor,
    teacher: torch.Tensor,
    row_indices: torch.Tensor,
) -> list[tuple[int, int]]:
    before_match = before.eq(teacher)
    after_match = after.eq(teacher)
    result = []
    for row in row_indices.unique(sorted=True):
        mask = row_indices.eq(row)
        result.append(
            (
                int((~before_match[mask] & after_match[mask]).sum()),
                int((before_match[mask] & ~after_match[mask]).sum()),
            )
        )
    return result


def cluster_bootstrap_log_ratio(
    row_counts: Sequence[tuple[int, int]], *, draws: int = 5000
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    counts = torch.tensor(row_counts, dtype=torch.float64)
    ratios = []
    for _ in range(draws):
        indices = torch.randint(len(counts), (len(counts),), generator=generator)
        helps, hurts = counts[indices].sum(dim=0).tolist()
        ratios.append(math.log((hurts + 0.5) / (helps + 0.5)))
    values = torch.tensor(ratios, dtype=torch.float64)
    lower, upper = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64))
    point_helps, point_hurts = counts.sum(dim=0).tolist()
    return {
        "harm_to_help_ratio": (point_hurts + 0.5) / (point_helps + 0.5),
        "log_ratio_ci95": [float(lower), float(upper)],
        "ratio_ci95": [math.exp(float(lower)), math.exp(float(upper))],
        "cluster_unit": "source_row",
        "draws": draws,
        "seed": BOOTSTRAP_SEED,
    }


def baseline_validity(
    *,
    fresh: dict[str, Any],
    fresh_rows: Sequence[tuple[int, int]],
    audit_rows: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    fresh_ci = cluster_bootstrap_log_ratio(fresh_rows)
    audit_ci = cluster_bootstrap_log_ratio(audit_rows)
    intervals_overlap = not (
        fresh_ci["log_ratio_ci95"][1] < audit_ci["log_ratio_ci95"][0]
        or audit_ci["log_ratio_ci95"][1] < fresh_ci["log_ratio_ci95"][0]
    )
    direction = int(fresh["hurts"]) > int(fresh["helps"])
    return {
        "status": "green" if direction and intervals_overlap else "red_stop_before_append_scoring",
        "fresh_hurts_exceed_helps": direction,
        "source_cluster_bootstrap_intervals_overlap": intervals_overlap,
        "fresh": fresh_ci,
        "banked_audit": {
            **audit_ci,
            "helps": AUDIT_HELPS,
            "hurts": AUDIT_HURTS,
        },
        "scope": "out-of-sample replication on the same post-D0 checkpoint only",
    }


def quartile_labels(values: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    boundaries = torch.quantile(values.float(), torch.tensor([0.25, 0.5, 0.75]))
    labels = torch.bucketize(values.float(), boundaries)
    return labels, [float(value) for value in boundaries]


def transition_report(
    predictions: torch.Tensor,
    teacher: torch.Tensor,
    *,
    strata: Sequence[str],
    entropy: torch.Tensor,
    drafter_logprob: torch.Tensor,
    drafter_rank: torch.Tensor,
) -> dict[str, Any]:
    entropy_q, entropy_bounds = quartile_labels(entropy)
    logprob_q, logprob_bounds = quartile_labels(drafter_logprob)
    rank_bins = torch.tensor(
        [0 if int(value) == 1 else 1 if int(value) <= 5 else 2 if int(value) <= 20 else 3 for value in drafter_rank]
    )
    result: dict[str, Any] = {
        "steps": int(predictions.shape[1]),
        "teacher_confidence_boundaries": {
            "entropy_quartiles": entropy_bounds,
            "drafter_logprob_quartiles": logprob_bounds,
            "rank_bins": ["1", "2-5", "6-20", ">20"],
        },
        "transitions": {},
    }
    stratum_values = sorted(set(strata))
    for step in range(predictions.shape[1] - 1):
        key = f"{step}_to_{step + 1}"
        cell: dict[str, Any] = {
            "pooled": transition_counts(predictions[:, step], predictions[:, step + 1], teacher),
            "by_stratum": {},
            "by_entropy_quartile": {},
            "by_drafter_logprob_quartile": {},
            "by_teacher_rank": {},
        }
        for stratum in stratum_values:
            mask = torch.tensor([value == stratum for value in strata])
            cell["by_stratum"][stratum] = transition_counts(
                predictions[mask, step], predictions[mask, step + 1], teacher[mask]
            )
        for index in range(4):
            mask = entropy_q.eq(index)
            cell["by_entropy_quartile"][f"q{index + 1}"] = transition_counts(
                predictions[mask, step], predictions[mask, step + 1], teacher[mask]
            )
            mask = logprob_q.eq(index)
            cell["by_drafter_logprob_quartile"][f"q{index + 1}"] = transition_counts(
                predictions[mask, step], predictions[mask, step + 1], teacher[mask]
            )
            mask = rank_bins.eq(index)
            cell["by_teacher_rank"][["1", "2-5", "6-20", ">20"][index]] = transition_counts(
                predictions[mask, step], predictions[mask, step + 1], teacher[mask]
            )
        result["transitions"][key] = cell
    return result


def group_batches(rows: Sequence[dict[str, Any]], batch_size: int) -> list[list[int]]:
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_length[len(row["input_ids"])].append(index)
    batches = []
    for length in sorted(by_length, reverse=True):
        indices = by_length[length]
        batches.extend(indices[start : start + batch_size] for start in range(0, len(indices), batch_size))
    return batches


@torch.inference_mode()
def evaluate_inplace(
    model: Any,
    rows: Sequence[dict[str, Any]],
    *,
    vocab_size: int,
    device: str,
    resume_dir: Path,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor | None] = [None] * len(rows)
    resume_dir.mkdir(parents=True, exist_ok=True)
    for number, indices in enumerate(group_batches(rows, 1), start=1):
        path = resume_dir / f"batch_{number:06d}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            values = torch.tensor([rows[index]["input_ids"] for index in indices], device=device)
            result = model(
                input_ids=values,
                attention_mask=torch.ones_like(values),
                max_loops=MAX_INPLACE,
                use_cache=False,
                return_loop_logits=True,
                return_dict=True,
            )
            logits = result.loop_logits[:, 0, :, :-1, :vocab_size]
            selected, _ties = deterministic_argmax_fp32(logits, dim=-1)
            payload = {"indices": indices, "predictions": selected.permute(0, 2, 1).cpu()}
            torch.save(payload, path)
        for local, row_index in enumerate(payload["indices"]):
            outputs[int(row_index)] = payload["predictions"][local]
        if number == 1 or number % 32 == 0:
            print(f"dc0_inplace_progress batches={number}", flush=True)
    if any(value is None for value in outputs):
        raise RuntimeError("DC0 in-place evaluation left rows missing")
    return [value for value in outputs if value is not None]


@torch.inference_mode()
def evaluate_append_arm(
    composite: CoconutRecurrentQwen,
    rows: Sequence[dict[str, Any]],
    *,
    arm: str,
    feedback_mode: str,
    reference_rms: float | None,
    neutral_token_id: int,
    vocab_size: int,
    device: str,
    batch_size: int,
    resume_dir: Path,
    read_at_t_query: bool = False,
) -> tuple[list[torch.Tensor], dict[str, int]]:
    outputs: list[torch.Tensor | None] = [None] * len(rows)
    totals = Counter()
    destination = resume_dir / arm
    destination.mkdir(parents=True, exist_ok=True)
    batches = group_batches(rows, batch_size)
    cached_at_start = sum(
        (destination / f"batch_{number:06d}.pt").exists()
        for number in range(1, len(batches) + 1)
    )
    pending_at_start = len(batches) - cached_at_start
    print(
        f"dc0_append_resume arm={arm} total_batches={len(batches)} "
        f"cached_batches={cached_at_start} pending_batches={pending_at_start}",
        flush=True,
    )
    started = time.monotonic()
    computed = 0
    compute_seconds = 0.0
    for number, indices in enumerate(batches, start=1):
        path = destination / f"batch_{number:06d}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            batch_started = time.monotonic()
            values = torch.tensor([rows[index]["input_ids"] for index in indices], device=device)
            result = composite.depth_by_append(
                input_ids=values,
                append_steps=MAX_APPEND,
                feedback_mode=feedback_mode,
                reference_rms=reference_rms,
                neutral_token_id=neutral_token_id,
                read_at_t_query=read_at_t_query,
                prediction_vocab_size=vocab_size,
            )
            payload = {
                "indices": indices,
                "predictions": result.predictions,
                "counters": result.assert_accounting(),
                "readout_grid_applications": result.readout_grid_applications,
                "eviction_assertions": result.eviction_assertions,
                "diagnostics": result.diagnostics,
            }
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
            computed += 1
            compute_seconds += time.monotonic() - batch_started
        for key, value in payload["counters"].items():
            if isinstance(value, int):
                totals[key] += value
        totals["readout_grid_applications"] += int(payload["readout_grid_applications"])
        totals["eviction_assertions"] += int(payload["eviction_assertions"])
        for local, row_index in enumerate(payload["indices"]):
            outputs[int(row_index)] = payload["predictions"][local]
        if number == 1 or number % 16 == 0 or number == len(batches):
            remaining_compute = max(0, pending_at_start - computed)
            eta_seconds = (
                compute_seconds / computed * remaining_compute if computed else None
            )
            eta_text = f"{eta_seconds / 3600:.2f}h" if eta_seconds is not None else "cached"
            print(
                f"dc0_append_progress arm={arm} batches={number}/{len(batches)} "
                f"new_batches={computed}/{pending_at_start} "
                f"elapsed={(time.monotonic() - started) / 3600:.2f}h eta={eta_text}",
                flush=True,
            )
    if any(value is None for value in outputs):
        raise RuntimeError(f"DC0 append arm {arm} left rows missing")
    return [value for value in outputs if value is not None], dict(totals)


def flatten_rows(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(values), dim=0)


def audit_row_counts(cache: dict[str, Any]) -> list[tuple[int, int]]:
    matches = cache["matches"].bool()
    rows = torch.tensor([int(item["row_index"]) for item in cache["metadata"]])
    return _row_transition_counts(matches[:, 0].long(), matches[:, 1].long(), torch.ones(len(matches), dtype=torch.long), rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_checkpoint_sha256", required=True)
    parser.add_argument("--audit_feature_cache", required=True)
    parser.add_argument("--composite_preflight_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--append_batch_size", type=int, default=8)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("DC0 checkpoint SHA-256 mismatch")
    composite_preflight = json.loads(
        Path(args.composite_preflight_summary).read_text(encoding="utf-8")
    )
    if composite_preflight["contracts"]["rg1"].get("passed") is not True:
        raise RuntimeError("DC0 requires the banked two-budget RG-1 identity contract")
    if composite_preflight["contracts"]["rg2"].get("passed") is not True:
        raise RuntimeError("DC0 requires the banked horizontal-bridge identity contract")
    rows = read_jsonl(args.data_jsonl)
    teacher_summary = json.loads(Path(args.teacher_cache_summary).read_text(encoding="utf-8"))
    teacher_rows = load_partition_cache(teacher_summary, "teacher_7b", "eval_b")
    tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    wrapper.eval()
    composite = CoconutRecurrentQwen(wrapper, latent_token_id=int(resize.control_token_ids[2])).eval()
    with torch.no_grad():
        if composite.horizontal_bridge.delta.weight.abs().max().item() != 0:
            raise RuntimeError("DC0 requires the horizontal bridge at exact identity")
    before = parameter_fingerprint(composite)
    private = Path(args.private_dir)
    private.mkdir(parents=True, exist_ok=True)

    # M2 uses the actual EVAL-B token distribution, fixed before any outcome scoring.
    embedding_square_sum = token_count = 0
    embedding = wrapper.qwen.embed_tokens
    with torch.no_grad():
        for row in rows:
            values = torch.tensor(row["input_ids"], device=args.device)
            states = embedding(values).float()
            embedding_square_sum += float(states.square().sum().cpu())
            token_count += states.numel()
    embedding_rms = math.sqrt(embedding_square_sum / token_count)

    probe_ids = torch.tensor([rows[0]["input_ids"][:8]], device=args.device)
    probe_plain = wrapper(
        input_ids=probe_ids,
        attention_mask=torch.ones_like(probe_ids),
        max_loops=1,
        use_cache=False,
        return_dict=True,
    ).logits[:, :-1, : resize.original_tokenizer_size].float().cpu()
    probe_k0 = composite.depth_by_append(
        input_ids=probe_ids,
        append_steps=0,
        feedback_mode="raw",
        capture_real_logits=True,
        prediction_vocab_size=resize.original_tokenizer_size,
    )
    if probe_k0.real_logits is None:
        raise RuntimeError("DC0 M7 probe did not preserve real logits")
    k0_max_diff = float((probe_plain - probe_k0.real_logits).abs().max())
    if k0_max_diff >= 1e-3:
        raise RuntimeError(f"DC0 k=0 identity failed: max_abs_diff={k0_max_diff}")
    probe_cached = composite.depth_by_append(
        input_ids=probe_ids,
        append_steps=1,
        feedback_mode="neutral",
        neutral_token_id=int(resize.control_token_ids[2]),
        capture_real_logits=True,
        prediction_vocab_size=resize.original_tokenizer_size,
    )
    probe_append = composite.depth_by_append(
        input_ids=probe_ids,
        append_steps=2,
        feedback_mode="raw",
        capture_real_logits=True,
        prediction_vocab_size=resize.original_tokenizer_size,
    )
    if probe_cached.real_logits is None or probe_append.real_logits is None:
        raise RuntimeError("DC0 cached eviction probe did not preserve real logits")
    cached_vs_registered_max_diff = float(
        (probe_k0.real_logits - probe_cached.real_logits).abs().max()
    )
    cached_vs_registered_prediction_disagreements = int(
        probe_k0.real_logits.argmax(dim=-1)
        .ne(probe_cached.real_logits.argmax(dim=-1))
        .sum()
    )
    if cached_vs_registered_prediction_disagreements:
        raise RuntimeError(
            "DC0 cached k=0 path changes registered predictions: "
            f"disagreements={cached_vs_registered_prediction_disagreements}"
        )
    downstream_max_diff = float(
        (probe_cached.real_logits - probe_append.real_logits).abs().max()
    )
    if downstream_max_diff >= 1e-3:
        raise RuntimeError(
            f"DC0 append-evict downstream identity failed: max_abs_diff={downstream_max_diff}"
        )

    inplace_rows = evaluate_inplace(
        wrapper,
        rows,
        vocab_size=resize.original_tokenizer_size,
        device=args.device,
        resume_dir=private / "inplace_batches",
    )
    inplace = flatten_rows(inplace_rows)
    targets = torch.cat(
        [teacher_rows[index]["teacher_greedy_token_id"].long() for index in range(len(rows))]
    )
    row_indices = torch.cat(
        [torch.full((len(row["input_ids"]) - 1,), index, dtype=torch.long) for index, row in enumerate(rows)]
    )
    strata = [
        str(row["stratum"])
        for row in rows
        for _ in range(len(row["input_ids"]) - 1)
    ]
    entropy = torch.cat([teacher_rows[index]["teacher_entropy"].float() for index in range(len(rows))])
    logprob = torch.cat(
        [teacher_rows[index]["drafter_token_logprob_under_teacher"].float() for index in range(len(rows))]
    )
    rank = torch.cat(
        [teacher_rows[index]["drafter_token_rank_under_teacher"].long() for index in range(len(rows))]
    )
    fresh_transition = transition_counts(inplace[:, 0], inplace[:, 1], targets)
    fresh_rows = _row_transition_counts(inplace[:, 0], inplace[:, 1], targets, row_indices)
    audit_cache = torch.load(args.audit_feature_cache, map_location="cpu", weights_only=False)
    audit_counts = audit_row_counts(audit_cache)
    if tuple(torch.tensor(audit_counts).sum(dim=0).tolist()) != (AUDIT_HELPS, AUDIT_HURTS):
        raise RuntimeError("banked audit feature cache does not reproduce 8564 helps / 30008 hurts")
    validity = baseline_validity(
        fresh=fresh_transition,
        fresh_rows=fresh_rows,
        audit_rows=audit_counts,
    )
    preconditions = {
        "banked_rg1_both_budgets_and_forced_l": composite_preflight["contracts"]["rg1"],
        "banked_rg2_bridge_identity": composite_preflight["contracts"]["rg2"],
        "banked_composite_preflight_sha256": sha256_file(args.composite_preflight_summary),
        "k0_registered_surgery_max_abs_logit_difference": k0_max_diff,
        "k0_identity_threshold": 1e-3,
        "cached_vs_registered_max_abs_logit_difference_descriptive": (
            cached_vs_registered_max_diff
        ),
        "cached_vs_registered_prediction_disagreements": (
            cached_vs_registered_prediction_disagreements
        ),
        "append_evict_later_real_max_abs_logit_difference": downstream_max_diff,
        "post_eviction_real_position_ids_match": True,
        "cache_length_equals_real_tokens_after_every_eviction": True,
        "horizontal_bridge_exact_identity": True,
        "embedding_rms": embedding_rms,
        "probe_fed_hidden_rms": probe_append.diagnostics["fed_hidden_rms_mean"],
        "probe_fed_to_embedding_rms_ratio": probe_append.diagnostics["fed_hidden_rms_mean"] / embedding_rms,
        "status": "green",
    }
    if validity["status"] != "green":
        summary = {
            "kind": "paper2_dc0_depth_by_append",
            "status": "blocked_baseline_validity",
            "training_started": False,
            "optimizer_steps": 0,
            "preconditions": preconditions,
            "baseline_validity": validity,
            "inplace": transition_report(
                inplace, targets, strata=strata, entropy=entropy, drafter_logprob=logprob, drafter_rank=rank
            ),
            "eval_b_read_once_scoring_spent": True,
        }
        write_json(Path(args.output_dir) / "summary.json", summary)
        return 2

    arm_specs = [
        ("append_raw", "raw", None, False),
        ("append_rms_matched", "rms_matched", embedding_rms, False),
        ("neutral_append", "neutral", None, False),
        ("append_read_at_t_query", "raw", None, True),
    ]
    arm_reports: dict[str, Any] = {}
    private_predictions: dict[str, torch.Tensor] = {"inplace": inplace}
    private_cached_k0: dict[str, torch.Tensor] = {}
    execution_path_diagnostics: dict[str, Any] = {}
    counters: dict[str, Any] = {}
    for name, mode, reference, read_at_t in arm_specs:
        predicted_rows, arm_counters = evaluate_append_arm(
            composite,
            rows,
            arm=name,
            feedback_mode=mode,
            reference_rms=reference,
            neutral_token_id=int(resize.control_token_ids[2]),
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            batch_size=args.append_batch_size,
            resume_dir=private / "append_batches",
            read_at_t_query=read_at_t,
        )
        cached_grid = flatten_rows(predicted_rows)
        predicted, cached_k0, path_diagnostics = anchor_registered_k0(
            cached_grid,
            inplace[:, 0],
        )
        private_predictions[name] = predicted
        private_cached_k0[name] = cached_k0
        execution_path_diagnostics[name] = {
            **path_diagnostics,
            "cached_k0_to_k1_descriptive": transition_counts(
                cached_k0,
                predicted[:, 1],
                targets,
            ),
            "registered_k0_to_k1_primary": transition_counts(
                predicted[:, 0],
                predicted[:, 1],
                targets,
            ),
        }
        counters[name] = arm_counters
        arm_reports[name] = transition_report(
            predicted,
            targets,
            strata=strata,
            entropy=entropy,
            drafter_logprob=logprob,
            drafter_rank=rank,
        )

    after = parameter_fingerprint(composite)
    if before != after:
        raise RuntimeError("DC0 mutated the frozen checkpoint or horizontal bridge")
    private_payload = {
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "data_jsonl_sha256": sha256_file(args.data_jsonl),
        "teacher_targets": targets,
        "row_indices": row_indices,
        "predictions": private_predictions,
        "cached_k0_predictions": private_cached_k0,
    }
    private_path = private / "dc0_predictions.pt"
    torch.save(private_payload, private_path)
    summary = {
        "kind": "paper2_dc0_depth_by_append",
        "status": "complete_diagnostic_requires_strategy_review",
        "training_started": False,
        "optimizer_steps": 0,
        "checkpoint_mutated": False,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "eval_b_jsonl_sha256": sha256_file(args.data_jsonl),
        "eval_b_positions": len(targets),
        "eval_b_read_once_scoring_spent": True,
        "preconditions": preconditions,
        "baseline_validity": validity,
        "inplace": transition_report(
            inplace, targets, strata=strata, entropy=entropy, drafter_logprob=logprob, drafter_rank=rank
        ),
        "append_arms": arm_reports,
        "execution_path_diagnostics": execution_path_diagnostics,
        "m7_counters": counters,
        "compute_accounting": {
            **layer_application_costs(),
            "read_at_t_query_extra_per_feedback_step": 24,
            "matched_first_marginal_comparator": "append k=1 versus in-place depth=3",
        },
        "matched_layer_application_comparison": {
            "common_extra_layer_applications_per_position": 24,
            "inplace_depth_3_vs_depth_1": transition_counts(
                inplace[:, 0], inplace[:, 2], targets
            ),
            "append_raw_k1_vs_k0": transition_counts(
                private_predictions["append_raw"][:, 0],
                private_predictions["append_raw"][:, 1],
                targets,
            ),
            "attention_overhead_excluded_from_layer_application_match": True,
        },
        "interpretation": {
            "bands_are_diagnostic_not_gates": True,
            "automatic_verdict": "withheld_for_strategy_review",
            "architecture_scope": "same post-D0 checkpoint; D1 utility-training counterfactual remains open",
            "bridge_adaptation_authorized": False,
            "registered_k0_is_primary_anchor": True,
            "cached_k0_drift_is_descriptive_not_suppressed": True,
            "neutral_append_is_required_execution_path_control": True,
        },
        "private_predictions_sha256": sha256_file(private_path),
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    build_report(summary, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del wrapper, composite
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def build_report(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DC0 Depth-by-Append Diagnostic",
        "",
        f"Status: `{summary['status']}`",
        "",
        "No training occurred. EVAL-B is now spent for this registered comparison.",
        "",
        "## First Transition",
        "",
        "| Arm | Helps | Hurts | Net delta | Harm/help |",
        "|---|---:|---:|---:|---:|",
    ]
    arms = {"in-place 1->2": summary["inplace"]["transitions"]["0_to_1"]["pooled"]}
    arms.update(
        {
            name: report["transitions"]["0_to_1"]["pooled"]
            for name, report in summary["append_arms"].items()
        }
    )
    for name, cell in arms.items():
        lines.append(
            f"| {name} | {cell['helps']} | {cell['hurts']} | {cell['net_correct_delta']} | "
            f"{cell['harm_to_help_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Execution-Path Diagnostic",
            "",
            "Append-grid k=0 is anchored to the registered full-sequence depth-1 prediction. "
            "The incremental-cache k=0 prediction is retained as a descriptive diagnostic, "
            "not silently substituted for the registered baseline.",
            "",
            "| Arm | Cached-vs-registered disagreements | Rate | Cached-path net | Registered-anchor net |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, diagnostic in summary["execution_path_diagnostics"].items():
        cached = diagnostic["cached_k0_to_k1_descriptive"]
        registered = diagnostic["registered_k0_to_k1_primary"]
        lines.append(
            f"| {name} | {diagnostic['prediction_disagreements']} | "
            f"{diagnostic['prediction_disagreement_rate']:.6f} | "
            f"{cached['net_correct_delta']} | {registered['net_correct_delta']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    labels = list(arms)
    helps = [arms[name]["helps"] for name in labels]
    hurts = [arms[name]["hurts"] for name in labels]
    x = torch.arange(len(labels)).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - 0.18, helps, width=0.36, label="helps")
    axes[0].bar(x + 0.18, hurts, width=0.36, label="hurts")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Positions")
    axes[0].set_title("First marginal transition")
    axes[0].legend(frameon=False)
    axes[1].bar(x, [helps[i] - hurts[i] for i in range(len(labels))])
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("Net correct-token delta")
    axes[1].set_title("Utility before compute penalty")
    figure.tight_layout()
    figure.savefig(output_dir / "dc0_first_transition.png", dpi=180, bbox_inches="tight")
    figure.savefig(output_dir / "dc0_first_transition.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
