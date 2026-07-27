"""Read-only D0 causal allocation audit and D1 utility-label construction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_speculative_depth_d0_floor import load_partition_cache
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_spec import deterministic_argmax_fp32


AUDIT_SEED = 20260727
FOLDS = 5
MAX_LOOPS = 4
PENALTIES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0 / 3.0, 0.50, 1.00)
SCALAR_SCHEMA = (
    "answer_top1_top2_margin",
    "answer_top1_logprob",
    "answer_top1_logit",
    "answer_prediction_changed",
    "state_rms",
    "state_update_rms",
    "state_previous_cosine",
    "control_stop_minus_continue_margin",
)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_parameter_fingerprint(module: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = parameter.detach().float()
            digest.update(
                repr(
                    (
                        name,
                        tuple(parameter.shape),
                        str(parameter.dtype),
                        float(value.sum().item()),
                        float(value.square().sum().item()),
                    )
                ).encode("utf-8")
            )
    return digest.hexdigest()


def transition_label(before: bool, after: bool) -> str:
    if not before and after:
        return "helps"
    if before and not after:
        return "hurts"
    return "neutral"


def transition_labels(matches: Sequence[bool]) -> list[str]:
    if len(matches) < 2:
        raise ValueError("transition labels require at least two loop outcomes")
    return [transition_label(bool(left), bool(right)) for left, right in zip(matches, matches[1:])]


def source_fold(row_index: int, stratum: str, *, seed: int = AUDIT_SEED) -> int:
    payload = f"{seed}:{stratum}:{row_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % FOLDS


def deterministic_sample_rows(
    rows: Sequence[dict[str, Any]], *, max_positions: int, seed: int = AUDIT_SEED
) -> tuple[list[int], int]:
    ranked = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{rows[index].get('row_id', index)}".encode("utf-8")
        ).digest(),
    )
    selected: list[int] = []
    positions = 0
    for index in ranked:
        selected.append(index)
        positions += max(0, len(rows[index]["input_ids"]) - 1)
        if positions >= max_positions:
            break
    return selected, positions


def _selected_depth(controls: Sequence[int]) -> int:
    for loop, decision in enumerate(controls, start=1):
        if int(decision) == 1:
            return loop
    return len(controls)


def _row_cache_valid(path: Path, signature: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return payload.get("signature") == signature


@torch.inference_mode()
def extract_partition_cache(
    *,
    data_jsonl: Path,
    cache_summary_path: Path,
    partition: str,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    output_cache: Path,
    row_cache_dir: Path,
    device: str,
    dtype: str,
    attn_implementation: str,
    capture_scalars: bool,
    selected_row_indices: Sequence[int] | None = None,
    max_positions: int | None = None,
) -> dict[str, Any]:
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise RuntimeError("D1 audit checkpoint SHA-256 mismatch")
    sources = read_jsonl(data_jsonl)
    cache_summary = read_json(cache_summary_path)
    teacher_rows = load_partition_cache(cache_summary, "teacher_7b", partition)
    selected = list(selected_row_indices) if selected_row_indices is not None else list(range(len(sources)))
    signature = {
        "kind": "paper2_d1_causal_allocation_feature_cache_v1",
        "checkpoint_sha256": checkpoint_sha,
        "data_jsonl_sha256": sha256_file(data_jsonl),
        "teacher_cache_summary_sha256": sha256_file(cache_summary_path),
        "partition": partition,
        "forced_depths": [1, 2, 3, 4],
        "capture_scalars": bool(capture_scalars),
        "selected_row_indices_sha256": hashlib.sha256(json.dumps(selected).encode("utf-8")).hexdigest(),
        "max_positions": max_positions,
        "scalar_schema": list(SCALAR_SCHEMA) if capture_scalars else [],
        "teacher_features_excluded_from_probe_inputs": True,
    }
    if _row_cache_valid(output_cache, signature):
        print(f"d1_audit_cache_reused={output_cache}", flush=True)
        return torch.load(output_cache, map_location="cpu", weights_only=False)

    _, wrapper, resize, _ = load_drafter(
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in wrapper.parameters()):
        raise AssertionError("D1 audit requires a frozen model")
    wrapper.eval()
    before = frozen_parameter_fingerprint(wrapper)
    row_cache_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    predictions_out: list[torch.Tensor] = []
    matches_out: list[torch.Tensor] = []
    controls_out: list[torch.Tensor] = []
    scalars_out: list[torch.Tensor] = []
    remaining = max_positions

    for selected_number, row_index in enumerate(selected, start=1):
        source = sources[row_index]
        teacher = teacher_rows[row_index]
        row_signature = {**signature, "row_index": row_index}
        row_path = row_cache_dir / f"row_{row_index:06d}.pt"
        if _row_cache_valid(row_path, row_signature):
            result = torch.load(row_path, map_location="cpu", weights_only=False)
        else:
            values = [int(value) for value in source["input_ids"]]
            input_ids = torch.tensor([values], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            recurrent_states: list[torch.Tensor] = []
            prelude_captures: list[torch.Tensor] = []

            def capture_prelude(
                _module: Any, _inputs: tuple[Any, ...], layer_output: Any
            ) -> None:
                state = layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output
                prelude_captures.append(state)

            hook = None
            if capture_scalars:
                prelude_layer = wrapper.qwen.layers[int(wrapper.layer_split.prelude_end) - 1]
                hook = prelude_layer.register_forward_hook(capture_prelude)
            call_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention,
                labels=None,
                max_loops=MAX_LOOPS,
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
            if capture_scalars:
                call_kwargs["recurrent_application_sink"] = recurrent_states
            try:
                output = wrapper(**call_kwargs)
            finally:
                if hook is not None:
                    hook.remove()
            if output.loop_logits is None:
                raise RuntimeError("D1 audit requires per-loop logits")
            logits = output.loop_logits[0, 0, :, : len(values) - 1]
            answer_logits = logits[..., : resize.original_tokenizer_size]
            predicted, _answer_ties = deterministic_argmax_fp32(answer_logits, dim=-1)
            control_logits = logits[..., list(resize.control_token_ids[:2])]
            controls, _control_ties = deterministic_argmax_fp32(control_logits, dim=-1)
            teacher_ids = teacher["teacher_greedy_token_id"].long().to(device)
            matches = predicted.eq(teacher_ids.unsqueeze(0))
            row_scalars = torch.empty((len(values) - 1, MAX_LOOPS, 0), dtype=torch.float32)
            if capture_scalars:
                if len(recurrent_states) != MAX_LOOPS:
                    raise RuntimeError("D1 audit did not capture four recurrent states")
                if len(prelude_captures) != 1:
                    raise RuntimeError("D1 audit did not capture exactly one Prelude state")
                states = torch.stack(recurrent_states, dim=1)[0].float()
                previous_state = prelude_captures[0][0, : len(values) - 1].float()
                previous_prediction: torch.Tensor | None = None
                scalar_loops: list[torch.Tensor] = []
                for loop_index in range(MAX_LOOPS):
                    loop_scores = answer_logits[loop_index].float()
                    top = loop_scores.topk(k=2, dim=-1)
                    prediction = predicted[loop_index]
                    current_state = states[:, loop_index]
                    delta = current_state - previous_state
                    scalar_loops.append(
                        torch.stack(
                            [
                                top.values[:, 0] - top.values[:, 1],
                                top.values[:, 0] - torch.logsumexp(loop_scores, dim=-1),
                                top.values[:, 0],
                                torch.zeros_like(prediction, dtype=torch.float32)
                                if previous_prediction is None
                                else prediction.ne(previous_prediction).float(),
                                current_state.square().mean(dim=-1).sqrt(),
                                delta.square().mean(dim=-1).sqrt(),
                                torch.nn.functional.cosine_similarity(
                                    current_state, previous_state, dim=-1, eps=1e-8
                                ),
                                control_logits[loop_index, :, 1].float()
                                - control_logits[loop_index, :, 0].float(),
                            ],
                            dim=-1,
                        )
                    )
                    previous_state = current_state
                    previous_prediction = prediction
                row_scalars = torch.stack(scalar_loops, dim=1).cpu()
            result = {
                "signature": row_signature,
                "predictions": predicted.transpose(0, 1).cpu(),
                "matches": matches.transpose(0, 1).cpu(),
                "controls": controls.transpose(0, 1).cpu(),
                "scalars": row_scalars,
            }
            temporary = row_path.with_suffix(".pt.tmp")
            torch.save(result, temporary)
            os.replace(temporary, row_path)
            del output, logits, answer_logits, control_logits

        count = len(result["predictions"])
        if remaining is not None:
            count = min(count, remaining)
        if count <= 0:
            break
        for local_position in range(count):
            metadata.append(
                {
                    "row_index": row_index,
                    "local_position": local_position,
                    "sequence_length": len(source["input_ids"]),
                    "stratum": str(source["stratum"]),
                    "plain_accepted": bool(teacher["accepted"][local_position]),
                    "teacher_entropy": float(teacher["teacher_entropy"][local_position]),
                    "teacher_to_plain_drafter_kl": float(
                        teacher["teacher_to_plain_drafter_kl"][local_position]
                    ),
                    "drafter_token_rank_under_teacher": int(
                        teacher["drafter_token_rank_under_teacher"][local_position]
                    ),
                    "drafter_token_logprob_under_teacher": float(
                        teacher["drafter_token_logprob_under_teacher"][local_position]
                    ),
                    "rejection_run_length": int(teacher["rejection_run_length"][local_position]),
                }
            )
        predictions_out.append(result["predictions"][:count])
        matches_out.append(result["matches"][:count])
        controls_out.append(result["controls"][:count])
        if capture_scalars:
            scalars_out.append(result["scalars"][:count])
        if remaining is not None:
            remaining -= count
        if selected_number == 1 or selected_number % 32 == 0 or remaining == 0:
            print(
                f"d1_audit_progress partition={partition} rows={selected_number}/{len(selected)} "
                f"positions={len(metadata)}",
                flush=True,
            )
        if remaining == 0:
            break

    after = frozen_parameter_fingerprint(wrapper)
    if before != after:
        raise RuntimeError("D1 audit mutated the frozen checkpoint")
    payload = {
        "signature": signature,
        "frozen_parameter_fingerprint_before": before,
        "frozen_parameter_fingerprint_after": after,
        "metadata": metadata,
        "predictions": torch.cat(predictions_out, dim=0),
        "matches": torch.cat(matches_out, dim=0),
        "controls": torch.cat(controls_out, dim=0),
        "scalars": torch.cat(scalars_out, dim=0)
        if capture_scalars
        else torch.empty((len(metadata), MAX_LOOPS, 0), dtype=torch.float32),
    }
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_cache.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_cache)
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def calibration_cache(
    *, private_rows_path: Path, cache_summary_path: Path, data_jsonl: Path
) -> dict[str, Any]:
    private = read_json(private_rows_path)
    cache_summary = read_json(cache_summary_path)
    seven = load_partition_cache(cache_summary, "teacher_7b", "calibration")
    fourteen = load_partition_cache(cache_summary, "teacher_14b", "calibration")
    sources = read_jsonl(data_jsonl)
    rows: list[dict[str, Any]] = []
    for item in private["all_position_rows"]:
        row_index = int(item["row_index"])
        local = int(item["local_position"])
        row7 = seven[row_index]
        row14 = fourteen[row_index]
        predictions = [int(value) for value in item["predictions"][:MAX_LOOPS]]
        teacher7 = int(item["teacher_7b"])
        rows.append(
            {
                "row_index": row_index,
                "local_position": local,
                "stratum": str(sources[row_index]["stratum"]),
                "predictions": predictions,
                "matches": [value == teacher7 for value in predictions],
                "plain_accepted": bool(row7["accepted"][local]),
                "teacher_entropy": float(row7["teacher_entropy"][local]),
                "teacher_to_plain_drafter_kl": float(row7["teacher_to_plain_drafter_kl"][local]),
                "drafter_token_rank_under_teacher": int(
                    row7["drafter_token_rank_under_teacher"][local]
                ),
                "drafter_token_logprob_under_teacher": float(
                    row7["drafter_token_logprob_under_teacher"][local]
                ),
                "teacher_14b_endorses_loop1": predictions[0] == int(item["teacher_14b"]),
                "rejected_7b": not bool(row7["accepted"][local]),
                "teacher_14b_endorses_plain_drafter": int(
                    row7["drafter_greedy_token_id"][local]
                )
                == int(item["teacher_14b"]),
            }
        )
    return {"rows": rows, "private_rows_sha256": sha256_file(private_rows_path)}


def _transition_counts(matches: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for loop in range(MAX_LOOPS - 1):
        counts = Counter(
            transition_label(bool(before), bool(after))
            for before, after in zip(matches[:, loop].tolist(), matches[:, loop + 1].tolist())
        )
        result[f"{loop + 1}_to_{loop + 2}"] = {
            name: {"count": counts[name], "share": counts[name] / len(matches)}
            for name in ("helps", "hurts", "neutral")
        }
    return result


def transition_summary(matches: torch.Tensor, metadata: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if matches.ndim != 2 or matches.shape[1] < MAX_LOOPS:
        raise ValueError("transition summary requires [positions, four loops]")
    result = _transition_counts(matches)
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({str(row["stratum"]) for row in metadata}):
        indices = [index for index, row in enumerate(metadata) if str(row["stratum"]) == stratum]
        if indices:
            by_stratum[stratum] = _transition_counts(matches[indices])
    return {"pooled": result, "by_stratum": by_stratum}


def oracle_frontier(matches: torch.Tensor, penalties: Sequence[float] = PENALTIES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = matches[:, :MAX_LOOPS].float()
    depths = torch.arange(MAX_LOOPS, dtype=torch.float32)
    for penalty in penalties:
        utilities = outcomes - float(penalty) * depths.unsqueeze(0)
        selected = utilities.argmax(dim=1)
        selected_match = outcomes.gather(1, selected.unsqueeze(1)).squeeze(1)
        mean_loops = float((selected.float() + 1.0).mean().item())
        accuracy = float(selected_match.mean().item())
        rows.append(
            {
                "penalty": float(penalty),
                "correct": int(selected_match.sum().item()),
                "total": len(matches),
                "accuracy": accuracy,
                "mean_loops": mean_loops,
                "net_utility": accuracy - float(penalty) * (mean_loops - 1.0),
                "selected_depth_counts": {
                    str(depth): int(selected.eq(depth - 1).sum().item())
                    for depth in range(1, MAX_LOOPS + 1)
                },
            }
        )
    return rows


def fixed_depth_baselines(matches: torch.Tensor) -> dict[str, Any]:
    return {
        str(depth): {
            "correct": int(matches[:, depth - 1].sum()),
            "total": len(matches),
            "accuracy": float(matches[:, depth - 1].float().mean()),
            "mean_loops": float(depth),
        }
        for depth in range(1, MAX_LOOPS + 1)
    }


def deployed_policy_frontier(
    matches: torch.Tensor, controls: torch.Tensor, penalties: Sequence[float] = PENALTIES
) -> list[dict[str, Any]]:
    selected = torch.tensor([_selected_depth(row.tolist()) for row in controls], dtype=torch.long)
    outcome = matches.gather(1, (selected - 1).unsqueeze(1)).squeeze(1).float()
    accuracy = float(outcome.mean())
    mean_loops = float(selected.float().mean())
    return [
        {
            "penalty": float(penalty),
            "correct": int(outcome.sum()),
            "total": len(matches),
            "accuracy": accuracy,
            "mean_loops": mean_loops,
            "net_utility": accuracy - float(penalty) * (mean_loops - 1.0),
        }
        for penalty in penalties
    ]


def d1_label_balance(matches: torch.Tensor) -> dict[str, Any]:
    helps = (~matches[:, : MAX_LOOPS - 1]) & matches[:, 1:MAX_LOOPS]
    total = int(helps.numel())
    positives = int(helps.sum())
    return {
        "continue": positives,
        "stop": total - positives,
        "total_transition_labels": total,
        "continue_share": positives / total if total else 0.0,
        "inverse_frequency_class_weight_ratio_stop_to_continue": (
            (total - positives) / positives if positives else None
        ),
        "per_transition": {
            f"{loop + 1}_to_{loop + 2}": {
                "continue": int(helps[:, loop].sum()),
                "stop": int((~helps[:, loop]).sum()),
                "continue_share": float(helps[:, loop].float().mean()),
            }
            for loop in range(MAX_LOOPS - 1)
        },
    }


def policy_confusion(matches: torch.Tensor, controls: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reachable = torch.ones(len(matches), dtype=torch.bool)
    for loop in range(MAX_LOOPS - 1):
        true_continue = (~matches[:, loop]) & matches[:, loop + 1]
        predicted_continue = controls[:, loop].eq(0)

        def table(mask: torch.Tensor) -> dict[str, int]:
            return {
                "continue_true_continue": int((mask & predicted_continue & true_continue).sum()),
                "continue_true_stop": int((mask & predicted_continue & ~true_continue).sum()),
                "stop_true_continue": int((mask & ~predicted_continue & true_continue).sum()),
                "stop_true_stop": int((mask & ~predicted_continue & ~true_continue).sum()),
                "positions": int(mask.sum()),
            }

        result[str(loop + 1)] = {
            "all_forced_positions": table(torch.ones_like(reachable)),
            "deployed_reachable_positions": table(reachable),
        }
        reachable &= predicted_continue
    return result


def simulate_threshold_policy(
    scores: torch.Tensor, matches: torch.Tensor, thresholds: Sequence[float], penalty: float
) -> dict[str, Any]:
    selected = torch.full((len(matches),), MAX_LOOPS - 1, dtype=torch.long)
    active = torch.ones(len(matches), dtype=torch.bool)
    for loop in range(MAX_LOOPS - 1):
        stop = active & scores[:, loop].lt(float(thresholds[loop]))
        selected[stop] = loop
        active &= ~stop
    outcome = matches.gather(1, selected.unsqueeze(1)).squeeze(1).float()
    mean_loops = float((selected.float() + 1).mean())
    accuracy = float(outcome.mean())
    return {
        "correct": int(outcome.sum()),
        "total": len(matches),
        "accuracy": accuracy,
        "mean_loops": mean_loops,
        "net_utility": accuracy - float(penalty) * (mean_loops - 1.0),
    }


def _ridge_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    train: torch.Tensor,
    score_indices: torch.Tensor,
) -> torch.Tensor:
    x_train = features[train].float()
    mean = x_train.mean(0)
    scale = x_train.std(0, unbiased=False).clamp_min(1e-5)
    x_train = (x_train - mean) / scale
    x_score = (features[score_indices].float() - mean) / scale
    y = labels[train].float().mul(2).sub(1)
    positives = int(labels[train].sum())
    negatives = len(train) - positives
    if positives == 0 or negatives == 0:
        return torch.full((len(score_indices),), float(positives > 0))
    weights = torch.where(
        labels[train],
        torch.full_like(y, len(y) / (2.0 * positives)),
        torch.full_like(y, len(y) / (2.0 * negatives)),
    )
    design = torch.cat([x_train, torch.ones(len(x_train), 1)], dim=1)
    weighted = design * weights.sqrt().unsqueeze(1)
    target = y * weights.sqrt()
    identity = torch.eye(design.shape[1])
    identity[-1, -1] = 0.0
    coefficients = torch.linalg.solve(weighted.T @ weighted + 0.1 * identity, weighted.T @ target)
    return torch.cat([x_score, torch.ones(len(x_score), 1)], dim=1) @ coefficients


def _candidate_thresholds(values: torch.Tensor) -> list[float]:
    quantiles = torch.linspace(0.0, 1.0, 41)
    candidates = torch.quantile(values.float(), quantiles).unique().tolist()
    return [float("-inf"), *[float(value) for value in candidates], float("inf")]


def _select_thresholds(
    validation_scores: torch.Tensor, validation_matches: torch.Tensor, penalty: float
) -> list[float]:
    thresholds = [float("inf")] * (MAX_LOOPS - 1)
    candidates = [_candidate_thresholds(validation_scores[:, loop]) for loop in range(MAX_LOOPS - 1)]
    for _ in range(3):
        changed = False
        for loop in range(MAX_LOOPS - 1):
            best = thresholds[loop]
            best_key = None
            for candidate in candidates[loop]:
                trial = list(thresholds)
                trial[loop] = candidate
                result = simulate_threshold_policy(
                    validation_scores, validation_matches, trial, penalty
                )
                key = (result["net_utility"], result["accuracy"], -result["mean_loops"])
                if best_key is None or key > best_key:
                    best_key = key
                    best = candidate
            if best != thresholds[loop]:
                thresholds[loop] = best
                changed = True
        if not changed:
            break
    return thresholds


def cross_fitted_utility(
    *,
    scalars: torch.Tensor,
    matches: torch.Tensor,
    metadata: Sequence[dict[str, Any]],
    penalties: Sequence[float] = PENALTIES,
) -> list[dict[str, Any]]:
    if scalars.shape[:2] != matches.shape[:2]:
        raise ValueError("scalar and match tensors are not aligned")
    structural = torch.tensor(
        [
            [
                math.log1p(float(row["sequence_length"])),
                float(row["local_position"]) / max(1.0, float(row["sequence_length"] - 1)),
                float(str(row["stratum"]) == "code"),
            ]
            for row in metadata
        ],
        dtype=torch.float32,
    )
    features = torch.cat(
        [scalars[:, : MAX_LOOPS - 1].float(), structural[:, None, :].repeat(1, MAX_LOOPS - 1, 1)],
        dim=-1,
    )
    folds = torch.tensor(
        [source_fold(int(row["row_index"]), str(row["stratum"])) for row in metadata]
    )
    aggregated: dict[float, dict[str, float]] = {
        float(penalty): {"correct": 0.0, "total": 0.0, "loop_sum": 0.0}
        for penalty in penalties
    }
    fold_receipts: list[dict[str, Any]] = []
    for outer in range(FOLDS):
        validation_fold = (outer + 1) % FOLDS
        train = (folds.ne(outer) & folds.ne(validation_fold)).nonzero().flatten()
        validation = folds.eq(validation_fold).nonzero().flatten()
        test = folds.eq(outer).nonzero().flatten()
        validation_scores = torch.empty((len(validation), MAX_LOOPS - 1))
        test_scores = torch.empty((len(test), MAX_LOOPS - 1))
        for loop in range(MAX_LOOPS - 1):
            labels = (~matches[:, loop]) & matches[:, loop + 1]
            validation_scores[:, loop] = _ridge_scores(features[:, loop], labels, train, validation)
            test_scores[:, loop] = _ridge_scores(features[:, loop], labels, train, test)
        fold_result: dict[str, Any] = {
            "outer_test_fold": outer,
            "validation_fold": validation_fold,
            "fit_positions": len(train),
            "validation_positions": len(validation),
            "test_positions": len(test),
            "penalties": {},
        }
        for penalty in penalties:
            thresholds = _select_thresholds(validation_scores, matches[validation], float(penalty))
            result = simulate_threshold_policy(test_scores, matches[test], thresholds, float(penalty))
            cell = aggregated[float(penalty)]
            cell["correct"] += result["correct"]
            cell["total"] += result["total"]
            cell["loop_sum"] += result["mean_loops"] * result["total"]
            fold_result["penalties"][str(float(penalty))] = {
                "thresholds": thresholds,
                "test": result,
            }
        fold_receipts.append(fold_result)
    curve: list[dict[str, Any]] = []
    for penalty in penalties:
        cell = aggregated[float(penalty)]
        total = int(cell["total"])
        accuracy = cell["correct"] / total
        mean_loops = cell["loop_sum"] / total
        curve.append(
            {
                "penalty": float(penalty),
                "correct": int(cell["correct"]),
                "total": total,
                "accuracy": accuracy,
                "mean_loops": mean_loops,
                "net_utility": accuracy - float(penalty) * (mean_loops - 1.0),
            }
        )
    return [{"curve": curve, "folds": fold_receipts}][0]


def accepted_loss_decomposition(
    matches: torch.Tensor, controls: torch.Tensor, metadata: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    selected = torch.tensor([_selected_depth(row.tolist()) for row in controls], dtype=torch.long)
    adaptive = matches.gather(1, (selected - 1).unsqueeze(1)).squeeze(1)
    plain_accepted = torch.tensor([bool(row["plain_accepted"]) for row in metadata])
    lost = plain_accepted & ~adaptive
    loop1_regression = lost & ~matches[:, 0]
    policy_loss = lost & matches[:, 0]
    transitions = Counter()
    strata = Counter()
    severity = Counter()
    for index in lost.nonzero().flatten().tolist():
        row = metadata[index]
        strata[str(row["stratum"])] += 1
        severity[str(row.get("severity_bin", "unbinned"))] += 1
        if loop1_regression[index]:
            transitions["before_policy_loop1_weight_regression"] += 1
            continue
        depth = int(selected[index])
        cause = "unlocalized"
        for loop in range(depth - 1):
            if matches[index, loop] and not matches[index, loop + 1]:
                cause = f"{loop + 1}_to_{loop + 2}"
                break
        transitions[cause] += 1
    total = int(lost.sum())
    preventable = int(policy_loss.sum())
    return {
        "accepted_positions_lost": total,
        "loop1_weight_regression": int(loop1_regression.sum()),
        "post_loop_policy_losses": preventable,
        "preventable_by_stop_on_nonhelp_label": preventable,
        "preventable_fraction": preventable / total if total else 0.0,
        "by_first_harmful_transition": dict(sorted(transitions.items())),
        "by_stratum": dict(sorted(strata.items())),
        "by_severity": dict(sorted(severity.items())),
    }


def _quartile_bins(values: Sequence[float]) -> tuple[list[float], list[str]]:
    tensor = torch.tensor(values, dtype=torch.float32)
    boundaries = [float(torch.quantile(tensor, q)) for q in (0.25, 0.50, 0.75)]
    names = []
    for value in values:
        names.append("q1" if value <= boundaries[0] else "q2" if value <= boundaries[1] else "q3" if value <= boundaries[2] else "q4")
    return boundaries, names


def rescue_confidence(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entropy_bounds, entropy_bins = _quartile_bins([float(row["teacher_entropy"]) for row in rows])
    logprob_bounds, logprob_bins = _quartile_bins(
        [float(row["drafter_token_logprob_under_teacher"]) for row in rows]
    )
    result = {
        "teacher_top1_top2_margin_available": False,
        "margin_unavailable_reason": "not cached under the registered single-pass teacher contract",
        "teacher_reload_performed": False,
        "entropy_quartile_boundaries": entropy_bounds,
        "drafter_token_logprob_quartile_boundaries": logprob_bounds,
        "by_entropy_quartile": Counter(),
        "by_drafter_token_logprob_quartile": Counter(),
        "by_teacher_rank": Counter(),
        "rescue_events": 0,
    }
    for index, row in enumerate(rows):
        labels = transition_labels(row["matches"][:MAX_LOOPS])
        rescue_count = labels.count("helps")
        if not rescue_count:
            continue
        result["rescue_events"] += rescue_count
        result["by_entropy_quartile"][entropy_bins[index]] += rescue_count
        result["by_drafter_token_logprob_quartile"][logprob_bins[index]] += rescue_count
        rank = int(row["drafter_token_rank_under_teacher"])
        rank_bin = "1" if rank == 1 else "2-5" if rank <= 5 else "6-20" if rank <= 20 else ">20"
        result["by_teacher_rank"][rank_bin] += rescue_count
    for key in ("by_entropy_quartile", "by_drafter_token_logprob_quartile", "by_teacher_rank"):
        result[key] = dict(sorted(result[key].items()))
    return result


def cache_rows_as_records(cache: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for index, metadata in enumerate(cache["metadata"]):
        records.append(
            {
                **metadata,
                "predictions": cache["predictions"][index].tolist(),
                "matches": cache["matches"][index].tolist(),
                "controls": cache["controls"][index].tolist(),
            }
        )
    return records


def replay_equivalence(cache: dict[str, Any], reference_rows_path: Path) -> dict[str, Any]:
    reference = read_json(reference_rows_path)
    rows = list(reference["rows"])
    if len(rows) != len(cache["metadata"]):
        raise RuntimeError("D1 audit replay and banked D0 private rows have different lengths")
    counters = Counter()
    for index, banked in enumerate(rows):
        metadata = cache["metadata"][index]
        if (
            int(banked["row_index"]) != int(metadata["row_index"])
            or int(banked["local_position"]) != int(metadata["local_position"])
        ):
            raise RuntimeError("D1 audit replay and banked D0 rows are not position-aligned")
        predictions = cache["predictions"][index]
        selected = _selected_depth(cache["controls"][index].tolist())
        adaptive = int(predictions[selected - 1])
        counters["loop1"] += int(int(predictions[0]) != int(banked["loop1_token_id"]))
        counters["loop4"] += int(int(predictions[3]) != int(banked["loop4_token_id"]))
        counters["selected_loop"] += int(selected != int(banked["selected_loop"]))
        counters["adaptive_answer"] += int(adaptive != int(banked["adaptive_token_id"]))
    exact = all(value == 0 for value in counters.values())
    result = {
        "reference_private_rows_sha256": sha256_file(reference_rows_path),
        "positions": len(rows),
        "mismatches": dict(counters),
        "exact": exact,
    }
    if not exact:
        raise RuntimeError(f"D1 audit replay differs from banked D0 A100 anchors: {result}")
    return result


def add_severity_bins(metadata: list[dict[str, Any]], boundaries: Sequence[float]) -> None:
    for row in metadata:
        value = float(row["teacher_to_plain_drafter_kl"])
        row["severity_bin"] = (
            "q1" if value <= boundaries[0] else "q2" if value <= boundaries[1] else "q3" if value <= boundaries[2] else "q4"
        )


def markdown_summary(summary: dict[str, Any]) -> str:
    evaluation = summary["evaluation"]
    lines = [
        "# D0 Causal Allocation Audit and D1 Label Construction",
        "",
        f"- Status: `{summary['status']}`",
        f"- Checkpoint: `{summary['checkpoint_sha256']}`",
        f"- Evaluation positions: {summary['evaluation_positions']:,}",
        f"- Label-train dry-run positions: {summary['label_train_dry_run']['positions']:,}",
        "- Training or optimizer steps: 0",
        "",
        "## Transition outcomes on evaluation",
        "",
        "| Transition | Helps | Hurts | Neutral |",
        "|---|---:|---:|---:|",
    ]
    for transition, cell in evaluation["transition_outcomes"]["pooled"].items():
        lines.append(
            f"| {transition.replace('_', ' ')} | {cell['helps']['count']:,} | "
            f"{cell['hurts']['count']:,} | {cell['neutral']['count']:,} |"
        )
    loss = evaluation["accepted_loss_decomposition"]
    lines.extend(
        [
            "",
            "## Accepted-position policy damage",
            "",
            f"- Total baseline-accepted losses: {loss['accepted_positions_lost']:,}",
            f"- Already lost at loop 1 after training: {loss['loop1_weight_regression']:,}",
            f"- Post-loop policy losses: {loss['post_loop_policy_losses']:,}",
            f"- Preventable by stopping on non-help labels: {loss['preventable_fraction']:.1%}",
            "",
            "## Scope",
            "",
            "This is a post-hoc read-only audit. It cannot alter D0's registered verdict and does not authorize D1 training.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_figure(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    outcomes = summary["evaluation"]["transition_outcomes"]["pooled"]
    transitions = list(outcomes)
    helps = [outcomes[key]["helps"]["share"] for key in transitions]
    hurts = [outcomes[key]["hurts"]["share"] for key in transitions]
    neutral = [outcomes[key]["neutral"]["share"] for key in transitions]
    oracle = summary["evaluation"]["oracle_frontier"]
    deployable = summary["evaluation"]["cross_fitted_scalar_policy"]["curve"]
    deployed_d0 = summary["evaluation"]["deployed_d0_policy_frontier"]
    fixed = summary["evaluation"]["fixed_depth_baselines"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(transitions, helps, label="helps", color="#178f6a")
    axes[0].bar(transitions, hurts, bottom=helps, label="hurts", color="#c34b43")
    axes[0].bar(
        transitions,
        neutral,
        bottom=[a + b for a, b in zip(helps, hurts)],
        label="neutral",
        color="#b7bdc5",
    )
    axes[0].set_ylabel("Share of positions")
    axes[0].set_title("One-step causal outcomes")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].plot(
        [row["mean_loops"] for row in oracle],
        [row["accuracy"] for row in oracle],
        marker="o",
        label="oracle",
        color="#1d5f91",
    )
    axes[1].plot(
        [row["mean_loops"] for row in deployable],
        [row["accuracy"] for row in deployable],
        marker="o",
        label="cross-fitted scalars",
        color="#bb6b22",
    )
    axes[1].scatter(
        [deployed_d0[0]["mean_loops"]],
        [deployed_d0[0]["accuracy"]],
        marker="X",
        s=70,
        label="deployed D0",
        color="#8a3f80",
    )
    axes[1].scatter(
        [fixed[str(depth)]["mean_loops"] for depth in range(1, MAX_LOOPS + 1)],
        [fixed[str(depth)]["accuracy"] for depth in range(1, MAX_LOOPS + 1)],
        marker="s",
        s=28,
        label="fixed depths",
        color="#4b5563",
    )
    axes[1].set_xlabel("Mean loops")
    axes[1].set_ylabel("Teacher agreement")
    axes[1].set_title("Utility frontier")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_jsonl", required=True)
    parser.add_argument("--calibration_private_rows", required=True)
    parser.add_argument("--evaluation_reference_rows", required=True)
    parser.add_argument("--calibration_jsonl", required=True)
    parser.add_argument("--label_train_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--floor_summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_checkpoint_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--private_cache_dir", required=True)
    parser.add_argument("--sample_positions", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    private = Path(args.private_cache_dir)
    evaluation = extract_partition_cache(
        data_jsonl=Path(args.evaluation_jsonl),
        cache_summary_path=Path(args.teacher_cache_summary),
        partition="evaluation",
        checkpoint=Path(args.checkpoint),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        output_cache=private / "evaluation_feature_cache.pt",
        row_cache_dir=private / "evaluation_rows",
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        capture_scalars=True,
    )
    label_rows = read_jsonl(args.label_train_jsonl)
    selected_rows, sampled_positions = deterministic_sample_rows(
        label_rows, max_positions=args.sample_positions
    )
    label_train = extract_partition_cache(
        data_jsonl=Path(args.label_train_jsonl),
        cache_summary_path=Path(args.teacher_cache_summary),
        partition="label_train",
        checkpoint=Path(args.checkpoint),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        output_cache=private / "label_train_100k_cache.pt",
        row_cache_dir=private / "label_train_rows",
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        capture_scalars=False,
        selected_row_indices=selected_rows,
        max_positions=args.sample_positions,
    )
    floor = read_json(args.floor_summary)
    boundaries = [float(value) for value in floor["severity_quartile_boundaries"]]
    add_severity_bins(evaluation["metadata"], boundaries)
    add_severity_bins(label_train["metadata"], boundaries)
    evaluation_records = cache_rows_as_records(evaluation)
    equivalence = replay_equivalence(evaluation, Path(args.evaluation_reference_rows))
    calibration = calibration_cache(
        private_rows_path=Path(args.calibration_private_rows),
        cache_summary_path=Path(args.teacher_cache_summary),
        data_jsonl=Path(args.calibration_jsonl),
    )
    calibration_matches = torch.tensor([row["matches"] for row in calibration["rows"]])
    label_transition = transition_summary(label_train["matches"], label_train["metadata"])
    calibration_transition = transition_summary(calibration_matches, calibration["rows"])
    evaluation_transition = transition_summary(evaluation["matches"], evaluation["metadata"])
    crossfit = cross_fitted_utility(
        scalars=evaluation["scalars"],
        matches=evaluation["matches"],
        metadata=evaluation["metadata"],
    )
    endorsed_rows = [
        row
        for row in calibration["rows"]
        if row["rejected_7b"] and row["teacher_14b_endorses_plain_drafter"]
    ]
    summary = {
        "kind": "paper2_d0_causal_allocation_audit_d1_label_construction",
        "status": "complete",
        "audit_seed": AUDIT_SEED,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "optimizer_steps": 0,
        "training_started": False,
        "checkpoint_mutated": False,
        "evaluation_partition_touched_posthoc": True,
        "d0_verdict_unchanged": "not_recoverable_at_pilot_scale",
        "d0_scope": "binary teacher-disagreement targets, 4000 steps, one seed",
        "evaluation_positions": len(evaluation["metadata"]),
        "evaluation": {
            "banked_replay_equivalence": equivalence,
            "transition_outcomes": evaluation_transition,
            "d1_label_balance": d1_label_balance(evaluation["matches"]),
            "fixed_depth_baselines": fixed_depth_baselines(evaluation["matches"]),
            "oracle_frontier": oracle_frontier(evaluation["matches"]),
            "deployed_d0_policy_frontier": deployed_policy_frontier(
                evaluation["matches"], evaluation["controls"]
            ),
            "d0_policy_confusion": policy_confusion(evaluation["matches"], evaluation["controls"]),
            "cross_fitted_scalar_policy": crossfit,
            "accepted_loss_decomposition": accepted_loss_decomposition(
                evaluation["matches"], evaluation["controls"], evaluation["metadata"]
            ),
            "rescue_confidence": rescue_confidence(evaluation_records),
        },
        "calibration": {
            "positions": len(calibration["rows"]),
            "transition_outcomes": calibration_transition,
            "d1_label_balance": d1_label_balance(calibration_matches),
            "rescue_confidence": rescue_confidence(calibration["rows"]),
            "fourteen_b_endorsed_plain_drafter_on_7b_rejections_subset": {
                "positions": len(endorsed_rows),
                "rescue_confidence": rescue_confidence(endorsed_rows) if endorsed_rows else None,
            },
        },
        "label_train_dry_run": {
            "positions": len(label_train["metadata"]),
            "requested_positions": args.sample_positions,
            "sampled_positions_pretruncate": sampled_positions,
            "selected_source_rows": len(selected_rows),
            "selection_seed": AUDIT_SEED,
            "forced_depths": [1, 2, 3, 4],
            "transition_outcomes": label_transition,
            "d1_label_balance": d1_label_balance(label_train["matches"]),
            "calibration_train_comparison": {
                transition: {
                    label: {
                        "train_share": label_transition["pooled"][transition][label]["share"],
                        "calibration_share": calibration_transition["pooled"][transition][label]["share"],
                        "absolute_difference": label_transition["pooled"][transition][label]["share"]
                        - calibration_transition["pooled"][transition][label]["share"],
                    }
                    for label in ("helps", "hurts", "neutral")
                }
                for transition in label_transition["pooled"]
            },
        },
        "teacher_confidence_contract": {
            "teacher_entropy_available": True,
            "drafter_token_rank_under_teacher_available": True,
            "drafter_token_logprob_under_teacher_available": True,
            "teacher_top1_top2_margin_available": False,
            "teacher_reload_forbidden_and_not_performed": True,
        },
        "penalty_grid": list(PENALTIES),
        "cross_fit": {
            "folds": FOLDS,
            "source_row_grouped": True,
            "seed": AUDIT_SEED,
            "teacher_features_used": False,
            "hidden_state_projections_used": False,
        },
        "private_artifacts": {
            "evaluation_feature_cache": str(private / "evaluation_feature_cache.pt"),
            "label_train_cache": str(private / "label_train_100k_cache.pt"),
            "calibration_private_rows_sha256": calibration["private_rows_sha256"],
        },
        "d1_training_authorized": False,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    build_figure(summary, output_dir / "causal_allocation_audit.png")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
