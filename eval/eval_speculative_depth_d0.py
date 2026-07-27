"""Evaluate the locked Paper Two D0 adaptive-depth pilot on untouched rows."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_speculative_depth_d0_floor import load_partition_cache, quartile
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import validate_cache_summary
from training.speculative_depth_d0_spec import (
    DRAFTER_CHECKPOINT_SHA256,
    deterministic_argmax_fp32,
    validate_locked_d0,
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def synchronized_time(device: str) -> float:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_scale = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else None


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def first_stop(control_predictions: torch.Tensor, max_loops: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-indexed selected loops and whether the loop budget was exhausted."""

    stop = control_predictions.eq(1)
    indices = torch.arange(1, max_loops + 1, device=stop.device).view(1, -1)
    sentinel = torch.full_like(indices, max_loops + 1)
    selected = torch.where(stop, indices, sentinel).amin(dim=1)
    exhausted = selected.eq(max_loops + 1)
    return selected.clamp_max(max_loops), exhausted


def summarize_binary(correct: int, total: int) -> dict[str, Any]:
    return {"correct": int(correct), "total": int(total), "accuracy": correct / total if total else None}


@torch.inference_mode()
def evaluate_partition(
    wrapper: Any,
    rows: list[dict[str, Any]],
    cache_rows: dict[int, dict[str, Any]],
    *,
    control_ids: tuple[int, int],
    original_vocab_size: int,
    max_loops: int,
    boundaries: list[float],
    device: str,
    collect_records: bool,
    resume_dir: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counters: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    records: list[dict[str, Any]] = []
    elapsed = 0.0
    positions = 0
    exhausted_total = 0
    loop_values: list[float] = []
    kl_values: list[float] = []
    rank_values: list[float] = []
    run_values: list[float] = []
    answer_tie_cells = 0
    control_tie_cells = 0
    evaluated_argmax_cells = 0
    if not int(control_ids[0]) < int(control_ids[1]):
        raise AssertionError("D0 control IDs must be presented in ascending token-id order")

    for row_index, source in enumerate(rows):
        cache_path = resume_dir / f"row_{row_index:06d}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                cached.get("row_index") != row_index
                or cached.get("max_loops") != max_loops
                or cached.get("checkpoint_sha256") != checkpoint_sha256
                or cached.get("tie_policy") != "fp32_lowest_token_id"
            ):
                raise RuntimeError(f"D0 evaluation resume cache mismatch: {cache_path}")
            loop1 = cached["loop1"].to(device)
            loop4 = cached["loop4"].to(device)
            selected_answers = cached["selected_answers"].to(device)
            selected = cached["selected"].to(device)
            exhausted = cached["exhausted"].to(device)
            selected_answer_tied = cached["selected_answer_tied"].to(device)
            selected_control_tied = cached["selected_control_tied"].to(device)
            loop1_answer_tied = cached["loop1_answer_tied"].to(device)
            row_answer_tie_cells = int(cached["answer_tie_cells"])
            row_control_tie_cells = int(cached["control_tie_cells"])
            row_argmax_cells = int(cached["argmax_cells"])
            row_elapsed = float(cached["elapsed_seconds"])
        else:
            values = torch.tensor([source["input_ids"]], dtype=torch.long, device=device)
            start = synchronized_time(device)
            output = wrapper(
                input_ids=values,
                attention_mask=torch.ones_like(values),
                labels=None,
                max_loops=max_loops,
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
            stop = synchronized_time(device)
            row_elapsed = stop - start
            if output.loop_logits is None:
                raise RuntimeError("D0 evaluation requires per-loop logits")
            logits = output.loop_logits[0, 0, :, :-1]
            answers, answer_ties = deterministic_argmax_fp32(
                logits[..., :original_vocab_size], dim=-1
            )
            controls, control_ties = deterministic_argmax_fp32(
                logits[..., list(control_ids)], dim=-1
            )
            answers = answers.transpose(0, 1)
            answer_ties = answer_ties.transpose(0, 1)
            controls = controls.transpose(0, 1)
            control_ties = control_ties.transpose(0, 1)
            selected, exhausted = first_stop(controls, max_loops)
            selected_index = (selected - 1).unsqueeze(1)
            selected_answers = answers.gather(1, selected_index).squeeze(1)
            selected_answer_tied = answer_ties.gather(1, selected_index).squeeze(1)
            selected_control_tied = control_ties.gather(1, selected_index).squeeze(1)
            loop1 = answers[:, 0]
            loop4 = answers[:, -1]
            loop1_answer_tied = answer_ties[:, 0]
            row_answer_tie_cells = int(answer_ties.sum().item())
            row_control_tie_cells = int(control_ties.sum().item())
            row_argmax_cells = int(answer_ties.numel() + control_ties.numel())
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "kind": "paper2_d0_eval_row_cache",
                    "row_index": row_index,
                    "max_loops": max_loops,
                    "checkpoint_sha256": checkpoint_sha256,
                    "tie_policy": "fp32_lowest_token_id",
                    "positions": len(source["input_ids"]) - 1,
                    "loop1": loop1.cpu(),
                    "loop4": loop4.cpu(),
                    "selected_answers": selected_answers.cpu(),
                    "selected": selected.cpu(),
                    "exhausted": exhausted.cpu(),
                    "selected_answer_tied": selected_answer_tied.cpu(),
                    "selected_control_tied": selected_control_tied.cpu(),
                    "loop1_answer_tied": loop1_answer_tied.cpu(),
                    "answer_tie_cells": row_answer_tie_cells,
                    "control_tie_cells": row_control_tie_cells,
                    "argmax_cells": row_argmax_cells,
                    "elapsed_seconds": row_elapsed,
                },
                temporary,
            )
            os.replace(temporary, cache_path)
            del output, logits, answers, controls, answer_ties, control_ties
        elapsed += row_elapsed
        answer_tie_cells += row_answer_tie_cells
        control_tie_cells += row_control_tie_cells
        evaluated_argmax_cells += row_argmax_cells
        row_cache = cache_rows[row_index]
        teacher = row_cache["teacher_greedy_token_id"].to(device=device, dtype=torch.long)
        plain = row_cache["drafter_greedy_token_id"].to(device=device, dtype=torch.long)
        accepted = row_cache["accepted"].to(device=device, dtype=torch.bool)
        stratum = str(source["stratum"])
        positions += int(teacher.numel())
        exhausted_total += int(exhausted.sum().item())

        kl = row_cache["teacher_to_plain_drafter_kl"].float().tolist()
        ranks = row_cache["drafter_token_rank_under_teacher"].float().tolist()
        runs = row_cache["rejection_run_length"].float().tolist()
        selected_list = selected.float().tolist()
        accepted_list = accepted.tolist()
        teacher_list = teacher.tolist()
        plain_list = plain.tolist()
        loop1_list = loop1.tolist()
        loop4_list = loop4.tolist()
        adaptive_list = selected_answers.tolist()
        exhausted_list = exhausted.tolist()
        selected_answer_tied_list = selected_answer_tied.tolist()
        selected_control_tied_list = selected_control_tied.tolist()
        loop1_answer_tied_list = loop1_answer_tied.tolist()

        for local_position in range(len(teacher_list)):
            severity = quartile(kl[local_position], boundaries)
            categories = ("pooled", stratum)
            plain_hit = plain_list[local_position] == teacher_list[local_position]
            loop1_hit = loop1_list[local_position] == teacher_list[local_position]
            loop4_hit = loop4_list[local_position] == teacher_list[local_position]
            adaptive_hit = adaptive_list[local_position] == teacher_list[local_position]
            for category in categories:
                for name, hit in (
                    ("plain_all", plain_hit),
                    ("loop1_all", loop1_hit),
                    ("adaptive_all", adaptive_hit),
                ):
                    cell = counters[(category, "all", name)]
                    cell[0] += int(hit)
                    cell[1] += 1
                if not accepted_list[local_position]:
                    for name, hit in (
                        ("loop1_rejected", loop1_hit),
                        ("loop4_rejected", loop4_hit),
                        ("adaptive_rejected", adaptive_hit),
                    ):
                        for bin_name in ("all", severity):
                            cell = counters[(category, bin_name, name)]
                            cell[0] += int(hit)
                            cell[1] += 1
                else:
                    cell = counters[(category, "all", "loop1_accepted")]
                    cell[0] += int(loop1_hit)
                    cell[1] += 1
            loop_values.append(selected_list[local_position])
            kl_values.append(kl[local_position])
            rank_values.append(ranks[local_position])
            run_values.append(runs[local_position])
            if collect_records:
                records.append(
                    {
                        "row_index": row_index,
                        "local_position": local_position,
                        "stratum": stratum,
                        "accepted_pretraining": bool(accepted_list[local_position]),
                        "teacher_token_id": int(teacher_list[local_position]),
                        "plain_token_id": int(plain_list[local_position]),
                        "loop1_token_id": int(loop1_list[local_position]),
                        "loop4_token_id": int(loop4_list[local_position]),
                        "adaptive_token_id": int(adaptive_list[local_position]),
                        "selected_loop": int(selected_list[local_position]),
                        "exhausted": bool(exhausted_list[local_position]),
                        "loop1_answer_tied": bool(loop1_answer_tied_list[local_position]),
                        "selected_answer_tied": bool(selected_answer_tied_list[local_position]),
                        "selected_control_tied": bool(selected_control_tied_list[local_position]),
                        "kl": float(kl[local_position]),
                        "rank": float(ranks[local_position]),
                        "rejection_run_length": int(runs[local_position]),
                        "severity_bin": severity,
                    }
                )
        del selected_answers
        if (row_index + 1) % 8 == 0 or row_index + 1 == len(rows):
            print(f"d0_eval_progress rows={row_index + 1}/{len(rows)} positions={positions}", flush=True)

    by_stratum: dict[str, Any] = {}
    for stratum in ("pooled", "general", "code"):
        metrics: dict[str, Any] = {}
        for bin_name in ("all", "q1", "q2", "q3", "q4"):
            metrics[bin_name] = {}
            for name in (
                "plain_all",
                "loop1_all",
                "adaptive_all",
                "loop1_rejected",
                "loop4_rejected",
                "adaptive_rejected",
                "loop1_accepted",
            ):
                correct, total = counters[(stratum, bin_name, name)]
                if total:
                    metrics[bin_name][name] = summarize_binary(correct, total)
        rejected = metrics["all"].get("adaptive_rejected", {"accuracy": None})["accuracy"]
        loop1_rejected = metrics["all"].get("loop1_rejected", {"accuracy": None})["accuracy"]
        metrics["depth_recoverable_fraction_R"] = (
            float(rejected) - float(loop1_rejected)
            if rejected is not None and loop1_rejected is not None
            else None
        )
        metrics["unrecovered_at_depth4_by_severity"] = {
            bin_name: (
                1.0 - metrics[bin_name]["loop4_rejected"]["accuracy"]
                if "loop4_rejected" in metrics[bin_name]
                else None
            )
            for bin_name in ("all", "q1", "q2", "q3", "q4")
        }
        by_stratum[stratum] = metrics

    return (
        {
            "forced_max_loops": max_loops,
            "positions": positions,
            "elapsed_seconds": elapsed,
            "positions_per_second": positions / elapsed if elapsed else None,
            "latency_ms_per_position": 1000.0 * elapsed / positions if positions else None,
            "exhausted": exhausted_total,
            "exhaustion_rate": exhausted_total / positions if positions else None,
            "mean_selected_loops": statistics.fmean(loop_values),
            "tie_diagnostics": {
                "policy": "fp32 logits; exact ties choose the lowest token id",
                "answer_tie_cells": answer_tie_cells,
                "control_tie_cells": control_tie_cells,
                "argmax_cells": evaluated_argmax_cells,
                "tie_rate": (
                    (answer_tie_cells + control_tie_cells) / evaluated_argmax_cells
                    if evaluated_argmax_cells
                    else None
                ),
            },
            "loop_usage_correlations": {
                "spearman_kl": spearman(loop_values, kl_values),
                "spearman_teacher_rank": spearman(loop_values, rank_values),
                "spearman_rejection_run_length": spearman(loop_values, run_values),
            },
            "by_stratum": by_stratum,
        },
        records,
    )


def simulate_windows(records: list[dict[str, Any]], gamma: int, *, adaptive: bool) -> dict[str, Any]:
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_row[int(record["row_index"])].append(record)
    proposed = accepted = windows = loop_cost = 0
    for row in by_row.values():
        index = 0
        while index < len(row):
            window = row[index : index + gamma]
            windows += 1
            proposed += len(window)
            loop_cost += sum(int(item["selected_loop"]) if adaptive else 1 for item in window)
            matched_prefix = 0
            for item in window:
                prediction = item["adaptive_token_id"] if adaptive else item["plain_token_id"]
                if prediction != item["teacher_token_id"]:
                    break
                accepted += 1
                matched_prefix += 1
            index += len(window) if matched_prefix == len(window) else matched_prefix + 1
    emitted = accepted + windows
    return {
        "gamma": gamma,
        "windows": windows,
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else None,
        "loop_cost": loop_cost,
        "loops_per_accepted_draft_token": loop_cost / accepted if accepted else None,
        "implied_tokens_emitted": emitted,
        "drafter_loop_passes_per_emitted_token": loop_cost / emitted if emitted else None,
        "teacher_verifications_per_emitted_token": windows / emitted if emitted else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--evaluation_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--floor_summary", required=True)
    parser.add_argument("--initial_checkpoint", required=True)
    parser.add_argument("--trained_checkpoint", required=True)
    parser.add_argument("--trained_checkpoint_sha256", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--private_rows_output", required=True)
    parser.add_argument("--resume_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    prereg = read_json(args.preregistration)
    validate_locked_d0(prereg)
    cache_summary = read_json(args.teacher_cache_summary)
    validate_cache_summary(cache_summary)
    floor = read_json(args.floor_summary)
    if floor.get("status") != "complete":
        raise RuntimeError("D0 final evaluation requires a complete floor")
    if sha256_file(args.initial_checkpoint) != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError("D0 final evaluation initial checkpoint mismatch")
    if sha256_file(args.trained_checkpoint) != args.trained_checkpoint_sha256:
        raise RuntimeError("D0 final evaluation trained checkpoint mismatch")
    rows = read_jsonl(args.evaluation_jsonl)
    cache_rows = load_partition_cache(cache_summary, "teacher_7b", "evaluation")
    boundaries = list(floor["severity_quartile_boundaries"])

    _, initial, initial_resize, _initial_original_vocab = load_drafter(
        checkpoint=Path(args.initial_checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    initial_metrics, initial_records = evaluate_partition(
        initial,
        rows,
        cache_rows,
        control_ids=tuple(int(value) for value in initial_resize.control_token_ids[:2]),
        original_vocab_size=initial_resize.original_tokenizer_size,
        max_loops=1,
        boundaries=boundaries,
        device=args.device,
        collect_records=True,
        resume_dir=Path(args.resume_dir) / "initial_loop1",
        checkpoint_sha256=DRAFTER_CHECKPOINT_SHA256,
    )
    mismatches = [row for row in initial_records if row["loop1_token_id"] != row["plain_token_id"]]
    non_tie_mismatches = [row for row in mismatches if not row["loop1_answer_tied"]]
    if non_tie_mismatches:
        raise RuntimeError(
            "D0 initial same-reader equivalence failed outside explicit tied cells at "
            f"{len(non_tie_mismatches)} positions"
        )
    initial_equivalence = {
        "positions": len(initial_records),
        "mismatches_vs_legacy_plain_cache": len(mismatches),
        "non_tie_mismatches": len(non_tie_mismatches),
        "all_mismatches_explained_by_current_fp32_ties": not non_tie_mismatches,
    }
    del initial_records
    del initial
    gc.collect()
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    _, trained, trained_resize, _trained_original_vocab = load_drafter(
        checkpoint=Path(args.trained_checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    trained_loop1_metrics, _ = evaluate_partition(
        trained,
        rows,
        cache_rows,
        control_ids=tuple(int(value) for value in trained_resize.control_token_ids[:2]),
        original_vocab_size=trained_resize.original_tokenizer_size,
        max_loops=1,
        boundaries=boundaries,
        device=args.device,
        collect_records=False,
        resume_dir=Path(args.resume_dir) / "trained_loop1",
        checkpoint_sha256=args.trained_checkpoint_sha256,
    )
    trained_metrics, records = evaluate_partition(
        trained,
        rows,
        cache_rows,
        control_ids=tuple(int(value) for value in trained_resize.control_token_ids[:2]),
        original_vocab_size=trained_resize.original_tokenizer_size,
        max_loops=4,
        boundaries=boundaries,
        device=args.device,
        collect_records=True,
        resume_dir=Path(args.resume_dir) / "trained_adaptive",
        checkpoint_sha256=args.trained_checkpoint_sha256,
    )
    private_path = Path(args.private_rows_output)
    write_json(private_path, {"kind": "paper2_d0_private_evaluation_rows", "rows": records})
    simulations = {
        str(gamma): {
            "plain": simulate_windows(records, gamma, adaptive=False),
            "adaptive": simulate_windows(records, gamma, adaptive=True),
        }
        for gamma in (2, 4, 8)
    }
    accepted_reference = 1.0
    accepted_trained = trained_metrics["by_stratum"]["pooled"]["all"]["loop1_accepted"]["accuracy"]
    natural_drop = float(accepted_reference) - float(accepted_trained)
    incremental_loop_seconds = max(
        0.0,
        (trained_metrics["elapsed_seconds"] - trained_loop1_metrics["elapsed_seconds"]) / 3.0,
    )
    implied_adaptive_seconds = trained_loop1_metrics["elapsed_seconds"] + incremental_loop_seconds * (
        trained_metrics["mean_selected_loops"] - 1.0
    )
    summary = {
        "kind": "paper2_d0_final_evaluation",
        "status": "complete" if natural_drop <= 0.01 else "blocked_guardrail",
        "initial_checkpoint_sha256": DRAFTER_CHECKPOINT_SHA256,
        "trained_checkpoint_sha256": args.trained_checkpoint_sha256,
        "initial_loop1": initial_metrics,
        "initial_same_reader_equivalence": initial_equivalence,
        "trained_loop1": trained_loop1_metrics,
        "trained_adaptive": trained_metrics,
        "simulation_grade_wall_clock": {
            "plain_elapsed_seconds": initial_metrics["elapsed_seconds"],
            "trained_loop1_elapsed_seconds": trained_loop1_metrics["elapsed_seconds"],
            "trained_forced4_elapsed_seconds": trained_metrics["elapsed_seconds"],
            "implied_adaptive_elapsed_seconds": implied_adaptive_seconds,
            "implied_adaptive_positions_per_second": (
                trained_metrics["positions"] / implied_adaptive_seconds if implied_adaptive_seconds else None
            ),
            "method": "same-session whole-chunk timing; adaptive time interpolated from trained loop-1 and forced-loop-4 calls",
        },
        "speculative_decoding_simulation": simulations,
        "natural_surface_guardrail": {
            "reference_loop1_match": accepted_reference,
            "trained_loop1_match_on_pretraining_accepted_positions": accepted_trained,
            "absolute_drop": natural_drop,
            "maximum_drop": 0.01,
            "passed": natural_drop <= 0.01,
        },
        "private_rows_sha256": sha256_file(private_path),
        "interpretation_band": (
            "strong_recoverability"
            if trained_metrics["by_stratum"]["pooled"]["depth_recoverable_fraction_R"] > 0.10
            else "partial_recoverability"
            if trained_metrics["by_stratum"]["pooled"]["depth_recoverable_fraction_R"] >= 0.02
            else "not_recoverable_at_pilot_scale"
        ),
        "scope": "teacher-forced next-token agreement; simulation-grade latency; single seed",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
