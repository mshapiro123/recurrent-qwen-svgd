"""Evaluate exact preimage validity and answer-head sampling coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper
from eval.eval_synthetic_depth_active_labels import (
    candidates_for_row,
    prompt_for_row,
    read_jsonl,
    score_candidates_all_loops,
)
from eval.phase_g_coverage import (
    categorical_entropy,
    exact_coverage,
    exact_valid_preimages,
    temperature_for_target_entropy,
)


def parse_sample_counts(text: str) -> list[int]:
    counts = sorted({int(item) for item in text.split(",") if item.strip()})
    if not counts or counts[0] < 1:
        raise ValueError("sample_counts must contain positive integers")
    return counts


def uniform_expected_coverage(*, n_symbols: int, samples: int) -> float:
    """Expected fraction of an exact preimage set found by uniform name sampling.

    Each valid name has the same probability of appearing at least once, so the
    expected fraction is independent of the number of valid preimages.
    """

    if int(n_symbols) < 1:
        raise ValueError("n_symbols must be positive")
    if int(samples) < 1:
        raise ValueError("samples must be positive")
    if int(samples) == 1:
        return 1.0 / int(n_symbols)
    return 1.0 - (1.0 - 1.0 / int(n_symbols)) ** int(samples)


def sample_names(
    scores: dict[str, float],
    *,
    count: int,
    temperature: float,
    generator: torch.Generator,
) -> list[str]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    names = list(scores)
    logits = torch.tensor([float(scores[name]) for name in names], dtype=torch.float64)
    probabilities = torch.softmax(logits / float(temperature), dim=0)
    indices = torch.multinomial(probabilities, int(count), replacement=True, generator=generator)
    return [names[int(index)] for index in indices.tolist()]


def score_sample_prefix(samples: list[str], valid_starts: set[str]) -> dict[str, Any]:
    valid_samples = [sample for sample in samples if sample in valid_starts]
    unique_valid = sorted(set(valid_samples))
    return {
        "samples": samples,
        "valid_samples": len(valid_samples),
        "invalid_samples": len(samples) - len(valid_samples),
        "unique_samples": len(set(samples)),
        "unique_valid": unique_valid,
        "unique_valid_count": len(unique_valid),
        "coverage": len(unique_valid) / len(valid_starts),
        "full_coverage": set(unique_valid) == valid_starts,
    }


def reverse_chain_validity(row: dict[str, Any], predictions: list[str]) -> dict[str, Any]:
    depth = int(row["depth"])
    mapping = {str(source): str(destination) for source, destination in row["mapping"].items()}
    current = str(row["observed_target"])
    edge_receipts = []
    for loop_index, predecessor in enumerate(predictions, start=1):
        predecessor = str(predecessor)
        valid = mapping.get(predecessor) == current
        edge_receipts.append(
            {
                "loop": loop_index,
                "predecessor": predecessor,
                "destination": current,
                "valid": valid,
            }
        )
        current = predecessor
    exact_starts = set(exact_valid_preimages(row))
    final_valid = len(predictions) == depth and current in exact_starts
    return {
        "predictions": [str(value) for value in predictions],
        "edges": edge_receipts,
        "valid_edges": sum(int(item["valid"]) for item in edge_receipts),
        "total_edges": depth,
        "final_valid": final_valid,
        "chain_valid": len(edge_receipts) == depth
        and all(bool(item["valid"]) for item in edge_receipts)
        and final_valid,
    }


def summarize_rows(rows: list[dict[str, Any]], sample_counts: list[int]) -> dict[str, Any]:
    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        result: dict[str, Any] = {
            "rows": total,
            "greedy_valid": sum(int(row["greedy_valid"]) for row in subset),
        }
        result["greedy_valid_rate"] = result["greedy_valid"] / total if total else 0.0
        if subset and all("greedy_chain_valid" in row for row in subset):
            result["greedy_chain_valid"] = sum(int(row["greedy_chain_valid"]) for row in subset)
            result["greedy_chain_valid_rate"] = result["greedy_chain_valid"] / total
            valid_edges = sum(int(row["greedy_chain_valid_edges"]) for row in subset)
            total_edges = sum(int(row["greedy_chain_total_edges"]) for row in subset)
            result["greedy_chain_edge_valid_rate"] = valid_edges / total_edges if total_edges else 0.0
        result["sampling"] = {}
        temperatures = [float(row.get("sampling_temperature", 0.7)) for row in subset]
        entropy_errors = [float(row.get("entropy_match_absolute_error", 0.0)) for row in subset]
        result["mean_sampling_temperature"] = sum(temperatures) / total if total else None
        result["mean_entropy_match_absolute_error"] = sum(entropy_errors) / total if total else None
        result["entropy_match_clamp_rate"] = (
            sum(int(row.get("entropy_match_clamped", False)) for row in subset) / total if total else None
        )
        for count in sample_counts:
            key = str(count)
            values = [row["sampling"][key] for row in subset]
            samples_total = total * count
            valid_samples = sum(int(value["valid_samples"]) for value in values)
            mean_coverage = sum(float(value["coverage"]) for value in values) / total if total else 0.0
            has_uniform_baseline = bool(subset) and all("n_symbols" in row for row in subset)
            uniform_coverages = (
                [
                    uniform_expected_coverage(n_symbols=int(row["n_symbols"]), samples=count)
                    for row in subset
                ]
                if has_uniform_baseline
                else []
            )
            uniform_valid_rates = (
                [
                    int(row["coverage_denominator"]) / int(row["n_symbols"])
                    for row in subset
                ]
                if has_uniform_baseline
                else []
            )
            uniform_coverage = sum(uniform_coverages) / total if uniform_coverages else None
            uniform_valid_rate = sum(uniform_valid_rates) / total if uniform_valid_rates else None
            result["sampling"][key] = {
                "K": count,
                "valid_sample_rate": valid_samples / samples_total if samples_total else 0.0,
                "mean_unique_valid": (
                    sum(int(value["unique_valid_count"]) for value in values) / total if total else 0.0
                ),
                "mean_coverage": mean_coverage,
                "uniform_expected_coverage": uniform_coverage,
                "coverage_minus_uniform": (
                    mean_coverage - uniform_coverage if uniform_coverage is not None else None
                ),
                "uniform_expected_valid_sample_rate": uniform_valid_rate,
                "full_coverage_rate": (
                    sum(int(value["full_coverage"]) for value in values) / total if total else 0.0
                ),
                "duplicate_rate": (
                    1.0
                    - sum(int(value["unique_samples"]) for value in values) / samples_total
                    if samples_total
                    else 0.0
                ),
            }
        return result

    by_depth = {
        str(depth): aggregate([row for row in rows if int(row["depth"]) == depth])
        for depth in sorted({int(row["depth"]) for row in rows})
    }
    by_solution_count = {
        str(count): aggregate([row for row in rows if int(row["coverage_denominator"]) == count])
        for count in sorted({int(row["coverage_denominator"]) for row in rows})
    }
    by_preimage_stratum = {
        str(stratum): aggregate([row for row in rows if str(row.get("preimage_stratum")) == stratum])
        for stratum in sorted({str(row.get("preimage_stratum")) for row in rows})
    }
    return {
        "kind": "abductive_exact_coverage",
        "rows": len(rows),
        "overall": aggregate(rows),
        "by_depth": by_depth,
        "by_solution_count": by_solution_count,
        "by_preimage_stratum": by_preimage_stratum,
    }


def read_target_entropies(path: str | None, *, field: str) -> dict[str, float]:
    if not path:
        return {}
    rows = read_jsonl(path)
    values: dict[str, float] = {}
    for row in rows:
        row_id = str(row.get("id") or row.get("instance_id"))
        if field not in row:
            raise KeyError(f"Entropy row {row_id} is missing field {field!r}")
        values[row_id] = float(row[field])
    return values


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.data_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    sample_counts = parse_sample_counts(args.sample_counts)
    max_samples = max(sample_counts)
    target_entropies = read_target_entropies(
        getattr(args, "target_entropy_jsonl", None),
        field=getattr(args, "target_entropy_field", "latent_candidate_entropy"),
    )
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if args.progress_every and (row_index == 0 or (row_index + 1) % args.progress_every == 0):
            print(f"abductive_eval_progress row={row_index + 1}/{len(rows)}", flush=True)
        prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
        candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="name:")
        depth = int(row["depth"])
        loop_counts = list(range(1, depth + 1))
        scores_by_loop = score_candidates_all_loops(
            wrapper,
            tokenizer,
            prompt,
            candidates,
            args,
            loop_counts=loop_counts,
        )
        scores = scores_by_loop[depth]
        greedy_chain = [
            max(scores_by_loop[loop_count].items(), key=lambda item: item[1])[0]
            for loop_count in loop_counts
        ]
        chain_receipt = reverse_chain_validity(row, greedy_chain)
        greedy = max(scores.items(), key=lambda item: item[1])[0]
        row_id = str(row.get("id") or row.get("instance_id"))
        exact_starts = exact_valid_preimages(row)
        if target_entropies:
            if row_id not in target_entropies:
                raise KeyError(f"No target entropy for row {row_id}")
            entropy_match = temperature_for_target_entropy(
                scores,
                target_entropies[row_id],
                minimum=args.temperature_min,
                maximum=args.temperature_max,
                tolerance=args.entropy_tolerance,
            )
            sampling_temperature = float(entropy_match["temperature"])
        else:
            sampling_temperature = float(args.temperature)
            achieved_entropy = categorical_entropy(scores, sampling_temperature)
            entropy_match = {
                "achieved_entropy": achieved_entropy,
                "target_entropy": achieved_entropy,
                "absolute_error": 0.0,
                "clamped": False,
            }
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + row_index)
        samples = sample_names(
            scores,
            count=max_samples,
            temperature=sampling_temperature,
            generator=generator,
        )
        output_rows.append(
            {
                "id": row_id,
                "depth": depth,
                "task_mode": row.get("task_mode"),
                "preimage_stratum": row.get("preimage_stratum"),
                "valid_starts": exact_starts,
                "coverage_denominator": len(exact_starts),
                "greedy_prediction": greedy,
                "greedy_valid": greedy in set(exact_starts),
                "greedy_chain_predictions": greedy_chain,
                "greedy_chain_valid": chain_receipt["chain_valid"],
                "greedy_chain_valid_edges": chain_receipt["valid_edges"],
                "greedy_chain_total_edges": chain_receipt["total_edges"],
                "greedy_chain_edge_receipts": chain_receipt["edges"],
                "scores": scores if args.include_scores else None,
                "sampling_temperature": sampling_temperature,
                "answer_head_entropy": float(entropy_match["achieved_entropy"]),
                "target_entropy": float(entropy_match["target_entropy"]),
                "entropy_match_absolute_error": float(entropy_match["absolute_error"]),
                "entropy_match_clamped": bool(entropy_match["clamped"]),
                "sampling": {
                    str(count): exact_coverage(samples[:count], row)
                    for count in sample_counts
                },
            }
        )
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--sample_counts", default="1,2,4,8,20")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--target_entropy_jsonl")
    parser.add_argument("--target_entropy_field", default="latent_candidate_entropy")
    parser.add_argument("--temperature_min", type=float, default=1e-3)
    parser.add_argument("--temperature_max", type=float, default=100.0)
    parser.add_argument("--entropy_tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2_718_281)
    parser.add_argument("--include_scores", action="store_true")
    parser.add_argument("--normalize_candidate_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_slow_candidate_score", action="store_true")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    output_rows = evaluate(args)
    sample_counts = parse_sample_counts(args.sample_counts)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8"
    )
    summary = summarize_rows(output_rows, sample_counts)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "temperature": args.temperature,
            "sample_counts": sample_counts,
            "seed": args.seed,
            "sampling_mode": (
                "answer_head_entropy_matched"
                if args.target_entropy_jsonl
                else "answer_head_fixed_temperature_provisional"
            ),
            "target_entropy_jsonl": args.target_entropy_jsonl,
        }
    )
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
