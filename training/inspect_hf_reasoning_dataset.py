"""Inspect Hugging Face reasoning traces before adding them to training mixes.

The audit is intentionally lightweight: it samples rows, checks which converter
adapters can read them, estimates token/trace length, and assigns a conservative
training role. Use it before spending Colab GPU time on a new trace corpus.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.prepare_hf_reasoning_jsonl import ADAPTERS, row_to_example, row_to_prompt_completion  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def length_stats(values: Iterable[int]) -> dict[str, float | int | None]:
    items = sorted(int(value) for value in values)
    if not items:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    p90_idx = min(len(items) - 1, int(round(0.9 * (len(items) - 1))))
    return {
        "count": len(items),
        "min": items[0],
        "p50": items[len(items) // 2],
        "p90": items[p90_idx],
        "max": items[-1],
        "mean": float(statistics.fmean(items)),
    }


def count_non_null(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if row.get(field) not in (None, "", [], {}))


def scalar_counts(rows: list[dict[str, Any]], field: str, limit: int = 20) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (str, int, float, bool)):
            counts[str(value)] += 1
    return dict(counts.most_common(limit))


def adapter_success_counts(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for adapter in ADAPTERS:
        counts[adapter] = sum(1 for row in rows if row_to_prompt_completion(row, tokenizer, adapter=adapter) is not None)
    return counts


def fit_rates(values: list[int], budgets: tuple[int, ...] = (512, 1024, 2048)) -> dict[str, float]:
    if not values:
        return {str(budget): 0.0 for budget in budgets}
    return {
        str(budget): sum(1 for value in values if value <= budget) / len(values)
        for budget in budgets
    }


def curriculum_signal(
    *,
    dataset_id: str,
    converted: list[dict[str, Any]],
    prompt_tokens: list[int],
    completion_tokens: list[int],
    total_tokens: list[int],
    role: dict[str, Any],
) -> dict[str, Any]:
    cot_tokens = [int(example.get("cot_tokens") or 0) for example in converted]
    direct_rows = sum(1 for cot, total in zip(cot_tokens, total_tokens) if cot <= 96 and total <= 512)
    deep_rows = sum(1 for cot, total in zip(cot_tokens, total_tokens) if cot >= 128 and total <= 2048)
    long_rows = sum(1 for total in total_tokens if total > 2048)
    wide_or_agentic = role.get("primary_role") == "agent_tool_trace_or_coding_diversity"
    converted_count = len(converted)
    direct_fraction = direct_rows / max(converted_count, 1)
    deep_fraction = deep_rows / max(converted_count, 1)
    long_fraction = long_rows / max(converted_count, 1)

    if wide_or_agentic:
        recommendation = "hold_for_wide_or_agentic_filter"
        reason = "Agent/tool traces should be filtered into wide or tool-trajectory experiments before SFT."
        target_mix = {"direct": 0, "deep_narrow": 0, "wide": converted_count}
    elif converted_count == 0:
        recommendation = "do_not_train"
        reason = "No rows converted with the selected adapter."
        target_mix = {"direct": 0, "deep_narrow": 0, "wide": 0}
    elif direct_fraction >= 0.6 and deep_fraction < 0.25:
        recommendation = "direct_recovery_candidate"
        reason = "Most converted traces are short enough for direct competence recovery."
        target_mix = {"direct": direct_rows, "deep_narrow": deep_rows, "wide": 0}
    elif deep_fraction >= 0.25:
        recommendation = "deep_narrow_candidate"
        reason = "A meaningful share of converted traces has longer reasoning suitable for depth supervision."
        target_mix = {"direct": direct_rows, "deep_narrow": deep_rows, "wide": 0}
    else:
        recommendation = "audit_filter_before_training"
        reason = "Rows convert, but the length mix is not clearly direct, deep, or wide."
        target_mix = {"direct": direct_rows, "deep_narrow": deep_rows, "wide": 0}

    if long_fraction > 0.25 and recommendation in {"direct_recovery_candidate", "deep_narrow_candidate"}:
        recommendation = f"{recommendation}_with_length_filter"
        reason += " Apply a max-total-token filter before GPU training."

    return {
        "dataset_id": dataset_id,
        "recommendation": recommendation,
        "reason": reason,
        "fit_rates_total_tokens": fit_rates(total_tokens),
        "fit_rates_completion_tokens": fit_rates(completion_tokens),
        "estimated_target_mix": target_mix,
        "direct_candidate_rows": direct_rows,
        "deep_narrow_candidate_rows": deep_rows,
        "long_context_rows": long_rows,
        "direct_candidate_fraction": direct_fraction,
        "deep_narrow_candidate_fraction": deep_fraction,
        "long_context_fraction": long_fraction,
    }


def infer_training_role(
    dataset_id: str,
    rows: list[dict[str, Any]],
    adapter_counts: dict[str, int],
) -> dict[str, Any]:
    lowered = dataset_id.lower()
    tool_rows = count_non_null(rows, "tools") + sum(1 for row in rows if int(row.get("num_tool_calls") or 0) > 0)
    has_fable = "fable" in lowered or tool_rows > 0 or count_non_null(rows, "trace") > 0
    has_opus = "opus" in lowered or count_non_null(rows, "thinking") > 0
    has_qwen_text = adapter_counts.get("qwen_text", 0) > 0

    if has_fable:
        return {
            "primary_role": "agent_tool_trace_or_coding_diversity",
            "priority": "later",
            "use_for": ["coding/tool trajectory experiments", "particle diversity diagnostics"],
            "avoid_for_now": ["ARC/GPQA competence recovery", "blind mixed SFT"],
            "reason": "Agent/tool traces need domain and tool-call filtering before ordinary causal SFT.",
        }
    if has_opus or has_qwen_text:
        return {
            "primary_role": "reasoning_trace_sft",
            "priority": "immediate_candidate",
            "use_for": ["modified-Opus recurrent fine-tuning", "hard-reasoning trace curriculum"],
            "avoid_for_now": ["unfiltered long-context runs above current max_length"],
            "reason": "Rows look like direct reasoning traces that can be converted to prompt/completion SFT.",
        }
    return {
        "primary_role": "unknown",
        "priority": "audit_first",
        "use_for": [],
        "avoid_for_now": ["training"],
        "reason": "No known adapter matched enough rows to assign a training role.",
    }


def audit_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    dataset_id: str,
    adapter: str = "auto",
) -> dict[str, Any]:
    columns = sorted({key for row in rows for key in row})
    adapter_counts = adapter_success_counts(rows, tokenizer)
    converted = [
        example
        for row in rows
        if (example := row_to_example(row, tokenizer, adapter=adapter, source_dataset_name=dataset_id)) is not None
    ]

    prompt_tokens = [len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]) for example in converted]
    completion_tokens = [
        len(tokenizer(example["completion"], add_special_tokens=False)["input_ids"]) for example in converted
    ]
    total_tokens = [prompt + completion for prompt, completion in zip(prompt_tokens, completion_tokens)]

    role = infer_training_role(dataset_id, rows, adapter_counts)
    return {
        "dataset_id": dataset_id,
        "sample_rows": len(rows),
        "columns": columns,
        "adapter": adapter,
        "adapter_success_counts": adapter_counts,
        "converted_rows": len(converted),
        "conversion_rate": len(converted) / max(len(rows), 1),
        "field_presence": {
            field: count_non_null(rows, field)
            for field in [
                "text",
                "messages",
                "thinking",
                "response",
                "context",
                "cot",
                "output",
                "completion",
                "tools",
                "trace",
                "num_tool_calls",
            ]
        },
        "value_counts": {
            "source_dataset": scalar_counts(rows, "source_dataset"),
            "first_source_dataset": scalar_counts(rows, "first_source_dataset"),
            "category": scalar_counts(rows, "category"),
            "domain": scalar_counts(rows, "domain"),
            "difficulty": scalar_counts(rows, "difficulty"),
            "output_type": scalar_counts(rows, "output_type"),
            "model": scalar_counts(rows, "model"),
            "harness": scalar_counts(rows, "harness"),
        },
        "token_stats": {
            "prompt_tokens": length_stats(prompt_tokens),
            "completion_tokens": length_stats(completion_tokens),
            "total_tokens": length_stats(total_tokens),
            "cot_tokens": length_stats(int(example["cot_tokens"]) for example in converted),
        },
        "training_role": role,
        "curriculum_signal": curriculum_signal(
            dataset_id=dataset_id,
            converted=converted,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            role=role,
        ),
    }


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_jsonl:
        rows = read_jsonl(args.input_jsonl)
        return rows[: args.limit] if args.limit else rows
    if args.hf_file:
        path = hf_hub_download(repo_id=args.dataset_id, repo_type="dataset", filename=args.hf_file)
        rows = read_jsonl(path)
        return rows[: args.limit] if args.limit else rows

    kwargs: dict[str, Any] = {"split": args.split}
    if args.name:
        kwargs["name"] = args.name
    if args.streaming:
        dataset = load_dataset(args.dataset_id, streaming=True, **kwargs)
        return list(islice(dataset, args.limit))
    dataset = load_dataset(args.dataset_id, **kwargs)
    rows = list(dataset)
    return rows[: args.limit] if args.limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", default="local-jsonl")
    parser.add_argument("--name", help="Optional Hugging Face dataset config/subset name.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input_jsonl", help="Audit a local JSONL file instead of Hugging Face.")
    parser.add_argument("--hf_file", help="Download and audit a specific JSONL file from the Hugging Face dataset repo.")
    parser.add_argument("--adapter", choices=ADAPTERS, default="auto")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    rows = load_rows(args)
    report = audit_rows(rows, tokenizer, dataset_id=args.dataset_id, adapter=args.adapter)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
