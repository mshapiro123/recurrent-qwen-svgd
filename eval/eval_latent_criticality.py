"""Probe latent criticality signatures for recurrent loop-depth selection.

This diagnostic reads forced-depth benchmark sweeps, recomputes prompt-only
recurrent hidden states, extracts label-free depth signatures, and tests whether
simple signature selectors transfer from a discovery sweep to a held-out sweep.
It intentionally avoids option-completion hidden states so the features do not
peek at candidate answers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_depth_sweep import joined_examples, load_loop_payloads, path_for_cli, resolve_path  # noqa: E402
from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype  # noqa: E402
from eval.eval_mcq import format_prompt  # noqa: E402
from eval.eval_reentry_drift import prepare_recurrent_inputs, run_recurrent_block  # noqa: E402
from models.halting import masked_mean  # noqa: E402
from models.lora import apply_lora_to_recurrent_block  # noqa: E402
from models.recurrent_wrapper import RecurrentQwenForCausalLM  # noqa: E402
from training.checkpointing import load_trainable_checkpoint  # noqa: E402


FEATURE_NAMES = [
    "participation_ratio",
    "effective_rank",
    "dragon_king_gap",
    "dragon_king_z",
    "state_rms",
    "pooled_norm",
    "move_from_prev",
    "move_to_next",
    "decel",
    "logit_entropy",
    "logit_top_prob",
    "logit_top2_margin",
    "jacobian_gain_mean",
    "jacobian_gain_max",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor")
        value = value.detach().float().cpu().item()
    value = float(value)
    return value if math.isfinite(value) else None


def masked_rms(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    values = hidden.float().square().mean(dim=-1)
    mask = attention_mask.to(device=hidden.device, dtype=values.dtype)
    return (values * mask).sum().div(mask.sum().clamp_min(1.0)).sqrt()


def spectral_signatures(hidden: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, float | None]:
    mask = attention_mask[0].to(device=hidden.device, dtype=torch.bool)
    tokens = hidden[0, mask].float()
    if tokens.shape[0] < 2:
        return {
            "participation_ratio": None,
            "effective_rank": None,
            "dragon_king_gap": None,
            "dragon_king_z": None,
        }
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    if singular.numel() < 2 or float(singular.square().sum()) <= 0:
        return {
            "participation_ratio": None,
            "effective_rank": None,
            "dragon_king_gap": None,
            "dragon_king_z": None,
        }
    energy = singular.square()
    total = energy.sum().clamp_min(1e-12)
    p = energy / total
    tail = singular[1:]
    return {
        "participation_ratio": finite_float(total.square() / energy.square().sum().clamp_min(1e-12)),
        "effective_rank": finite_float(torch.exp(-(p * (p + 1e-12).log()).sum())),
        "dragon_king_gap": finite_float(singular[0] / singular[1].clamp_min(1e-12)),
        "dragon_king_z": finite_float((singular[0] - tail.mean()) / tail.std(unbiased=False).clamp_min(1e-12)),
    }


def pooled_state(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(hidden, attention_mask).float()


def vector_distance(a: torch.Tensor, b: torch.Tensor) -> float | None:
    denom = b.norm(dim=-1).clamp_min(1e-12)
    return finite_float(((b - a).norm(dim=-1) / denom).mean())


def logit_lens_signatures(
    wrapper: RecurrentQwenForCausalLM,
    recurrent_state: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
) -> dict[str, float | None]:
    with torch.no_grad():
        coda_hidden, _ = wrapper._run_layer_range(  # noqa: SLF001
            start=wrapper.layer_split.recurrent_end,
            end=len(wrapper.qwen.layers),
            hidden_states=recurrent_state,
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
        normed = wrapper.qwen.norm(coda_hidden)
        logits = wrapper.lm_head(normed)
        last_index = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0)
        selected = logits[torch.arange(logits.shape[0], device=logits.device), last_index].float()
        probs = torch.softmax(selected, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    return {
        "logit_entropy": finite_float(entropy.mean()),
        "logit_top_prob": finite_float(top2[:, 0].mean()),
        "logit_top2_margin": finite_float((top2[:, 0] - top2[:, 1]).mean()),
    }


def next_recurrent_map(
    wrapper: RecurrentQwenForCausalLM,
    state: torch.Tensor,
    prelude_state: torch.Tensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
) -> torch.Tensor:
    loop_input = wrapper.bridge(state, prelude_hidden=prelude_state)
    return run_recurrent_block(
        wrapper,
        loop_input,
        causal_mask,
        position_ids,
        cache_position,
        position_embeddings,
    )


def finite_difference_jacobian_signatures(
    wrapper: RecurrentQwenForCausalLM,
    state: torch.Tensor,
    prelude_state: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
    *,
    random_probes: int,
    epsilon: float,
) -> dict[str, float | None]:
    if random_probes <= 0:
        return {"jacobian_gain_mean": None, "jacobian_gain_max": None}
    gains: list[float] = []
    with torch.no_grad():
        base_out = next_recurrent_map(
            wrapper,
            state,
            prelude_state,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        state_rms = masked_rms(state, attention_mask).clamp_min(1e-8)
        for _ in range(random_probes):
            direction = torch.randn_like(state)
            direction = direction / masked_rms(direction, attention_mask).clamp_min(1e-8).to(dtype=direction.dtype)
            perturb = (float(epsilon) * state_rms).to(dtype=state.dtype) * direction
            perturbed_out = next_recurrent_map(
                wrapper,
                state + perturb,
                prelude_state,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
            )
            gain = masked_rms(perturbed_out - base_out, attention_mask) / masked_rms(perturb, attention_mask).clamp_min(
                1e-12
            )
            value = finite_float(gain)
            if value is not None:
                gains.append(value)
    if not gains:
        return {"jacobian_gain_mean": None, "jacobian_gain_max": None}
    return {
        "jacobian_gain_mean": sum(gains) / len(gains),
        "jacobian_gain_max": max(gains),
    }


def load_wrapper(args: argparse.Namespace, checkpoint: str) -> RecurrentQwenForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    adapter_dtype = resolve_dtype(args.adapter_dtype)
    replaced = apply_lora_to_recurrent_block(
        wrapper,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        adapter_dtype=adapter_dtype,
    )
    print(f"lora_recurrent_modules={replaced}", flush=True)
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    load_info = load_trainable_checkpoint(wrapper, checkpoint)
    print(f"loaded_checkpoint={checkpoint} loaded_keys={len(load_info['loaded_keys'])}", flush=True)
    if load_info["skipped"]:
        print(f"skipped_keys={len(load_info['skipped'])}", flush=True)
    wrapper.eval()
    return wrapper


def question_rows_from_prepared_file(data_path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(data_path)}


def local_prepared_file(run_id: str, benchmark: str) -> Path | None:
    data_dir = ROOT / "data" / "stage5_benchmark_suite" / run_id
    if not data_dir.exists():
        return None
    candidates = [
        path
        for path in data_dir.glob(f"{benchmark}_*.jsonl")
        if "cyclic_permuted" not in path.name and path.is_file()
    ]
    if candidates:
        return sorted(candidates)[0]
    if benchmark == "arc_challenge":
        candidates = [
            path
            for path in data_dir.glob("arc_challenge_*.jsonl")
            if "cyclic_permuted" not in path.name and path.is_file()
        ]
        return sorted(candidates)[0] if candidates else None
    return None


def fallback_arc_rows(benchmark: str, required_ids: set[str], output_dir: Path) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    if benchmark == "arc_easy":
        config, split = "ARC-Easy", "validation"
    elif benchmark == "open_hard_arc_challenge":
        config, split = "ARC-Challenge", "test"
    else:
        config, split = "ARC-Challenge", "validation"
    from eval.prepare_arc_mcq import row_to_mcq

    rows: dict[str, dict[str, Any]] = {}
    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    for idx, row in enumerate(dataset):
        prepared = row_to_mcq(dict(row), index=idx, seed=0, shuffle_choices=True)
        row_id = str(prepared["id"])
        if row_id in required_ids:
            rows[row_id] = prepared
    missing = sorted(required_ids - set(rows))
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} question rows for benchmark={benchmark}; first_missing={missing[:5]}"
        )
    cache_path = output_dir / f"{benchmark}_{config}_{split}_reconstructed.jsonl"
    write_jsonl(cache_path, [rows[key] for key in sorted(rows)])
    return rows


def resolve_question_rows(
    loop_payloads: dict[int, dict[str, Any]],
    benchmark: str,
    required_ids: set[str],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    run_id = str(loop_payloads[min(loop_payloads)].get("run_id") or "")
    if run_id:
        data_file = local_prepared_file(run_id, benchmark)
        if data_file:
            rows = question_rows_from_prepared_file(data_file)
            if required_ids.issubset(rows):
                return {row_id: rows[row_id] for row_id in required_ids}
    return fallback_arc_rows(benchmark, required_ids, output_dir)


def recurrent_depth_states(
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    prompt: str,
    *,
    max_loops: int,
    device: str,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    hidden, attention_mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
        wrapper,
        encoded["input_ids"],
        encoded["attention_mask"],
    )
    states: list[torch.Tensor] = []
    prelude_state = hidden
    recurrent_state = hidden
    with torch.no_grad():
        for loop_idx in range(max_loops):
            loop_input = (
                recurrent_state
                if loop_idx == 0
                else wrapper.bridge(recurrent_state, prelude_hidden=prelude_state)
            )
            recurrent_state = run_recurrent_block(
                wrapper,
                loop_input,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
            )
            states.append(recurrent_state)
    return {
        "states": states,
        "prelude_state": prelude_state,
        "attention_mask": attention_mask,
        "causal_mask": causal_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
    }


def feature_rows_for_examples(
    *,
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    question_rows: dict[str, dict[str, Any]],
    split_name: str,
    benchmark: str,
    loops: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_loops = max(loops)
    jacobian_budget = int(args.jacobian_examples_per_benchmark)
    for example_idx, example in enumerate(examples[: int(args.max_examples_per_benchmark or len(examples))]):
        question_row = question_rows[str(example["id"])]
        prompt = format_prompt(read_examples_from_rows([question_row])[0], "question_only")
        state_bundle = recurrent_depth_states(wrapper, tokenizer, prompt, max_loops=max_loops, device=args.device)
        states = state_bundle["states"]
        attention_mask = state_bundle["attention_mask"]
        pooled = [pooled_state(state, attention_mask) for state in states]
        moves = [
            vector_distance(pooled[idx - 1], pooled[idx])
            for idx in range(1, len(pooled))
        ]
        hit_pattern = "".join("1" if example["loop_hits"][loop] else "0" for loop in loops)
        use_jacobian = example_idx < jacobian_budget
        for depth_index, loop in enumerate(loops):
            state = states[depth_index]
            row: dict[str, Any] = {
                "split": split_name,
                "benchmark": benchmark,
                "id": str(example["id"]),
                "depth": loop,
                "hit_pattern": hit_pattern,
                "base_hit": bool(example["base_hit"]),
                "loop_hit": bool(example["loop_hits"][loop]),
                "is_correct_depth": bool(example["loop_hits"][loop]),
                "state_rms": finite_float(masked_rms(state, attention_mask)),
                "pooled_norm": finite_float(pooled[depth_index].norm(dim=-1).mean()),
                "move_from_prev": moves[depth_index - 1] if depth_index > 0 else None,
                "move_to_next": moves[depth_index] if depth_index < len(moves) else None,
                "decel": (moves[0] - moves[1]) if len(moves) >= 2 and depth_index == 1 else None,
            }
            row.update(spectral_signatures(state, attention_mask))
            row.update(
                logit_lens_signatures(
                    wrapper,
                    state,
                    attention_mask,
                    state_bundle["causal_mask"],
                    state_bundle["position_ids"],
                    state_bundle["cache_position"],
                    state_bundle["position_embeddings"],
                )
            )
            if use_jacobian:
                row.update(
                    finite_difference_jacobian_signatures(
                        wrapper,
                        state,
                        state_bundle["prelude_state"],
                        attention_mask,
                        state_bundle["causal_mask"],
                        state_bundle["position_ids"],
                        state_bundle["cache_position"],
                        state_bundle["position_embeddings"],
                        random_probes=int(args.jacobian_random_probes),
                        epsilon=float(args.jacobian_epsilon),
                    )
                )
            else:
                row.update({"jacobian_gain_mean": None, "jacobian_gain_max": None})
            rows.append(row)
    return rows


def read_examples_from_rows(rows: list[dict[str, Any]]):
    tmp = []
    from eval.eval_mcq import MCQExample, option_items, normalize_answer

    for idx, row in enumerate(rows):
        choices = option_items(row)
        tmp.append(
            MCQExample(
                id=str(row.get("id") or idx),
                question=str(row["question"]),
                choices=choices,
                answer=normalize_answer(row.get("answer"), choices),
            )
        )
    return tmp


def auc_score(rows: list[dict[str, Any]], feature: str) -> float | None:
    values = [(row.get(feature), bool(row.get("is_correct_depth"))) for row in rows]
    clean = [(float(value), label) for value, label in values if isinstance(value, (int, float))]
    positives = [value for value, label in clean if label]
    negatives = [value for value, label in clean if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def grouped_examples_from_feature_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["benchmark"]), str(row["id"])), []).append(row)
    return {
        key: sorted(value, key=lambda item: int(item["depth"]))
        for key, value in grouped.items()
    }


def fit_feature_selector(rows: list[dict[str, Any]], feature: str) -> dict[str, Any] | None:
    grouped = grouped_examples_from_feature_rows(rows)
    best: dict[str, Any] | None = None
    for direction in ("max", "min"):
        correct = 0
        total = 0
        for depth_rows in grouped.values():
            clean = [row for row in depth_rows if isinstance(row.get(feature), (int, float))]
            if len(clean) != len(depth_rows) or not clean:
                continue
            selected = max(clean, key=lambda row: float(row[feature])) if direction == "max" else min(
                clean,
                key=lambda row: float(row[feature]),
            )
            correct += int(bool(selected["loop_hit"]))
            total += 1
        if total:
            candidate = {
                "feature": feature,
                "direction": direction,
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
            if best is None or candidate["accuracy"] > best["accuracy"]:
                best = candidate
    return best


def eval_feature_selector(rows: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any]:
    grouped = grouped_examples_from_feature_rows(rows)
    correct = 0
    loop1_correct = 0
    oracle_correct = 0
    total = 0
    feature = selector["feature"]
    for depth_rows in grouped.values():
        clean = [row for row in depth_rows if isinstance(row.get(feature), (int, float))]
        if len(clean) != len(depth_rows) or not clean:
            continue
        selected = max(clean, key=lambda row: float(row[feature])) if selector["direction"] == "max" else min(
            clean,
            key=lambda row: float(row[feature]),
        )
        correct += int(bool(selected["loop_hit"]))
        loop1_correct += int(bool(depth_rows[0]["loop_hit"]))
        oracle_correct += int(any(bool(row["loop_hit"]) for row in depth_rows))
        total += 1
    oracle_gap = oracle_correct - loop1_correct
    return {
        "feature": feature,
        "direction": selector["direction"],
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "loop1_correct": loop1_correct,
        "oracle_correct": oracle_correct,
        "delta_vs_loop1": correct - loop1_correct,
        "oracle_gap_capture": (correct - loop1_correct) / oracle_gap if oracle_gap > 0 else None,
    }


def harmed_rescued_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = grouped_examples_from_feature_rows(rows)
    buckets = {"harmed": [], "rescued": []}
    for depth_rows in grouped.values():
        hits = [bool(row["loop_hit"]) for row in depth_rows]
        if hits[0] and any(not item for item in hits[1:]):
            buckets["harmed"].append(depth_rows)
        if (not hits[0]) and any(hits[1:]):
            buckets["rescued"].append(depth_rows)
    result: dict[str, Any] = {
        "harmed_examples": len(buckets["harmed"]),
        "rescued_examples": len(buckets["rescued"]),
        "features": {},
    }
    for feature in FEATURE_NAMES:
        feature_result: dict[str, Any] = {}
        for bucket_name, examples in buckets.items():
            deltas = []
            for depth_rows in examples:
                if len(depth_rows) < 2:
                    continue
                a = depth_rows[0].get(feature)
                b = depth_rows[1].get(feature)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    deltas.append(float(b) - float(a))
            feature_result[f"{bucket_name}_depth2_minus_depth1_mean"] = (
                sum(deltas) / len(deltas) if deltas else None
            )
        h = feature_result.get("harmed_depth2_minus_depth1_mean")
        r = feature_result.get("rescued_depth2_minus_depth1_mean")
        feature_result["opposite_sign_depth2_minus_depth1"] = (
            bool(h is not None and r is not None and h * r < 0)
        )
        result["features"][feature] = feature_result
    return result


def summarize_features(discovery_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selectors = [
        selector
        for feature in FEATURE_NAMES
        if (selector := fit_feature_selector(discovery_rows, feature)) is not None
    ]
    transfer = [eval_feature_selector(heldout_rows, selector) for selector in selectors]
    transfer = sorted(
        transfer,
        key=lambda row: (
            row["delta_vs_loop1"],
            -999 if row["oracle_gap_capture"] is None else row["oracle_gap_capture"],
            row["correct"],
        ),
        reverse=True,
    )
    return {
        "feature_auc": {
            "discovery": {feature: auc_score(discovery_rows, feature) for feature in FEATURE_NAMES},
            "heldout": {feature: auc_score(heldout_rows, feature) for feature in FEATURE_NAMES},
        },
        "selector_transfer": transfer,
        "harmed_rescued": {
            "discovery": harmed_rescued_summary(discovery_rows),
            "heldout": harmed_rescued_summary(heldout_rows),
        },
    }


def rows_for_sweep(
    *,
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    sweep_summary: Path,
    split_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    benchmarks = list(loop_payloads[loops[0]].get("benchmarks", []))
    requested = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    if requested:
        benchmarks = [benchmark for benchmark in benchmarks if benchmark in requested]
    all_rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        examples = joined_examples(loop_payloads, benchmark, args.score_target, args.aggregate)
        if not examples:
            continue
        required_ids = {str(example["id"]) for example in examples[: int(args.max_examples_per_benchmark or len(examples))]}
        question_rows = resolve_question_rows(loop_payloads, benchmark, required_ids, output_dir / "question_cache")
        print(
            f"latent_criticality split={split_name} benchmark={benchmark} examples={len(required_ids)}",
            flush=True,
        )
        all_rows.extend(
            feature_rows_for_examples(
                wrapper=wrapper,
                tokenizer=tokenizer,
                examples=examples,
                question_rows=question_rows,
                split_name=split_name,
                benchmark=benchmark,
                loops=loops,
                args=args,
            )
        )
    return all_rows


def checkpoint_from_sweep(sweep_summary: Path) -> str:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    for payload in loop_payloads.values():
        checkpoint = payload.get("checkpoint")
        if checkpoint:
            return str(checkpoint)
    raise ValueError(f"No checkpoint found under sweep {sweep_summary}")


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Latent Criticality Probe - {summary['run_id']}",
        "",
        f"- Discovery sweep: `{summary['discovery_sweep_summary']}`",
        f"- Held-out sweep: `{summary['heldout_sweep_summary']}`",
        f"- Max examples per benchmark: `{summary['max_examples_per_benchmark']}`",
        f"- Jacobian examples per benchmark: `{summary['jacobian_examples_per_benchmark']}`",
        f"- Jacobian method: `finite_difference_random_gain`",
        "",
        "## Top Transfer Selectors",
        "",
    ]
    for row in summary["selector_transfer"][:10]:
        lines.append(
            "- "
            f"`{row['feature']}`/{row['direction']}: selected `{row['correct']}/{row['total']}`, "
            f"loop1 `{row['loop1_correct']}/{row['total']}`, oracle `{row['oracle_correct']}/{row['total']}`, "
            f"delta `{row['delta_vs_loop1']}`, capture `{row['oracle_gap_capture']}`"
        )
    lines.extend(["", "## Feature AUC", ""])
    for split_name, values in summary["feature_auc"].items():
        lines.append(f"### {split_name}")
        for feature, value in values.items():
            lines.append(f"- `{feature}`: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery_sweep_summary", required=True)
    parser.add_argument("--heldout_sweep_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--score_target", default="content_question_only")
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--max_examples_per_benchmark", type=int, default=64)
    parser.add_argument("--jacobian_examples_per_benchmark", type=int, default=8)
    parser.add_argument("--jacobian_random_probes", type=int, default=1)
    parser.add_argument("--jacobian_epsilon", type=float, default=0.02)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    discovery = resolve_path(args.discovery_sweep_summary)
    heldout = resolve_path(args.heldout_sweep_summary)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or checkpoint_from_sweep(heldout)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_wrapper(args, checkpoint)
    discovery_rows = rows_for_sweep(
        wrapper=wrapper,
        tokenizer=tokenizer,
        sweep_summary=discovery,
        split_name="discovery",
        output_dir=output_dir,
        args=args,
    )
    heldout_rows = rows_for_sweep(
        wrapper=wrapper,
        tokenizer=tokenizer,
        sweep_summary=heldout,
        split_name="heldout",
        output_dir=output_dir,
        args=args,
    )
    write_jsonl(output_dir / "features_discovery.jsonl", discovery_rows)
    write_jsonl(output_dir / "features_heldout.jsonl", heldout_rows)
    analysis = summarize_features(discovery_rows, heldout_rows)
    summary = {
        "kind": "stage5_latent_criticality_probe",
        "run_id": output_dir.name,
        "discovery_sweep_summary": path_for_cli(discovery),
        "heldout_sweep_summary": path_for_cli(heldout),
        "checkpoint": checkpoint,
        "score_target": args.score_target,
        "aggregate": args.aggregate,
        "max_examples_per_benchmark": args.max_examples_per_benchmark,
        "jacobian_examples_per_benchmark": args.jacobian_examples_per_benchmark,
        "jacobian_random_probes": args.jacobian_random_probes,
        **analysis,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
