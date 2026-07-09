"""Evaluate active intermediate labels for the synthetic depth task.

The final-answer matrix is intentionally wrong off the diagonal for chain
supervision: loop k <= depth is trained to emit f^k(x), not f^depth(x).  This
script scores the active cells against their per-loop targets and reports
above-diagonal behavior separately.
"""

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

from eval.eval_mcq import load_recurrent_wrapper, sequence_logprobs, select_forced_loop_logits

LETTER_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NAME_SYMBOLS = (
    "Ben",
    "Sam",
    "Tom",
    "Max",
    "Ada",
    "Lee",
    "Ana",
    "Joe",
    "Amy",
    "Dan",
    "Ray",
    "Ted",
    "Una",
    "Val",
    "Bob",
    "Ann",
    "Tim",
    "Jan",
    "Kim",
    "Jon",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def row_symbol_names(row: dict[str, Any] | None = None) -> tuple[str, ...]:
    if row is not None and isinstance(row.get("symbol_names"), list):
        names = tuple(str(item) for item in row["symbol_names"])
        if names:
            return names
    return NAME_SYMBOLS


def symbol(value: int | str, *, prefix: str = "", row_symbols: tuple[str, ...] | None = None) -> str:
    if prefix == "letter:":
        text = str(value)
        if text in LETTER_SYMBOLS:
            return text
        idx = int(text)
        if idx < 0 or idx >= len(LETTER_SYMBOLS):
            raise ValueError(f"letter: value_prefix supports values 0-{len(LETTER_SYMBOLS) - 1}; got {value}")
        return LETTER_SYMBOLS[idx]
    if prefix == "name:":
        names = row_symbols or NAME_SYMBOLS
        text = str(value)
        if text in names:
            return text
        try:
            idx = int(text)
        except ValueError:
            return text
        if idx < 0 or idx >= len(names):
            raise ValueError(f"name: value_prefix supports values 0-{len(names) - 1}; got {value}")
        return names[idx]
    text = str(value)
    if prefix and not text.startswith(prefix):
        return f"{prefix}{text}"
    return text


def parse_int_symbol(value: Any, *, prefix: str = "", row_symbols: tuple[str, ...] | None = None) -> int:
    text = str(value)
    if prefix == "letter:":
        if text in LETTER_SYMBOLS:
            return LETTER_SYMBOLS.index(text)
        return int(text)
    if prefix == "name:":
        names = row_symbols or NAME_SYMBOLS
        if text in names:
            return names.index(text)
        return int(text)
    if prefix and text.startswith(prefix):
        text = text[len(prefix) :]
    return int(text)


def row_mapping(row: dict[str, Any], *, value_prefix: str = "") -> dict[int, int] | None:
    raw = row.get("mapping")
    if not isinstance(raw, dict):
        return None
    mapping: dict[int, int] = {}
    symbols = row_symbol_names(row)
    for key, value in raw.items():
        mapping[parse_int_symbol(key, prefix=value_prefix, row_symbols=symbols)] = parse_int_symbol(
            value,
            prefix=value_prefix,
            row_symbols=symbols,
        )
    return mapping


def apply_mapping(mapping: dict[int, int], start: int, loop: int) -> int:
    current = int(start)
    for _ in range(int(loop)):
        current = int(mapping[current])
    return current


def prompt_for_row(row: dict[str, Any], *, prediction_space: str, prompt_style: str) -> str:
    question_source = row.get("question", row.get("prompt"))
    if question_source is None:
        raise KeyError("Expected row to contain either 'question' or 'prompt'")
    question = str(question_source).rstrip()
    if prompt_style == "with_options" and prediction_space == "choice_labels":
        choices = row.get("choices") or {}
        rendered = "\n".join(f"{label}. {text}" for label, text in choices.items())
        return f"{question}\n{rendered}\nAnswer:"
    return f"{question}\nAnswer:"


def candidates_for_row(row: dict[str, Any], *, prediction_space: str, value_prefix: str) -> dict[str, str]:
    if prediction_space == "choice_labels":
        return {str(label): f" {label}" for label in (row.get("choices") or {}).keys()}
    if prediction_space == "full_symbols":
        n_symbols = int(row["n_symbols"])
        symbols = row_symbol_names(row)
        return {
            symbol(idx, prefix=value_prefix, row_symbols=symbols): f" {symbol(idx, prefix=value_prefix, row_symbols=symbols)}"
            for idx in range(n_symbols)
        }
    raise ValueError("prediction_space must be one of: choice_labels, full_symbols")


def single_token_candidate_ids(tokenizer: Any, prompt: str, candidates: dict[str, str]) -> dict[str, int] | None:
    """Return candidate token ids when every completion is exactly one prompt suffix token.

    The check tokenizes ``prompt + completion`` and compares against the prompt
    prefix, matching the slower sequence-scoring path.  If a tokenizer merges
    across the prompt/completion boundary, this returns ``None`` and callers
    should fall back to exact sequence scoring.
    """

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    token_ids: dict[str, int] = {}
    for name, completion in candidates.items():
        full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
        suffix = full_ids[len(prompt_ids) :]
        if len(suffix) != 1:
            return None
        token_ids[name] = int(suffix[0])
    return token_ids


def active_target_for_loop(
    row: dict[str, Any],
    loop: int,
    *,
    prediction_space: str,
    value_prefix: str,
) -> str | None:
    depth = int(row["depth"])
    if loop > depth:
        return None
    if prediction_space == "choice_labels":
        labels = row.get("chain_answer_by_loop") or {}
        return None if str(loop) not in labels else str(labels[str(loop)]).strip()
    orbit = list(row.get("orbit") or [])
    if len(orbit) <= loop:
        return None
    return symbol(orbit[loop], prefix=value_prefix, row_symbols=row_symbol_names(row))


def continued_symbol_for_loop(row: dict[str, Any], loop: int, *, value_prefix: str) -> str | None:
    mapping = row_mapping(row, value_prefix=value_prefix)
    if mapping is None:
        return None
    symbols = row_symbol_names(row)
    start = parse_int_symbol(row["start"], prefix=value_prefix, row_symbols=symbols)
    return symbol(apply_mapping(mapping, start, loop), prefix=value_prefix, row_symbols=symbols)


def score_candidates_all_loops(
    wrapper: Any,
    tokenizer: Any,
    prompt: str,
    candidates: dict[str, str],
    args: argparse.Namespace,
    *,
    loop_counts: list[int],
) -> dict[int, dict[str, float]]:
    scores_by_loop: dict[int, dict[str, float]] = {loop: {} for loop in loop_counts}
    max_loops = max(loop_counts)
    fast_token_ids = None if getattr(args, "force_slow_candidate_score", False) else single_token_candidate_ids(
        tokenizer,
        prompt,
        candidates,
    )
    if fast_token_ids is not None:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(args.device)
        with torch.no_grad():
            output = wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                max_loops=max_loops,
                num_trajectories=1,
                particle_update_mode="none",
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
        for loop in loop_counts:
            logits = select_forced_loop_logits(output, loop)
            next_token_logits = logits[0, -1]
            scores_by_loop[loop] = {
                name: float(next_token_logits[token_id].detach().cpu().item())
                for name, token_id in fast_token_ids.items()
            }
        return scores_by_loop

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    for name, completion in candidates.items():
        encoded = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=True).to(args.device)
        labels = encoded["input_ids"].clone()
        labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
        with torch.no_grad():
            output = wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                max_loops=max_loops,
                num_trajectories=1,
                particle_update_mode="none",
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
        for loop in loop_counts:
            logits = select_forced_loop_logits(output, loop)
            score = sequence_logprobs(logits, labels, normalize=args.normalize_candidate_score)
            scores_by_loop[loop][name] = float(score.detach().cpu().item())
    return scores_by_loop


def summarize_active_rows(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    active_counts: dict[tuple[int, int], list[int]] = {}
    above = {"iterate": 0, "hold": 0, "other": 0, "unknown_iterate_target": 0, "n": 0}
    for row in rows:
        depth = int(row["depth"])
        loop = int(row["forced_loop_count"])
        if bool(row["active_cell"]):
            hits = active_counts.setdefault((depth, loop), [0, 0])
            hits[0] += int(bool(row["hit"]))
            hits[1] += 1
            continue
        above["n"] += 1
        behavior = str(row.get("above_diagonal_behavior") or "other")
        if behavior in above:
            above[behavior] += 1
        else:
            above["other"] += 1

    depths = sorted({depth for depth, _ in active_counts})
    loops = sorted({loop for _, loop in active_counts})
    matrix: dict[str, dict[str, dict[str, float | int]]] = {}
    for depth in depths:
        matrix[str(depth)] = {}
        for loop in loops:
            correct, total = active_counts.get((depth, loop), [0, 0])
            matrix[str(depth)][str(loop)] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }

    diagonal = {
        str(depth): matrix.get(str(depth), {}).get(str(depth), {"accuracy": 0.0})["accuracy"]
        for depth in depths
    }
    active_total = {
        "correct": sum(value[0] for value in active_counts.values()),
        "total": sum(value[1] for value in active_counts.values()),
    }
    active_total["accuracy"] = active_total["correct"] / active_total["total"] if active_total["total"] else 0.0
    above_rates = {
        key: (value / above["n"] if above["n"] else 0.0)
        for key, value in above.items()
        if key != "n"
    }
    return {
        "kind": "synthetic_depth_active_label_matrix",
        "threshold": threshold,
        "depths": depths,
        "loops": loops,
        "active_matrix": matrix,
        "active_diagonal": diagonal,
        "active_diagonal_clears_bar": bool(diagonal) and all(float(value) >= threshold for value in diagonal.values()),
        "active_total": active_total,
        "above_diagonal": {**above, "rates": above_rates},
    }


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.data_jsonl)
    loop_counts = [int(item) for item in args.loop_counts.split(",") if item.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(args, args.checkpoint)
    output_rows: list[dict[str, Any]] = []
    total_rows = len(rows)
    for row_index, row in enumerate(rows, start=1):
        if args.progress_every and (row_index == 1 or row_index % args.progress_every == 0 or row_index == total_rows):
            print(
                f"active_eval_progress row={row_index}/{total_rows} depth={row.get('depth')}",
                flush=True,
            )
        prompt = prompt_for_row(row, prediction_space=args.prediction_space, prompt_style=args.prompt_style)
        candidates = candidates_for_row(row, prediction_space=args.prediction_space, value_prefix=args.value_prefix)
        scores_by_loop = score_candidates_all_loops(
            wrapper,
            tokenizer,
            prompt,
            candidates,
            args,
            loop_counts=loop_counts,
        )
        depth = int(row["depth"])
        final_symbol = symbol(row["target"], prefix=args.value_prefix, row_symbols=row_symbol_names(row))
        final_label = str(row.get("answer", "")).strip()
        for loop in loop_counts:
            scores = scores_by_loop[loop]
            prediction = max(scores.items(), key=lambda item: item[1])[0]
            target = active_target_for_loop(
                row,
                loop,
                prediction_space=args.prediction_space,
                value_prefix=args.value_prefix,
            )
            active_cell = target is not None
            behavior = None
            if not active_cell:
                if args.prediction_space == "choice_labels":
                    value_by_label = {str(label): str(value) for label, value in (row.get("choices") or {}).items()}
                    predicted_value = value_by_label.get(prediction)
                    continued = continued_symbol_for_loop(row, loop, value_prefix=args.value_prefix)
                    if continued is None:
                        behavior = "unknown_iterate_target"
                    elif predicted_value == continued:
                        behavior = "iterate"
                    elif prediction == final_label:
                        behavior = "hold"
                    else:
                        behavior = "other"
                else:
                    continued = continued_symbol_for_loop(row, loop, value_prefix=args.value_prefix)
                    if continued is None:
                        behavior = "unknown_iterate_target"
                    elif prediction == continued:
                        behavior = "iterate"
                    elif prediction == final_symbol:
                        behavior = "hold"
                    else:
                        behavior = "other"
            out = {
                "id": row.get("id") or row.get("instance_id"),
                "depth": depth,
                "forced_loop_count": loop,
                "prediction": prediction,
                "target": target,
                "active_cell": active_cell,
                "hit": bool(active_cell and prediction == target),
                "scores": scores,
                "prediction_space": args.prediction_space,
                "above_diagonal_behavior": behavior,
            }
            for passthrough_key in (
                "instance_id",
                "paired_instance_id",
                "verbal_surface_family",
                "template_variant",
                "symbol_names",
            ):
                if passthrough_key in row:
                    out[passthrough_key] = row[passthrough_key]
            output_rows.append(out)
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--loop_counts", default="1,2,3,4")
    parser.add_argument("--threshold", type=float, default=0.71)
    parser.add_argument("--prediction_space", choices=("choice_labels", "full_symbols"), default="full_symbols")
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="question_only")
    parser.add_argument("--value_prefix", default="")
    parser.add_argument("--normalize_candidate_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_slow_candidate_score", action="store_true")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", choices=("concat", "split"), default="concat")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).write_text("", encoding="utf-8")
    rows = evaluate(args)
    with Path(args.output_jsonl).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = summarize_active_rows(rows, threshold=args.threshold)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "prediction_space": args.prediction_space,
            "prompt_style": args.prompt_style,
            "rows": len(rows),
        }
    )
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
