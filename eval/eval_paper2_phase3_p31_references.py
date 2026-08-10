"""Score Phase 3.1 DEV and verified-train rows with frozen base and 14B models.

CONFIRM is rejected before model loading.  Results are resumable JSONL files;
each model is loaded and released separately so the pass fits one A100 session.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.paper2_phase3_p31_completion import sha256_file


MODEL_SPECS = {
    "base": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    },
    "teacher_14b": {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _mcq(row: Mapping[str, Any]) -> tuple[str, list[tuple[str, str]], str]:
    prompt = row["prompt"]
    if row["battery"] in {"arc_easy", "arc_challenge"}:
        labels = [str(value) for value in prompt["choice_labels"]]
        texts = [str(value) for value in prompt["choice_text"]]
        question = str(prompt["question"])
        answer = str(row["answer"])
    elif row["battery"] == "mmlu":
        labels = ["A", "B", "C", "D"]
        texts = [str(value) for value in prompt["choices"]]
        question = str(prompt["question"])
        answer = labels[int(row["answer"])]
    else:
        raise ValueError(f"not an MCQ row: {row['battery']}")
    return question, list(zip(labels, texts)), answer


def _mcq_prompt(question: str, choices: Sequence[tuple[str, str]]) -> str:
    rendered = "\n".join(f"{label}. {text}" for label, text in choices)
    return f"{question.rstrip()}\n{rendered}\nAnswer:"


def _chat_prompt(tokenizer: Any, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _sequence_scores(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted = logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    safe = targets.masked_fill(~mask, 0)
    values = torch.log_softmax(shifted, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


@torch.inference_mode()
def score_mcq_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    candidate_batch_size: int,
) -> list[dict[str, Any]]:
    """Apply Paper One's cyclic-label, permutation-mean MCQ reader."""

    candidates = []
    metadata = []
    answers = {}
    for row in rows:
        question, choices, answer = _mcq(row)
        item_id = str(row["item_id"])
        answers[item_id] = answer
        labels = [label for label, _text in choices]
        for shift in range(len(choices)):
            permuted = []
            label_map = {}
            for new_index, new_label in enumerate(labels):
                original_label, original_text = choices[(new_index - shift) % len(choices)]
                permuted.append((new_label, original_text))
                label_map[new_label] = original_label
            prompt = _mcq_prompt(question, permuted)
            for new_label in labels:
                candidates.append(prompt + f" {new_label}")
                metadata.append(
                    (item_id, label_map[new_label], new_label, shift, prompt)
                )
    option_score_lists: dict[str, dict[str, list[float]]] = {}
    for start in range(0, len(candidates), candidate_batch_size):
        stop = min(len(candidates), start + candidate_batch_size)
        encoded = tokenizer(
            candidates[start:stop],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)
        labels = encoded["input_ids"].clone()
        labels = labels.masked_fill(encoded["attention_mask"].eq(0), -100)
        for local, (_item_id, _original_label, _new_label, _shift, prompt) in enumerate(
            metadata[start:stop]
        ):
            prompt_length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            labels[local, :prompt_length] = -100
        output = model(**encoded, use_cache=False, return_dict=True)
        scores = _sequence_scores(output.logits, labels).cpu().tolist()
        for (item_id, original_label, _new_label, _shift, _prompt), score in zip(
            metadata[start:stop], scores
        ):
            option_score_lists.setdefault(item_id, {}).setdefault(
                original_label, []
            ).append(float(score))
    results = []
    for row in rows:
        item_id = str(row["item_id"])
        score_lists = option_score_lists[item_id]
        scores = {
            label: sum(values) / len(values)
            for label, values in score_lists.items()
        }
        prediction = max(scores, key=scores.get)
        permutation_count = min(len(values) for values in score_lists.values())
        results.append(
            {
                "item_id": item_id,
                "prediction": prediction,
                "correct": prediction == answers[item_id],
                "option_scores": scores,
                "reader": "cyclic_label_aggregated_permutation_mean_v1",
                "num_permutations": permutation_count,
            }
        )
    return results


def _generation_prompt(row: Mapping[str, Any]) -> tuple[str, int]:
    if row["battery"] == "gsm8k":
        return (
            "Solve the problem carefully. End with a line of the form 'Final answer: <number>'.\n\n"
            + str(row["prompt"]),
            256,
        )
    if row["battery"] == "mbpp":
        tests = "\n".join(str(test) for test in row["tests"])
        return (
            "Write Python code that solves the task. Return only one Python code block.\n\n"
            f"Task: {row['prompt']}\n\nRequired tests:\n{tests}",
            384,
        )
    if row["battery"] == "tier1":
        return (f"Answer concisely.\n\n{row['prompt']}", 64)
    raise ValueError(f"not a generation row: {row['battery']}")


@torch.inference_mode()
def generate_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    batch_size: int,
) -> list[tuple[Mapping[str, Any], str]]:
    output = []
    by_cap: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        _, cap = _generation_prompt(row)
        by_cap.setdefault(cap, []).append(row)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for cap, selected in sorted(by_cap.items()):
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            prompts = [_chat_prompt(tokenizer, _generation_prompt(row)[0]) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=cap,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            padding_width = encoded["input_ids"].shape[1]
            for row, tokens, _length in zip(batch, generated, lengths):
                text = tokenizer.decode(tokens[padding_width:], skip_special_tokens=True)
                output.append((row, text))
            print(
                f"phase3_p31_generation model={model.config.name_or_path} "
                f"battery={batch[0]['battery']} rows={min(start + len(batch), len(selected))}/{len(selected)}",
                flush=True,
            )
    return output


def _normalize_number(value: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.12g}"


def _extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fenced[-1] if fenced else text).strip()


def _execute_mbpp(code: str, tests: Sequence[str], *, timeout_seconds: int = 6) -> bool:
    script = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory(prefix="phase3_mbpp_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return completed.returncode == 0


def score_generated(row: Mapping[str, Any], text: str) -> tuple[bool, str | None]:
    if row["battery"] == "gsm8k":
        prediction = _normalize_number(text)
        return prediction == _normalize_number(str(row["answer"])), prediction
    if row["battery"] == "mbpp":
        code = _extract_code(text)
        return _execute_mbpp(code, [str(test) for test in row["tests"]]), code
    if row["battery"] == "tier1":
        prediction = _normalize_number(text)
        answer = _normalize_number(str(row["answer"]))
        if answer is not None:
            return prediction == answer, prediction
        normalized = " ".join(text.casefold().split())
        target = " ".join(str(row["answer"]).casefold().split())
        return target in normalized, normalized[-256:]
    raise ValueError(f"unknown generated battery: {row['battery']}")


def score_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    output_jsonl: Path,
    device: str,
    dtype: torch.dtype,
    mcq_candidate_batch_size: int,
    generation_batch_size: int,
    confirm_seal_sha256: str,
) -> dict[str, Any]:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown P3.1 model key {model_key}")
    if any(row["partition"] == "confirm" for row in rows):
        raise RuntimeError("P3.1 scorer received CONFIRM rows")
    spec = MODEL_SPECS[model_key]
    existing_rows = read_jsonl(output_jsonl) if output_jsonl.exists() else []
    if len(existing_rows) != len({str(row["item_id"]) for row in existing_rows}):
        raise RuntimeError(f"P3.1 resumable score file has duplicate item ids: {output_jsonl}")
    source_lookup = {str(row["item_id"]): row for row in rows}
    for existing_row in existing_rows:
        source = source_lookup.get(str(existing_row["item_id"]))
        if source is None:
            raise RuntimeError(f"P3.1 resumable score row is outside the current source: {output_jsonl}")
        if (
            existing_row.get("model_key") != model_key
            or existing_row.get("model") != spec["model"]
            or existing_row.get("revision") != spec["revision"]
            or existing_row.get("reader") != source["reader"]
        ):
            raise RuntimeError(f"P3.1 resumable score lineage or reader changed: {output_jsonl}")
    existing = {str(row["item_id"]): row for row in existing_rows}
    pending = [row for row in rows if str(row["item_id"]) not in existing]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()

    mcq = [row for row in pending if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}]
    generated = [row for row in pending if row["battery"] in {"gsm8k", "mbpp", "tier1"}]
    if mcq:
        results = score_mcq_rows(
            model,
            tokenizer,
            mcq,
            device=device,
            candidate_batch_size=mcq_candidate_batch_size,
        )
        lookup = {str(row["item_id"]): row for row in mcq}
        append_jsonl(
            output_jsonl,
            [
                {
                    "kind": "paper2_phase3_p31_model_score_v1",
                    "model_key": model_key,
                    "model": spec["model"],
                    "revision": spec["revision"],
                    "battery": lookup[result["item_id"]]["battery"],
                    "battery_role": lookup[result["item_id"]]["battery_role"],
                    "partition": lookup[result["item_id"]]["partition"],
                    "document_id": lookup[result["item_id"]]["document_id"],
                    "content_sha256": lookup[result["item_id"]]["content_sha256"],
                    **result,
                }
                for result in results
            ],
        )
    if generated:
        results = []
        for row, text in generate_rows(
            model,
            tokenizer,
            generated,
            device=device,
            batch_size=generation_batch_size,
        ):
            correct, prediction = score_generated(row, text)
            results.append(
                {
                    "kind": "paper2_phase3_p31_model_score_v1",
                    "model_key": model_key,
                    "model": spec["model"],
                    "revision": spec["revision"],
                    "battery": row["battery"],
                    "battery_role": row["battery_role"],
                    "partition": row["partition"],
                    "document_id": row["document_id"],
                    "content_sha256": row["content_sha256"],
                    "item_id": row["item_id"],
                    "reader": row["reader"],
                    "prediction": prediction,
                    "correct": bool(correct),
                    "generated_text": text,
                }
            )
            if len(results) >= 32:
                append_jsonl(output_jsonl, results)
                results = []
        append_jsonl(output_jsonl, results)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final = read_jsonl(output_jsonl)
    if len(final) != len({str(row["item_id"]) for row in final}):
        raise RuntimeError(f"P3.1 resumable score file has duplicate item ids: {output_jsonl}")
    expected = {str(row["item_id"]) for row in rows}
    observed = {str(row["item_id"]) for row in final}
    if observed != expected:
        raise RuntimeError(
            f"P3.1 resumable score coverage mismatch model={model_key} "
            f"missing={len(expected - observed)} extra={len(observed - expected)}"
        )
    return {
        "model_key": model_key,
        "model": spec["model"],
        "revision": spec["revision"],
        "rows": len(final),
        "path": str(output_jsonl),
        "sha256": sha256_file(output_jsonl),
        "confirm_seal_sha256": confirm_seal_sha256,
        "confirm_rows": sum(row["partition"] == "confirm" for row in final),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_key", choices=tuple(MODEL_SPECS), action="append")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--mcq_candidate_batch_size", type=int, default=32)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--confirm_seal_ledger", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads(args.confirm_seal_ledger.read_text(encoding="utf-8"))
    if seal.get("status") != "sealed_before_model_scoring":
        raise RuntimeError("P3.1 scorer requires a valid pre-model CONFIRM seal")
    if not seal.get("assertions", {}).get("confirm_membership_sealed"):
        raise RuntimeError("P3.1 CONFIRM seal assertion is absent or false")
    confirm_seal_sha256 = sha256_file(args.confirm_seal_ledger)
    rows = [row for row in read_jsonl(args.rows_jsonl) if row["partition"] in {"dev", "verified_train"}]
    if not rows or any(row["partition"] == "confirm" for row in rows):
        raise RuntimeError("P3.1 reference input must be nonempty and CONFIRM-free")
    dtype = getattr(torch, args.dtype)
    receipts = []
    for model_key in args.model_key or list(MODEL_SPECS):
        receipts.append(
            score_model(
                rows,
                model_key=model_key,
                output_jsonl=args.output_dir / f"{model_key}_scores.jsonl",
                device=args.device,
                dtype=dtype,
                mcq_candidate_batch_size=args.mcq_candidate_batch_size,
                generation_batch_size=args.generation_batch_size,
                confirm_seal_sha256=confirm_seal_sha256,
            )
        )
    result = {
        "kind": "paper2_phase3_p31_reference_model_score_receipts_v1",
        "status": "complete_dev_and_verified_train_confirm_unscored",
        "models": receipts,
        "confirm_seal_ledger": str(args.confirm_seal_ledger),
        "confirm_seal_sha256": confirm_seal_sha256,
        "scorer_source_sha256": sha256_file(Path(__file__).resolve()),
        "optimizer_steps": 0,
        "confirm_scoring_spent": False,
    }
    write_json(args.output_dir / "model_score_receipts.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
