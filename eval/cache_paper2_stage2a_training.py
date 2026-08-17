"""Build the score-blind Stage 2A training population and teacher lattice."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p31_references import (
    MODEL_SPECS,
    _chat_prompt,
    _extract_code,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
    _normalize_number,
)


TEACHER_KEY = "teacher_14b"
STUDENT_KEY = "base"
TEACHER_TOP_K = 128
MAX_ANSWER_TOKENS = 128


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TeacherForcedExample:
    battery: str
    item_id: str
    content_sha256: str
    text: str
    span_start: int
    span_end: int
    strict_boundaries: bool
    owner_slot: int | None


def _last_span(text: str, value: str) -> tuple[int, int]:
    start = text.rfind(value)
    if start < 0:
        raise ValueError("answer-bearing value is absent from the teacher response")
    return start, start + len(value)


def _gsm8k_registered_span(text: str, prediction: str) -> tuple[int, int]:
    """Return the numeric span selected by the registered GSM8K reader."""

    matches = list(re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", text))
    if not matches:
        raise ValueError("GSM8K teacher response has no numeric reader target")
    match = matches[-1]
    selected = _normalize_number(match.group(0))
    if selected != _normalize_number(prediction):
        raise ValueError(
            "GSM8K registered reader span does not match the cached prediction: "
            f"selected={selected!r} prediction={prediction!r}"
        )
    return match.start(), match.end()


def build_teacher_forced_example(
    source: Mapping[str, Any],
    teacher: Mapping[str, Any],
    *,
    tokenizer: Any,
    owner_slot: int | None,
) -> TeacherForcedExample:
    """Reconstruct the registered teacher-forced sequence and answer span."""

    battery = str(source["battery"])
    if teacher.get("correct") is not True:
        raise ValueError("teacher-forced examples must be 14B-correct")
    if str(teacher.get("revision")) != str(MODEL_SPECS[TEACHER_KEY]["revision"]):
        raise ValueError("teacher-forced example revision changed")

    if battery == "arc_challenge":
        question, choices, _answer = _mcq(source)
        prompt = _mcq_prompt(question, choices)
        target = str(teacher["prediction"])
        text = prompt + " " + target
        start, end = _last_span(text, target)
        # Qwen commonly tokenizes the answer label with adjacent whitespace.
        # Only MBPP's code-span contract requires exact token boundaries.
        strict = False
    elif battery == "gsm8k":
        prompt = _chat_prompt(tokenizer, _generation_prompt(source)[0])
        generated = str(teacher["generated_text"])
        target = str(teacher["prediction"])
        text = prompt + generated
        local_start, local_end = _gsm8k_registered_span(generated, target)
        start, end = len(prompt) + local_start, len(prompt) + local_end
        strict = False
    elif battery == "mbpp":
        prompt = _chat_prompt(tokenizer, _generation_prompt(source)[0])
        generated = str(teacher["generated_text"])
        code = _extract_code(generated)
        if code != str(teacher["prediction"]):
            raise ValueError("MBPP executed code differs from the cached teacher prediction")
        text = prompt + generated
        local_start, local_end = _last_span(generated, code)
        start, end = len(prompt) + local_start, len(prompt) + local_end
        strict = True
    else:
        raise ValueError(f"unsupported Stage 2A training battery: {battery}")
    return TeacherForcedExample(
        battery=battery,
        item_id=str(source["item_id"]),
        content_sha256=str(source["content_sha256"]),
        text=text,
        span_start=start,
        span_end=end,
        strict_boundaries=strict,
        owner_slot=owner_slot,
    )


def answer_token_positions(
    offsets: Sequence[Sequence[int]],
    *,
    span_start: int,
    span_end: int,
    strict_boundaries: bool,
) -> tuple[list[int], bool]:
    """Map a character span to causal-input token positions.

    MBPP uses strict boundaries. ARC and GSM8K permit the tokenizer token that
    carries adjacent whitespace or punctuation, because their registered
    targets are short answer strings inside a formatted response.
    """

    if not 0 <= int(span_start) < int(span_end):
        raise ValueError("invalid answer character span")
    positions = [
        index
        for index, pair in enumerate(offsets)
        if int(pair[1]) > int(pair[0])
        and int(pair[1]) > int(span_start)
        and int(pair[0]) < int(span_end)
    ]
    if not positions:
        return [], False
    first = offsets[positions[0]]
    last = offsets[positions[-1]]
    stable = int(first[0]) == int(span_start) and int(last[1]) == int(span_end)
    if strict_boundaries and not stable:
        return [], False
    return positions, stable


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["battery"]), str(row["item_id"]), str(row["content_sha256"])


def build_population(
    *,
    firm_rows: Sequence[Mapping[str, Any]],
    memory_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    teacher_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[TeacherForcedExample], list[dict[str, Any]], dict[str, Any]]:
    source = {_identity(row): row for row in source_rows}
    teacher = {_identity(row): row for row in teacher_rows}
    owners = {_identity(row): index for index, row in enumerate(memory_rows)}
    admitted = [
        row
        for row in firm_rows
        if bool(row.get("stage2a_firm_knowledge_admitted"))
        and str(row.get("partition")) == "verified_train"
    ]
    examples: list[TeacherForcedExample] = []
    owner_rows: list[dict[str, Any]] = []
    for row in admitted:
        key = _identity(row)
        if key not in source or key not in teacher:
            raise RuntimeError(f"admitted row is absent from source or teacher cache: {key}")
        slot = owners.get(key)
        example = build_teacher_forced_example(
            source[key], teacher[key], tokenizer=tokenizer, owner_slot=slot
        )
        examples.append(example)
        owner_rows.append(
            {
                "battery": key[0],
                "item_id": key[1],
                "content_sha256": key[2],
                "owns_memory_slot": slot is not None,
                "owner_slot": slot,
                "retrieval_contract": "leave_one_out" if slot is not None else "unrestricted",
            }
        )
    if len(examples) != len({_identity(row) for row in admitted}):
        raise RuntimeError("Stage 2A admitted training population contains duplicates")
    receipt = {
        "kind": "paper2_stage2a_training_population_v1",
        "status": "all_admitted_nondev_outside_validation",
        "rows": len(examples),
        "owners": sum(row["owns_memory_slot"] for row in owner_rows),
        "non_owners": sum(not row["owns_memory_slot"] for row in owner_rows),
        "identity_sha256": canonical_sha256(
            [
                {
                    "battery": example.battery,
                    "item_id": example.item_id,
                    "content_sha256": example.content_sha256,
                }
                for example in examples
            ]
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    return examples, owner_rows, receipt


@torch.inference_mode()
def cache_teacher_lattice(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[TeacherForcedExample],
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    row_manifest: list[dict[str, Any]] = []
    flat_targets: list[torch.Tensor] = []
    flat_top_ids: list[torch.Tensor] = []
    flat_top_logits: list[torch.Tensor] = []
    offsets = [0]
    boundary_failures = 0
    truncated_rows = 0
    mbpp_rows = 0
    mbpp_boundary_failures = 0
    mbpp_truncated = 0

    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start : start + int(batch_size)]
        encoded = tokenizer(
            [example.text for example in batch],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=True,
        )
        offset_mapping = encoded.pop("offset_mapping")
        positions_by_row: list[list[int]] = []
        truncated_by_row: list[bool] = []
        keep_rows: list[int] = []
        for local, example in enumerate(batch):
            positions, stable = answer_token_positions(
                offset_mapping[local].tolist(),
                span_start=example.span_start,
                span_end=example.span_end,
                strict_boundaries=example.strict_boundaries,
            )
            if example.battery == "mbpp":
                mbpp_rows += 1
            if not positions:
                boundary_failures += 1
                if example.battery == "mbpp":
                    mbpp_boundary_failures += 1
                positions_by_row.append([])
                truncated_by_row.append(False)
                continue
            was_truncated = len(positions) > MAX_ANSWER_TOKENS
            if was_truncated:
                positions = positions[:MAX_ANSWER_TOKENS]
                truncated_rows += 1
                if example.battery == "mbpp":
                    mbpp_truncated += 1
            if min(positions) < 1:
                raise RuntimeError("Stage 2A answer mask reached position zero")
            positions_by_row.append(positions)
            truncated_by_row.append(was_truncated)
            keep_rows.append(local)

        if not keep_rows:
            continue
        logits_positions = sorted(
            {position - 1 for local in keep_rows for position in positions_by_row[local]}
        )
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        output = model(
            **model_inputs,
            use_cache=False,
            return_dict=True,
            logits_to_keep=torch.tensor(logits_positions, device=device, dtype=torch.long),
        )
        position_index = {position: index for index, position in enumerate(logits_positions)}
        for local, example in enumerate(batch):
            positions = positions_by_row[local]
            if not positions:
                continue
            selected = torch.stack(
                [output.logits[local, position_index[position - 1]] for position in positions]
            ).float()
            values, token_ids = torch.topk(selected, k=TEACHER_TOP_K, dim=-1)
            targets = encoded["input_ids"][local, positions]
            row_start = offsets[-1]
            row_stop = row_start + len(positions)
            flat_targets.append(targets.to(torch.int32).cpu())
            flat_top_ids.append(token_ids.to(torch.int32).cpu())
            flat_top_logits.append(values.to(torch.bfloat16).cpu())
            offsets.append(row_stop)
            row_manifest.append(
                {
                    "battery": example.battery,
                    "item_id": example.item_id,
                    "content_sha256": example.content_sha256,
                    "row_index": len(row_manifest),
                    "position_start": row_start,
                    "position_stop": row_stop,
                    "answer_tokens": len(positions),
                    "boundary_exact": bool(
                        int(offset_mapping[local, positions[0], 0]) == example.span_start
                        and int(offset_mapping[local, positions[-1], 1]) == example.span_end
                    ),
                    "truncated": truncated_by_row[local],
                    "owns_memory_slot": example.owner_slot is not None,
                    "owner_slot": example.owner_slot,
                }
            )
        print(
            f"stage2a_teacher_lattice_progress rows={min(start + len(batch), len(examples))}/{len(examples)} "
            f"cached={len(row_manifest)} positions={offsets[-1]}",
            flush=True,
        )

    if not row_manifest:
        raise RuntimeError("Stage 2A teacher lattice contains no rows")
    artifact = {
        "kind": "paper2_stage2a_teacher_lattice_v1",
        "teacher_model": MODEL_SPECS[TEACHER_KEY]["model"],
        "teacher_revision": MODEL_SPECS[TEACHER_KEY]["revision"],
        "temperature": 1.0,
        "top_k": TEACHER_TOP_K,
        "row_offsets": torch.tensor(offsets, dtype=torch.int64),
        "teacher_token_ids": torch.cat(flat_targets, dim=0),
        "teacher_topk_token_ids": torch.cat(flat_top_ids, dim=0),
        "teacher_topk_logits": torch.cat(flat_top_logits, dim=0),
    }
    receipt = {
        "kind": "paper2_stage2a_teacher_lattice_receipt_v1",
        "status": "complete_score_blind_pre_training",
        "source_rows": len(examples),
        "cached_rows": len(row_manifest),
        "cached_positions": offsets[-1],
        "boundary_failures": boundary_failures,
        "truncated_rows": truncated_rows,
        "mbpp": {
            "source_rows": mbpp_rows,
            "boundary_failures": mbpp_boundary_failures,
            "truncated_rows": mbpp_truncated,
        },
        "optimizer_constructed": False,
        "training_started": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return artifact, row_manifest, receipt


@torch.inference_mode()
def cache_student_prefix_features(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[TeacherForcedExample],
    teacher_manifest: Sequence[Mapping[str, Any]],
    teacher_token_ids: torch.Tensor,
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cache frozen 0.5B states needed by the registered training graph."""

    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    expected = {
        (str(row["battery"]), str(row["item_id"]), str(row["content_sha256"])): row
        for row in teacher_manifest
    }
    ordered = [
        example
        for example in examples
        if (example.battery, example.item_id, example.content_sha256) in expected
    ]
    order_index = {
        (str(row["battery"]), str(row["item_id"]), str(row["content_sha256"])): index
        for index, row in enumerate(teacher_manifest)
    }
    ordered.sort(
        key=lambda example: order_index[
            (example.battery, example.item_id, example.content_sha256)
        ]
    )
    if len(ordered) != len(teacher_manifest):
        raise RuntimeError("student feature population differs from teacher lattice")

    sequence_offsets = [0]
    answer_offsets = [0]
    flat_input_ids: list[torch.Tensor] = []
    flat_final_hidden: list[torch.Tensor] = []
    flat_layer6_queries: list[torch.Tensor] = []
    flat_current_hidden: list[torch.Tensor] = []
    flat_prefix_positions: list[torch.Tensor] = []
    flat_student_targets: list[torch.Tensor] = []

    for start in range(0, len(ordered), int(batch_size)):
        batch = ordered[start : start + int(batch_size)]
        encoded = tokenizer(
            [example.text for example in batch],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=True,
        )
        offsets = encoded.pop("offset_mapping")
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        output = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        layer6 = output.hidden_states[6]
        final = output.hidden_states[-1]
        for local, example in enumerate(batch):
            positions, _stable = answer_token_positions(
                offsets[local].tolist(),
                span_start=example.span_start,
                span_end=example.span_end,
                strict_boundaries=example.strict_boundaries,
            )
            positions = positions[:MAX_ANSWER_TOKENS]
            key = (example.battery, example.item_id, example.content_sha256)
            expected_count = int(expected[key]["answer_tokens"])
            if len(positions) != expected_count:
                raise RuntimeError(f"student/teacher answer-mask mismatch: {key}")
            prefix_positions = torch.tensor(
                [position - 1 for position in positions], dtype=torch.int32
            )
            prefix_stop = int(max(positions))
            if prefix_stop < 1:
                raise RuntimeError("student feature cache reached position zero")
            flat_input_ids.append(encoded["input_ids"][local, :prefix_stop].to(torch.int32))
            flat_final_hidden.append(final[local, :prefix_stop].to(torch.bfloat16).cpu())
            flat_layer6_queries.append(
                layer6[local, prefix_positions.long()].to(torch.bfloat16).cpu()
            )
            flat_current_hidden.append(
                final[local, prefix_positions.long()].to(torch.bfloat16).cpu()
            )
            flat_prefix_positions.append(prefix_positions)
            flat_student_targets.append(
                encoded["input_ids"][local, positions].to(torch.int32)
            )
            sequence_offsets.append(sequence_offsets[-1] + prefix_stop)
            answer_offsets.append(answer_offsets[-1] + len(positions))
        print(
            f"stage2a_student_feature_progress rows={min(start + len(batch), len(ordered))}/{len(ordered)} "
            f"tokens={sequence_offsets[-1]} positions={answer_offsets[-1]}",
            flush=True,
        )

    observed_targets = torch.cat(flat_student_targets, dim=0)
    if not torch.equal(observed_targets.cpu(), teacher_token_ids.to(torch.int32).cpu()):
        raise RuntimeError("0.5B and 14B tokenizers disagree on Stage 2A target IDs")
    artifact = {
        "kind": "paper2_stage2a_student_prefix_features_v1",
        "student_model": MODEL_SPECS[STUDENT_KEY]["model"],
        "student_revision": MODEL_SPECS[STUDENT_KEY]["revision"],
        "layer6_index": 6,
        "final_layer_index": -1,
        "sequence_offsets": torch.tensor(sequence_offsets, dtype=torch.int64),
        "answer_offsets": torch.tensor(answer_offsets, dtype=torch.int64),
        "input_ids": torch.cat(flat_input_ids, dim=0),
        "final_hidden": torch.cat(flat_final_hidden, dim=0),
        "layer6_queries": torch.cat(flat_layer6_queries, dim=0),
        "current_hidden": torch.cat(flat_current_hidden, dim=0),
        "answer_prefix_positions": torch.cat(flat_prefix_positions, dim=0),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    receipt = {
        "kind": "paper2_stage2a_student_prefix_features_receipt_v1",
        "status": "complete_frozen_base_cache",
        "rows": len(ordered),
        "sequence_tokens": sequence_offsets[-1],
        "answer_positions": answer_offsets[-1],
        "layer6": 6,
        "final_layer": -1,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return artifact, receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_rows", type=Path, required=True)
    parser.add_argument("--teacher_14b_scores", type=Path, required=True)
    parser.add_argument("--firm_manifest", type=Path, required=True)
    parser.add_argument("--memory_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[TEACHER_KEY]
    teacher_cache = args.model_cache / "teacher_14b"
    student_cache = args.model_cache / "student_0p5b"
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["revision"], cache_dir=teacher_cache
    )
    examples, owner_rows, population_receipt = build_population(
        firm_rows=read_jsonl(args.firm_manifest),
        memory_rows=read_jsonl(args.memory_manifest),
        source_rows=read_jsonl(args.source_rows),
        teacher_rows=read_jsonl(args.teacher_14b_scores),
        tokenizer=tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        cache_dir=teacher_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    artifact, manifest, receipt = cache_teacher_lattice(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=args.batch_size,
        device=args.device,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    owner_by_identity = {
        (str(row["battery"]), str(row["item_id"]), str(row["content_sha256"])): row
        for row in owner_rows
    }
    filtered_owners = [
        owner_by_identity[
            (str(row["battery"]), str(row["item_id"]), str(row["content_sha256"]))
        ]
        for row in manifest
    ]
    owner_path = args.private_dir / "stage2a_memory_owner_manifest.jsonl"
    write_jsonl(owner_path, filtered_owners)
    population_path = args.private_dir / "stage2a_training_population.jsonl"
    write_jsonl(
        population_path,
        [
            {
                "battery": row["battery"],
                "item_id": row["item_id"],
                "content_sha256": row["content_sha256"],
                "owns_memory_slot": row["owns_memory_slot"],
                "owner_slot": row["owner_slot"],
            }
            for row in filtered_owners
        ],
    )
    population_receipt.update(
        {
            "admitted_rows_before_answer_mask": population_receipt["rows"],
            "rows": len(manifest),
            "dropped_answer_boundary_rows": population_receipt["rows"] - len(manifest),
            "owners": sum(row["owns_memory_slot"] for row in filtered_owners),
            "non_owners": sum(not row["owns_memory_slot"] for row in filtered_owners),
            "identity_sha256": canonical_sha256(
                [
                    {
                        "battery": row["battery"],
                        "item_id": row["item_id"],
                        "content_sha256": row["content_sha256"],
                    }
                    for row in manifest
                ]
            ),
        }
    )

    student_spec = MODEL_SPECS[STUDENT_KEY]
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_spec["model"],
        revision=student_spec["revision"],
        cache_dir=student_cache,
    )
    student = AutoModelForCausalLM.from_pretrained(
        student_spec["model"],
        revision=student_spec["revision"],
        cache_dir=student_cache,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    features, feature_receipt = cache_student_prefix_features(
        model=student,
        tokenizer=student_tokenizer,
        examples=examples,
        teacher_manifest=manifest,
        teacher_token_ids=artifact["teacher_token_ids"],
        batch_size=max(1, args.batch_size * 4),
        device=args.device,
    )
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    artifact_path = args.private_dir / "stage2a_teacher_lattice.pt"
    torch.save(artifact, artifact_path)
    manifest_path = args.private_dir / "stage2a_teacher_lattice_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    feature_path = args.private_dir / "stage2a_student_prefix_features.pt"
    torch.save(features, feature_path)
    mbpp_audit_path = args.output_dir / "mbpp_span_audit.json"
    write_json(mbpp_audit_path, receipt["mbpp"])
    receipt.update(
        {
            "population": population_receipt,
            "population_manifest_sha256": sha256_file(population_path),
            "memory_owner_manifest_sha256": sha256_file(owner_path),
            "teacher_lattice_manifest_sha256": sha256_file(manifest_path),
            "teacher_lattice_artifact_sha256": sha256_file(artifact_path),
            "mbpp_span_audit_sha256": sha256_file(mbpp_audit_path),
            "student_feature_cache": feature_receipt,
            "student_feature_cache_sha256": sha256_file(feature_path),
        }
    )
    write_json(args.output_dir / "summary.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
