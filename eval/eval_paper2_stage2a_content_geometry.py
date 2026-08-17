"""Materialize the score-blind Stage 2A memory population and geometry.

This pass is deliberately pre-training. It scores only the verified-train
partition, selects the validation and memory manifests under locked rules,
and extracts frozen prompt states. It never constructs an optimizer or reads
CONFIRM/EVAL-E.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_kp1_t1 import canonical_prompt, last_active
from eval.eval_paper2_phase3_kp1r_t1_teacher import parameter_sentinel_fingerprint
from eval.eval_paper2_phase3_p31_references import score_model
from training.paper2_phase3_kp1r_t1_teacher import stratified_alignment_split
from training.paper2_phase3_p31_completion import sha256_file
from training.sidecar_v2_data_spine import (
    STAGE2A_TEACHER_14B,
    apply_stage2a_firm_knowledge_rule,
    build_fingerprint_memory_manifest,
    fit_nondev_fingerprint_geometry,
    select_stage2a_geometry_population,
    select_stage2a_validation_split,
)


EXPECTED_MERGED_SHA256 = "1aa4391deefaee6e1a70f4e99ac20e0728dae1a4e45fec482f38e76edf5fa54b"
EXPECTED_SOURCE_ROWS_SHA256 = "5e32eb1905b05076a59b2c5b315ccf9319c04eda18af450565128fd34c18ffa5"
EXPECTED_TEACHER_14B_SCORES_SHA256 = (
    "5e9c0d2cd4e4097dc14aefb1d49c5ff74e86c84f44a38296f10ff60014cbb74e"
)
EXPECTED_PANEL_SHA256 = "c0e15a890b598544059ac337cc475123f97c05e3c1626febcdee1c6d8fe02615"
CONFIRM_SEAL_SHA256 = "f404c1dbc2cd13a8937f8e87a1f14bc4f8de5d94b6fab4d11ad82648c6d2eb18"
SELECTION_SEED = 20_260_817


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_digest(row: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(row))


def concurrence_value(row: Mapping[str, Any]) -> str:
    """Return the registered answer representation for family concurrence."""

    battery = str(row["battery"])
    if battery == "mbpp":
        if bool(row["correct"]):
            return "passes_required_tests"
        return f"fails_required_tests:{score_digest(row)}"
    value = row.get("prediction")
    return "<no-normalized-answer>" if value is None else str(value)


def join_sources(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    merged_rows: Sequence[Mapping[str, Any]],
    teacher_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_ids = [str(row["item_id"]) for row in source_rows]
    merged_ids_list = [str(row["item_id"]) for row in merged_rows]
    teacher_ids = [str(row["item_id"]) for row in teacher_scores]
    if (
        len(source_ids) != len(set(source_ids))
        or len(merged_ids_list) != len(set(merged_ids_list))
        or len(teacher_ids) != len(set(teacher_ids))
    ):
        raise RuntimeError("Stage 2A source tables contain duplicate item ids")
    merged_ids = {str(row["item_id"]) for row in merged_rows}
    source = {
        str(row["item_id"]): dict(row)
        for row in source_rows
        if str(row["item_id"]) in merged_ids
    }
    merged = {str(row["item_id"]): dict(row) for row in merged_rows}
    teacher = {str(row["item_id"]): dict(row) for row in teacher_scores}
    if set(source) != set(merged) or set(source) != set(teacher):
        raise RuntimeError("Stage 2A source, merged, and 14B score row sets differ")
    rows = []
    for item_id in sorted(source):
        row = source[item_id] | merged[item_id]
        score = teacher[item_id]
        if bool(row["teacher_14b_correct"]) != bool(score["correct"]):
            raise RuntimeError(f"14B correctness mismatch for {item_id}")
        row.update(
            {
                "teacher_14b_prediction": score.get("prediction"),
                "teacher_14b_normalized_answer": concurrence_value(score),
                "teacher_14b_output_sha256": score_digest(score),
            }
        )
        rows.append(row)
    return rows


def make_firm_rows(
    rows: Sequence[Mapping[str, Any]], verifier_scores: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    verifier = {str(row["item_id"]): dict(row) for row in verifier_scores}
    if len(verifier) != len(verifier_scores):
        raise RuntimeError("32B verifier score table contains duplicate item ids")
    result = []
    for source in rows:
        item_id = str(source["item_id"])
        if item_id not in verifier:
            raise RuntimeError(f"32B verifier lacks candidate {item_id}")
        score = verifier[item_id]
        row = dict(source)
        row.update(
            {
                "teacher_32b_normalized_answer": concurrence_value(score),
                "teacher_32b_output_sha256": score_digest(score),
                "teacher_32b_correct": bool(score["correct"]),
                "correctness_reader": str(source["reader"]),
            }
        )
        result.append(row)
    return result


def _state_cache_valid(path: Path, identities: Sequence[tuple[str, str]], layer: int) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload.get("identities") == list(identities)
        and int(payload.get("layer", -1)) == int(layer)
        and isinstance(payload.get("states"), torch.Tensor)
        and int(payload["states"].shape[0]) == len(identities)
    )


def extract_prompt_states(
    *,
    rows: Sequence[Mapping[str, Any]],
    model_id: str,
    revision: str,
    layer: int,
    batch_size: int,
    cache_dir: Path,
    output_path: Path,
    device: torch.device,
) -> torch.Tensor:
    identities = [(str(row["battery"]), str(row["item_id"])) for row in rows]
    if _state_cache_valid(output_path, identities, layer):
        print(f"stage2a_state_cache_hit path={output_path}", flush=True)
        return torch.load(output_path, map_location="cpu", weights_only=False)["states"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = parameter_sentinel_fingerprint(model)
    prompts = [canonical_prompt(row, tokenizer) for row in rows]
    parts: list[torch.Tensor] = []
    for start in range(0, len(rows), int(batch_size)):
        stop = min(len(rows), start + int(batch_size))
        encoded = tokenizer(
            prompts[start:stop], return_tensors="pt", padding=True, add_special_tokens=True
        ).to(device)
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        parts.append(
            last_active(output.hidden_states[int(layer)], encoded["attention_mask"])
            .detach()
            .cpu()
            .to(torch.float16)
        )
        if stop % 256 == 0 or stop == len(rows):
            print(
                f"stage2a_state_progress model={model_id} layer={layer} rows={stop}/{len(rows)}",
                flush=True,
            )
    if before != parameter_sentinel_fingerprint(model):
        raise RuntimeError(f"frozen state extraction mutated {model_id}")
    states = torch.cat(parts, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "paper2_stage2a_prompt_state_cache_v1",
            "model": model_id,
            "revision": revision,
            "layer": int(layer),
            "state_position": "last_active_prompt_token",
            "identities": identities,
            "states": states,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        },
        output_path,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_rows", type=Path, required=True)
    parser.add_argument("--merged_rows", type=Path, required=True)
    parser.add_argument("--teacher_14b_scores", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--verifier_mcq_batch_size", type=int, default=4)
    parser.add_argument("--verifier_generation_batch_size", type=int, default=2)
    parser.add_argument("--student_state_batch_size", type=int, default=32)
    parser.add_argument("--teacher_state_batch_size", type=int, default=8)
    args = parser.parse_args()

    if sha256_file(args.source_rows) != EXPECTED_SOURCE_ROWS_SHA256:
        raise RuntimeError("Stage 2A partition source SHA changed")
    if sha256_file(args.merged_rows) != EXPECTED_MERGED_SHA256:
        raise RuntimeError("Stage 2A merged reference SHA changed")
    if sha256_file(args.teacher_14b_scores) != EXPECTED_TEACHER_14B_SCORES_SHA256:
        raise RuntimeError("Stage 2A 14B score SHA changed")
    if sha256_file(args.panel) != EXPECTED_PANEL_SHA256:
        raise RuntimeError("Stage 2A DEV panel SHA changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(args.source_rows)
    merged_rows = read_jsonl(args.merged_rows)
    teacher_scores = read_jsonl(args.teacher_14b_scores)
    panel = read_jsonl(args.panel)
    joined = join_sources(
        source_rows=source_rows, merged_rows=merged_rows, teacher_scores=teacher_scores
    )
    panel_ids = {(str(row["battery"]), str(row["item_id"])) for row in panel}
    validation, validation_receipt = select_stage2a_validation_split(
        joined, panel_item_ids=panel_ids, count=512, seed=SELECTION_SEED
    )
    validation_ids = {(str(row["battery"]), str(row["item_id"])) for row in validation}
    validation_path = args.private_dir / "stage2a_validation_manifest.jsonl"
    write_jsonl(validation_path, validation)
    validation_receipt["path"] = str(validation_path)
    validation_receipt["sha256"] = sha256_file(validation_path)
    write_json(args.output_dir / "validation_receipt.json", validation_receipt)

    verifier_candidates = [
        row
        for row in joined
        if bool(row["teacher_14b_correct"])
        and str(row["partition"]) == "verified_train"
        and (str(row["battery"]), str(row["item_id"])) not in validation_ids
    ]
    verifier_source = args.private_dir / "stage2a_verifier_source_rows.jsonl"
    write_jsonl(verifier_source, verifier_candidates)
    verifier_scores_path = args.private_dir / "verifier_32b_scores.jsonl"
    verifier_receipt = score_model(
        verifier_candidates,
        model_key="verifier_32b",
        output_jsonl=verifier_scores_path,
        device="cuda",
        dtype=torch.bfloat16,
        mcq_candidate_batch_size=args.verifier_mcq_batch_size,
        generation_batch_size=args.verifier_generation_batch_size,
        confirm_seal_sha256=CONFIRM_SEAL_SHA256,
    )
    verifier_scores = read_jsonl(verifier_scores_path)
    firm_rows, firm_receipt = apply_stage2a_firm_knowledge_rule(
        make_firm_rows(verifier_candidates, verifier_scores)
    )
    firm_path = args.private_dir / "stage2a_firm_knowledge_manifest.jsonl"
    write_jsonl(firm_path, firm_rows)
    firm_receipt.update({"path": str(firm_path), "sha256": sha256_file(firm_path)})
    write_json(args.output_dir / "firm_knowledge_receipt.json", firm_receipt)

    memory_rows, memory_receipt = build_fingerprint_memory_manifest(
        firm_rows,
        panel_item_ids=panel_ids,
        reserved_item_ids=validation_ids,
        admitted_field="stage2a_firm_knowledge_admitted",
        slots=None,
        seed=SELECTION_SEED,
    )
    memory_path = args.private_dir / "stage2a_memory_manifest.jsonl"
    write_jsonl(memory_path, memory_rows)
    memory_receipt.update({"path": str(memory_path), "sha256": sha256_file(memory_path)})
    write_json(args.output_dir / "memory_manifest_receipt.json", memory_receipt)

    geometry_population, geometry_population_receipt = select_stage2a_geometry_population(
        memory_rows, count=min(1_024, len(memory_rows)), seed=SELECTION_SEED
    )
    geometry_ids = [str(row["item_id"]) for row in geometry_population]
    geometry_batteries = [str(row["battery"]) for row in geometry_population]
    geometry_split = stratified_alignment_split(
        geometry_ids, geometry_batteries, seed=SELECTION_SEED, fit_fraction=0.8
    )
    fit_mask = torch.tensor([value == "alignment_fit" for value in geometry_split])
    holdout_mask = ~fit_mask
    fit_manifest = [
        {
            "battery": str(row["battery"]),
            "item_id": str(row["item_id"]),
            "content_sha256": str(row["content_sha256"]),
            "split": split,
        }
        for row, split in zip(geometry_population, geometry_split, strict=True)
    ]
    fit_manifest_path = args.private_dir / "stage2a_geometry_fit_manifest.jsonl"
    write_jsonl(fit_manifest_path, fit_manifest)

    device = torch.device("cuda")
    student_states = extract_prompt_states(
        rows=memory_rows,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        layer=6,
        batch_size=args.student_state_batch_size,
        cache_dir=args.model_cache / "student_0p5b",
        output_path=args.private_dir / "student_layer6_prompt_states.pt",
        device=device,
    )
    teacher_states = extract_prompt_states(
        rows=memory_rows,
        model_id=STAGE2A_TEACHER_14B["model"],
        revision=STAGE2A_TEACHER_14B["revision"],
        layer=12,
        batch_size=args.teacher_state_batch_size,
        cache_dir=args.model_cache / "teacher_14b",
        output_path=args.private_dir / "teacher_layer12_prompt_states.pt",
        device=device,
    )
    memory_index = {
        (str(row["battery"]), str(row["item_id"])): index
        for index, row in enumerate(memory_rows)
    }
    geometry_indexes = torch.tensor(
        [memory_index[(str(row["battery"]), str(row["item_id"]))] for row in geometry_population],
        dtype=torch.long,
    )
    geometry_student = student_states[geometry_indexes]
    geometry_teacher = teacher_states[geometry_indexes]
    geometry, geometry_receipt = fit_nondev_fingerprint_geometry(
        student_fit=geometry_student[fit_mask],
        teacher_fit=geometry_teacher[fit_mask],
        student_holdout=geometry_student[holdout_mask],
        teacher_holdout=geometry_teacher[holdout_mask],
        rank=128,
    )
    artifact_path = args.private_dir / "stage2a_memory_geometry.pt"
    torch.save(
        {
            "kind": "paper2_stage2a_memory_geometry_v1",
            "memory_identities": [
                (str(row["battery"]), str(row["item_id"])) for row in memory_rows
            ],
            "memory_keys": geometry.student_keys(student_states).to(torch.float16),
            "teacher_values": geometry.teacher_values(teacher_states).to(torch.float16),
            "student_mean": geometry.student_mean,
            "student_basis": geometry.student_basis,
            "teacher_mean": geometry.teacher_mean,
            "teacher_basis": geometry.teacher_basis,
            "diagnostic_rotation": geometry.diagnostic_rotation,
            "state_position": "last_active_prompt_token",
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        },
        artifact_path,
    )
    geometry_receipt.update(
        {
            "fit_manifest_path": str(fit_manifest_path),
            "fit_manifest_sha256": sha256_file(fit_manifest_path),
            "population": geometry_population_receipt,
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "fit_rows_by_battery": dict(
                sorted(Counter(row["battery"] for row in fit_manifest if row["split"] == "alignment_fit").items())
            ),
            "holdout_rows_by_battery": dict(
                sorted(Counter(row["battery"] for row in fit_manifest if row["split"] == "alignment_eval").items())
            ),
        }
    )
    write_json(args.output_dir / "geometry_receipt.json", geometry_receipt)

    summary = {
        "kind": "paper2_stage2a_content_geometry_summary_v1",
        "status": "complete_score_blind_pre_signature",
        "source": {
            "partition_rows_sha256": sha256_file(args.source_rows),
            "merged_sha256": sha256_file(args.merged_rows),
            "teacher_14b_scores_sha256": sha256_file(args.teacher_14b_scores),
            "panel_sha256": sha256_file(args.panel),
        },
        "validation": validation_receipt,
        "verifier_32b": verifier_receipt,
        "firm_knowledge": firm_receipt,
        "concurrence_readers": {
            "arc_challenge": "same cyclic-reader answer label",
            "gsm8k": "same normalized final number",
            "mbpp": "both programs pass the identical required tests",
        },
        "memory": memory_receipt,
        "geometry": geometry_receipt,
        "training_authorized": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "dev_scores_computed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
